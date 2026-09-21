"""Explicit lifecycle for recording live strategy decisions.

The framework creates :class:`DecisionRecorder` only for live v0.5 pandas
strategies that enable ``Parameters.record_strategy_inputs``. The strategy owns
the lifecycle: call :meth:`DecisionRecorder.begin` before calculations, record
zero or more explicit observations, then call :meth:`DecisionRecorder.finish`
or :meth:`DecisionRecorder.fail`. The recorder writes decision inputs and
research diagnostics only; it neither selects trades nor mutates executor
state.
"""

from __future__ import annotations

import datetime
import platform
from collections.abc import Iterable
from decimal import Decimal
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

import numpy as np

from eth_defi.compat import native_datetime_utc_now
from tradeexecutor.strategy.recorder.indicators import capture_indicators
from tradeexecutor.strategy.recorder.observations import observation
from tradeexecutor.strategy.recorder.serialisation import to_json_value
from tradeexecutor.strategy.recorder.storage import RecorderStorage, validate_recorder_strategy_id
from tradeexecutor.strategy.recorder.universe import capture_universe

if TYPE_CHECKING:
    from tradeexecutor.state.trade import TradeExecution
    from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput


def _model_projection(model: Any) -> dict[str, Any]:
    """Capture type identity and public scalar configuration only.

    Runtime models commonly expose caches and API responses as dictionaries or
    lists. Deliberately skip all containers before serialising them so a model
    projection never turns into an accidental API archive. The projection
    schema is ``implementation`` plus public scalar ``values``.
    """

    values: dict[str, Any] = {}
    for name, value in vars(model).items() if hasattr(model, "__dict__") else ():
        if name.startswith("_") or isinstance(value, (dict, list, set, tuple, np.ndarray)):
            continue
        if value is None or isinstance(
            value,
            (bool, str, int, float, Decimal, datetime.date, datetime.timedelta, Enum, Path, UUID, np.generic),
        ):
            values[name] = to_json_value(value)

    return {
        "implementation": f"{type(model).__module__}.{type(model).__qualname__}",
        "values": values,
    }


def _open_position_references(strategy_input: "StrategyInput") -> list[dict[str, Any]]:
    """Return position, pair, and trade identifiers without copying state.

    These references let a researcher join a decision with the authoritative
    state file while keeping the recorder independent of state serialisation.
    """

    portfolio = getattr(getattr(strategy_input, "state", None), "portfolio", None)
    if portfolio is None:
        return []

    references = []
    for position in portfolio.get_open_positions():
        trades = getattr(position, "trades", {})
        trade_values = trades.values() if hasattr(trades, "values") else trades
        references.append({
            "position_id": position.position_id,
            "pair_id": position.pair.internal_id,
            "trade_ids": [trade.trade_id for trade in trade_values],
        })
    return references


class DecisionRecorder:
    """Write one run and its explicitly recorded live decisions to DuckDB.

    The framework constructs this object only when a live strategy enables
    ``record_strategy_inputs``. The strategy owns the decision lifecycle:
    call :meth:`begin`, zero or more :meth:`record` calls, then :meth:`finish`
    or :meth:`fail`.
    """

    def __init__(
        self,
        path: Path,
        strategy_id: str,
        strategy_source: str,
        *,
        strategy_file: str,
        executor_revision: str | None = None,
    ) -> None:
        """Create a recorder for one executor process.

        :param path:
            Persistent DuckDB path beside the executor state file.
        :param strategy_id:
            Executor identifier used in the recorder filename and run metadata.
        :param strategy_source:
            Exact loaded strategy source code.
        :param strategy_file:
            Source strategy path for human inspection.
        :param executor_revision:
            Executor build revision, when available.
        """

        self.path = Path(path)
        self.strategy_id = validate_recorder_strategy_id(strategy_id)
        self.storage = RecorderStorage(self.path)
        source_ref = self.storage.put_object(
            "source",
            {"path": strategy_file, "text": strategy_source},
        )
        package_versions: dict[str, str] = {}
        for package_name in (
            "trade-executor",
            "duckdb",
            "pandas",
            "tradingstrategy",
            "web3-ethereum-defi",
        ):
            try:
                package_versions[package_name] = version(package_name)
            except PackageNotFoundError:
                continue

        self.run_id = self.storage.start_run(
            self.strategy_id,
            {
                "strategy_file": strategy_file,
                "strategy_source": source_ref,
                "executor_revision": executor_revision,
                "packages": package_versions,
                "python_version": platform.python_version(),
            },
        )
        self.invocation_id: UUID | None = None
        self._observations: list[dict[str, Any]] = []
        self._sequence = 0

    def begin(self, strategy_input: "StrategyInput") -> "DecisionRecorder":
        """Persist immutable inputs and start one live decision.

        Captures parameters, execution context, constructed universe, indicator
        fingerprints, selected model projections, limited Web3 identity,
        ``other_data``, and state references before strategy calculations run.
        Raises if a decision is already active, recorder use is not live, or an
        input cannot be serialised.
        """

        if self.invocation_id is not None:
            raise RuntimeError("Cannot begin a decision while another decision is active")
        if not strategy_input.execution_context.mode.is_live_trading():
            raise RuntimeError("DecisionRecorder is live-only")

        parameters = {
            key: to_json_value(value)
            for key, value in dict(strategy_input.parameters).items()
        }
        parameter_ref = self.storage.put_object("parameters", {"values": parameters})
        context = strategy_input.execution_context
        context_ref = self.storage.put_object(
            "execution_context",
            {
                "mode": context.mode.value,
                "engine_version": str(context.engine_version) if context.engine_version else None,
                "parameters": parameter_ref,
                "flags": {
                    name: getattr(context, name)
                    for name in (
                        "grid_search",
                        "optimiser",
                        "jupyter",
                        "progress_bars",
                        "force_visualisation",
                    )
                },
            },
        )
        universe_ref = capture_universe(strategy_input.strategy_universe, self.storage.put_object)
        indicator_refs = capture_indicators(
            strategy_input.indicators,
            self.storage.put_object,
            inputs=[universe_ref],
        )

        def capture_model(kind: str, model: Any) -> dict[str, str] | None:
            return self.storage.put_object(kind, _model_projection(model)) if model is not None else None

        manifest = {
            "parameters": parameter_ref,
            "execution_context": context_ref,
            "universe": universe_ref,
            "indicators": indicator_refs,
            # These models are present on the production StrategyInput.  Use
            # ``getattr`` nevertheless: the public recorder is also useful in
            # small strategy tests which deliberately supply only the inputs
            # their decision needs.
            "pricing": capture_model("pricing", getattr(strategy_input, "pricing_model", None)),
            "routing_model": capture_model("routing_model", getattr(strategy_input, "routing_model", None)),
            "routing_state": capture_model("routing_state", getattr(strategy_input, "routing_state", None)),
            "web3": (
                {"implementation": f"{type(web3).__module__}.{type(web3).__qualname__}"}
                if (web3 := getattr(strategy_input, "web3", None)) is not None
                else None
            ),
            "other_data": to_json_value(strategy_input.other_data),
            "state_refs": to_json_value(_open_position_references(strategy_input)),
        }
        state_path = str(strategy_input.state_path) if strategy_input.state_path else ""
        self.invocation_id = self.storage.start_decision(
            self.run_id,
            strategy_input.cycle,
            strategy_input.timestamp,
            state_path,
            manifest,
        )
        self._observations = []
        self._sequence = 0
        return self

    def record(self, kind: str, name: str, value: Any, **kwargs: Any) -> None:
        """Append one serialisable, decision-relevant observation.

        :param kind:
            Stable observation category, for example ``signal`` or
            ``allocation``.
        :param name:
            Stable observation name within ``kind``.
        :param value:
            Decision-relevant value to record.
        :param kwargs:
            Optional :func:`observation` fields such as ``pair_key``,
            ``arguments``, ``source_at``, ``state_refs``, and ``provenance``.
        :raises RuntimeError:
            If no decision has been started with :meth:`begin`.
        """

        self._require_active_decision()
        self._observations.append(
            observation(
                self._sequence,
                kind,
                name,
                value,
                observed_at=native_datetime_utc_now(),
                **kwargs,
            )
        )
        self._sequence += 1

    def finish(self, trades: Iterable["TradeExecution"] | None = None) -> None:
        """Mark the active decision complete, record trade IDs, and checkpoint.

        :param trades:
            Trades returned from this decision. Only their existing state trade
            identifiers are stored; trade data remains in the state file.
        """

        invocation_id = self._require_active_decision()
        trade_ids = [
            trade.trade_id
            for trade in trades or []
            if getattr(trade, "trade_id", None) is not None
        ]
        self.storage.finish_decision(invocation_id, "completed", self._observations, trade_ids)
        self.storage.checkpoint()
        self.invocation_id = None

    def fail(self, error: Exception) -> None:
        """Mark the active decision failed, preserve observations, and checkpoint.

        :param error:
            Exception raised by the decision. Its class name and a bounded
            message are recorded, after which the original exception continues
            through normal strategy error handling.
        """

        if self.invocation_id is None:
            return
        self.storage.finish_decision(
            self.invocation_id,
            "failed",
            self._observations,
            [],
            {"type": type(error).__name__, "message": str(error)[:2_000]},
        )
        self.storage.checkpoint()
        self.invocation_id = None

    def close(self) -> None:
        """Close the underlying DuckDB connection after the executor stops."""

        self.storage.close()

    def _require_active_decision(self) -> UUID:
        if self.invocation_id is None:
            raise RuntimeError("Call recorder.begin() before recording a decision")
        return self.invocation_id
