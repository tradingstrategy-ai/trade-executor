"""Coordinate recording around one live strategy decision at a time.

CLI bootstrap creates :class:`DecisionRecorder` for a live v0.5 pandas strategy
that enables ``Parameters.record_strategy_inputs``. ``PandasTraderRunner`` then
injects it into ``StrategyInput``. The strategy decorates ``decide_trades()``
with :func:`record_decision` and emits optional observations from the callback.
This boundary captures inputs at the instant the strategy consumes them without
adding generic framework hooks. The recorder writes research diagnostics only;
it neither selects trades nor mutates executor state.
"""

import datetime
import platform
from collections.abc import Callable, Iterable
from decimal import Decimal
from enum import Enum
from functools import wraps
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


_DecisionFunction = Callable[["StrategyInput"], list["TradeExecution"]]


def _model_projection(model: Any) -> dict[str, Any]:
    """Capture type identity and public scalar configuration only.

    :meth:`DecisionRecorder.begin` calls this for pricing, routing model, and
    routing state objects. Their implementation and scalar settings can explain
    a decision, while copying caches, clients, or API responses would create a
    large and misleading archive of data the strategy may not have used.

    Runtime models commonly expose caches and API responses as dictionaries or
    lists. Deliberately skip all containers before serialising them so a model
    projection never turns into an accidental API archive. The projection
    schema is ``implementation`` plus public scalar ``values``.

    :param model:
        Runtime pricing or routing object exposed to the strategy.
    :return:
        JSON-ready implementation identity and public scalar settings.
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

    :meth:`DecisionRecorder.begin` calls this while building the input manifest.
    The identifiers let a researcher join the decision to the authoritative
    state file while avoiding a second, potentially divergent copy of portfolio
    state in DuckDB.

    :param strategy_input:
        Active live decision input containing authoritative executor state.
    :return:
        Open position, pair, and trade identifiers suitable for state joins.
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


def record_decision(func: _DecisionFunction) -> _DecisionFunction:
    """Wrap a v0.5 ``decide_trades()`` callback in recorder lifecycle handling.

    Strategy modules use this decorator on their public ``decide_trades()``
    function. The live runner has already attached a :class:`DecisionRecorder`
    before invoking the callback, so the wrapper can capture inputs immediately,
    complete successful decisions with their returned trade IDs, and persist
    callback failures before re-raising them. When recording is disabled or the
    strategy is backtesting, ``StrategyInput.recorder`` is ``None`` and the
    callback runs unchanged.

    Recorder startup or completion failures deliberately propagate without a
    recovery path. Only exceptions raised by the decorated strategy callback
    are passed to :meth:`DecisionRecorder.fail`.

    :param func:
        Synchronous v0.5 strategy callback accepting one ``StrategyInput`` and
        returning its proposed trades.
    :return:
        Callback with the same public signature and metadata, wrapped by the
        optional recorder lifecycle.
    """
    @wraps(func)
    def wrapper(strategy_input: "StrategyInput") -> list["TradeExecution"]:
        recorder = strategy_input.recorder
        if recorder is None:
            return func(strategy_input)

        recorder.begin(strategy_input)
        try:
            trades = func(strategy_input)
        except Exception as error:
            recorder.fail(error)
            raise
        recorder.finish(trades)
        return trades

    return wrapper


class DecisionRecorder:
    """Write one run and its explicitly recorded live decisions to DuckDB.

    CLI strategy bootstrap constructs this object only when a live strategy
    enables ``record_strategy_inputs``. ``PandasTraderRunner`` exposes it on
    each ``StrategyInput`` so :func:`record_decision` can own the exact callback
    boundary. Strategy calculations may call :meth:`record` for explicit
    observations. One instance covers one executor process and accepts only one
    active decision at a time.

    Strategies should not construct a recorder themselves. Enable
    ``Parameters.record_strategy_inputs`` and decorate the decision callback::

        @record_decision
        def decide_trades(input: StrategyInput) -> list[TradeExecution]:
            candidate_scores = calculate_candidate_scores(input)
            if input.recorder is not None:
                input.recorder.record("calculation", "candidate_scores", candidate_scores)
            return create_trades(input, candidate_scores)

    :func:`record_decision` preserves ordinary backtest and non-recording call
    paths, and it re-raises callback failures after persisting their diagnostics.

    See ``strategy/hyper-ai-v8.py`` in the strategies repository for a real
    strategy integration. See
    ``strategies/test_only/strategy_input_recorder.py`` for the minimal test
    strategy and ``tests/cli/test_cli_strategy_input_recorder.py`` for its
    end-to-end Typer call site.
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

        The strategy factory calls this during live bootstrap, before the
        runner starts cycling. Recording source and runtime provenance once per
        process lets analysts distinguish changes across restarts without
        repeating that metadata in every decision row.

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

        A recording-enabled ``decide_trades()`` wrapper calls this before any
        signal or allocation calculations. That placement captures the actual
        constructed universe, indicators, parameters, models, and state
        references consumed by this invocation rather than a later snapshot.

        Captures parameters, execution context, constructed universe, indicator
        fingerprints, selected model projections, limited Web3 identity,
        ``other_data``, and state references before strategy calculations run.
        Raises if a decision is already active, recorder use is not live, or an
        input cannot be serialised.

        :param strategy_input:
            Active live input that the strategy is about to consume.
        :return:
            This recorder, allowing an optional context-style local assignment.
        :raises RuntimeError:
            If another decision is active or the execution mode is not live.
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
            """Store one optional runtime model projection for the manifest.

            ``begin()`` calls this for each model supplied by the runner. The
            helper keeps optional-value handling and content-addressed storage
            identical across model kinds.

            :param kind:
                Object-store category for the runtime model.
            :param model:
                Model to project, or ``None`` when the runner has none.
            :return:
                Content reference for a present model, otherwise ``None``.
            """
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

        Strategy calculation helpers call this between :meth:`begin` and
        :meth:`finish` for values that are not already in the state file, such
        as candidate scores, exclusion reasons, or proposed allocations. The
        monotonically increasing sequence preserves calculation order.

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

        The successful branch of ``decide_trades()`` calls this immediately
        before returning its trades. IDs link the recorded calculations to
        authoritative trade data in state, and the checkpoint makes the cycle
        available to external inspection.

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

        The exception branch around a recording-enabled ``decide_trades()``
        calls this and then re-raises the same exception. Persisting values
        recorded before the failure makes live-only calculation errors
        diagnosable even though no trades were returned.

        :param error:
            Exception raised by the decision. Its class name and a bounded
            message are recorded. The caller remains responsible for re-raising
            the original exception through normal strategy error handling.
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
        """Release recorder storage when the live runner or loop stops.

        ``PandasTraderRunner.close()`` and the execution loop's ``finally`` path
        may both reach this method, so the delegated storage close is
        intentionally idempotent.
        """
        self.storage.close()

    def _require_active_decision(self) -> UUID:
        """Return the current invocation or reject an orphan lifecycle call.

        :meth:`record` and :meth:`finish` call this before writing anything.
        Failing immediately prevents observations or outputs from being
        associated with the wrong live cycle.
        """
        if self.invocation_id is None:
            raise RuntimeError("Call recorder.begin() before recording a decision")
        return self.invocation_id
