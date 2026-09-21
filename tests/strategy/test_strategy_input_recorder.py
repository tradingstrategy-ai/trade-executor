"""Tests for live strategy-input recording.

1. Build a small deterministic live decision input.
2. Record it through the public recorder API.
3. Reopen DuckDB and verify JSON objects, lifecycle rows and exact values.
"""

from pathlib import Path
from types import SimpleNamespace
import json

import pandas as pd
from duckdb import connect

from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.recorder import DecisionRecorder
from tradeexecutor.strategy.recorder.serialisation import (
    canonical_json,
    decode_json_value,
    to_json_value,
)
from tradeexecutor.strategy.recorder.storage import RecorderStorage


class _Pair:
    internal_id = 1
    pair_id = 1
    pool_address = "0xabc"
    chain_id = 999
    other_data = {"decision_value": "kept", "token_metadata": {"symbol": "FIX"}}

    def to_dict(self, encode_json=False):
        # Mirrors TradingPairIdentifier's in-place custom encoder behaviour.
        self.other_data.pop("token_metadata", None)
        return {"internal_id": self.internal_id, "pool_address": self.pool_address}


class _Pairs:
    def __init__(self):
        self.df = pd.DataFrame({"pair_id": [1], "name": ["fixture"]})
        self.pair = _Pair()

    def iterate_pairs(self):
        return iter([self.pair])


def _make_input(state_path: Path):
    data = SimpleNamespace(
        pairs=_Pairs(),
        candles=SimpleNamespace(df=pd.DataFrame({
            "pair_id": [1],
            "timestamp": [pd.Timestamp("2026-01-01")],
            "close": [1.25],
        })),
        liquidity=None,
        resampled_liquidity=None,
        lending_candles=None,
        time_bucket="1d",
        chains={999},
        forward_filled=None,
        vault_specs=None,
        start_hint=None,
        end_hint=None,
    )
    universe = SimpleNamespace(
        data_universe=data,
        reserve_assets=[],
        pair_cache={},
        options=None,
        primary_chain=None,
        required_history_period=None,
        price_data_delay_tolerance=None,
        other_data={},
        vault_history_diagnostics=None,
        vault_window_overrides=None,
        ignore_routing=False,
        backtest_stop_loss_time_bucket=None,
        vault_state=None,
    )
    return SimpleNamespace(
        execution_context=ExecutionContext(ExecutionMode.unit_testing_trading),
        parameters=StrategyParameters({"record_strategy_inputs": True, "decimal": "1.2300"}),
        strategy_universe=universe,
        indicators=SimpleNamespace(indicator_results={}),
        other_data={"fixture": "ok"},
        timestamp=pd.Timestamp("2026-01-01 00:00:00.123456789"),
        cycle=1,
        state_path=state_path,
    )


def test_strategy_input_recorder_round_trip(tmp_path: Path):
    """Record one live decision and inspect its JSON payloads after reopening."""
    state_path = tmp_path / "hyper-ai.json"
    recorder = DecisionRecorder(
        tmp_path / "hyper-ai-record.duckdb",
        "hyper-ai",
        "fixture strategy",
        strategy_file="fixture.py",
    )
    input = _make_input(state_path)

    recorder.begin(input)
    assert input.strategy_universe.data_universe.pairs.pair.other_data == {
        "decision_value": "kept",
        "token_metadata": {"symbol": "FIX"},
    }
    recorder.record("calculation", "fixture", {"decimal": to_json_value("1.2300")})
    recorder.finish([])
    recorder.close()

    connection = connect(str(tmp_path / "hyper-ai-record.duckdb"))
    assert connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 1
    assert connection.execute("SELECT count(*) FROM decisions WHERE status = 'completed'").fetchone()[0] == 1
    assert connection.execute("SELECT epoch_ns(decision_at) FROM decisions").fetchone()[0] == 1767225600123456789
    kinds = {row[0] for row in connection.execute("SELECT kind FROM objects").fetchall()}
    assert {"source", "parameters", "execution_context", "frame_chunk", "universe"}.issubset(kinds)
    observations = json.loads(connection.execute("SELECT observations FROM decisions").fetchone()[0])
    assert observations[0]["name"] == "fixture"
    assert decode_json_value(to_json_value(pd.Timestamp("2026-01-01 00:00:00.123456789"))).value == 1767225600123456789
    connection.close()


def test_strategy_input_recorder_uses_zstd_for_persisted_history(tmp_path: Path):
    """Large, non-constant JSON history is stored in native Zstandard segments."""
    path = tmp_path / "compression-record.duckdb"
    storage = RecorderStorage(path)
    for index in range(1_000):
        # One realistic frame-like history object is intentionally large enough
        # to cross DuckDB's segment compression threshold.  The values are
        # unique so this does not only exercise constant compression.
        storage.put_object("history", {"rows": f"{index}:" + ("x" * 20_000)})
    storage.checkpoint()
    storage.close()

    connection = connect(str(path))
    compressed_columns = {
        row[0]
        for row in connection.execute(
            "SELECT column_name FROM pragma_storage_info('objects') WHERE compression = 'ZSTD'"
        ).fetchall()
    }
    connection.close()
    assert "payload" in compressed_columns
    reopened = RecorderStorage(path)
    reopened.close()


def test_strategy_input_recorder_persists_callback_failure(tmp_path: Path):
    """A strategy exception is visible as a failed invocation and re-raises normally."""
    path = tmp_path / "failed-record.duckdb"
    recorder = DecisionRecorder(
        path,
        "hyper-ai",
        "fixture strategy",
        strategy_file="fixture.py",
    )
    recorder.begin(_make_input(tmp_path / "hyper-ai.json"))
    recorder.record("calculation", "before_failure", 1)
    recorder.fail(ValueError("fixture failure"))
    recorder.close()

    connection = connect(str(path))
    status, error = connection.execute("SELECT status, error FROM decisions").fetchone()
    connection.close()
    assert status == "failed"
    assert json.loads(error)["type"] == "ValueError"


def test_strategy_input_recorder_canonicalises_unordered_values() -> None:
    """Object hashes stay stable when unordered values change insertion order."""

    assert canonical_json({2: "two", 1: "one"}) == canonical_json({1: "one", 2: "two"})
    assert canonical_json(frozenset({"a", "b"})) == canonical_json(frozenset({"b", "a"}))
