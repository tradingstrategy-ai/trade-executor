"""Protect recorder schemas, serialisation, and lifecycle independently of CLI.

1. Build a small deterministic live decision input.
2. Record it through the public recorder API.
3. Reopen DuckDB and verify JSON objects, lifecycle rows and exact values.

These focused tests make storage failures easy to diagnose before the slower
black-box CLI test checks framework wiring.
"""

from pathlib import Path
from decimal import Decimal
from types import SimpleNamespace
import json
from collections.abc import Iterator

import pandas as pd
import pytest
from duckdb import connect

from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.recorder import DecisionRecorder, record_decision
from tradeexecutor.strategy.recorder.serialisation import (
    canonical_json,
    decode_json_value,
    encode_frame_chunks,
    to_json_value,
)
from tradeexecutor.strategy.recorder.storage import RecorderStorage


class _Pair:
    """Mimic the pair fields and mutating codec used by universe capture.

    ``capture_universe()`` serialises this fixture to prove recording cannot
    mutate decision-relevant ``other_data`` on a live pair object.
    """

    internal_id = 1
    pair_id = 1
    pool_address = "0xabc"
    chain_id = 999
    other_data = {"decision_value": "kept", "token_metadata": {"symbol": "FIX"}}

    def to_dict(self, encode_json: bool = False) -> dict[str, int | str]:
        """Simulate the domain codec called by recorder serialisation.

        ``_domain_to_dict()`` invokes this method on a copy. Removing metadata
        in place mirrors ``TradingPairIdentifier`` and exposes any accidental
        mutation of the source universe.
        """
        del encode_json
        self.other_data.pop("token_metadata", None)
        return {"internal_id": self.internal_id, "pool_address": self.pool_address}


class _Pairs:
    """Provide the pair-collection interface consumed by universe capture.

    ``_make_input()`` attaches this to the synthetic data universe so recorder
    tests exercise both dataframe chunking and domain-object serialisation.
    """

    def __init__(self) -> None:
        """Create the deterministic frame and pair used by ``_make_input()``.

        One pair is enough to expose mutation and storage-shape regressions
        without introducing irrelevant universe fixtures.
        """
        self.df = pd.DataFrame({"pair_id": [1], "name": ["fixture"]})
        self.pair = _Pair()

    def iterate_pairs(self) -> Iterator[_Pair]:
        """Yield the domain object requested by ``capture_universe()``.

        The method mirrors the production pair-universe call site so the test
        does not depend on recorder-specific branches.
        """
        return iter([self.pair])


def _make_input(
    state_path: Path,
    recorder: DecisionRecorder | None = None,
) -> SimpleNamespace:
    """Create the smallest realistic input accepted by ``DecisionRecorder.begin``.

    Recorder unit tests call this instead of constructing the full live runner.
    The namespace includes every capture surface needed to test universe,
    parameters, timestamps, state references, and empty indicators in isolation.
    """
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
        parameters=StrategyParameters({"decimal": "1.2300"}),
        strategy_universe=universe,
        indicators=SimpleNamespace(indicator_results={}),
        other_data={"fixture": "ok"},
        timestamp=pd.Timestamp("2026-01-01 00:00:00.123456789"),
        cycle=1,
        state_path=state_path,
        recorder=recorder,
    )


@pytest.mark.timeout(300)
def test_strategy_input_recorder_round_trip(tmp_path: Path) -> None:
    """Persist and reopen one completed live decision without mutating its universe.

    This guards the core analyst use case: a completed decision must be
    inspectable with exact nanosecond timing while capture leaves live inputs
    unchanged.

    1. Exercise the decorated callback with recording disabled and enabled.
    2. Close the writer so DuckDB checkpoints the completed decision.
    3. Reopen the file and verify the lifecycle row, input objects, and timestamp.
    """
    # 1. Exercise the decorated callback with recording disabled and enabled.
    state_path = tmp_path / "hyper-ai.json"
    recorder = DecisionRecorder(
        tmp_path / "hyper-ai-record.duckdb",
        "hyper-ai",
        "fixture strategy",
        strategy_file="fixture.py",
    )
    strategy_input = _make_input(state_path, recorder)

    @record_decision
    def decide_trades(input: StrategyInput) -> list[TradeExecution]:
        if input.recorder is not None:
            input.recorder.record("calculation", "fixture", {"decimal": Decimal("1.2300"), "timestamp": input.timestamp})
        return []

    # Bootstrap can discover the opt-in on the callback without consulting
    # strategy parameters; backtests still call through with no recorder.
    assert decide_trades.__record_decision__ is True
    assert not getattr(decide_trades.__wrapped__, "__record_decision__", False)
    assert decide_trades(SimpleNamespace(recorder=None)) == []
    assert decide_trades(strategy_input) == []
    assert strategy_input.strategy_universe.data_universe.pairs.pair.other_data == {
        "decision_value": "kept",
        "token_metadata": {"symbol": "FIX"},
    }

    # 2. Close the writer so DuckDB checkpoints the completed decision.
    recorder.close()

    # 3. Reopen the file and verify the lifecycle row, input objects, and timestamp.
    connection = connect(str(tmp_path / "hyper-ai-record.duckdb"))
    assert connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 1
    assert connection.execute("SELECT count(*) FROM decisions WHERE status = 'completed'").fetchone()[0] == 1
    assert connection.execute("SELECT epoch_ns(decision_at) FROM decisions").fetchone()[0] == 1767225600123456789
    kinds = {row[0] for row in connection.execute("SELECT kind FROM objects").fetchall()}
    assert {"source", "parameters", "execution_context", "frame_chunk", "universe"}.issubset(kinds)
    observations = json.loads(connection.execute("SELECT observations FROM decisions").fetchone()[0])
    assert observations[0]["name"] == "fixture"
    decoded = decode_json_value(observations[0]["value"])
    assert decoded["decimal"] == Decimal("1.2300")
    assert decoded["timestamp"].value == 1767225600123456789
    universe = json.loads(connection.execute("SELECT payload FROM objects WHERE kind = 'universe'").fetchone()[0])
    candle_ref = universe["frames"]["data_universe.candles.df"]["chunks"][0]["object"]
    candle_chunk = json.loads(connection.execute("SELECT payload FROM objects WHERE content_hash = ?", [candle_ref]).fetchone()[0])
    assert decode_json_value(candle_chunk["rows"])[0][1] == pd.Timestamp("2026-01-01")
    assert decode_json_value(to_json_value(pd.Timestamp("2026-01-01 00:00:00.123456789"))).value == 1767225600123456789
    connection.close()


@pytest.mark.timeout(300)
def test_strategy_input_recorder_uses_zstd_for_persisted_history(tmp_path: Path) -> None:
    """Persist non-constant history with the requested Zstandard compression.

    Large universe and indicator histories make compression an operational
    requirement, so this checks DuckDB's physical storage rather than merely a
    connection setting.

    1. Write enough unique JSON history to form DuckDB storage segments.
    2. Checkpoint and reopen the database as a reader.
    3. Confirm DuckDB used Zstandard for the JSON payload column.
    """
    # 1. Write enough unique JSON history to form DuckDB storage segments.
    path = tmp_path / "compression-record.duckdb"
    storage = RecorderStorage(path)
    for index in range(1_000):
        # One realistic frame-like history object is intentionally large enough
        # to cross DuckDB's segment compression threshold.  The values are
        # unique so this does not only exercise constant compression.
        storage.put_object("history", {"rows": f"{index}:" + ("x" * 20_000)})

    # 2. Checkpoint and reopen the database as a reader.
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

    # 3. Confirm DuckDB used Zstandard for the JSON payload column.
    assert "payload" in compressed_columns
    reopened = RecorderStorage(path)
    reopened.close()


@pytest.mark.timeout(300)
def test_strategy_input_recorder_persists_callback_failure(tmp_path: Path) -> None:
    """Persist a strategy exception as a failed decision for later diagnosis.

    Live failures are the decisions most likely to need forensic data; this
    ensures the terminal failure and preceding observations survive shutdown.

    1. Invoke a decorated live decision that records before raising an exception.
    2. Confirm the decorator re-raises the original callback exception.
    3. Reopen the file and verify the terminal status and error type.
    """
    # 1. Invoke a decorated live decision that records before raising an exception.
    path = tmp_path / "failed-record.duckdb"
    recorder = DecisionRecorder(
        path,
        "hyper-ai",
        "fixture strategy",
        strategy_file="fixture.py",
    )

    @record_decision
    def decide_trades(input: StrategyInput) -> list[TradeExecution]:
        input.recorder.record("calculation", "before_failure", 1)
        raise ValueError("fixture failure")

    # 2. Confirm the decorator re-raises the original callback exception.
    with pytest.raises(ValueError, match="fixture failure"):
        decide_trades(_make_input(tmp_path / "hyper-ai.json", recorder))
    recorder.close()

    # 3. Reopen the file and verify the terminal status and error type.
    connection = connect(str(path))
    status, error = connection.execute("SELECT status, error FROM decisions").fetchone()
    connection.close()
    assert status == "failed"
    assert json.loads(error)["type"] == "ValueError"


@pytest.mark.timeout(300)
def test_strategy_input_recorder_canonicalises_unordered_values() -> None:
    """Keep content hashes stable when unordered values change insertion order.

    Content-addressed deduplication depends on logically equal inputs producing
    identical bytes, regardless of Python mapping or set construction order.

    1. Canonicalise mappings containing non-string keys in both insertion orders.
    2. Canonicalise frozensets containing the same values in both insertion orders.
    3. Compare the canonical JSON used as the content-hash input.
    4. Read back tagged values and preserve the original Series schema.
    """
    # 1. Canonicalise mappings containing non-string keys in both insertion orders.
    first_mapping = canonical_json(to_json_value({2: "two", 1: "one"}))
    second_mapping = canonical_json(to_json_value({1: "one", 2: "two"}))

    # 2. Canonicalise frozensets containing the same values in both insertion orders.
    first_frozenset = canonical_json(to_json_value(frozenset({"a", "b"})))
    second_frozenset = canonical_json(to_json_value(frozenset({"b", "a"})))

    # 3. Compare the canonical JSON used as the content-hash input.
    assert first_mapping == second_mapping
    assert first_frozenset == second_frozenset
    # 4. Read back tagged values and preserve the original Series schema.
    # Tagged scalar values remain decodable after canonical storage, while a
    # user dictionary containing "$type" stays an ordinary dictionary.
    literal_mapping = {"$type": "timestamp_ns", "value": "123"}
    assert decode_json_value(json.loads(canonical_json(to_json_value(literal_mapping)))) == literal_mapping
    enum_set = {ExecutionMode.unit_testing_trading}
    assert decode_json_value(json.loads(canonical_json(to_json_value(enum_set)))) == {ExecutionMode.unit_testing_trading.value}
    series = pd.Series([1.0], index=pd.DatetimeIndex(["2026-01-01"]), name="tvl")
    schema, chunks = encode_frame_chunks(series)
    assert schema["container"] == "series"
    assert schema["name"] == "tvl"
    assert decode_json_value(json.loads(canonical_json(chunks[0]["payload"])))["index_values"][0][0] == series.index[0]
