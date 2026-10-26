"""Keep routine reporting quiet without concealing accounting or rendering faults."""

import datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from tradeexecutor.analysis.trade_analyser import build_trade_analysis
from tradeexecutor.cli.loop import ExecutionLoop
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import HYPERCORE_DEPOSIT_CAPITAL_AT_RISK_KEY, TradeType
from tradeexecutor.statistics.in_memory_statistics import redraw_visualisations
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.pandas_trader.runner import PandasTraderRunner
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.strategy.run_state import RunState


@pytest.mark.parametrize("activity", ["dust", "failed_opening", "funded", "allocated", "unresolved", "started"])
def test_missing_realised_profit_warns_only_for_accounting_activity(
    activity: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Distinguish empty administrative closures from missing funded accounting.

    1. Create a real position closed without a successful reduction.
    2. Represent a dust placeholder, failed opening, or outstanding capital.
    3. Calculate statistics twice and retain warnings only for outstanding capital.
    """
    # 1. Real portfolio objects exercise accounting rather than mocking its answer.
    timestamp = datetime.datetime(2026, 4, 7)
    usdc = AssetIdentifier(1, "0x" + "01" * 20, "USDC", 6)
    token = AssetIdentifier(1, "0x" + "02" * 20, "TEST", 18)
    pair = TradingPairIdentifier(token, usdc, "0x" + "03" * 20, "0x" + "04" * 20, internal_id=1)
    state = State()
    reserve = state.portfolio.initialise_reserves(usdc, reserve_token_price=1.0)
    reserve.quantity = reserve.initial_deposit = Decimal(1000)
    reserve.initial_deposit_reserve_token_price = 1.0
    position, trade, _ = state.create_trade(
        timestamp, pair, Decimal(1), None, 100.0, TradeType.rebalance, usdc, 1.0,
    )
    position.closed_at = timestamp + datetime.timedelta(days=1)
    state.portfolio.closed_positions[position.position_id] = state.portfolio.open_positions.pop(position.position_id)

    # 2. These are the persisted shapes of administrative and incomplete closures.
    if activity in ("dust", "funded"):
        quantity = Decimal(1) if activity == "funded" else Decimal(0)
        trade.mark_success(timestamp, 100.0, quantity, quantity * 100, 0.0, 1.0, force=True)
    else:
        trade.failed_at = timestamp
        if activity == "allocated":
            trade.reserve_currency_allocated = Decimal(100)
        elif activity == "unresolved":
            trade.other_data[HYPERCORE_DEPOSIT_CAPITAL_AT_RISK_KEY] = "100"
        elif activity == "started":
            trade.failed_at = None
            trade.started_at = timestamp
    assert position.get_realised_profit_usd() is None

    # 3. Repeated reporting must stay quiet for empty records, without changing PnL.
    with caplog.at_level("WARNING"):
        summaries = [build_trade_analysis(state.portfolio).calculate_summary_statistics() for _ in range(2)]
    warnings = [r for r in caplog.records if "Realised profit calculation returned None" in r.message]
    assert len(warnings) == (0 if activity in ("dust", "failed_opening") else 2)
    assert all(summary.realised_profit == pytest.approx(0) for summary in summaries)


def test_live_freshness_context_tracks_custody() -> None:
    """Include frozen and pending holdings so low TVL cannot hide a custody fault.

    1. Supply open, frozen, pending and closed vault identities.
    2. Build live options, then move a held vault into closed positions.
    3. Keep only current custody addresses and preserve snapshot selection.
    """
    # 1. Only pair identity is needed here; external universe loading is unnecessary.
    usdc = AssetIdentifier(1, "0x" + "01" * 20, "USDC", 6)
    positions = {}
    for index in range(1, 5):
        pair = TradingPairIdentifier(
            usdc, usdc, f"0x{index:040x}", "0x" + "04" * 20, kind=TradingPairKind.vault,
        )
        positions[index] = SimpleNamespace(pair=pair)
    state = State()
    state.portfolio.open_positions = {1: positions[1]}
    state.portfolio.frozen_positions = {2: positions[2]}
    state.portfolio.pending_positions = {3: positions[3]}
    state.portfolio.closed_positions = {4: positions[4]}
    options = UniverseOptions(end_at=datetime.datetime(2026, 10, 15), vault_price_snapshot=Path("verified.parquet"))
    loop = SimpleNamespace(parameters={"min_tvl_usd": 7500}, universe_options=options)

    # 2. Refresh the context from accounting state rather than caching old holdings.
    first = ExecutionLoop._get_live_universe_options(loop, state)
    state.portfolio.closed_positions[1] = state.portfolio.open_positions.pop(1)
    second = ExecutionLoop._get_live_universe_options(loop, state)

    # 3. Preserve the verified price snapshot and fail conservatively without a floor.
    assert first.vault_history_min_tvl == 7500
    assert first.vault_history_held_addresses == frozenset(f"0x{i:040x}" for i in (1, 2, 3))
    assert second.vault_history_held_addresses == frozenset(f"0x{i:040x}" for i in (2, 3))
    assert second.end_at == options.end_at
    assert second.vault_price_snapshot == options.vault_price_snapshot
    assert options.vault_history_held_addresses is None
    loop.parameters = None
    assert ExecutionLoop._get_live_universe_options(loop, state).vault_history_min_tvl is None


@pytest.mark.parametrize("entrypoint", ["statistics", "runner"])
def test_unsupported_legacy_images_are_informational(
    entrypoint: str,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep expected renderer limits quiet while preserving genuine failure warnings.

    1. Supply a large universe to each legacy image refresh entry point.
    2. Verify its supported pair limit is informational.
    3. Force a supported image render to fail and retain its warning.
    """
    # 1. Mock only universe shape: no downloaded candles are needed to skip images.
    universe = SimpleNamespace(get_pair_count=lambda: 523, is_empty=lambda: False, has_candle_data=lambda: True)
    context = ExecutionContext(ExecutionMode.unit_testing_trading)
    run_state = RunState()
    state = State()
    runner = SimpleNamespace(run_state=run_state, visualisation=True, execution_context=context)
    module = "tradeexecutor.statistics.in_memory_statistics" if entrypoint == "statistics" else "tradeexecutor.strategy.pandas_trader.runner"

    def refresh() -> None:
        """Exercise the selected public reporting path."""
        if entrypoint == "statistics":
            redraw_visualisations(run_state, state, universe, context)
        else:
            PandasTraderRunner.refresh_visualisations(runner, state, universe)

    # 2. The renderer's limit is expected even when separate custom charts exist.
    with caplog.at_level("INFO"):
        refresh()
    assert "Skipping legacy technical images for 523 pairs" in caplog.text
    assert not any(record.levelno >= 30 for record in caplog.records)

    # 3. A genuine drawing failure must still be visible to operators.
    def fail_render(*args, **kwargs):
        """Replace plotting to test failure handling without invoking a browser."""
        raise RuntimeError("broken chart input")

    caplog.clear()
    universe.get_pair_count = lambda: 1
    monkeypatch.setattr(f"{module}.draw_single_pair_strategy_state", fail_render)
    with caplog.at_level("WARNING"):
        refresh()
    assert any("broken chart input" in record.message and record.levelno == 30 for record in caplog.records)
