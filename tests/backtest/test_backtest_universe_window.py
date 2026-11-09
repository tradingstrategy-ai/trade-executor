"""Reject execution beyond a prebuilt universe's requested data window."""

import datetime
from pathlib import Path
import pickle

import pytest

from tradeexecutor.backtest.backtest_runner import run_backtest, run_backtest_inline, setup_backtest_for_universe
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionMode
from tradeexecutor.strategy.strategy_module import read_strategy_module
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.testing.vault_fee_sample import load_fee_sample


def test_backtest_rejects_end_beyond_universe_load_window() -> None:
    """Fail before indicator calculation or trading on a misleading flat tail.

    1. Load the ordinary offline vault universe and declare its requested load end.
    2. Request a later execution end through the normal setup helper.
    3. Verify the runner refuses before it needs indicators or executes any trade.
    """
    # 1. Load the ordinary offline vault universe and declare its requested load end.
    universe = load_fee_sample(Path(__file__).parent / "fixtures/vault_externalised_fees", "hyperliquid")
    start, end = universe.data_universe.candles.get_timestamp_range()
    universe.data_load_end_at = end.to_pydatetime()
    universe = universe.clone()
    assert universe.data_load_end_at == end.to_pydatetime()
    module = read_strategy_module(Path("strategies/test_only/vault_externalised_fees.py"))
    # 2. Request a later execution end through the normal setup helper.
    setup = setup_backtest_for_universe(
        module, start_at=start.to_pydatetime(), end_at=end.to_pydatetime() + datetime.timedelta(days=14),
        cycle_duration=module.parameters.cycle_duration, initial_deposit=10_000,
        universe=universe, parameters=module.parameters,
    )
    # 3. Verify the runner refuses before it needs indicators or executes any trade.
    with pytest.raises(AssertionError, match="exceeds universe load end.*rebuild the universe"):
        run_backtest(setup)
    assert list(setup.state.portfolio.get_all_trades()) == []


@pytest.mark.parametrize("legacy_pickle", [False, True])
def test_backtest_preserves_load_window_and_legacy_pickles(legacy_pickle: bool) -> None:
    """Run normally with a shorter prior run window or a pickle predating the field.

    1. Load the ordinary vault fixture with a legacy options window shorter than its data.
    2. Optionally round-trip a slotted universe without the new field, matching an older pickle.
    3. Run the real fee strategy through the declared data end and verify successful trades.
    """
    # 1. Load the ordinary vault fixture with a legacy options window shorter than its data.
    universe = load_fee_sample(Path(__file__).parent / "fixtures/vault_externalised_fees", "hyperliquid")
    start, end = universe.data_universe.candles.get_timestamp_range()
    universe.data_load_end_at = end.to_pydatetime()
    universe.options = UniverseOptions(end_at=start.to_pydatetime() + datetime.timedelta(days=1))
    module = read_strategy_module(Path("strategies/test_only/vault_externalised_fees.py"))
    # 2. Optionally round-trip a slotted universe without the new field, matching an older pickle.
    if legacy_pickle:
        delattr(universe, "data_load_end_at")
        universe = pickle.loads(pickle.dumps(universe))
        assert universe.clone().data_load_end_at is None
    # 3. Run the real fee strategy through the declared data end and verify successful trades.
    result = run_backtest_inline(
        start_at=start.to_pydatetime(), end_at=end.to_pydatetime(), client=None,
        decide_trades=module.decide_trades, create_indicators=module.create_indicators,
        universe=universe, cycle_duration=module.parameters.cycle_duration,
        initial_deposit=10_000, parameters=module.parameters, trade_routing=TradeRouting.default,
        engine_version="0.5", mode=ExecutionMode.unit_testing, max_workers=1,
    )
    trades = list(result.state.portfolio.get_all_trades())
    assert len(trades) == 4
    assert all(trade.is_success() for trade in trades)
