"""Test token tax (buy tax, sell tax) inside decide_trades loop and stop loss.

"""
import os

import datetime

import pytest

from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet, DiskIndicatorStorage
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.strategy_module import StrategyParameters
from tradingstrategy.client import Client
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket

from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_partial_data
from tradeexecutor.strategy.execution_context import ExecutionContext, unit_test_execution_context, ExecutionMode
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.universe_model import UniverseOptions, default_universe_options
from tradeexecutor.state.trade import TradeExecution


# https://docs.pytest.org/en/latest/how-to/skipping.html#skip-all-test-functions-of-a-class-or-module
pytestmark = pytest.mark.skipif(os.environ.get("TRADING_STRATEGY_API_KEY") is None, reason="Set TRADING_STRATEGY_API_KEY environment variable to run this test")



def create_trading_universe(
    ts: datetime.datetime,
    client: Client,
    execution_context: ExecutionContext,
    universe_options: UniverseOptions,
) -> TradingStrategyUniverse:

    assert universe_options.start_at

    # TRUMP has ~1% buy and sell tax
    pairs = [
        (ChainId.ethereum, "uniswap-v2", "TRUMP", "WETH"),
        (ChainId.ethereum, "uniswap-v3", "WETH", "USDC", 0.0005)  # Needed for routing USDC->WETH->TRUMP
    ]

    dataset = load_partial_data(
        client,
        execution_context=unit_test_execution_context,
        time_bucket=TimeBucket.d1,
        pairs=pairs,
        universe_options=default_universe_options,
        start_at=universe_options.start_at,
        end_at=universe_options.end_at,
        pair_extra_metadata=True,  # Enable loading of TokenSniffer data
    )

    strategy_universe = TradingStrategyUniverse.create_from_dataset(
        dataset,
        reserve_asset="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
    )
    return strategy_universe


@pytest.fixture()
def strategy_universe(persistent_test_client: Client):
    return create_trading_universe(
        datetime.datetime.now(),
        persistent_test_client,
        unit_test_execution_context,
        UniverseOptions(start_at=datetime.datetime(2023, 1, 1), end_at=datetime.datetime(2023, 2, 1))
    )



def decide_trades(input: StrategyInput) -> list[TradeExecution]:
    # Example decide trades that generates buy, sell and stop loss sell events for the given real price data

    position_manager = input.get_position_manager()

    if input.cycle % 3 == 0:
        # Set a buy
        pass

    if position_manager.is_any_open():
        # Natural sell
        position_manager.close_all()

    return []


def create_indicators(
    timestamp,
    parameters,
    strategy_universe,
    execution_context,
) -> IndicatorSet:
    indicator_set = IndicatorSet()
    return indicator_set


def test_token_tax(strategy_universe, tmp_path):
    """Test DecideTradesProtocolV4

    - Check that StrategyInput is passed correctly in backtesting (only backtesting, not live trading)
    """

    class Parameters:
        cycle_duration = CycleDuration.cycle_1d
        initial_cash = 10_000
        backtest_start = datetime.datetime(2024, 6, 1)
        backtest_end = datetime.datetime(2024, 7, 1)

    indicator_storage = DiskIndicatorStorage(tmp_path, strategy_universe.get_cache_key())

    # Check that we have token tax data for TRUMP
    trump_weth = strategy_universe.get_pair_by_human_description(
        (ChainId.ethereum, "uniswap-v2", "TRUMP", "WETH")
    )

    # Is accurate 1.0, as rounded to 2 decimals
    assert trump_weth.get_buy_tax() == 1.0
    assert trump_weth.get_sell_tax() == 1.0

    # Run the test
    result = run_backtest_inline(
        client=None,
        decide_trades=decide_trades,
        create_indicators=create_indicators,
        universe=strategy_universe,
        reserve_currency=ReserveCurrency.usdc,
        engine_version="0.5",
        parameters=StrategyParameters.from_class(Parameters),
        mode=ExecutionMode.unit_testing,
        indicator_storage=indicator_storage,
    )

    state, universe, debug_dump = result
    assert len(state.portfolio.closed_positions) == 0
