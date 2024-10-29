"""Test token tax (buy tax, sell tax) inside decide_trades loop and stop loss.

See

- load_partial_data(pair_extra_metadata=True)

- Parameters.slippage_tolerance

"""
import itertools
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
        stop_loss_time_bucket=TimeBucket.h1,
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
def parameters() -> StrategyParameters:
    class Parameters:
        cycle_duration = CycleDuration.cycle_1d
        initial_cash = 10_000
        backtest_start = datetime.datetime(2024, 9, 1)
        backtest_end = datetime.datetime(2024, 10, 1)
        slippage_tolerance = 0.005  # Default slippage tolerance 0.5%
    return StrategyParameters.from_class(Parameters)


@pytest.fixture()
def strategy_universe(persistent_test_client: Client, parameters: StrategyParameters):
    return create_trading_universe(
        datetime.datetime.now(),
        persistent_test_client,
        unit_test_execution_context,
        universe_options=UniverseOptions.from_strategy_parameters_class(parameters, unit_test_execution_context),
    )


def decide_trades(input: StrategyInput) -> list[TradeExecution]:
    # Example decide trades that generates buy, sell and stop loss sell events for the given real price data

    position_manager = input.get_position_manager()
    strategy_universe = input.strategy_universe

    assert position_manager.default_slippage_tolerance == 0.005

    trump_weth = strategy_universe.get_pair_by_human_description(
        (ChainId.ethereum, "uniswap-v2", "TRUMP", "WETH")
    )

    trades = []

    if not position_manager.is_any_open():
        if input.cycle % 3 == 0:
            # Set a buy
            cash = position_manager.get_current_cash()
            trades += position_manager.open_spot(
                trump_weth,
                value = cash * 0.99,
                stop_loss_pct=0.99,  # Will trigger
            )
    else:
        # Natural sell
        trades += position_manager.close_all()

    return trades


def create_indicators(
    timestamp,
    parameters,
    strategy_universe,
    execution_context,
) -> IndicatorSet:
    indicator_set = IndicatorSet()
    return indicator_set


def test_token_tax(
    strategy_universe: TradingStrategyUniverse,
    parameters: StrategyParameters,
    tmp_path,
):
    """Buy/sell token tax is included in the slippage tolerance.

    - PricingModel.calculate_trade_adjusted_slippage_tolerance is used to set slippage tolerance for all trades

    - This accounts for the token tax
    """

    indicator_storage = DiskIndicatorStorage(tmp_path, strategy_universe.get_cache_key())

    # Check that we have token tax data for TRUMP
    trump_weth = strategy_universe.get_pair_by_human_description(
        (ChainId.ethereum, "uniswap-v2", "TRUMP", "WETH")
    )

    # Is accurate 1.0, as rounded to 2 decimals
    assert trump_weth.get_buy_tax() == 0.01
    assert trump_weth.get_sell_tax() == 0.01

    # We loaded OHLCV for TRUMP
    candles = strategy_universe.data_universe.candles.get_candles_by_pair(trump_weth.internal_id)
    assert candles is not None
    assert len(candles) > 30

    # Run the test
    result = run_backtest_inline(
        client=None,
        decide_trades=decide_trades,
        create_indicators=create_indicators,
        universe=strategy_universe,
        reserve_currency=ReserveCurrency.usdc,
        engine_version="0.5",
        parameters=parameters,
        mode=ExecutionMode.unit_testing,
        indicator_storage=indicator_storage,
    )

    state = result.state

    # Go through all trades we made and see they had correct
    # slippage tolerance with tax included
    buy_trades = [t for t in state.portfolio.get_all_trades() if t.is_buy()]
    sell_trades = [t for t in state.portfolio.get_all_trades() if t.is_sell() and not t.is_stop_loss()]
    stop_loss_sell_trades = [t for t in state.portfolio.get_all_trades() if t.is_sell() and t.is_stop_loss()]

    assert len(buy_trades) > 0
    assert len(sell_trades) > 0
    assert len(stop_loss_sell_trades) > 0

    for t in itertools.chain(buy_trades, sell_trades, stop_loss_sell_trades):
        # Slippage tolerance is 1% buy/sell tax + our default slippage tolerance
        assert t.slippage_tolerance == 0.01 + 0.005, f"Trade has bad slippage tolerance {t}"

