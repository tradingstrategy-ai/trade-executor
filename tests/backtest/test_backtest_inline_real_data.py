"""Test backtesting where decide_trades and create_universe functions are passed directly.

"""
import os
import logging
import datetime

import pytest

import pandas as pd

from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.backtest.data_preload import preload_data
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.state.visualisation import PlotKind
from tradeexecutor.strategy.trading_strategy_universe import load_all_data, TradingStrategyUniverse
from tradeexecutor.strategy.execution_context import ExecutionContext, unit_test_execution_context
from tradingstrategy.client import Client


from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.strategy_type import StrategyType
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.universe_model import UniverseOptions

CI = os.environ.get("CI") == "true"

# https://docs.pytest.org/en/latest/how-to/skipping.html#skip-all-test-functions-of-a-class-or-module
pytestmark = pytest.mark.skipif(os.environ.get("TRADING_STRATEGY_API_KEY") is None, reason="Set TRADING_STRATEGY_API_KEY environment variable to run this test")

# Tell what trade execution engine version this strategy needs to use
trading_strategy_engine_version = "0.1"

# What kind of strategy we are running.
# This tells we are going to use
trading_strategy_type = StrategyType.managed_positions

# How our trades are routed.
# PancakeSwap basic routing supports two way trades with BUSD
# and three way trades with BUSD-BNB hop.
trade_routing = TradeRouting.pancakeswap_busd

# How often the strategy performs the decide_trades cycle.
# We do it for every 16h.
trading_strategy_cycle = CycleDuration.cycle_1d

# Strategy keeps its cash in BUSD
reserve_currency = ReserveCurrency.busd

# Time bucket for our candles
candle_time_frame = TimeBucket.d1

# Which chain we are trading
chain_id = ChainId.bsc

# Which exchange we are trading on.
exchange_slug = "pancakeswap-v2"

# Which trading pair we are trading
trading_pair = ("WBNB", "BUSD")

# How much of the cash to put on a single trade
position_size = 0.10

#
# Strategy thinking specific parameter
#

batch_size = 90

slow_ema_candle_count = 39

fast_ema_candle_count = 15



from tradeexecutor.state.trade import TradeExecution
from typing import List, Dict
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.state.state import State
from tradingstrategy.universe import Universe


def decide_trades(
        timestamp: pd.Timestamp,
        universe: Universe,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: Dict) -> List[TradeExecution]:
    """The brain function to decide the trades on each trading strategy cycle."""

    # The pair we are trading
    pair = universe.pairs.get_single()

    # How much cash we have in the hand
    cash = state.portfolio.get_cash()

    # Get OHLCV candles for our trading pair as Pandas Dataframe.
    # We could have candles for multiple trading pairs in a different strategy,
    # but this strategy only operates on single pair candle.
    # We also limit our sample size to N latest candles to speed up calculations.
    candles: pd.DataFrame = universe.candles.get_single_pair_data(timestamp)

    # We have data for open, high, close, etc.
    # We only operate using candle close values in this strategy.
    close = candles["close"]

    # Calculate exponential moving averages based on slow and fast sample numbers.
    # More information about calculating expotential averages in Pandas:
    #
    # - https://www.statology.org/exponential-moving-average-pandas/
    #
    # - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
    #
    slow_ema = close.ewm(span=slow_ema_candle_count).mean().iloc[-1]
    fast_ema = close.ewm(span=fast_ema_candle_count).mean().iloc[-1]

    # print("Slow EMA is", slow_ema)

    # Get the last close price from close time series
    # that's Pandas's Series object
    # https://pandas.pydata.org/docs/reference/api/pandas.Series.iat.html
    current_price = close.iloc[-1]

    # List of any trades we decide on this cycle.
    # Because the strategy is simple, there can be
    # only zero (do nothing) or 1 (open or close) trades
    # decides
    trades = []

    # Create a position manager helper class that allows us easily to create
    # opening/closing trades for different positions
    position_manager = PositionManager(timestamp, universe, state, pricing_model)

    if current_price >= slow_ema:
        # Entry condition:
        # Close price is higher than the slow EMA
        if not position_manager.is_any_open():
            buy_amount = cash * position_size
            trades += position_manager.open_spot(pair, buy_amount)
    elif fast_ema >= slow_ema:
        # Exit condition:
        # Fast EMA crosses slow EMA
        if position_manager.is_any_open():
            trades += position_manager.close_all()

    # Visualize strategy
    # See available Plotly colours here
    # https://community.plotly.com/t/plotly-colours-list/11730/3?u=miohtama
    visualisation = state.visualisation
    visualisation.plot_indicator(timestamp, "Slow EMA", PlotKind.technical_indicator_on_price, slow_ema, colour="darkblue")
    visualisation.plot_indicator(timestamp, "Fast EMA", PlotKind.technical_indicator_on_price, fast_ema, colour="mediumpurple")

    return trades


def create_trading_universe(
        ts: datetime.datetime,
        client: Client,
        execution_context: ExecutionContext,
        universe_options: UniverseOptions,
) -> TradingStrategyUniverse:
    dataset = load_all_data(client, candle_time_frame, execution_context, universe_options)

    # Filter down to the single pair we are interested in
    universe = TradingStrategyUniverse.create_single_pair_universe(
        dataset,
        chain_id,
        exchange_slug,
        trading_pair[0],
        trading_pair[1],
    )

    return universe


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


@pytest.mark.skipif(CI, reaosn="Too much data / slow / flaky on Github")
def test_backtest_data_preload(
        logger: logging.Logger,
        persistent_test_client: Client,
    ):
    """Run the data preload step before a strategy backtesting begins."""

    client = persistent_test_client

    preload_data(
        client,
        create_trading_universe=create_trading_universe,
        universe_options=UniverseOptions(),
        execution_context=unit_test_execution_context,
    )


def test_run_inline_real_backtest(
    persistent_test_client: Client,
):
    """Run the strategy backtest using inline decide_trades function.
    """

    # Run the test
    state, universe, debug_dump = run_backtest_inline(
        start_at=datetime.datetime(2021, 6, 1),
        end_at=datetime.datetime(2022, 1, 1),
        client=persistent_test_client,
        cycle_duration=CycleDuration.cycle_1d,  # Override to use 24h cycles despite what strategy file says
        decide_trades=decide_trades,
        create_trading_universe=create_trading_universe,
        initial_deposit=10_000,
        reserve_currency=ReserveCurrency.busd,
        trade_routing=TradeRouting.pancakeswap_busd,
    )

    assert len(debug_dump) == 216



