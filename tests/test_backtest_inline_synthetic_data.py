"""Test backtesting where decide_trades and create_universe functions are passed directly.

"""
import logging
import random
import datetime
from typing import List, Dict

import pytest

import pandas as pd
from pandas_ta.overlap import ema

from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.visualisation import PlotKind
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, \
    create_pair_universe_from_code
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange, generate_simple_routing_model
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradingstrategy.candle import GroupedCandleUniverse
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.state.state import State
from tradingstrategy.universe import Universe
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.strategy_module import StrategyType, TradeRouting, ReserveCurrency, \
    pregenerated_create_trading_universe

# How much of the cash to put on a single trade
position_size = 0.10

#
# Strategy thinking specific parameter
#

batch_size = 90

slow_ema_candle_count = 39

fast_ema_candle_count = 15


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
    cash = state.portfolio.get_current_cash()

    # Get OHLCV candles for our trading pair as Pandas Dataframe.
    # We could have candles for multiple trading pairs in a different strategy,
    # but this strategy only operates on single pair candle.
    # We also limit our sample size to N latest candles to speed up calculations.
    candles: pd.DataFrame = universe.candles.get_single_pair_data(timestamp, sample_count=batch_size)

    # We have data for open, high, close, etc.
    # We only operate using candle close values in this strategy.
    close = candles["close"]

    # Calculate exponential moving averages based on slow and fast sample numbers.
    # https://github.com/twopirllc/pandas-ta
    slow_ema_series = ema(close, length=slow_ema_candle_count)
    fast_ema_series = ema(close, length=fast_ema_candle_count)

    if slow_ema_series is None or fast_ema_series is None:
        # Cannot calculate EMA, because
        # not enough samples in backtesting
        return []

    slow_ema = slow_ema_series.iloc[-1]
    fast_ema = fast_ema_series.iloc[-1]

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
            trades += position_manager.open_1x_long(pair, buy_amount)
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


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


@pytest.fixture(scope="module")
def universe() -> TradingStrategyUniverse:

    start_at = datetime.datetime(2021, 6, 1)
    end_at = datetime.datetime(2022, 1, 1)

    # Set up fake assets
    mock_chain_id = ChainId.ethereum
    mock_exchange = generate_exchange(
        exchange_id=random.randint(1, 1000),
        chain_id=mock_chain_id,
        address=generate_random_ethereum_address())
    usdc = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "USDC", 6, 1)
    weth = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "WETH", 18, 2)
    weth_usdc = TradingPairIdentifier(
        weth,
        usdc,
        generate_random_ethereum_address(),
        mock_exchange.address,
        internal_id=random.randint(1, 1000),
        internal_exchange_id=mock_exchange.exchange_id)

    time_bucket = TimeBucket.d1

    pair_universe = create_pair_universe_from_code(mock_chain_id, [weth_usdc])

    candles = generate_ohlcv_candles(time_bucket, start_at, end_at, pair_id=weth_usdc.internal_id)
    candle_universe = GroupedCandleUniverse.create_from_single_pair_dataframe(candles)

    universe = Universe(
        time_bucket=time_bucket,
        chains={mock_chain_id},
        exchanges={mock_exchange},
        pairs=pair_universe,
        candles=candle_universe,
        liquidity=None
    )

    return TradingStrategyUniverse(universe=universe, reserve_assets=[usdc])


def test_run_inline_synthetic_backtest(
        logger: logging.Logger,
        universe: TradingStrategyUniverse,
    ):
    """Run the strategy backtest using inline decide_trades function.
    """

    start_at, end_at = universe.universe.candles.get_timestamp_range()

    routing_model = generate_simple_routing_model(universe)

    # Run the test
    state, debug_dump = run_backtest_inline(
        start_at=start_at.to_pydatetime(),
        end_at=end_at.to_pydatetime(),
        client=None,  # None of downloads needed, because we are using synthetic data
        cycle_duration=CycleDuration.cycle_24h,  # Override to use 24h cycles despite what strategy file says
        decide_trades=decide_trades,
        create_trading_universe=None,
        universe=universe,
        initial_deposit=10_000,
        reserve_currency=ReserveCurrency.busd,
        trade_routing=TradeRouting.routing_model,
        routing_model=routing_model,
        log_level=logging.WARNING,
    )

    assert len(debug_dump) == 213
