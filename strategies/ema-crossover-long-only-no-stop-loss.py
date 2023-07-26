"""Live trading strategy implementation fo ra single pair exponential moving average model.

- Trades a single trading pair - BNB/USD

- Does long positions only, based on exponential moving average indicators

"""
import datetime
from typing import Dict, List, Optional

import pandas as pd

from tradeexecutor.strategy.pricing_model import PricingModel
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.strategy_type import StrategyType
from tradeexecutor.state.visualisation import Visualisation, PlotKind
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_all_data
from tradingstrategy.client import Client

from tradeexecutor.strategy.universe_model import UniverseOptions

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
trading_strategy_cycle = CycleDuration.cycle_16h

# Strategy keeps its cash in BUSD
reserve_currency = ReserveCurrency.busd

# Time bucket for our candles
candle_time_bucket = TimeBucket.h4

# Which chain we are trading
chain_id = ChainId.bsc

# Which exchange we are trading on.
exchange_slug = "pancakeswap-v2"

# Which trading pair we are trading
trading_pair = ("WBNB", "BUSD")

# Use 4h candles for trading
candle_time_frame = TimeBucket.h4

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
    """The brain function to decide the trades on each trading strategy cycle.

    - Reads incoming execution state (positions, past trades)

    - Reads the current universe (candles)

    - Decides what to do next

    - Outputs strategy thinking for visualisation and debug messages

    :param timestamp:
        The Pandas timestamp object for this cycle. Matches
        trading_strategy_cycle division.
        Always truncated to the zero seconds and minutes, never a real-time clock.

    :param universe:
        Trading universe that was constructed earlier.

    :param state:
        The current trade execution state.
        Contains current open positions and all previously executed trades, plus output
        for statistics, visualisation and diangnostics of the strategy.

    :param pricing_model:
        Pricing model can tell the buy/sell price of the particular asset at a particular moment.

    :param cycle_debug_data:
        Python dictionary for various debug variables you can read or set, specific to this trade cycle.
        This data is discarded at the end of the trade cycle.

    :return:
        List of trade instructions in the form of :py:class:`TradeExecution` instances.
        The trades can be generated using `position_manager` but strategy could also hand craft its trades.
    """

    # The pair we are trading
    pair = universe.pairs.get_single()

    # How much cash we have in the hand
    cash = state.portfolio.get_current_cash()

    # Get OHLCV candles for our trading pair as Pandas Dataframe.
    # We could have candles for multiple trading pairs in a different strategy,
    # but this strategy only operates on single pair candle.
    # We also limit our sample size to N latest candles to speed up calculations.
    candles: pd.DataFrame = universe.candles.get_single_pair_data(sample_count=batch_size, raise_on_not_enough_data=False)

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
    # See available colours here
    # https://matplotlib.org/stable/gallery/color/named_colors.html
    visualisation = state.visualisation
    visualisation.plot_indicator(timestamp, "Slow EMA", PlotKind.technical_indicator_on_price, slow_ema, colour="forestgreen")
    visualisation.plot_indicator(timestamp, "Fast EMA", PlotKind.technical_indicator_on_price, fast_ema, colour="limegreen")

    return trades


def create_trading_universe(
        ts: datetime.datetime,
        client: Client,
        execution_context: ExecutionContext,
        universe_options: UniverseOptions,
) -> TradingStrategyUniverse:
    """Creates the trading universe where the strategy trades.

    If `execution_context.live_trading` is true then this function is called for
    every execution cycle. If we are backtesting, then this function is
    called only once at the start of backtesting and the `decide_trades`
    need to deal with new and deprecated trading pairs.

    As we are only trading a single pair, load data for the single pair only.

    :param ts:
        The timestamp of the trading cycle. For live trading,
        `create_trading_universe` is called on every cycle.
        For backtesting, it is only called at the start

    :param client:
        Trading Strategy Python client instance.

    :param execution_context:
        Information how the strategy is executed. E.g.
        if we are live trading or not.

    :param universe_options:
        TODO

    :return:
        This function must return :py:class:`TradingStrategyUniverse` instance
        filled with the data for exchanges, pairs and candles needed to decide trades.
        The trading universe also contains information about the reserve asset,
        usually stablecoin, we use for the strategy.
    """

    # Load all datas we can get for our candle time bucket
    dataset = load_all_data(
        client,
        candle_time_bucket,
        execution_context,
        universe_options)

    # Filter down to the single pair we are interested in
    universe = TradingStrategyUniverse.create_single_pair_universe(
        dataset,
        chain_id,
        exchange_slug,
        trading_pair[0],
        trading_pair[1],
    )

    return universe

