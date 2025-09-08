"""Pancake EMA example strategy.

To "speedrun" backtest locally to see if the code does not have implementation issues:

.. code-block:: shell

    # Set your API key
    export TRADING_STRATEGY_API_KEY=...

    # Run the backtest of this module using local trade-executor command
    # Tick size and stop loss check frequencies are less from what the strategy
    # is expected (1h -> 1d). We call decide_trades less often,
    # allowing us to complete the test faster, albeit with incorrect
    # results.
    trade-executor start \
        --strategy-file=strategies/pancake-eth-usdc-sma.py \
        --execution-type=backtest \
        --trading-strategy-api-key=$TRADING_STRATEGY_API_KEY \
        --cycle-duration=1d \
        --stop-loss-check-frequency=1d \
        --backtest-start=2021-06-01 \
        --backtest-end=2022-09-01


You can also do the full backtest run locally. This might take a long time depending on your CPU speed:

.. code-block:: shell

    # Set your API key
    export TRADING_STRATEGY_API_KEY=...

    # Run the backtest of this module using local trade-executor command
    # Tick size must match what the strategy is expecting
    trade-executor start \
        --strategy-file=strategies/pancake-eth-usdc-sma.py \
        --execution-type=backtest \
        --trading-strategy-api-key=$TRADING_STRATEGY_API_KEY \
        --backtest-start=2021-06-01 \
        --backtest-end=2022-09-01

"""

import logging
import datetime
import pandas as pd
from pandas_ta_classic import ema

from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.visualisation import PlotKind
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.strategy_module import StrategyType, TradeRouting, ReserveCurrency

from typing import Optional, List, Dict
from tradeexecutor.strategy.trading_strategy_universe import load_pair_data_for_single_exchange, TradingStrategyUniverse
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradingstrategy.client import Client

from tradeexecutor.strategy.universe_model import UniverseOptions
from tradingstrategy.utils.groupeduniverse import NoDataAvailable

# Tell what trade execution engine version this strategy needs to use
# NOTE: this setting has currently no effect
TRADING_STRATEGY_ENGINE_VERSION = "0.1"

# What kind of strategy we are running.
# This tells we are going to use
# NOTE: this setting has currently no effect
TRADING_STRATEGY_TYPE = StrategyType.managed_positions

# How our trades are routed.
# PancakeSwap basic routing supports two way trades with BUSD
# and three way trades with BUSD-BNB hop.
TRADE_ROUTING = TradeRouting.pancakeswap_usdc
# How often the strategy performs the decide_trades cycle.
# We do it for every 4h.
TRADING_STRATEGY_CYCLE = CycleDuration.cycle_1h

# Strategy keeps its cash in USDC
RESERVE_CURRENCY = ReserveCurrency.usdc

# Time bucket for our candles
CANDLE_TIME_BUCKET = TimeBucket.h1

# Which chain we are trading
CHAIN_ID = ChainId.bsc

# Which exchange we are trading on.
EXCHANGE_SLUG = "pancakeswap-v2"

# Which trading pair we are trading
TRADING_PAIR = ("ETH", "USDC")

# How much of the cash to put on a single trade
POSITION_SIZE = 0.70

#
# Strategy thinking specific parameter
#

BATCH_SIZE = 90

SLOW_EMA_CANDLE_COUNT = 10
FAST_EMA_CANDLE_COUNT = 3


# Range of backtesting and synthetic data generation.
# Because we are using synthetic data actual dates do not really matter -
# only the duration

START_AT = datetime.datetime(2022, 1, 1)

END_AT = datetime.datetime(2022, 10, 18)

# Start with 10,000 USD
INITIAL_DEPOSIT = 10_000

# If the price drops 0.7% we trigger a stop loss
STOP_LOSS_PCT = 0.993

STOP_LOSS_TIME_BUCKET = TimeBucket.m15


logger = logging.getLogger(__name__)


def decide_trades(
        timestamp: pd.Timestamp,
        universe: Universe,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: Dict) -> List[TradeExecution]:
    """The brain function to decide the trades on each trading strategy cycle.

    - Reads incoming execution state (positions, past trades)

    - Reads the current universe (candles)

    - Decides what trades to do next, if any, at current timestamp.

    - Outputs strategy thinking for visualisation and debug messages

    :param timestamp:
        The Pandas timestamp object for this cycle. Matches
        TRADING_STRATEGY_CYCLE division.
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
    cash = state.portfolio.get_cash()

    # Get OHLCV candles for our trading pair as Pandas Dataframe.
    # We could have candles for multiple trading pairs in a different strategy,
    # but this strategy only operates on single pair candle.
    # We also limit our sample size to N latest candles to speed up calculations.
    try:
        candles: pd.DataFrame = universe.candles.get_single_pair_data(timestamp, sample_count=BATCH_SIZE)
    except NoDataAvailable:
        return []  # Cannot amke trades yet

    # We have data for open, high, close, etc.
    # We only operate using candle close values in this strategy.
    close_prices = candles["close"]

    # Calculate exponential moving averages based on slow and fast sample numbers.
    slow_ema_series = ema(close_prices, length=SLOW_EMA_CANDLE_COUNT)
    fast_ema_series = ema(close_prices, length=FAST_EMA_CANDLE_COUNT)

    if slow_ema_series is None or fast_ema_series is None:
        # Cannot calculate EMA, because
        # not enough samples in backtesting
        logger.warning("slow_ema_series or fast_ema_series None")
        return []

    if len(slow_ema_series) < 2 or len(fast_ema_series) < 2:
        # We need at least two data points to determine if EMA crossover (or crossunder)
        # occurred at current timestamp.
        logger.warning("series too short")
        return []

    slow_ema_latest = slow_ema_series.iloc[-1]
    fast_ema_latest = fast_ema_series.iloc[-1]
    price_latest = close_prices.iloc[-1]

    # Compute technical indicators needed for trade decisions.
    slow_ema_crossover = (
            close_prices.iloc[-3] < slow_ema_series.iloc[-2]
            and price_latest > slow_ema_latest
    )
    slow_ema_crossunder = (
            close_prices.iloc[-2] > slow_ema_series.iloc[-2]
            and price_latest < slow_ema_latest
    )
    fast_ema_crossunder = (
            close_prices.iloc[-2] > fast_ema_series.iloc[-2]
            and price_latest < fast_ema_latest
    )

    #    entry_price = tradeposition.open_price
    #    entry_price = float(get_buy_price['amount'])
    # position_manager = PositionManager(timestamp, universe, state, pricing_model)

    trades = []

    # entry_price = position_manager.get_current_position().get_current_price()

    # entry_price = tradeexecutor.analysis.trade_analyser.SpotTrade

    # List of any trades we decide on this cycle.
    # trades = []

    # Create a position manager helper class that allows us easily to create
    # opening/closing trades for different positions
    position_manager = PositionManager(timestamp, universe, state, pricing_model)

    if not position_manager.is_any_open():
        # No open positions, decide if BUY in this cycle.
        # We buy if we just crossed over the slow EMA or if this is a very first
        # trading cycle and the price is already above the slow EMA.

        logger.info("No positions open, check for new trades")

        if (
                slow_ema_crossunder
                or price_latest < slow_ema_latest and timestamp == START_AT
        ):
            buy_amount = cash * POSITION_SIZE
            new_trades = position_manager.open_1x_long(pair, buy_amount, stop_loss_pct=STOP_LOSS_PCT)
            trades.extend(new_trades)
            logger.info("New trade decided")
        else:
            logger.info("Did not decide a new trade")

    else:

        logger.info("Checking for close")

        # We have an open position, decide if SELL in this cycle.
        # We do that if we fall below any of the two moving averages.
        if slow_ema_crossover or (fast_ema_crossunder and fast_ema_latest > slow_ema_latest):
            new_trades = position_manager.close_all()
            assert len(new_trades) == 1
            trades.extend(new_trades)

        # else:
        #    current_position = position_manager.get_current_position()
        #    current_price = current_position.get_current_price()
        # if price_latest <= current_price * STOP_LOSS:
        #    print(f"Stop loss. Now {close}, opened at {entry_price}")
        #    new_trades = position_manager.close_all()
        #    assert len(new_trades) == 1
        #    trades.extend(new_trades)

    # Visualize strategy
    # See available Plotly colours here
    # https://community.plotly.com/t/plotly-colours-list/11730/3?u=miohtama
    visualisation = state.visualisation
    visualisation.plot_indicator(timestamp, "Slow EMA", PlotKind.technical_indicator_on_price, slow_ema_latest,
                                 colour="green")
    visualisation.plot_indicator(timestamp, "Fast EMA", PlotKind.technical_indicator_on_price, fast_ema_latest,
                                 colour="red")

    return trades


def create_trading_universe(
        ts: datetime.datetime,
        client: Client,
        execution_context: ExecutionContext,
        universe_options: UniverseOptions,
) -> TradingStrategyUniverse:

    dataset = load_pair_data_for_single_exchange(
        client,
        execution_context,
        CANDLE_TIME_BUCKET,
        CHAIN_ID,
        EXCHANGE_SLUG,
        [TRADING_PAIR],
        universe_options,
        stop_loss_time_bucket=STOP_LOSS_TIME_BUCKET,
    )

    # Filter down to the single pair we are interested in
    universe = TradingStrategyUniverse.create_single_pair_universe(
        dataset,
        CHAIN_ID,
        EXCHANGE_SLUG,
        TRADING_PAIR[0],
        TRADING_PAIR[1],
    )

    logger.trade("Universe created, we have %d pairs", universe.get_pair_count())

    return universe


