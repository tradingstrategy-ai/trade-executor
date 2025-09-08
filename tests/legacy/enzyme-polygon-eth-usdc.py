"""Used with test_legacy_repair_broadcast.py
"""

from typing import List, Dict

import pandas as pd
from pandas_ta_classic import bbands
from pandas_ta_classic.momentum import rsi

from tradeexecutor.state.visualisation import PlotKind
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.strategy_type import StrategyType
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradingstrategy.client import Client
from tradingstrategy.universe import Universe

import datetime

from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.strategy_module import TradeRouting, ReserveCurrency
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.state.state import State
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_pair_data_for_single_exchange

#
# Strategy properties
#

# Tell what trade execution engine version this strategy needs to use
# NOTE: this setting has currently no effect
TRADING_STRATEGY_ENGINE_VERSION = "0.1"

# What kind of strategy we are running.
# This tells we are going to use
# NOTE: this setting has currently no effect
TRADING_STRATEGY_TYPE = StrategyType.managed_positions

# We trade on Polygon
CHAIN_ID = ChainId.polygon

# How our trades are routed.
# PancakeSwap basic routing supports two way trades with BUSD
# and three way trades with BUSD-BNB hop.
TRADE_ROUTING = TradeRouting.uniswap_v3_usdc_poly

# How often the strategy performs the decide_trades cycle.
TRADING_STRATEGY_CYCLE = CycleDuration.cycle_1h

# Time bucket for our candles
CANDLE_TIME_BUCKET = TimeBucket.h1

# Candle time granularity we use to trigger stop loss checks
STOP_LOSS_TIME_BUCKET = TimeBucket.m15

# Strategy keeps its cash in USDC
RESERVE_CURRENCY = ReserveCurrency.usdc

# Which trading pair we are backtesting on
# (Might be different from the live trading pair)
# https://tradingstrategy.ai/trading-view/polygon/quickswap/eth-usdc
TRADING_PAIR = (ChainId.polygon, "uniswap-v3", "WETH", "USDC")

# How much % of the cash to put on a single trade
POSITION_SIZE = 0.50

# Start with this amount of USD
INITIAL_DEPOSIT = 5_000

#
# Strategy inputs
#

# How many candles we load in the decide_trades() function for calculating indicators
LOOKBACK_WINDOW = 90

# How many candles we use to calculate the Relative Strength Indicator
RSI_LENGTH = 14

#
# Grid searched parameters
#_loss

# Bollinger band's standard deviation options
#
# STDDEV = [1.0, 1.5, 1.7, 2.0, 2.5, 2.8]
STDDEV = 2.5

# RSI must be above this value to open a new position.
RSI_THRESHOLD = 65

# What's the moving average length in candles for Bollinger bands
MOVING_AVERAGE_LENGTH = 30

# Backtest range
#
# Note that for this example notebook we deliberately choose a very short period,
# as the backtest completes faster, charts are more readable
# and tables shorter for the demostration.
#
START_AT = datetime.datetime(2020, 9, 1)

# Backtest range
END_AT = datetime.datetime(2023, 4, 1)

# Stop loss relative to the mid price during the time when the position is opened
#
# If the price drops below this level, trigger a stop loss
STOP_LOSS_PCT = 0.98

# What is the trailing stop loss level
TRAILING_STOP_LOSS_PCT = 0.9975

# Activate trailing stop loss when this level is reached
TRAILING_STOP_LOSS_ACTIVATION_LEVEL=1.03


def decide_trades(
        timestamp: pd.Timestamp,
        universe: Universe,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: Dict) -> List[TradeExecution]:
    # Trades generated in this cycle
    trades = []

    # We have only a single trading pair for this strategy.
    pair = universe.pairs.get_single()

    # Assume 0.05% fee on live trading
    pair.fee = 0.0005

    # How much cash we have in a hand
    cash = state.portfolio.get_current_cash()

    # Get OHLCV candles for our trading pair as Pandas Dataframe.
    # We could have candles for multiple trading pairs in a different strategy,
    # but this strategy only operates on single pair candle.
    # We also limit our sample size to N latest candles to speed up calculations.
    candles: pd.DataFrame = universe.candles.get_single_pair_data(timestamp, sample_count=LOOKBACK_WINDOW)

    if len(candles) == 0:
        # We are looking back so far in the history that the pair is not trading yet
        return trades

    # We have data for open, high, close, etc.
    # We only operate using candle close values in this strategy.
    close_prices = candles["close"]

    price_latest = close_prices.iloc[-1]

    # Create a position manager helper class that allows us easily to create
    # opening/closing trades for different positions
    position_manager = PositionManager(timestamp, universe, state, pricing_model)

    # Calculate RSI for candle close
    # https://tradingstrategy.ai/docs/programming/api/technical-analysis/momentum/help/pandas_ta.momentum.rsi.html#rsi
    rsi_series = rsi(close_prices, length=RSI_LENGTH)
    if rsi_series is None:
        # Not enough data in the backtesting buffer yet
        return trades

    # Calculate Bollinger Bands with a 20-day SMA and 2 standard deviations using pandas_ta
    # See documentation here https://tradingstrategy.ai/docs/programming/api/technical-analysis/volatility/help/pandas_ta.volatility.bbands.html#bbands
    bollinger_bands = bbands(close_prices, length=MOVING_AVERAGE_LENGTH, std=STDDEV)

    if bollinger_bands is None:
        # Not enough data in the backtesting buffer yet
        return trades

    # bbands() returns a dictionary of items with different name mangling
    bb_upper = bollinger_bands[f"BBU_{MOVING_AVERAGE_LENGTH}_{STDDEV}"]
    bb_lower = bollinger_bands[f"BBL_{MOVING_AVERAGE_LENGTH}_{STDDEV}"]
    bb_mid = bollinger_bands[f"BBM_{MOVING_AVERAGE_LENGTH}_{STDDEV}"]  # Moving average

    if not position_manager.is_any_open():
        # No open positions, decide if BUY in this cycle.
        # We buy if the price on the daily chart closes above the upper Bollinger Band.
        if price_latest > bb_upper.iloc[-1] and rsi_series[-1] >= RSI_THRESHOLD:
            buy_amount = cash * POSITION_SIZE
            new_trades = position_manager.open_spot(pair, buy_amount, stop_loss_pct=STOP_LOSS_PCT)
            trades.extend(new_trades)

    else:
        # We have an open position, decide if SELL in this cycle.
        # We close the position when the price closes below the 20-day moving average.
        if price_latest < bb_mid.iloc[-1]:
            new_trades = position_manager.close_all()
            trades.extend(new_trades)

        # Check if we have reached out level where we activate trailing stop loss
        position = position_manager.get_current_position()
        if price_latest >= position.get_opening_price() * TRAILING_STOP_LOSS_ACTIVATION_LEVEL:
            position.trailing_stop_loss_pct = TRAILING_STOP_LOSS_PCT
            position.stop_loss = float(price_latest * TRAILING_STOP_LOSS_PCT)

    # Visualise our technical indicators
    visualisation = state.visualisation
    visualisation.plot_indicator(timestamp, "BB upper", PlotKind.technical_indicator_on_price, bb_upper.iloc[-1],
                                 colour="darkblue")
    visualisation.plot_indicator(timestamp, "BB lower", PlotKind.technical_indicator_on_price, bb_lower.iloc[-1],
                                 colour="darkblue")
    visualisation.plot_indicator(timestamp, "BB mid", PlotKind.technical_indicator_on_price, bb_mid.iloc[-1],
                                 colour="blue")

    # Draw the RSI indicator on a separate chart pane.
    # Visualise the high RSI threshold we must exceed to take a position.
    visualisation.plot_indicator(timestamp, "RSI", PlotKind.technical_indicator_detached, rsi_series[-1])
    visualisation.plot_indicator(timestamp, "RSI threshold", PlotKind.technical_indicator_overlay_on_detached,
                                 RSI_THRESHOLD, colour="red", detached_overlay_name="RSI")

    return trades


def create_trading_universe(
        ts: datetime.datetime,
        client: Client,
        execution_context: ExecutionContext,
        universe_options: UniverseOptions,
):
    assert isinstance(client, Client), f"Looks like we are not running on the real data. Got: {client}"

    # Download live data from the oracle
    dataset = load_pair_data_for_single_exchange(
        client,
        time_bucket=CANDLE_TIME_BUCKET,
        pair_tickers=[TRADING_PAIR],
        execution_context=execution_context,
        universe_options=universe_options,
    )

    # Convert loaded data to a trading pair universe
    universe = TradingStrategyUniverse.create_single_pair_universe(
        dataset,
        pair=TRADING_PAIR,
    )

    return universe
