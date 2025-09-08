"""Test spot-short strategy."""

import datetime
from typing import List, Dict

import pandas as pd

from pandas_ta_classic import bbands
from pandas_ta_classic.overlap import ema
from pandas_ta_classic.momentum import rsi

from tradingstrategy.client import Client
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.lending import LendingProtocolType

from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_partial_data
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.state.visualisation import PlotKind
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.state.state import State
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradingstrategy.utils.groupeduniverse import NoDataAvailable

TRADING_STRATEGY_ENGINE_VERSION = "0.3"

NAME = "ETH mean reversion bounce"

SHORT_DESCRIPTION = "Capture mean reversion volatility of ETH on the 4h timeframe"

LONG_DESCRIPTION = """
The strategy focuses on capturing 4h volatility of ETH by utilising mean reversion to either long or short direction, following the on the direction of the current price action trend. 
<br>
The strategy trades Uniswap ETH spot market up, and Aave leveraged short positions with 2x leverage down.  
<br>
The strategy uses a fairly large relative position size, however there is a fixed % stop loss implemented for each position to minimise risks of too heavy volatility 
"""

# How often the strategy performs the decide_trades cycle.
# We do it for every 4h.
TRADING_STRATEGY_CYCLE = CycleDuration.cycle_4h

# Time bucket for our candles
CANDLE_TIME_BUCKET = TimeBucket.h4

# Which trading pair we are backtesting on
TRADING_PAIR = (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005)

# Which lending reserves we are using for supplying/borrowing assets
# NEW
LENDING_RESERVES = [
    (ChainId.polygon, LendingProtocolType.aave_v3, "WETH"),
    (ChainId.polygon, LendingProtocolType.aave_v3, "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"),
]

# How much % of the cash to put on a single trade
POSITION_SIZE = 0.8

# Start with this amount of USD
INITIAL_CASH = 50_000

# Candle time granularity we use to trigger stop loss checks
STOP_LOSS_TIME_BUCKET = TimeBucket.h1

#
# Strategy thinking specific parameter
#

# How many candles we load in the decide_trades() function for calculating indicators
LOOKBACK_WINDOW = 200

# Moving average
# How many candles to smooth out for Bollinger band's middle line
EMA_CANDLE_COUNT = 20


# How many candles we use to calculate the Relative Strength Indicator
RSI_LENGTH = 14

# RSI must be below this value to open a new position
RSI_THRESHOLD = 48

# RSI must be above this value to open a new position
RSI_THRESHOLD_SHORT = 52

# Backtest range
BACKTEST_START = datetime.datetime(2022, 9, 1)
START_AT_DATA = datetime.datetime(2022, 9, 1) #This is only for Binance data

# Backtest range
BACKTEST_END = datetime.datetime(2023, 12, 13)

# Stop loss relative to the mid price during the time when the position is opened
# If the price drops below this level, trigger a stop loss
STOP_LOSS_PCT = 0.96

STOP_LOSS_SHORT_PCT = 0.96

# Take profit percentage
TAKE_PROFIT_PCT = 1.055

TAKE_PROFIT_SHORT_PCT = 1.055

# Leverage ratio
LEVERAGE = 2



def decide_trades(
        timestamp: pd.Timestamp,
        strategy_universe: TradingStrategyUniverse,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: Dict) -> List[TradeExecution]:

    universe = strategy_universe.universe

    # We have only a single trading pair for this strategy.
    pair = universe.pairs.get_single()

    # How much cash we have in a hand
    cash = state.portfolio.get_current_cash()

    # Get OHLCV candles for our trading pair as Pandas Dataframe.
    # We could have candles for multiple trading pairs in a different strategy,
    # but this strategy only operates on single pair candle.
    # We also limit our sample size to N latest candles to speed up calculations.
    try:
        candles: pd.DataFrame = universe.candles.get_single_pair_data(timestamp, sample_count=LOOKBACK_WINDOW)
    except NoDataAvailable:
        # Return no trades at the start of the backtesting period
        return []

    # We have data for open, high, close, etc.
    # We only operate using candle close values in this strategy.
    close_prices = candles["close"]

    # Calculate exponential moving for candle close
    # https://tradingstrategy.ai/docs/programming/api/technical-analysis/overlap/help/pandas_ta.overlap.ema.html#ema
    moving_average = ema(close_prices, length=EMA_CANDLE_COUNT)
    trend_ema = ema(close_prices, length=200)

#    trend_moving_average = ema(close_prices, length=TREND_EMA_CANDLE_COUNT)

    # Calculate RSI for candle close
    # https://tradingstrategy.ai/docs/programming/api/technical-analysis/momentum/help/pandas_ta.momentum.rsi.html#rsi
    current_rsi = rsi(close_prices, length=RSI_LENGTH)[-1]

    trades = []

    if moving_average is None or trend_ema is None:
        # Cannot calculate EMA, because
        # not enough samples in backtesting buffer yet.
        return trades

    price_close = close_prices.iloc[-1]

    # Create a position manager helper class that allows us easily to create
    # opening/closing trades for different positions
    position_manager = PositionManager(timestamp, strategy_universe, state, pricing_model)

    # Calculate narrow Bollinger Bands using the typical 20-day SMA and 1 standard deviations
    bollinger_bands = bbands(close_prices, length=20, std=1)
    bb_upper = bollinger_bands["BBU_20_1.0"]
    bb_lower = bollinger_bands["BBL_20_1.0"]

    # Calculate wider Bollinger Bands using the typical 20-day SMA and 2 standard deviations
    bollinger_bands_wide = bbands(close_prices, length=20, std=2)
    wide_bb_upper = bollinger_bands_wide["BBU_20_2.0"]
    wide_bb_lower = bollinger_bands_wide["BBL_20_2.0"]

    ### LONGING
    if not position_manager.is_any_long_position_open():
        # We open long if the latest candle has upper wick above BB upper line and close under this line
        if price_close < wide_bb_lower.iloc[-1] and close_prices.iloc[-2] > wide_bb_lower.iloc[-2] and current_rsi < RSI_THRESHOLD and (price_close > (1.01 * trend_ema.iloc[-1])):
            amount = cash * POSITION_SIZE
            new_trades = position_manager.open_1x_long(pair, amount, stop_loss_pct=STOP_LOSS_PCT, take_profit_pct=TAKE_PROFIT_PCT)
            trades.extend(new_trades)
    else:
        # LONGING: We close the position when the price closes above the 20-day moving average.
        if price_close > wide_bb_upper.iloc[-1] and close_prices.iloc[-2] < wide_bb_upper.iloc[-2]:
            current_position = position_manager.get_current_long_position()
            new_trades = position_manager.close_position(current_position)
            trades.extend(new_trades)

    ### SHORTING
    if not position_manager.is_any_short_position_open():
        # No open positions, decide if open a position in this cycle.
        # We open short if the latest candle has upper wick above BB upper line and close under this line
        if price_close > wide_bb_upper.iloc[-1] and close_prices.iloc[-2] < wide_bb_upper.iloc[-2] and current_rsi > RSI_THRESHOLD_SHORT and (price_close < (0.99 * trend_ema.iloc[-1])):
            amount = cash * POSITION_SIZE
            new_trades = position_manager.open_short(pair, amount, leverage=LEVERAGE, stop_loss_pct=STOP_LOSS_SHORT_PCT, take_profit_pct=TAKE_PROFIT_SHORT_PCT)
            trades.extend(new_trades)
    else:
        # We close the position when the price closes below the 20-day moving average.
        if price_close < wide_bb_lower.iloc[-1] and close_prices.iloc[-2] > wide_bb_lower.iloc[-2]:
            current_position = position_manager.get_current_short_position()
            new_trades = position_manager.close_position(current_position)
            trades.extend(new_trades)

    # Visualise our technical indicators
    visualisation = state.visualisation
    visualisation.plot_indicator(timestamp, "Wide BB upper", PlotKind.technical_indicator_on_price, wide_bb_upper.iloc[-1], colour="red")
    visualisation.plot_indicator(timestamp, "Wide BB lower", PlotKind.technical_indicator_on_price, wide_bb_lower.iloc[-1], colour="red")
    visualisation.plot_indicator(timestamp, "EMA", PlotKind.technical_indicator_on_price, moving_average.iloc[-1], colour="black")
    visualisation.plot_indicator(timestamp, "Trend EMA", PlotKind.technical_indicator_on_price, trend_ema.iloc[-1], colour="grey")
    visualisation.plot_indicator(timestamp, "RSI", PlotKind.technical_indicator_detached, current_rsi)
    visualisation.plot_indicator(timestamp, "RSI Threshold", PlotKind.technical_indicator_overlay_on_detached, RSI_THRESHOLD, detached_overlay_name="RSI")
    visualisation.plot_indicator(timestamp, "RSI Short Threshold", PlotKind.technical_indicator_overlay_on_detached, RSI_THRESHOLD_SHORT, detached_overlay_name="RSI")


    return trades


def create_trading_universe(
        ts: datetime.datetime,
        client: Client,
        execution_context: ExecutionContext,
        universe_options: UniverseOptions,
) -> TradingStrategyUniverse:

    # CHoose between live data and historical backtesting data
    if execution_context.mode.is_live_trading():
        start = None
        end = None
        required_history_period = datetime.timedelta(days=180)
    else:
        start = universe_options.start_at
        end = universe_options.end_at
        required_history_period = None

    dataset = load_partial_data(
        client,
        execution_context=execution_context,
        time_bucket=CANDLE_TIME_BUCKET,
        pairs=[TRADING_PAIR],
        universe_options=universe_options,
        start_at=start,
        end_at=end,
        required_history_period=required_history_period,
        lending_reserves=LENDING_RESERVES,
        stop_loss_time_bucket=STOP_LOSS_TIME_BUCKET,
    )

    # Filter down to the single pair we are interested in
    strategy_universe = TradingStrategyUniverse.create_single_pair_universe(dataset)

    if not execution_context.mode.is_live_trading():
        assert strategy_universe.backtest_stop_loss_candles is not None

    return strategy_universe