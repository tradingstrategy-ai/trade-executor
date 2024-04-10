"""Trade WMATIC-USDC on Uniswap v3 on Polygon using 1h candles.

To locally backtest this file:

.. code-block:: shell

    trade-executor \
        start \
        --strategy-file=strategy/enzyme-polygon-matic-usdc.py \
        --asset-management-mode=backtest \
        --backtest-start=2021-01-01 \
        --backtest-end=2022-01-01 \
        --trading-strategy-api-key=$TRADING_STRATEGY_API_KEY
"""

from typing import List, Dict

import pandas as pd
from pandas_ta import bbands
from pandas_ta.momentum import rsi

from tradeexecutor.state.visualisation import PlotKind
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.strategy_type import StrategyType
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.utils.binance import create_binance_universe
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
TRADING_STRATEGY_ENGINE_VERSION = "0.2"

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
TRADING_PAIR = (ChainId.polygon, "uniswap-v3", "WMATIC", "USDC")

# How much % of the cash to put on a single trade
POSITION_SIZE = 0.50

# Start with this amount of USD
INITIAL_CASH = 10_000

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
STDDEV = 2.8

# RSI must be above this value to open a new position.
RSI_THRESHOLD = 50

# What's the moving average length in candles for Bollinger bands
MOVING_AVERAGE_LENGTH = 19

# Backtest range
#
# Note that for this example notebook we deliberately choose a very short period,
# as the backtest completes faster, charts are more readable
# and tables shorter for the demostration.
#
BACKTEST_START = datetime.datetime(2022, 8, 1)

# Backtest range
BACKTEST_END = datetime.datetime(2023, 7, 1)

# Stop loss relative to the mid price during the time when the position is opened
#
# If the price drops below this level, trigger a stop loss
STOP_LOSS_PCT = 0.98

# What is the trailing stop loss level
TRAILING_STOP_LOSS_PCT = 0.9975

# Activate trailing stop loss when this level is reached
TRAILING_STOP_LOSS_ACTIVATION_LEVEL=1.05

STOP_LOSS_TIME_BUCKET = TimeBucket.m15


class Parameters:
    """Parameteres for this strategy.

    - Collect parameters used for this strategy here

    - Both live trading and backtesting parameters
    """

    cycle_duration = CycleDuration.cycle_1d  # Run decide_trades() every 8h
    source_time_bucket = TimeBucket.d1  # Use 1h candles as the raw data
    target_time_bucket = TimeBucket.d1  # Create synthetic 8h candles
    clock_shift_bars = 0  # Do not do shifted candles

    rsi_bars = 8  # Number of bars to calculate RSI for each tradingbar
    eth_btc_rsi_bars = 5  # Number of bars for the momentum factor

    # RSI parameters for bull and bear market
    bearish_rsi_entry = 65
    bearish_rsi_exit = 70
    bullish_rsi_entry = 80
    bullish_rsi_exit = 65

    regime_filter_ma_length = 200  # Bull/bear MA begime filter in days
    regime_filter_only_btc = 1   # Use BTC or per-pair regime filter

    trailing_stop_loss = None  # Trailing stop loss as 1 - x
    trailing_stop_loss_activation_level = 1.05  # How much above opening price we must be before starting to use trailing stop loss
    stop_loss = None  # 0.80  # Hard stop loss when opening a new position
    momentum_exponent = 3.5  # How much momentum we capture when rebalancing between open positions

    #
    # Live trading only
    #
    chain_id = ChainId.polygon
    routing = TradeRouting.default  # Pick default routes for trade execution
    required_history_period = datetime.timedelta(days=regime_filter_ma_length) * 2  # Ask some extra history just in case

    #
    # Backtesting only
    #

    backtest_start = datetime.datetime(2019, 1, 1)
    backtest_end = datetime.datetime(2024, 3, 15)
    stop_loss_time_bucket = TimeBucket.m15  # use 1h close as the stop loss signal
    backtest_trading_fee = 0.0005  # Switch to QuickSwap 30 BPS free from the default Binance 5 BPS fee


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
            new_trades = position_manager.open_1x_long(pair, buy_amount, stop_loss_pct=STOP_LOSS_PCT)
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
        stop_loss_time_bucket=STOP_LOSS_TIME_BUCKET,
    )

    # Convert loaded data to a trading pair universe
    universe = TradingStrategyUniverse.create_single_pair_universe(
        dataset,
        pair=TRADING_PAIR,
    )

    return universe



def get_strategy_trading_pairs(execution_mode: ExecutionMode) -> list[HumanReadableTradingPairDescription]:
    """Switch between backtest and live trading pairs.

    Because the live trading DEX venues do not have enough history (< 2 years)
    for meaningful backtesting, we test with Binance CEX data.
    """
    if execution_mode.is_live_trading():
        # Live trade
        return [
            (ChainId.polygon, "uniswap-v3", "WAMTIC", "USDC", 0.0005),
        ]
    else:
        # Backtest - Binance fee matched to DEXes with Parameters.backtest_trading_fee
        return [
            (ChainId.centralised_exchange, "binance", "MATIC", "USDT"),
        ]



def create_trading_universe(
    timestamp: datetime.datetime,
    client: Client,
    execution_context: ExecutionContext,
    universe_options: UniverseOptions,
) -> TradingStrategyUniverse:
    """Create the trading universe.

    - For live trading, we load DEX data

    - We backtest with Binance data, as it has more history
    """

    pair_ids = get_strategy_trading_pairs(execution_context.mode)

    if execution_context.mode.is_backtesting():
        # Backtesting - load Binance data
        start_at = universe_options.start_at
        end_at = universe_options.end_at
        strategy_universe = create_binance_universe(
            [f"{p[2]}{p[3]}" for p in pair_ids],
            candle_time_bucket=Parameters.source_time_bucket,
            stop_loss_time_bucket=Parameters.stop_loss_time_bucket,
            start_at=start_at,
            end_at=end_at,
            trading_fee_override=Parameters.backtest_trading_fee,
        )
    else:
        # Live trading - load DEX data
        universe_options = UniverseOptions(
            history_period=Parameters.required_history_period,
            start_at=None,
            end_at=None,
        )

        dataset = load_partial_data(
            client=client,
            time_bucket=Parameters.source_time_bucket,
            pairs=pair_ids,
            execution_context=execution_context,
            universe_options=universe_options,
            liquidity=False,
            stop_loss_time_bucket=Parameters.stop_loss_time_bucket,
        )
        # Construct a trading universe from the loaded data,
        # and apply any data preprocessing needed before giving it
        # to the strategy and indicators
        strategy_universe = TradingStrategyUniverse.create_from_dataset(
            dataset,
            reserve_asset="USDC",
            forward_fill=True,
        )

    return strategy_universe