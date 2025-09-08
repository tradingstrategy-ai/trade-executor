"""Correctly handle gaps in lending data (no Aave activity, no lending candle)."""
import warnings
from typing import List, Dict
from pandas_ta_classic.overlap import ema
from pandas_ta_classic.momentum import rsi, stoch
from pandas_ta_classic.volume import mfi
import pandas as pd
import datetime
import numpy as np

from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.lending import LendingProtocolType
from tradeexecutor.strategy.trading_strategy_universe import (
    TradingStrategyUniverse,
    load_partial_data,
)
from tradingstrategy.client import Client
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode, unit_test_execution_context
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.strategy_type import StrategyType
from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradingstrategy.universe import Universe
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.state.visualisation import PlotKind, PlotShape
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.state.state import State
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager


# following two functions not used in this test
def get_indices_of_uneven_intervals(df: pd.DataFrame | pd.Series) -> bool:
    """Checks if a time series contains perfectly evenly spaced time intervals with no gaps.

    :param df: Pandas dataframe or series
    :return: True if time series is perfectly evenly spaced, False otherwise
    """
    assert type(df.index) == pd.DatetimeIndex, "Index must be a DatetimeIndex"

    numeric_representation = df.index.astype(np.int64)

    differences = np.diff(numeric_representation)

    not_equal_to_first = differences != differences[0]

    return np.where(not_equal_to_first)[0]


def is_missing_data(df: pd.DataFrame | pd.Series) -> bool:
    """Checks if a time series contains perfectly evenly spaced time intervals with no gaps.

    :param df: Pandas dataframe or series
    :return: False if time series is perfectly evenly spaced, True otherwise
    """
    return len(get_indices_of_uneven_intervals(df)) > 0


CANDLE_TIME_BUCKET = TimeBucket.h1

TRADING_PAIR = (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005)

LENDING_RESERVES = [
    (ChainId.polygon, LendingProtocolType.aave_v3, "WETH"),
    (ChainId.polygon, LendingProtocolType.aave_v3, "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"),
]

STOP_LOSS_TIME_BUCKET = TimeBucket.m15

START_AT = datetime.datetime(2023, 1, 9)
END_AT = datetime.datetime(2023, 1, 12)

TRADING_STRATEGY_TYPE = StrategyType.managed_positions

# How our trades are routed.
TRADE_ROUTING = TradeRouting.uniswap_v3_usdc

INITIAL_DEPOSIT = 50_000

TRADING_STRATEGY_CYCLE = CycleDuration.cycle_1h

# Strategy keeps its cash in USDC
RESERVE_CURRENCY = ReserveCurrency.usdc


# strategy variables
POSITION_SIZE = 0.75
LOOKBACK_WINDOW = 20
EMA_CANDLE_COUNT = 2
RSI_LENGTH = 5
RSI_THRESHOLD = 25
RSI_THRESHOLD_SHORT = 75
STOP_LOSS_PCT = 0.992
TAKE_PROFIT_PCT = 1.15
TRAILING_STOP_LOSS_PCT = 0.985
TAKE_PROFIT_SHORT_PCT = 1.15
STOP_LOSS_SHORT_PCT = 0.992
TRAILING_STOP_LOSS_SHORT_PCT = 0.985
LEVERAGE = 2

def decide_trades(
    timestamp: pd.Timestamp,
    strategy_universe: TradingStrategyUniverse,
    state: State,
    pricing_model: PricingModel,
    cycle_debug_data: Dict,
) -> List[TradeExecution]:
    universe = strategy_universe.universe

    # We have only a single trading pair for this strategy.
    pair = universe.pairs.get_single()

    # How much cash we have in a hand
    cash = state.portfolio.get_current_cash()

    # Get OHLCV candles for our trading pair as Pandas Dataframe.
    # We could have candles for multiple trading pairs in a different strategy,
    # but this strategy only operates on single pair candle.
    # We also limit our sample size to N latest candles to speed up calculations.
    candles: pd.DataFrame = universe.candles.get_single_pair_data(
        timestamp, sample_count=LOOKBACK_WINDOW
    )

    # We have data for open, high, close, etc.
    # We only operate using candle close values in this strategy.
    close_prices = candles["close"]

    # Calculate exponential moving from candle close prices
    # https://tradingstrategy.ai/docs/programming/api/technical-analysis/overlap/help/pandas_ta.overlap.ema.html#ema
    ema_series = ema(close_prices, length=EMA_CANDLE_COUNT)

    # Calculate RSI from candle close prices
    # https://tradingstrategy.ai/docs/programming/api/technical-analysis/momentum/help/pandas_ta.momentum.rsi.html#rsi
    current_rsi = rsi(close_prices, length=RSI_LENGTH).iloc[-1]

    # Calculate Stochastic
    try:
        stoch_series = stoch(
            candles["high"], candles["low"], candles["close"], k=14, d=3, smooth_k=1
        )
    except Exception as e:
        # TALib issue
        # TA_SMA function failed with error code 2: Bad Parameter (TA_BAD_PARAM)
        # Raises ugly C exception
        if len(e.args) == 1:
            msg = e.args[0]
            if "TA_SMA function failed with error code" in msg:
                return []

    # Calculate MFI (Money Flow Index)
    with warnings.catch_warnings():
        # Pandas 2.0 hack
        #  tdf.loc[tdf["diff"] == 1, "+mf"] = raw_money_flow
        # FutureWarning: Setting an item of incompatible dtype is deprecated and will raise an error in a future version of pandas. Value '[  49232.40376278   18944.37769333  140253.87353008   20198.58223039
        #    80910.24155829  592340.20548335   43023.68515471  309655.74963533
        #   284633.70414388 2330901.62747533]' has dtype incompatible with int64, please explicitly cast to a compatible dtype first.

        warnings.simplefilter("ignore")
        mfi_series = mfi(
            candles["high"], candles["low"], candles["close"], candles["volume"] , length=17
        )

    trades = []

    ema_latest = ema_series.iloc[-1]  # Let's take the latest EMA value from the series
    price_latest = close_prices.iloc[
        -1
    ]  # Let's take the latest close price value from the series
    stoch_latest = stoch_series.iloc[
        -1, 0
    ]  # Let's take the latest Stoch value from the series
    stoch_second_latest = stoch_series.iloc[
        -2, 0
    ]  # Let's take the latest Stoch value from the series
    mfi_latest = mfi_series.iloc[-1]  # Let's take the latest MFI value from the series

    if ema_series is None:
        # Check if we cannot calculate EMA, because there's not enough data in backtesting buffer yet.
        return trades

    if stoch_series is None:
        # Check if we cannot calculate Stochastic, because there's not enough data in backtesting buffer yet.
        return trades

    if mfi_series is None:
        # Check if we cannot calculate MFI, because there's not enough data in backtesting buffer yet.
        return trades

    # Create a position manager helper class that allows us easily to create
    # opening/closing trades for different positions
    position_manager = PositionManager(
        timestamp, strategy_universe, state, pricing_model
    )

    ### LONGING ###

    stoploss_price_long = None  # We use this to track Stop Loss price for Long positions and draw it to the price chart

    if not position_manager.is_any_long_position_open():
        if (
            stoch_latest > stoch_second_latest
            and stoch_second_latest > stoch_series.iloc[-3, 0]
            and mfi_latest > mfi_series.iloc[-2]
            and mfi_latest < 45
        ):
            amount = cash * POSITION_SIZE
            new_trades = position_manager.open_spot(
                pair,
                amount,
                stop_loss_pct=STOP_LOSS_PCT,
                take_profit_pct=TAKE_PROFIT_PCT,
            )
            trades.extend(new_trades)
            stoploss_price_long = position_manager.get_current_long_position().stop_loss

    else:
        current_position = position_manager.get_current_long_position()
        # LONGING: We activate trailing stop loss when the price closes above the EMA line.

        if price_latest > current_position.get_opening_price():
            # adjust trailing stop loss level for the open long position
            # Stop loss is the only way we sell in this set up, unless TAKE_PROFIT_PCT level has been reached

            current_position.trailing_stop_loss_pct = TRAILING_STOP_LOSS_PCT
            stoploss_price_long = position_manager.get_current_long_position().stop_loss
            if position_manager.get_current_long_position().stop_loss <= float(
                price_latest * TRAILING_STOP_LOSS_PCT
            ):  # Move the trailing stop loss level only of the new value is higher
                current_position.stop_loss = float(
                    price_latest * TRAILING_STOP_LOSS_PCT
                )
                stoploss_price_long = (
                    position_manager.get_current_long_position().stop_loss
                )

    ### SHORTING ###

    stoploss_price_short = None  # We use this to track Stop Loss price for Short positions and draw it to the price chart

    if not position_manager.is_any_short_position_open():
        # No open positions, decide if open a position in this cycle.
        # We open short if the latest candle has upper wick above BB upper line and close under this line

        if (
            stoch_latest < stoch_second_latest
            and stoch_second_latest < stoch_series.iloc[-3, 0]
            and mfi_latest < mfi_series.iloc[-2]
            and mfi_latest > 55
        ):
            amount = cash * POSITION_SIZE
            new_trades = position_manager.open_short(
                pair,
                amount,
                leverage=LEVERAGE,
                stop_loss_pct=STOP_LOSS_SHORT_PCT,
                take_profit_pct=TAKE_PROFIT_SHORT_PCT,
            )
            trades.extend(new_trades)

            stoploss_price_short = (
                position_manager.get_current_short_position().stop_loss
            )

    else:
        current_position = position_manager.get_current_short_position()
        # SHORTING: We activate trailing stop loss when the price closes below the EMA line.

        if price_latest < current_position.get_opening_price():
            # adjust trailing stop loss level for the open short position
            # Stop loss is the only way we sell in this set up, unless TAKE_PROFIT_SHORT_PCT level has been reached
            current_position.trailing_stop_loss_pct = TRAILING_STOP_LOSS_SHORT_PCT
            stoploss_price_short = (
                position_manager.get_current_short_position().stop_loss
            )
            if position_manager.get_current_short_position().stop_loss >= float(
                price_latest * TRAILING_STOP_LOSS_SHORT_PCT
            ):  # Move the trailing stop loss level only of the new value is lower
                current_position.stop_loss = float(
                    price_latest * TRAILING_STOP_LOSS_SHORT_PCT
                )
                stoploss_price_short = (
                    position_manager.get_current_short_position().stop_loss
                )

    # Visualise our technical indicators
    visualisation = state.visualisation
    visualisation.plot_indicator(
        timestamp,
        "Stoch",
        PlotKind.technical_indicator_detached,
        stoch_latest,
        colour="black",
    )
    visualisation.plot_indicator(
        timestamp, "MFI", PlotKind.technical_indicator_detached, mfi_latest
    )
    #    visualisation.plot_indicator(timestamp, "RSI Threshold", PlotKind.technical_indicator_detached, RSI_THRESHOLD, detached_overlay_name="RSI")
    #    visualisation.plot_indicator(timestamp, "RSI Threshold", PlotKind.technical_indicator_detached, RSI_THRESHOLD_SHORT, detached_overlay_name="RSI")

    visualisation.plot_indicator(
        timestamp,
        "Stop Loss long",
        PlotKind.technical_indicator_on_price,
        stoploss_price_long,
        colour="purple",
        plot_shape=PlotShape.horizontal_vertical,
    )
    visualisation.plot_indicator(
        timestamp,
        "Stop Loss short",
        PlotKind.technical_indicator_on_price,
        stoploss_price_short,
        colour="blue",
        plot_shape=PlotShape.horizontal_vertical,
    )

    return trades


# np.where(df['timestamp'] == '2023-01-10 23:00:00')


def test_missing_lending_data(persistent_test_client):
    """See that backtest does not fail if lending candles are missing."""

    client = persistent_test_client

    universe_options = UniverseOptions(
        start_at=START_AT - datetime.timedelta(days=50),
        end_at=END_AT,
    )

    execution_context = unit_test_execution_context

    dataset = load_partial_data(
        client,
        execution_context=execution_context,
        time_bucket=CANDLE_TIME_BUCKET,
        pairs=[TRADING_PAIR],
        universe_options=universe_options,
        start_at=universe_options.start_at,
        end_at=universe_options.end_at,
        lending_reserves=LENDING_RESERVES,  # NEW
        stop_loss_time_bucket=STOP_LOSS_TIME_BUCKET,
    )

    strategy_universe = TradingStrategyUniverse.create_single_pair_universe(dataset)

    state, universe, debug_dump = run_backtest_inline(
        name="SMI and MFI strategy",
        start_at=START_AT,
        end_at=END_AT,
        client=client,
        cycle_duration=TRADING_STRATEGY_CYCLE,
        decide_trades=decide_trades,
        universe=strategy_universe,
        initial_deposit=INITIAL_DEPOSIT,
        reserve_currency=RESERVE_CURRENCY,
        trade_routing=TRADE_ROUTING,
        engine_version="0.3",
        three_leg_resolution=False,
    )

    trade_count = len(list(state.portfolio.get_all_trades()))
    print(f"Backtesting completed, backtested strategy made {trade_count} trades")

    # problem_df = df[7294:7300]

    # assert not is_missing_data(problem_df)

