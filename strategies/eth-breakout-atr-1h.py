"""ETH ATR based breakout strategy.

Based on eth-breakout 1h notebook in getting-started repo.

To backtest this strategy module locally:

.. code-block:: console

    trade-executor \
        backtest \
        --strategy-file=strategies/eth-breakout-atr-1h.py \
        --trading-strategy-api-key=$TRADING_STRATEGY_API_KEY \
        --python-profile-report=backtest.cprof


"""

import datetime
import os

import pandas as pd
import pandas_ta

from tradingstrategy.utils.groupeduniverse import resample_candles
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket

from tradeexecutor.analysis.regime import Regime
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet, IndicatorSource
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.trading_strategy_universe import load_partial_data
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.state.visualisation import PlotKind
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.utils.binance import create_binance_universe


trading_strategy_engine_version = "0.5"

# List of trading pairs we use in the backtest
# In this backtest, we use Binance data as it has more price history than DEXes
trading_pairs = [
    (ChainId.polygon, "uniswap-v3", "WETH", "USDC"),
]


class Parameters:
    """Parameteres for this strategy.

    - Collect parameters used for this strategy here

    - Both live trading and backtesting parameters
    """

    id = "eth-breakout-atr-1h" # Used in cache paths

    cycle_duration = CycleDuration.h1
    candle_time_bucket = TimeBucket.h1
    allocation = 0.98

    atr_length = 10  #
    fract = 3.0  # Fraction between last hourly close and breakout level

    adx_length = 14  # 14 days
    adx_filter_threshold = 15

    trailing_stop_loss_pct = 0.99
    trailing_stop_loss_activation_level = 1.10
    stop_loss_pct = 0.98

    #
    # Live trading only
    #
    chain_id = ChainId.polygon
    routing = TradeRouting.default  # Pick default routes for trade execution
    required_history_period = datetime.timedelta(hours=200)

    #
    # Backtesting only
    #

    use_binance_data = True
    if use_binance_data:
        # Perform backesting on binance data instead of DEX data, allows longer backtesting period
        backtest_start = datetime.datetime(2019, 8, 1)
        backtest_end = datetime.datetime(2024, 5, 1)
    else:
        # WETH-USDC 5 BPS is not available on Polygon unti 2022-08
        backtest_start = datetime.datetime(2022, 8, 1)
        backtest_end = datetime.datetime(2024, 5, 15)
    stop_loss_time_bucket = TimeBucket.m5
    backtest_trading_fee = 0.0005  # Override the default Binance data trading fee and assume we can trade 5 BPS fee on WMATIC-USDC on Polygon on Uniswap v3
    initial_cash = 10_000



def daily_price(open, high, low, close) -> pd.DataFrame:
    """Resample pricees to daily for ADX filtering."""
    original_df = pd.DataFrame({
        "open": open,
        "high": high,
        "low": low,
        "close": close,
    })
    daily_df = resample_candles(original_df, pd.Timedelta(days=1))
    return daily_df


def daily_adx(open, high, low, close, length):
    daily_df = daily_price(open, high, low, close)
    adx_df = pandas_ta.adx(
        close=daily_df.close,
        high=daily_df.high,
        low=daily_df.low,
        length=length,
    )
    return adx_df


def regime(open, high, low, close, length, regime_threshold) -> pd.Series:
    """A regime filter based on ADX indicator.

    Get the trend of BTC applying ADX on a daily frame.

    - -1 is bear
    - 0 is sideways
    - +1 is bull
    """
    adx_df = daily_adx(open, high, low, close, length)
    def regime_filter(row):
        # ADX, DMP, # DMN
        average_direction_index, directional_momentum_positive, directional_momentum_negative = row.values
        if directional_momentum_positive > regime_threshold:
            return Regime.bull.value
        elif directional_momentum_negative > regime_threshold:
            return Regime.bear.value
        else:
            return Regime.crab.value
    regime_signal = adx_df.apply(regime_filter, axis="columns")
    return regime_signal



def create_indicators(
    timestamp: datetime.datetime | None,
    parameters: StrategyParameters,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext
):
    indicators = IndicatorSet()

    # https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/volatility/atr.py
    indicators.add(
        "atr",
        pandas_ta.atr,
        {"length": parameters.atr_length},
        IndicatorSource.ohlcv,
    )

    # ADX https://www.investopedia.com/articles/trading/07/adx-trend-indicator.asp
    # https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/trend/adx.py
    indicators.add(
        "adx",
        daily_adx,
        {"length": parameters.adx_length},
        IndicatorSource.ohlcv,
    )

    # Price OHLC resampled to daily
    indicators.add(
        "daily_price",
        daily_price,
        {},
        IndicatorSource.ohlcv,
    )

    # A regime filter to detect the trading pair bear/bull markets
    indicators.add(
        "regime",
        regime,
        {"length": parameters.adx_length, "regime_threshold": parameters.adx_filter_threshold},
        IndicatorSource.ohlcv,
    )

    return indicators




def decide_trades(
    input: StrategyInput,
) -> list[TradeExecution]:

    #
    # Decidion cycle setup.
    # Read all variables we are going to use for the decisions.
    #
    parameters = input.parameters
    position_manager = input.get_position_manager()
    state = input.state
    timestamp = input.timestamp
    indicators = input.indicators
    strategy_universe = input.strategy_universe

    pair = strategy_universe.get_single_pair()
    cash = position_manager.get_current_cash()

    #
    # Indicators
    #

    close_price = indicators.get_price()  # Price the previous 15m candle closed for this decision cycle timestamp
    atr = indicators.get_indicator_value("atr")  # The ATR value at the time of close price
    point_of_interest = (timestamp - parameters.cycle_duration.to_pandas_timedelta()).floor(freq="D")  # POI (point of interest): Account 15m of lookahead bias whehn using decision cycle timestamp
    previous_price = indicators.get_price(timestamp=point_of_interest)  # The price at the start of this hour
    regime_val = indicators.get_indicator_value("regime", data_delay_tolerance=pd.Timedelta(hours=24))  # Because the regime filter is calculated only daily, we allow some lookback

    if None in (atr, close_price, previous_price):
        # Not enough historic data,
        # cannot make decisions yet
        return []

    # If regime filter does not have enough data at the start of the backtest,
    # default to bull market
    if regime_val is None:
        regime = Regime.bull

    else:
        regime = Regime(regime_val)  # Convert to enum for readability

    #
    # Trading logic
    #

    trades = []

    # We assume a breakout if our current 15m candle has closed
    # above the 1h starting price + (atr * fraction) target level
    long_breakout_entry_level = previous_price + atr * parameters.fract

    # Check for open condition - is the price breaking out
    #
    if not position_manager.is_any_open():
        if regime == Regime.bull:
            if close_price > long_breakout_entry_level:
                trades += position_manager.open_spot(
                    pair,
                    value=cash * parameters.allocation,
                    stop_loss_pct=parameters.stop_loss_pct,
                )
    else:
        # Enable trailing stop loss after we reach the profit taking level
        #
        for position in state.portfolio.open_positions.values():
            if position.trailing_stop_loss_pct is None:
                close_price = indicators.get_price(position.pair)
                if close_price >= position.get_opening_price() * parameters.trailing_stop_loss_activation_level:
                    position.trailing_stop_loss_pct = parameters.trailing_stop_loss_pct

    # Visualisations
    #
    if input.is_visualisation_enabled():
        visualisation = state.visualisation
        visualisation.plot_indicator(timestamp, "ATR", PlotKind.technical_indicator_detached, atr)

    return trades  # Return the list of trades we made in this cycle


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


    if Parameters.use_binance_data:
        global trading_pairs

        trading_pairs = [
            (ChainId.centralised_exchange, "binance", "ETH", "USDT")
        ]

        start_at = universe_options.start_at
        end_at = universe_options.end_at
        strategy_universe = create_binance_universe(
            [f"{p[2]}{p[3]}" for p in trading_pairs],
            candle_time_bucket=Parameters.candle_time_bucket,
            stop_loss_time_bucket=Parameters.stop_loss_time_bucket,
            start_at=start_at,
            end_at=end_at,
            trading_fee_override=Parameters.backtest_trading_fee,
        )
    else:
        dataset = load_partial_data(
            client=client,
            time_bucket=Parameters.candle_time_bucket,
            pairs=trading_pairs,
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

