"""Market regime filter visualisation tests."""
import datetime
import os

import pandas as pd
import pandas_ta
import pytest

from tradeexecutor.analysis.regime import Regime
from tradeexecutor.strategy.execution_context import unit_test_execution_context, ExecutionContext
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSource, IndicatorSet, calculate_and_load_indicators, DiskIndicatorStorage
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInputIndicators
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.utils.binance import fetch_binance_dataset, create_binance_universe
from tradeexecutor.visual.bullbear import visualise_raw_market_regime_indicator, visualise_market_regime_filter
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.utils.groupeduniverse import resample_candles, resample_series

pytestmark = pytest.mark.skipif(os.environ.get("GITHUB_ACTIONS", None) == "true", reason="Github US servers are blocked by Binance with HTTP 451")


@pytest.fixture(scope="module")
def dataset(persistent_test_client: Client):
    dataset = fetch_binance_dataset(
        ["BTCUSDT"],
        candle_time_bucket=TimeBucket.d1,
        stop_loss_time_bucket=None,
        start_at=datetime.datetime(2020, 1, 1),
        end_at=datetime.datetime(2024, 4, 1),
        include_lending=False,
    )
    return dataset


def test_visualise_adx_binance_btc(dataset):
    """Resample price series to a higher time frame and shift at the same time using Binance data."""

    adx_df = pandas_ta.adx(
        length=21,
        close=dataset.candles.close,
        high=dataset.candles.high,
        low=dataset.candles.low,
    )

    def regime_filter(row):
        average_direction_index, direction_index_positive, direction_index_negative = row.values
        if direction_index_positive > 25:
            return 1
        elif direction_index_negative > 25:
            return -1
        else:
            return 0

    regime_signal = adx_df.apply(regime_filter, axis="columns")
    fig = visualise_market_regime_filter(
        dataset.candles.close,
        regime_signal,
    )
    assert fig.data  # No other checks here for now, just that code runs


def test_visualise_raw_market_regime_indicator(dataset):
    """Resample price series to a higher time frame and shift at the same time using Binance data."""

    adx_df = pandas_ta.adx(
        length=21,
        close=dataset.candles.close,
        high=dataset.candles.high,
        low=dataset.candles.low,
    )

    fig = visualise_raw_market_regime_indicator(dataset.candles.close, adx_df)
    assert fig.data  # No other checks here for now, just that code runs


def test_access_regime_filter_data_15m(tmp_path):
    """Mix 15m price data with 1d regime filter."""

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
        """Calculate 0-100 ADX value and +DMI and -DMI values for every day."""
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
            # ADX, DMP, DMN
            average_direction_index, direction_index_positive, direction_index_negative = row.values
            if direction_index_positive > regime_threshold:
                return Regime.bull.value
            elif direction_index_negative > regime_threshold:
                return Regime.bear.value
            else:
                return Regime.crab.value
        regime_signal = adx_df.apply(regime_filter, axis="columns")
        return regime_signal

    class Parameters:
        adx_filter_threshold = 25
        adx_length = 20

    parameters = StrategyParameters.from_class(Parameters)

    indicators = IndicatorSet()
    indicators.add(
        "adx",
        daily_adx,
        {"length": parameters.adx_length},
        IndicatorSource.ohlcv,
    )
    indicators.add(
        "daily_price",
        daily_price,
        {},
        IndicatorSource.ohlcv,
    )
    indicators.add(
        "regime",
        regime,
        {"length": parameters.adx_length, "regime_threshold": parameters.adx_filter_threshold},
        IndicatorSource.ohlcv,
    )

    strategy_universe = create_binance_universe(
        ["BTCUSDT"],
        candle_time_bucket=TimeBucket.m15,
        stop_loss_time_bucket=None,
        start_at=datetime.datetime(2023, 1, 1),
        end_at=datetime.datetime(2024, 1, 1),
        include_lending=False,
    )

    indicator_storage = DiskIndicatorStorage(tmp_path, strategy_universe.get_cache_key())

    indicator_results = calculate_and_load_indicators(
        strategy_universe,
        indicator_storage,
        indicators=indicators,
        execution_context=unit_test_execution_context,
        parameters=StrategyParameters({}),
        max_workers=1,
        max_readers=1,
    )

    first_day, last_day = strategy_universe.data_universe.candles.get_timestamp_range()
    input_indicators = StrategyInputIndicators(
        strategy_universe=strategy_universe,
        available_indicators=indicators,
        indicator_results=indicator_results,
        timestamp=last_day,
    )

    # assert len(input_indicators.get_indicator_series("regime")) == 365

    # 2023-01-20    0
    # 2023-01-21    1
    # 2023-01-22    1
    # 2023-01-23    1

    # data = input_indicators.get_indicator_series("regime")
    # pd.set_option("display.max_rows", None)
    # print(data[pd.Timestamp("2023-01-20"):pd.Timestamp("2023-01-22")])

    input_indicators.timestamp = pd.Timestamp("2023-01-20 00:15")
    assert input_indicators.get_indicator_value("regime") == 0

    input_indicators.timestamp = pd.Timestamp("2023-01-21 00:00")
    assert input_indicators.get_indicator_value("regime") == 0
    input_indicators.timestamp = pd.Timestamp("2023-01-21 00:15")
    assert input_indicators.get_indicator_value("regime") == 0
    input_indicators.timestamp = pd.Timestamp("2023-01-21 23:45")
    assert input_indicators.get_indicator_value("regime") == 0

    input_indicators.timestamp = pd.Timestamp("2023-01-22 00:00")
    assert input_indicators.get_indicator_value("regime") == 1
    input_indicators.timestamp = pd.Timestamp("2023-01-22 00:15")
    assert input_indicators.get_indicator_value("regime") == 1
    input_indicators.timestamp = pd.Timestamp("2023-01-22 23:45")
    assert input_indicators.get_indicator_value("regime") == 1
