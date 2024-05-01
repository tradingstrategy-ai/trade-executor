"""Market regime filter visualisation tests."""
import datetime
import os

import pandas_ta
import pytest

from tradeexecutor.utils.binance import fetch_binance_dataset
from tradeexecutor.visual.bullbear import visualise_raw_market_regime_indicator, visualise_market_regime_filter
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket


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


@pytest.mark.skipif(os.environ.get("GITHUB_ACTIONS", None) == "true", reason="Github US servers are blocked by Binance with HTTP 451")
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


@pytest.mark.skipif(os.environ.get("GITHUB_ACTIONS", None) == "true", reason="Github US servers are blocked by Binance with HTTP 451")
def test_visualise_raw_market_regime_indicator(dataset):
    """Resample price series to a higher time frame and shift at the same time using Binance data."""

    adx_df = pandas_ta.adx(
        length=21,
        close=dataset.candles.close,
        high=dataset.candles.high,
        low=dataset.candles.low,
    )

    fig = visualise_raw_market_regime_indicator(dataset.candles, adx_df)
    assert fig.data  # No other checks here for now, just that code runs
