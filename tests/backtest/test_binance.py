"""Binance data related tests."""
import datetime
import os

import pandas as pd
import pytest
from pandas import Timestamp

from tradeexecutor.utils.binance import fetch_binance_dataset
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.utils.groupeduniverse import resample_price_series


@pytest.mark.skipif(os.environ.get("GITHUB_ACTIONS", None) == "true", reason="Github US servers are blocked by Binance with HTTP 451")
def test_price_series_resample_and_shift_binance(persistent_test_client: Client):
    """Resample price series to a higher time frame and shift at the same time using Binance data."""

    dataset = fetch_binance_dataset(
        ["BTCUSDT"],
        candle_time_bucket=TimeBucket.h1,
        stop_loss_time_bucket=None,
        start_at=datetime.datetime(2020, 1, 1),
        end_at=datetime.datetime(2020, 2, 1),
        include_lending=False,
        force_download=True,
    )

    raw_candles = dataset.candles["close"]
    assert raw_candles.iloc[0:3].to_dict() == {
        Timestamp('2020-01-01 00:00:00'): 7177.02,
        Timestamp('2020-01-01 01:00:00'): 7216.27,
        Timestamp('2020-01-01 02:00:00'): 7242.85
    }

    unshifted_8h_close = resample_price_series(raw_candles, pd.Timedelta(hours=8))
    shifted_plus_one_8h_close = resample_price_series(raw_candles, pd.Timedelta(hours=8), shift=1)
    shifted_minus_one_8h_close = resample_price_series(raw_candles, pd.Timedelta(hours=8), shift=-1)

    diagnostics_df = pd.DataFrame({
        "raw": raw_candles,
        "unshifted_8h_close": unshifted_8h_close,
        "shifted_plus_one_8h_close": shifted_plus_one_8h_close,
        "shifted_minus_one_8h_close": shifted_minus_one_8h_close,
    })

    #                          raw  unshifted_8h_close  shifted_plus_one_8h_close  shifted_minus_one_8h_close
    # 2020-01-01 00:00:00  7177.02             7209.83                    7225.62                     7200.64
    # 2020-01-01 01:00:00  7216.27                 NaN                        NaN                         NaN
    # 2020-01-01 02:00:00  7242.85                 NaN                        NaN                         NaN
    # 2020-01-01 03:00:00  7225.01                 NaN                        NaN                         NaN
    # 2020-01-01 04:00:00  7217.27                 NaN                        NaN                         NaN
    # 2020-01-01 05:00:00  7224.21                 NaN                        NaN                         NaN
    # 2020-01-01 06:00:00  7225.62                 NaN                        NaN                         NaN
    # 2020-01-01 07:00:00  7209.83                 NaN                        NaN                         NaN
    # 2020-01-01 08:00:00  7200.64             7234.19                    7221.43                     7245.37
    # 2020-01-01 09:00:00  7188.77                 NaN                        NaN                         NaN
    # 2020-01-01 10:00:00  7202.00                 NaN                        NaN                         NaN
    # 2020-01-01 11:00:00  7197.20                 NaN                        NaN                         NaN

    with pd.option_context('display.min_rows', 24):
        # display(diagnostics_df)
        pass