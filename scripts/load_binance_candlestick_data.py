"""Get candlestick price and volume data from Binance.
"""

import requests
import datetime
import pandas as pd

from tradingstrategy.timebucket import TimeBucket
from pathlib import Path


def get_parquet_path(
    symbol: str,
    time_bucket: TimeBucket,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
):
    return Path(
        f"../tests/binance_data/{symbol}-{time_bucket.value}-{start_date}-{end_date}.parquet"
    )


def get_binance_candlestick_data(
    symbol: str,
    time_bucket: TimeBucket,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
):
    """Get candlestick price and volume data from Binance.

    .. code-block:: python
        five_min_data = get_binance_candlestick_data("ETHUSDC", TimeBucket.m5, datetime.datetime(2021, 1, 1), datetime.datetime(2021, 4, 1))

    :param symbol:
        Trading pair symbol E.g. ETHUSDC

    :param interval:
        Can be one of `1s, 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M`

    :param start_date:
        Start date of the data

    :param end_date:
        End date of the data

    :return:
        Pandas dataframe with the OHLCV data for the columns and datetimes as the index
    """

    # to include the end date, we need to add one day
    end_date = end_date + datetime.timedelta(days=1)

    try:
        df = pd.read_parquet(
            get_parquet_path(symbol, time_bucket, start_date, end_date)
        )
        return df
    except:
        pass

    params_str = f"symbol={symbol}&interval={time_bucket.value}"

    if start_date:
        assert (
            end_date
        ), "If you specify a start_date, you must also specify an end_date"
        assert isinstance(
            start_date, datetime.datetime
        ), "start_date must be a datetime.datetime object"
        assert isinstance(
            end_date, datetime.datetime
        ), "end_date must be a datetime.datetime object"
        start_timestamp = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)

    # generate timestamps for each iteration
    dates = [start_date]
    current_date = start_date
    while current_date < end_date:
        if (end_date - current_date) / time_bucket.to_timedelta() > 999:
            dates.append((current_date + time_bucket.to_timedelta() * 999))
            current_date += time_bucket.to_timedelta() * 999
        else:
            dates.append(end_date)
            current_date = end_date

    timestamps = [int(date.timestamp() * 1000) for date in dates]
    open_prices, high_prices, low_prices, close_prices, volume, dates = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for i in range(0, len(timestamps) - 1, 2):
        start_timestamp = timestamps[i]
        end_timestamp = timestamps[i + 1]
        full_params_str = (
            f"{params_str}&startTime={start_timestamp}&endTime={end_timestamp}"
        )
        url = f"https://api.binance.com/api/v3/klines?{full_params_str}&limit=1000"
        response = requests.get(url)
        if response.status_code == 200:
            json_data = response.json()
            if len(json_data) > 0:
                for item in json_data:
                    dates.append(datetime.datetime.fromtimestamp(item[0] / 1000))
                    open_prices.append(float(item[1]))
                    high_prices.append(float(item[2]))
                    low_prices.append(float(item[3]))
                    close_prices.append(float(item[4]))
                    volume.append(float(item[5]))
        else:
            print(f"Error fetching data between {start_timestamp} and {end_timestamp}")
            print(f"Response: {response.status_code} {response.text}")

    df = pd.DataFrame(
        {
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume,
        },
        index=dates,
    )

    df.to_parquet(get_parquet_path(symbol, time_bucket, start_date, end_date))

    return df
