"""Generate synthetic price data."""
import datetime
import random
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.types import USDollarPrice
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.types import USDollarAmount


def generate_ohlcv_candles(
    bucket: TimeBucket,
    start: datetime.datetime,
    end: datetime.datetime,
    start_price: int | float=1800,
    daily_drift=(0.95, 1.05),
    high_drift=1.05,
    low_drift=0.90,
    random_seed=1,
    pair_id=1,
    exchange_id=1,
) -> pd.DataFrame:
    """Generate some sample time series data.

    The output candles are deterministic: the same input parameters result to the same output parameters.

    :param bucket: 
        Time bucket to use for the candles

    :param start: 
        Start time for the candles

    :param end: 
        End time for the candles

    :param start_price: 
        Starting price for the candles

    :param daily_drift: 
        Tuple of (min, max) daily drift for the candles

    :param high_drift: 
        High drift for the candles

    :param low_drift: 
        Low drift for the candles

    :param random_seed: 
        Random seed to use for the candles

    :param pair_id: 
        Pair ID to use for the candles

    :param exchange_id: 
        Exchange ID to use for the candles
    """

    random_gen = random.Random(random_seed)
    time_delta = bucket.to_timedelta()
    open = start_price
    now = start

    assert pair_id is not None

    data = []

    while now < end:

        close = random_gen.uniform(open * daily_drift[0], open * daily_drift[1])
        high = random_gen.uniform(open, open * high_drift)
        low = random_gen.uniform(open, open * low_drift)

        data.append({
            "pair_id": pair_id,
            "timestamp": now,
            "open": open,
            "close": close,
            "high": high,
            "low": low,
            "buy_volume": 0,
            "sell_volume": 0,
            "buys": 0,
            "sells": 0,
            "start_block": int(now.timestamp()),
            "end_block": int(now.timestamp()),
        })

        open = close
        now += time_delta

    df = pd.DataFrame(data)
    df.set_index("timestamp", drop=False, inplace=True)
    df["volume"] = df["buy_volume"] + df["sell_volume"]   # Convert from Uni v2 style volume
    return df


def generate_multi_pair_candles(
    time_bucket: TimeBucket,
    start: datetime.datetime,
    end: datetime.datetime,
    pairs: Dict[TradingPairIdentifier, USDollarPrice],
    random_seed=1,
) -> pd.DataFrame:
    """Generate synthetic tarding data for multiple trading pairs.

    :param pairs:
        Map of trading pairs and their starting prices
    """

    segments = []

    for pair, price in pairs.items():
        df = generate_ohlcv_candles(
            time_bucket,
            start,
            end,
            start_price=price,
            pair_id=pair.internal_id,
            exchange_id=pair.internal_exchange_id,
            random_seed=random_seed,
        )
        segments.append(df)

    return pd.concat(segments)


def generate_fixed_price_candles(
    bucket: TimeBucket,
    start: datetime.datetime,
    end: datetime.datetime,
    pair_price_map: Dict[TradingPairIdentifier, USDollarAmount],
) -> pd.DataFrame:
    """Generate flat prices for several assets.

    Creates fake fixed price data where prices are stable over a period of time.
    """

    data = []
    for pair, price in pair_price_map.items():
        time_delta = bucket.to_timedelta()

        open = price
        close = price
        high = price
        low = price

        now = start

        while now < end:
            data.append({
                "pair_id": pair.internal_id,
                "timestamp": now,
                "open": open,
                "close": close,
                "high": high,
                "low": low,
                "buy_volume": 0,
                "sell_volume": 0,
                "buys": 0,
                "sells": 0,
                "start_block": int(now.timestamp()),
                "end_block": int(now.timestamp()),
            })

            open = close
            now += time_delta

    df = pd.DataFrame(data)
    df.set_index("timestamp", drop=False, inplace=True)
    return df


def load_ohlcv_parquet_file(path: Path, chain_id: ChainId, exchange_id: int, pair_id: int) -> pd.DataFrame:
    """Load OHLCV data directly from a parquet file.

    - Assume one file for one pair

    - Columns: timestamp, open, close, high, low, volume
    """

    assert isinstance(path, Path)
    assert path.exists()
    df = pd.read_parquet(path)
    df.set_index("timestamp", drop=False, inplace=True)

    # Fill data we need for the universe
    # https://stackoverflow.com/a/69822548/315168
    df.loc[:, "pair_id"] = pair_id
    df.loc[:, "exchange_id"] = exchange_id
    df.loc[:, "chain_id"] = chain_id.value

    return df

