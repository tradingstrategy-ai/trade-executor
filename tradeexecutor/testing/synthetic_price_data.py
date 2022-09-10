"""Generate synthetic price data."""
import datetime
import random
from pathlib import Path
from typing import Dict

import pandas as pd

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.types import USDollarAmount


def generate_ohlcv_candles(
    bucket: TimeBucket,
    start: datetime.datetime,
    end: datetime.datetime,
    start_price=1800,
    daily_drift=(0.95, 1.05),
    high_drift=1.05,
    low_drift=0.90,
    random_seed=1,
    pair_id=1,
    exchange_id=1,
) -> pd.DataFrame:
    """Generate some sample time series data.

    The output candles are deterministic: the same input parameters result to the same output parameters.
    """

    random_gen = random.Random(random_seed)
    time_delta = bucket.to_timedelta()
    open = start_price
    now = start

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
    return df


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

