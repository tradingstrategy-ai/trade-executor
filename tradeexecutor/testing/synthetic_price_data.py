"""Generate synthetic price data."""
import datetime
import random

import pandas as pd

from tradingstrategy.timebucket import TimeBucket


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


