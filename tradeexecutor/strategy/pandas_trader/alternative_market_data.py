"""Alternative market data sources.

Functions to use data from centralised exchanges, other sources,
for testing out trading strategies.

"""
from pathlib import Path
from typing import Dict

import pandas as pd

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.timebucket import TimeBucket


COLUMN_MAP = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
}


def load_pair_candles_from_parquet(
    pair: TradingPairIdentifier,
    file: Path,
    column_map: Dict[str, str] = COLUMN_MAP,
) -> GroupedCandleUniverse:
    """Load a single pair price feed from an alternative file.

    Overrides the current price candle feed with an alternative version,
    usually from a centralised exchange. This allows
    strategy testing to see there is no price feed data issues
    or specificity with it.

    :raise NoMatchingBucket:
        Could not match candle time frame to any of our timeframes.

    :return:
        A candle universe for a single pair
    """

    assert isinstance(pair, TradingPairIdentifier)
    assert isinstance(file, Path)

    df = pd.read_parquet(file)

    assert isinstance(df.index, pd.DatetimeIndex), f"Parquet did not have DateTime index: {df.index}"

    df = df.rename(columns=column_map)

    # What's the spacing of candles
    granularity = df.index[1] - df.index[0]

    bucket = TimeBucket.from_pandas_timedelta(granularity)

    # Add pair column
    df["pair_id"] = pair.internal_id

    # Because we assume multipair data from now on,
    # with group index instead of timestamp index,
    # we make timestamp a column
    df["timestamp"] = df.index.to_series()

    return GroupedCandleUniverse(
        df,
        time_bucket=bucket,
        index_automatically=False,
        fix_wick_threshold=None,
    )


def replace_candles(
        universe: TradingStrategyUniverse,
        candles: GroupedCandleUniverse):
    """Replace the candles in the data."""

    assert isinstance(universe, TradingStrategyUniverse)
    assert isinstance(candles, GroupedCandleUniverse)

    assert candles.time_bucket == universe.universe.candles.time_bucket, f"TimeBucket mismatch. Old {universe.universe.candles.time_bucket}, new: {candles.time_bucket}"

    universe.universe.candles = candles
