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


def resample_single_pair(df, bucket: TimeBucket) -> pd.DataFrame:
    """Upsample a single pair DataFrame to a lower time bucket.

    - Resample in OHLCV manner
    - Forward fill any gaps in data
    """

    # https://stackoverflow.com/a/68487354/315168

    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }

    # Do forward fill, as missing values in the source data
    # may case NaN to appear as price
    resampled = df.resample(bucket.to_frequency()).agg(ohlc_dict)
    filled = resampled.ffill()
    return filled


def _fix_nans(df: pd.DataFrame) -> pd.DataFrame:
    """External data sources might have NaN values for prices."""

    # TODO: Add NaN fixing logic here
    # https://stackoverflow.com/a/29530303/315168
    assert not df.isnull().any().any(), "DataFrame contains NaNs"
    return df


def load_pair_candles_from_parquet(
    pair: TradingPairIdentifier,
    file: Path,
    column_map: Dict[str, str] = COLUMN_MAP,
    resample: TimeBucket | None = None,
) -> GroupedCandleUniverse:
    """Load a single pair price feed from an alternative file.

    Overrides the current price candle feed with an alternative version,
    usually from a centralised exchange. This allows
    strategy testing to see there is no price feed data issues
    or specificity with it.

    :param pair:
        The trading pair data this Parquet file contains.

        E.g. ticker symbols and trading fee are read from this argument.

    :param resample:
        Resample OHLCV data to a higher timeframe

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

    if resample:
        df = resample_single_pair(df, resample)
        bucket = resample
    else:
        # What's the spacing of candles
        granularity = df.index[1] - df.index[0]
        bucket = TimeBucket.from_pandas_timedelta(granularity)

    df = _fix_nans(df)

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
        candles: GroupedCandleUniverse,
        ignore_time_bucket_mismatch=False,
):
    """Replace the candles in the data.

    Any stop loss price feed data is cleared.

    :param universe:
        Trading universe to modify

    :param candles:
        New price data feeds

    :param ignore_time_bucket_mismatch:
        Do not fail if new and old candles have different granularity
    """

    assert isinstance(universe, TradingStrategyUniverse)
    assert isinstance(candles, GroupedCandleUniverse)

    if not ignore_time_bucket_mismatch:
        assert candles.time_bucket == universe.universe.candles.time_bucket, f"TimeBucket mismatch. Old {universe.universe.candles.time_bucket}, new: {candles.time_bucket}"

    universe.universe.candles = candles
    universe.backtest_stop_loss_candles = None
    universe.backtest_stop_loss_time_bucket = None
