"""Alternative market data sources.

Functions to use data from centralised exchanges, other sources,
for testing out trading strategies.

"""
from pathlib import Path
from typing import Dict, Tuple, Final

import pandas as pd

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.timebucket import TimeBucket


DEFAULT_COLUMN_MAP: Final = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
}

# Every trading pair has the same value for these columns
PAIR_STATIC_COLUMNS = ('pair_id', 'base_token_symbol', 'quote_token_symbol', 'exchange_slug', 'chain_id', 'fee','token0_address', 'token1_address', 'token0_symbol', 'token1_symbol', 'token0_decimals', 'token1_decimals')


def get_resample_frequency(bucket: TimeBucket) -> str | pd.DateOffset:
    """Get Pandas resample frequency for our candles.

    - Map 7d to "weekly" instead of 7d + arbitrary starting date
    """
    match bucket:
        case TimeBucket.d7:
            return "W"
        case _:
            return bucket.to_frequency()



def resample_single_pair(df: pd.DataFrame, bucket: TimeBucket) -> pd.DataFrame:
    """Upsample a single pair DataFrame to a lower time bucket.

    - Resample in OHLCV manner
    - Forward fill any gaps in data

    .. warning ::

        Calling this function for a multipair data will corrupt the data.
        Use :py:func:`resample_multi_pair` instead.
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
    freq = get_resample_frequency(bucket)
    resampled = df.resample(freq).agg(ohlc_dict)
    filled = resampled.ffill()

    # Remove rows that appear before first row in df
    filled = filled[filled.index >= df.index[0]]

    return filled


def resample_multi_pair(
    df: pd.DataFrame,
    bucket: TimeBucket,
    pair_id_column="pair_id",
    copy_columns=PAIR_STATIC_COLUMNS,
) -> pd.DataFrame:
    """Upsample a OHLCV trading pair data to a lower time bucket.

    - Slower than :py:func:`resample_single_pair`
    - Resample in OHLCV manner
    - Forward fill any gaps in data

    :param pair_id_column:
        DataFrame column to group the data by pair

    :param copy_columns:
        Columns we simply copy over.

        We assume every pair has the same value for these columns.

    :return:
        Concatenated DataFrame of individually resampled pair data
    """
    by_pair = df.groupby(pair_id_column)
    segments = []
    for group_id in by_pair.groups:
        pair_df = by_pair.get_group(group_id)
        if len(pair_df) > 0:
            segment = resample_single_pair(pair_df, bucket)
            for c in copy_columns:
                if c in pair_df.columns:
                    first_row = pair_df.iloc[0]
                    segment[c] = first_row[c]
            segments.append(segment)

    return pd.concat(segments)


def _fix_nans(df: pd.DataFrame) -> pd.DataFrame:
    """External data sources might have NaN values for prices.
    
    Apply forward fill to columns not in [ 'open', 'high', 'low', 'close', 'volume']. This is fine if used for single pair data only where values are simply repeated down all rows.
    """
    
    exclude_columns= [ 'open', 'high', 'low', 'close', 'volume']
    
    for column in df.columns:
        if column not in exclude_columns:
            df[column].fillna(method='ffill', inplace=True)

    # TODO: Add NaN fixing logic here
    # https://stackoverflow.com/a/29530303/315168
    assert not df.isnull().any().any(), "DataFrame contains NaNs"
    return df


def load_candles_from_parquet(
    file: Path,
    column_map: Dict[str, str] = DEFAULT_COLUMN_MAP,
    resample: TimeBucket | None = None,
    identifier_column: str = "pair_id",
    pair_id: int | None = None,  # legacy
) -> Tuple[pd.DataFrame, pd.DataFrame, TimeBucket, TimeBucket]:
    """Loads OHLCV candle data from a Parquest file.

    Designed to load Parquest compressed external data dumps,
    like ones from Binance and Kraken.

    :return:
        Tuple (Original dataframe, processed candles dataframe, resampled time bucket, original time bucket) tuple
    """

    assert isinstance(file, Path)

    df = pd.read_parquet(file)

    if pair_id:
        df[identifier_column] = pair_id

    df, orig, bucket, original_bucket = load_candles_from_dataframe(column_map, df, resample, identifier_column)

    return df, orig, bucket, original_bucket


def load_candle_universe_from_parquet(
    file: Path,
    column_map: Dict[str, str] = DEFAULT_COLUMN_MAP,
    resample: TimeBucket | None = None,
    include_as_trigger_signal=True,
    identifier_column: str = "pair_id",
    pair: TradingPairIdentifier | None = None,  # legacy
) -> Tuple[GroupedCandleUniverse, GroupedCandleUniverse | None]:
    """Load a single pair price feed from an alternative file.

    Overrides the current price candle feed with an alternative version,
    usually from a centralised exchange. This allows
    strategy testing to see there is no price feed data issues
    or specificity with it.

    For example see :py:func:`replace_candles`.

    :param pair:
        The trading pair data this Parquet file contains.

        E.g. ticker symbols and trading fee are read from this argument.

    :param resample:
        Resample OHLCV data to a higher timeframe

    :param include_as_trigger_signal:
        Create take profit/stop loss signal from the data.

        For this, any upsampling is not used.

    :param pair:
        Trading pair identifier. Legacy option for backwards compatibility, no need for this anymore.

    :raise NoMatchingBucket:
        Could not match candle time frame to any of our timeframes.

    :return:
        (Price feed universe, stop loss trigger candls universe) tuple.

        Stop loss data is only generated if `include_as_trigger_signal` is True.
        Stop loss data is never resampled and is in the most accurate available resolution.

    """

    df, orig, bucket, original_bucket = load_candles_from_parquet(
        file,
        column_map,
        resample,
        identifier_column,
        pair_id=pair.internal_id if pair else None,
    )

    candles, stop_loss_candles = load_candle_universe_from_dataframe(
        orig,
        column_map,
        resample,
        include_as_trigger_signal,
        identifier_column,
    )

    return candles, stop_loss_candles


def load_candles_from_dataframe(
    column_map: Dict[str, str],
    df: pd.DataFrame,
    resample: TimeBucket | None,
    identifier_column: str = "pair_id"
) -> Tuple[pd.DataFrame, pd.DataFrame, TimeBucket, TimeBucket]:
    """Load OHLCV candle data from a DataFrame.

    Remap and resample data when loading.

    .. warning::

        Outside supported columns, any other columns are destroyed in the resampling.
    
    :param column_map:
        Column name mapping from the DataFrame to our internal format.

        E.g. { "open": "open", "high": "high", ... }

    :param df:
        DataFrame to load from

    :param pair_id:
        Pair id to set for the DataFrame

    :param resample:
        Resample OHLCV data to a higher timeframe

    :return:
        Tuple (Original dataframe, processed candles dataframe, resampled time bucket, original time bucket) tuple
    """
    assert identifier_column in df.columns, f"DataFrame does not have {identifier_column} column"

    assert isinstance(df.index, pd.DatetimeIndex), f"Parquet did not have DateTime index: {df.index}"

    df = df.rename(columns=column_map)

    orig = df.copy()

    # What's the spacing of candles
    granularity = df.index[1] - df.index[0]
    original_bucket = TimeBucket.from_pandas_timedelta(granularity)

    if resample:

        uniq = df["pair_id"].unique()
        if len(uniq) == 1:
            _df = resample_single_pair(df, resample)

            # to preserve addtional columns beyond OHLCV, we need to left join
            if len(df.columns) > 5:
                del df['open']
                del df['high']
                del df['low']
                del df['close']
                del df['volume']

                df = _df.join(df, how='left')
        else:
            df = resample_multi_pair(df, resample)

        bucket = resample

    else:
        bucket = TimeBucket.from_pandas_timedelta(granularity)

    df = _fix_nans(df)

    # Because we assume multipair data from now on,
    # with group index instead of timestamp index,
    # we make timestamp a column
    df["timestamp"] = df.index.to_series()

    return df, orig, bucket, original_bucket


def load_candle_universe_from_dataframe(
    df: pd.DataFrame,
    column_map: Dict[str, str] = DEFAULT_COLUMN_MAP,
    resample: TimeBucket | None = None,
    include_as_trigger_signal=True,
    identifier_column: str = "pair_id",
) -> Tuple[GroupedCandleUniverse, GroupedCandleUniverse | None]:
    """Load a single pair price feed from a DataFrame.
    
    Same as :py:func:`load_candle_universe_from_parquet` but from a DataFrame.

    Overrides the current price candle feed with an alternative version,
    usually from a centralised exchange. This allows
    strategy testing to see there is no price feed data issues
    or specificity with it.
    
    :param pair:
        The trading pair data this Parquet file contains.

        E.g. ticker symbols and trading fee are read from this argument.

    :param df:
        DataFrame to load from

    :param resample:
        Resample OHLCV data to a higher timeframe

    :param include_as_trigger_signal:
        Create take profit/stop loss signal from the data.

        For this, any upsampling is not used.

    :raise NoMatchingBucket:
        Could not match candle time frame to any of our timeframes.

    :return:
        (Price feed universe, stop loss trigger candls universe) tuple.

        Stop loss data is only generated if `include_as_trigger_signal` is True.
        Stop loss data is never resampled and is in the most accurate available resolution.
    """

    df, orig, bucket, original_bucket = load_candles_from_dataframe(
        column_map,
        df,
        resample,
        identifier_column=identifier_column,
    )

    candles = GroupedCandleUniverse(
        df,
        time_bucket=bucket,
        index_automatically=False,
        fix_wick_threshold=None,
    )

    if include_as_trigger_signal:
        orig.index.name = "timestamp"
        if "timestamp" not in orig.columns:
            orig = orig.reset_index()
        stop_loss_candles = GroupedCandleUniverse(
            orig,
            time_bucket=original_bucket,
            index_automatically=False,
            fix_wick_threshold=None,
        )
    else:
        stop_loss_candles = None

    return candles, stop_loss_candles


def replace_candles(
        universe: TradingStrategyUniverse,
        candles: GroupedCandleUniverse,
        stop_loss_candles: GroupedCandleUniverse | None = None,
        ignore_time_bucket_mismatch=False,
):
    """Replace the candles in the trading universe with an alternative version.

    - This is a simple trick to allow backtesting strategies against CEX
      and other price feed data that is not built into system.

    - You can compare if the outcome our the strategy would be different
      with a different price source

    Example:

    .. code-block:: python

        #
        # First load DEX data for a single pair as you would do normally
        #

        TRADING_PAIR = (ChainId.arbitrum, "uniswap-v3", "WBTC", "USDC", 0.0005)

        CANDLE_TIME_BUCKET = TimeBucket.h1

        def create_trading_universe(
            ts: datetime.datetime,
            client: Client,
            execution_context: ExecutionContext,
            universe_options: UniverseOptions,
        ):
            assert isinstance(
                client, Client
            ), f"Looks like we are not running on the real data. Got: {client}"

            # Download live data from the oracle
            dataset = load_pair_data_for_single_exchange(
                client,
                time_bucket=CANDLE_TIME_BUCKET,
                pair_tickers=[TRADING_PAIR],
                execution_context=execution_context,
                universe_options=universe_options,
            )

            # Convert loaded data to a trading pair universe
            universe = TradingStrategyUniverse.create_single_pair_universe(
                dataset,
                pair=TRADING_PAIR,
            )

            return universe

        client = Client.create_jupyter_client()
        universe = create_trading_universe(
            datetime.datetime.utcnow(),
            client,
            ExecutionContext(mode=ExecutionMode.backtesting),
            universe_options=UniverseOptions(),
        )

        #
        # Replace the single pair price feed with a data from Binance,
        # distributed as Parquet file.
        #
        # Also set the same 1h candle fee to be used as stop loss trigger
        # signal.
        #
        pair = universe.get_single_pair()
        new_candles, stop_loss_candles = load_pair_candles_from_parquet(
            pair,
            Path("tests/binance-BTCUSDT-1h.parquet"),
            include_as_trigger_signal=True,
        )
        replace_candles(universe, new_candles, stop_loss_candles)

    :param universe:
        Trading universe to modify

    :param candles:
        New price data feeds

    :param stop_loss_candles:
        Trigger signal for stop loss backtesting.

    :param ignore_time_bucket_mismatch:
        Do not fail if new and old candles have different granularity
    """

    assert isinstance(universe, TradingStrategyUniverse)
    assert isinstance(candles, GroupedCandleUniverse)

    if not ignore_time_bucket_mismatch:
        assert candles.time_bucket == universe.data_universe.candles.time_bucket, f"TimeBucket mismatch. Old {universe.data_universe.candles.time_bucket}, new: {candles.time_bucket}"

    universe.data_universe.candles = candles
    if stop_loss_candles:
        universe.backtest_stop_loss_candles = stop_loss_candles
        universe.backtest_stop_loss_time_bucket = stop_loss_candles.time_bucket
    else:
        universe.backtest_stop_loss_candles = None
        universe.backtest_stop_loss_time_bucket = None
