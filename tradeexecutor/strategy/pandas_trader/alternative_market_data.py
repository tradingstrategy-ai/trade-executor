"""Alternative market data sources.

Functions to use data from centralised exchanges, other sources,
for testing out trading strategies.

"""
from pathlib import Path
from typing import Dict, Tuple

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
    include_as_trigger_signal=True,
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

    :raise NoMatchingBucket:
        Could not match candle time frame to any of our timeframes.

    :return:
        (Price feed universe, stop loss trigger candls universe) tuple.

        Stop loss data is only generated if `include_as_trigger_signal` is True.
        Stop loss data is never resampled and is in the most accurate available resolution.

    """

    assert isinstance(pair, TradingPairIdentifier)
    assert isinstance(file, Path)

    df = pd.read_parquet(file)

    assert isinstance(df.index, pd.DatetimeIndex), f"Parquet did not have DateTime index: {df.index}"

    orig = df = df.rename(columns=column_map)

    # What's the spacing of candles
    granularity = df.index[1] - df.index[0]
    original_bucket = TimeBucket.from_pandas_timedelta(granularity)

    if resample:
        df = resample_single_pair(df, resample)
        bucket = resample
    else:
        bucket = TimeBucket.from_pandas_timedelta(granularity)

    df = _fix_nans(df)

    # Add pair column
    df["pair_id"] = pair.internal_id

    # Because we assume multipair data from now on,
    # with group index instead of timestamp index,
    # we make timestamp a column
    df["timestamp"] = df.index.to_series()

    candles = GroupedCandleUniverse(
        df,
        time_bucket=bucket,
        index_automatically=False,
        fix_wick_threshold=None,
    )

    if include_as_trigger_signal:
        orig["pair_id"] = pair.internal_id
        orig["timestamp"] = df.index.to_series()
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
        assert candles.time_bucket == universe.universe.candles.time_bucket, f"TimeBucket mismatch. Old {universe.universe.candles.time_bucket}, new: {candles.time_bucket}"

    universe.universe.candles = candles
    if stop_loss_candles:
        universe.backtest_stop_loss_candles = stop_loss_candles
        universe.backtest_stop_loss_time_bucket = stop_loss_candles.time_bucket
    else:
        universe.backtest_stop_loss_candles = None
        universe.backtest_stop_loss_time_bucket = None
