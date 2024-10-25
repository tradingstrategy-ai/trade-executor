"""Decision trigger.

- Wait for the latest candle data become available to act on it immediately

- Update :py:class:`TradingStrategyUniverse` with the latest data needed for the current strategy cycle

"""
import datetime
import logging
import time
from dataclasses import dataclass
from typing import Set, Optional, Dict

import pandas as pd
from tradingstrategy.candle import GroupedCandleUniverse, TradingPairDataAvailability

from tradingstrategy.client import Client
from tradingstrategy.lending import LendingReserve, LendingReserveUniverse, LendingCandleType, LendingCandleResult, LendingCandleUniverse
from tradingstrategy.pair import DEXPair
from tradingstrategy.timebucket import TimeBucket

from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradingstrategy.types import PrimaryKey

logger = logging.getLogger(__name__)


class NoNewDataReceived(Exception):
    """We never got new data despite max wait."""


class BadNewDataReceived(Exception):
    """Candles are off."""



@dataclass(slots=True)
class UpdatedUniverseResult:
    """Describe the result of universe waiting operation."""

    #: Trading Universe with updated candles
    updated_universe: TradingStrategyUniverse

    #: When we finished waiting
    ready_at: datetime.datetime

    #: How long we waited
    time_waited: datetime.timedelta

    #: How many cycles we did waiting
    poll_cycles: int

    #: Maximum difference between timestamp and last available candle.
    #:
    #: None if there was no poll cycles
    max_diff: Optional[datetime.datetime]


def fetch_price_data(
    client: Client,
    bucket: TimeBucket,
    timestamp: datetime.datetime,
    pairs: Set[DEXPair],
    required_history_period: datetime.timedelta,
) -> pd.DataFrame:
    """Download the pair data.

    TODO: Add an API to disable progress bars.

    :param client:
    :param bucket:
    :param timestamp:
    :param pairs:
    :param required_history_period:

    :return:
        A candle containing a mix of pair data for all pairs.
    """
    pair_ids = {p.pair_id for p in pairs}
    start_time = timestamp - required_history_period - datetime.timedelta(seconds=1)
    return client.fetch_candles_by_pair_ids(
        pair_ids,
        bucket=bucket,
        start_time=start_time,
        end_time=timestamp,
    )


def fetch_lending_data(
    client: Client,
    bucket: TimeBucket,
    timestamp: datetime.datetime,
    lending_reserve_universe: LendingReserveUniverse,
    required_history_period: datetime.timedelta,
) -> LendingCandleResult:
    """Download the new lending candles.

    :param client:
    :param bucket:
    :param timestamp:
    :param pairs:
    :param required_history_period:

    :return:
        DataFrame containing updated lending reserves.
    """
    start_time = timestamp - required_history_period - datetime.timedelta(seconds=1)
    return client.fetch_lending_candles_for_universe(
        lending_reserve_universe,
        bucket=bucket,
        start_time=start_time,
        end_time=timestamp,
    )


def fetch_availability(
        client: Client,
        bucket: TimeBucket,
        pairs: Set[DEXPair],
) -> Dict[PrimaryKey, TradingPairDataAvailability]:
    """Fetch the trading data availability from the oracle.

    :return:
        A candle containing a mix of pair data for all pairs.
    """
    pair_ids = {p.pair_id for p in pairs}
    return client.fetch_trading_data_availability(
        pair_ids,
        bucket=bucket,
    )


def update_universe(
    universe: TradingStrategyUniverse,
    price_df: pd.DataFrame,
    lending_candles: LendingCandleResult | None = None,
) -> TradingStrategyUniverse:
    """Update a Trading Universe with a new candle data.

    :param price_df:
        Unsorted DataFrame containing data for all trading pairs we are interested in.

    :param lending_candles:
        New lending candles fetches from the server

    """
    updated_universe = universe.clone()
    updated_universe.data_universe.candles = GroupedCandleUniverse(price_df)

    if universe.has_lending_data():
        assert lending_candles is not None
        updated_universe.data_universe.lending_candles = LendingCandleUniverse(
            lending_candles,
            lending_reserve_universe=universe.data_universe.lending_reserves
        )

    return updated_universe


def validate_latest_candles(
    pairs: Set[DEXPair],
    df: pd.DataFrame,
    timestamp: datetime.datetime,
):
    """Ensure that the oracle served us correct up-to-date candles.

    - The last timestamp of a pair must match
      what we requested earlier.

    - The timestamp cannot be sooner or later

    .. note ::

        This cam be only called for highly active pairs,
        as many low and middle cap tokens may not see trades
        in hours.

    :param pairs:
        Set of pairs our strategy is trading

    :param df:
        Dataframe of candles.

        May contain candes for a single or multiple pairs.

    :param timestamp:

        What is the latest timestamp we need to have avilable for every pair.
        This is the strategy decision timestamp - current candle time frame.

    :raise:
        AssertionError
    """
    timestamp = pd.Timestamp(timestamp)
    assert len(df) > 0, f"Empty dataframe. Pairs: {pairs}"
    for p in pairs:
        last_timestamp = df.loc[df["pair_id"] == p.pair_id].max()["timestamp"]
        assert last_timestamp == timestamp, f"Did not receive wanted latest candle timestamp: {timestamp}. Pair {p} has timestamp {last_timestamp}"


def wait_for_universe_data_availability_jsonl(
        timestamp: datetime.datetime,
        client: Client,
        current_universe: TradingStrategyUniverse,
        required_history_period=datetime.timedelta(days=90),
        max_wait=datetime.timedelta(minutes=30),
        max_poll_cycles: Optional[int] = None,
        poll_delay = datetime.timedelta(seconds=15),
) -> UpdatedUniverseResult:
    """Wait for the data to be available for the latest strategy cycle.

    - Used in live execution only

    - Uses Trading Strategy oracle real-time JSONL API for the data.

    - Uses simple polling appraoch

    :param timestamp:
        The current strategy decision timestamp.

        The latest available data we can have is the previous full candle.

    :param current_universe:
        The current trading universe with old candles.

    :param required_history_period:
        How much historical data we need to load.

        Depends on the strategy. Defaults to 90 days.

        If there is `current_universe.required_history_period` ignore this argument
        and use the value from the trading universe instead.

    :param max_wait:
        Unless data is seen, die with an exception after this period.

    :param max_poll_cycles:
        Can be set in integration testing.

        Return after this many cycles despite new data being incomplete.

    :return:
        An updated trading universe
    """

    assert client is not None
    assert timestamp.second == 0
    # List of monitored pairs by this strategy
    pairs = current_universe.data_universe.pairs
    assert len(pairs.pair_map) > 0, "No pairs in the pair_map"
    bucket = current_universe.data_universe.time_bucket
    completed_pairs: Set[DEXPair] = set()
    incompleted_pairs: Set[DEXPair] = {pairs.get_pair_by_id(id) for id in pairs.pair_map.keys()}
    started_at = datetime.datetime.utcnow()
    deadline = started_at + max_wait
    poll_cycle = 1

    wanted_timestamp = timestamp - bucket.to_timedelta()
    logger.info("Waiting for data availability for pairs %s, strategy cycle timestamp is %s, wanted timestamp is %s",
                pairs,
                timestamp,
                wanted_timestamp
                )

    if max_poll_cycles is None:
        # Make sure we can do int comparison
        max_poll_cycles = 99999

    # Use the required look back value from the trading
    # universe if available.
    if current_universe.required_history_period is not None:
        required_history_period = current_universe.required_history_period

    max_diff = datetime.timedelta(0)

    while datetime.datetime.utcnow() < deadline:

        # Get the availability of the trading for candles
        avail_map = fetch_availability(
            client,
            bucket,
            incompleted_pairs,
        )

        last_timestamps_log = {}

        # Move any pairs with new complete data to the completed set
        pairs_to_move = set()
        diff = None
        for p in incompleted_pairs:    
            latest_timestamp = avail_map[p.pair_id]["last_candle_at"]
            last_supposed_candle_at = avail_map[p.pair_id]["last_supposed_candle_at"]

            if last_supposed_candle_at > latest_timestamp:
                latest_timestamp = last_supposed_candle_at
        
            last_timestamps_log[p.get_ticker()] = latest_timestamp

            # This pair received its data and is ready
            if latest_timestamp >= wanted_timestamp:
                pairs_to_move.add(p)

            diff = wanted_timestamp - latest_timestamp
            max_diff = max(diff, max_diff)

        # Some pairs become ready with their data
        for p in pairs_to_move:
            incompleted_pairs.remove(p)
            completed_pairs.add(p)

        # Add we done with all incomplete pairs
        if not incompleted_pairs or poll_cycle >= max_poll_cycles:
            # We have latest data for all pairs and can now update the universe
            logger.info("Fetching candle data for the history period of %s", required_history_period)

            df = fetch_price_data(
                client,
                bucket,
                wanted_timestamp,
                completed_pairs,
                required_history_period,
            )

            if current_universe.has_lending_data():
                lending_data = fetch_lending_data(
                    client,
                    TimeBucket.h1,
                    wanted_timestamp,
                    current_universe.data_universe.lending_reserves,
                    required_history_period
                )
            else:
                lending_data = None

            updated_universe = update_universe(
                current_universe,
                df,
                lending_data
            )

            time_waited = datetime.datetime.utcnow() - started_at

            return UpdatedUniverseResult(
                updated_universe=updated_universe,
                ready_at=datetime.datetime.utcnow(),
                time_waited=time_waited,
                poll_cycles=poll_cycle,
                max_diff=max_diff,
            )

        # Avoid excessive logging output if > 10 pairs
        last_timestamps_log = last_timestamps_log[0:400]

        logger.info("Timestamp wanted %s, Completed pairs: %d, Incompleted pairs: %d, last candles %s, diff is %s, sleeping %s",
                    wanted_timestamp,
                    len(completed_pairs),
                    len(incompleted_pairs),
                    last_timestamps_log,
                    diff,
                    poll_delay)

        time.sleep(poll_delay.total_seconds())
        poll_cycle += 1

    raise NoNewDataReceived(
        f"Waited {max_wait} to get the data to make a trading strategy decision.\n"
        f"Decision cycle: {timestamp}.\n"
        f"Wanted candle timestamp: {wanted_timestamp}.\n"
        f"Latest candle we received: {latest_timestamp}.\n"
        f"Diff: {diff}.\n"
        f"Wait cycles: {poll_cycle}.\n"
        f"Pairs incomplete: {incompleted_pairs}.\n"
        f"Pairs complete: {completed_pairs}\n"
    )
