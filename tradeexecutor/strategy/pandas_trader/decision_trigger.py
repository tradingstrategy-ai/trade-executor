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

    updated_universe: TradingStrategyUniverse

    ready_at: datetime.datetime

    time_waited: datetime.timedelta

    poll_cycles: int


def fetch_data(
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
        df: pd.DataFrame
) -> TradingStrategyUniverse:
    """Update a Trading Universe with a new candle data.

    :param df:
        Unsorted DataFrame containing data for all trading pairs we are interested in.
    """
    updated_universe = universe.clone()
    updated_universe.universe.candles = GroupedCandleUniverse(df)
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
        What is the latest candle timestamp our strategy decision cycle needs to consume

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
        poll_delay = datetime.timedelta(seconds=30),
) -> UpdatedUniverseResult:
    """Wait for the data to be available for the latest strategy cycle.

    - Used in live execution only

    - Uses Trading Strategy oracle real-time JSONL API for the data.

    - Uses simple polling appraoch

    :param timestamp:
        The strategy decision timestamp.

    :param current_universe:
        The current trading universe with old candles.

    :param required_history_period:
        How much historical data we need to load.

        Depends on the strategy. Defaults to 90 days.

    :param max_wait:
        Unless data is seen, die with an exception after this period.

    :param max_poll_cycles:
        Can be set in integration testing.

        Return after this many cycles despite new data being incomplete.

    :return:
        An updated trading universe
    """

    # List of monitored pairs by this strategy
    pairs = current_universe.universe.pairs
    assert len(pairs.pair_map) > 0, "No pairs in the pair_map"

    bucket = current_universe.universe.time_bucket

    completed_pairs: Set[DEXPair] = set()
    incompleted_pairs: Set[DEXPair] = {pairs.get_pair_by_id(id) for id in pairs.pair_map.keys()}

    started_at = datetime.datetime.utcnow()
    deadline = started_at + max_wait

    poll_cycle = 1

    while datetime.datetime.utcnow() < deadline:

        # Get new candles for all pairs for which we do not have new enough data yet
        avail_map = fetch_availability(
            client,
            bucket,
            incompleted_pairs,
        )

        # Move any pairs with new complete data to the completed set
        for p in incompleted_pairs:
            latest_timestamp = avail_map[p.pair_id]["last_candle_at"]
            if latest_timestamp >= timestamp:
                incompleted_pairs.remove(p)
                completed_pairs.add(p)

            if not incompleted_pairs or poll_cycle >= max_poll_cycles:
                # We have latest data for all pairs and can now update the universe
                df = fetch_data(
                    client,
                    bucket,
                    timestamp,
                    completed_pairs,
                    required_history_period,
                )
                updated_universe = update_universe(current_universe, df)
                time_waited = datetime.datetime.utcnow() - started_at
                return UpdatedUniverseResult(
                    updated_universe=updated_universe,
                    ready_at=datetime.datetime.utcnow(),
                    time_waited=time_waited,
                    poll_cycles=poll_cycle,
                )

        time.sleep(poll_delay.total_seconds())
        poll_cycle += 1

    raise NoNewDataReceived(
        f"Waited {max_wait} to get the data to make a trading strategy decision.\n"
        f"Decision cycle: {timestamp}.\n"
        f"Pairs incomplete: {incompleted_pairs}.\n"
        f"Pairs complete: {completed_pairs}\n"
    )
