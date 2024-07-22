"""Strategy cycle definitions.

See :ref:`strategy cycle` for more information.
"""
import datetime
import enum
from typing import Optional

import pandas as pd
from tradingstrategy.timebucket import TimeBucket


class CycleDuration(enum.Enum):
    """Strategy cycle duration options.

    This enum defines what strategy cycle durations backtesting and live
    testing engine supports.

    It is also the value you can enter as `trading_strategy_cycle`
    option for your strategies.

    All cycles are aligned to the wall clock time.
    E.g. 24h cycle is always run at 00:00.

    See :ref:`strategy cycle` for more information.
    """

    #: Run `decide_trades()` one second
    #:
    #: Only used in unit testing.
    #: See `strategies/test_only_/enzymy_end_to_end.py`.
    #:
    cycle_1s = "1s"

    #: Run `decide_trades()` every minute
    cycle_1m = "1m"

    #: Run `decide_trades()` every 5 minutes
    cycle_5m = "5m"

    #: Run `decide_trades()` every 15 minutes
    cycle_15m = "15m"

    #: Run `decide_trades()` every 30 minutes
    cycle_30m = "30m"

    #: Run `decide_trades()` every hour
    cycle_1h = "1h"

    #: Run `decide_trades()` every 2 hours
    cycle_2h = "2h"

    #: Run `decide_trades()` every 4 hours
    cycle_4h = "4h"

    #: Run `decide_trades()` every 6 hours
    cycle_6h = "6h"

    #: Run `decide_trades()` for every 8 hours
    cycle_8h = "8h"

    #: Run `decide_trades()` for every 10 hours
    cycle_10h = "10h"

    #: Run `decide_trades()` for every 12 hours
    cycle_12h = "12h"

    #: Run `decide_trades()` for every 16 hours
    cycle_16h = "16h"

    #: Run `decide_trades()` for every 24h hours
    cycle_1d = "1d"

    #: Run `decide_trades()` for every 2 days
    cycle_2d = "2d"

    #: Run `decide_trades()` for every 2 days
    cycle_3d = "3d"

    #: Run `decide_trades()` for every 4 days
    cycle_4d = "4d"

    #: Run `decide_trades()` for every week
    cycle_7d = "7d"

    #: Run `decide_trades()` for 2 weeks cycl
    cycle_10d = "10d"

    #: Run `decide_trades()` for 2 weeks cycl
    cycle_14d = "14d"

    #: Run `decide_trades()` for every month
    cycle_30d = "30d"

    #: Random cycle that's prime number in hours
    cycle_97h = "97h"

    #: Don't really know or care about the trade cycle duration.
    #:
    #: Used when doing a simulated execution loop
    #: with `set_up_simulated_execution_loop`
    #: and where the time is ticked through manually by producing
    #: new blocks with EthereumTester chain.
    cycle_unknown = "unknown"

    #: Alias to match :py:class:`TimeBucket`
    s1 = cycle_1s

    #: Alias to match :py:class:`TimeBucket`
    m1 = cycle_1m

    #: Alias to match :py:class:`TimeBucket`
    m15 = cycle_15m

    #: Alias to match :py:class:`TimeBucket`
    h1 = cycle_1h

    #: Alias to match :py:class:`TimeBucket`
    h4 = cycle_4h

    #: Alias to match :py:class:`TimeBucket`
    d1 = cycle_1d

    #: Alias to match :py:class:`TimeBucket`
    d7 = cycle_7d

    #: Alias
    unknown = cycle_unknown

    def to_timedelta(self) -> datetime.timedelta:
        """Get the duration of the strategy cycle as Python timedelta object."""
        return _TICK_DURATIONS[self]

    def to_pandas_timedelta(self) -> pd.Timedelta:
        return pd.Timedelta(self.to_timedelta())

    def to_timebucket(self) -> Optional[TimeBucket]:
        """Convert to trading-strategy client format.

        TODO: Try to avoid tightly coupling and leaking trading-strategy client here.

        Unlike TimeBucket, CycleDuration may have "unknown" value that is presented by None
        """
        return TimeBucket(self.value) if self  != CycleDuration.cycle_unknown else None

    def get_yearly_periods(self) -> float:
        """How many decision cycle periods a year has.

        This metric is used to calculate Sharpe, other metrics.

        See :py:func:`tradeexecutor.analysis.advanced_metrics.calculate_advanced_metrics`
        for more information.
        """
        return pd.Timedelta(days=365.0) / self.to_timedelta()

    @staticmethod
    def from_timebucket(bucket: TimeBucket) -> Optional["CycleDuration"]:
        """Convert from OHLCV time frame."""
        return CycleDuration(bucket.value)


def round_datetime_up(
        ts: datetime.datetime,
        delta: datetime.timedelta,
        offset: datetime.timedelta = datetime.timedelta(minutes=0)) -> datetime.datetime:
    """Snap to next available timedelta.

    Preserve any timezone info on `ts`.

    If we are at the the given exact delta, then do not round, only add offset.

    :param ts: Timestamp we want to round
    :param delta: Our snap grid
    :param offset: Add a fixed time offset at the top of rounding
    :return: When to wake up from the sleep next time
    """
    rounded = ts + (datetime.datetime.min.replace(tzinfo=ts.tzinfo) - ts) % delta
    return rounded + offset


def round_datetime_down(
        ts: datetime.datetime,
        delta: datetime.timedelta,
        offset: datetime.timedelta = datetime.timedelta(minutes=0)) -> datetime.datetime:
    """Snap to previous available timedelta.

    Preserve any timezone info on `ts`.

    If we are at the the given exact delta, then do not round, only add offset.

    :param ts: Timestamp we want to round
    :param delta: Our snap grid
    :param offset: Add a fixed time offset at the top of rounding
    :return: When to wake up from the sleep next time
    """
    assert isinstance(ts, datetime.datetime)
    mod = (datetime.datetime.min.replace(tzinfo=ts.tzinfo) - ts) % delta
    if mod == datetime.timedelta(0):
        return ts
    rounded = ts - delta + mod
    return rounded + offset


def snap_to_next_tick(
        ts: datetime.datetime,
        tick_size: CycleDuration,
        offset: datetime.timedelta = datetime.timedelta(minutes=0)) -> datetime.datetime:
    """Calculate when the trading logic should wake up from the sleep next time.

    If cycle duration is unknown do nothing.

    :param ts: Current timestamp
    :param tick_size: How big leaps we are taking
    :param offset: How many minutes of offset we assume to ensure we have candle data generated after the timestamp
    :return: When to wake up from the sleep next time
    """

    if tick_size == CycleDuration.cycle_unknown:
        return ts

    delta = tick_size.to_timedelta()
    return round_datetime_up(ts, delta, offset)


def snap_to_previous_tick(
        ts: datetime.datetime,
        tick_size: CycleDuration,
        offset: datetime.timedelta = datetime.timedelta(minutes=0)) -> datetime.datetime:
    """Calculate what should the tick time for given real time.

    If `ts` matches the tick, do nothing.

    If cycle duration is unknown do nothing.

    :param ts: Current timestamp
    :param tick_size: How big leaps we are taking
    :param offset: How many minutes of offset we assume to ensure we have candle data generated after the timestamp
    :return: What tick are we living in
    """

    if tick_size == CycleDuration.cycle_unknown:
        return ts

    delta = tick_size.to_timedelta()
    return round_datetime_down(ts, delta, offset)


_TICK_DURATIONS = {
    CycleDuration.cycle_1s: datetime.timedelta(seconds=1),
    CycleDuration.cycle_1m: datetime.timedelta(minutes=1),
    CycleDuration.cycle_5m: datetime.timedelta(minutes=5),
    CycleDuration.cycle_15m: datetime.timedelta(minutes=15),
    CycleDuration.cycle_30m: datetime.timedelta(minutes=30),
    CycleDuration.cycle_1h: datetime.timedelta(hours=1),
    CycleDuration.cycle_2h: datetime.timedelta(hours=2),
    CycleDuration.cycle_4h: datetime.timedelta(hours=4),
    CycleDuration.cycle_6h: datetime.timedelta(hours=6),
    CycleDuration.cycle_8h: datetime.timedelta(hours=8),
    CycleDuration.cycle_10h: datetime.timedelta(hours=10),
    CycleDuration.cycle_12h: datetime.timedelta(hours=12),
    CycleDuration.cycle_16h: datetime.timedelta(hours=16),
    CycleDuration.cycle_1d: datetime.timedelta(hours=24),
    CycleDuration.cycle_2d: datetime.timedelta(days=2),
    CycleDuration.cycle_3d: datetime.timedelta(days=3),
    CycleDuration.cycle_4d: datetime.timedelta(days=4),
    CycleDuration.cycle_7d: datetime.timedelta(days=7),
    CycleDuration.cycle_10d: datetime.timedelta(days=10),
    CycleDuration.cycle_14d: datetime.timedelta(days=14),
    CycleDuration.cycle_30d: datetime.timedelta(days=30),
    CycleDuration.cycle_unknown: datetime.timedelta(days=0),
    CycleDuration.cycle_97h: datetime.timedelta(hours=97),
}

assert len(_TICK_DURATIONS) == len(CycleDuration)  # sanity check
