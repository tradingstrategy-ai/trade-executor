"""Strategy cycle definitions.

See :ref:`strategy cycle` for more information.
"""
import datetime
import enum


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

    #: Run `decide_trades()` every minute
    cycle_1m = "1m"

    #: Run `decide_trades()` every 5 minutes
    cycle_5m = "5m"

    #: Run `decide_trades()` every 15 minutes
    cycle_15m = "15m"

    #: Run `decide_trades()` every hour
    cycle_1h = "1h"

    #: Run `decide_trades()` every 4 hours
    cycle_4h = "4h"

    #: Run `decide_trades()` for every 8 hours
    cycle_8h = "8h"

    #: Run `decide_trades()` for every 16 hours
    cycle_16h = "16h"

    #: Run `decide_trades()` for every 24h hours
    cycle_24h = "24h"

    def to_timedelta(self) -> datetime.timedelta:
        """Get the duration of the strategy cycle as Python timedelta object."""
        return _TICK_DURATIONS[self]


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

    :param ts: Current timestamp
    :param tick_size: How big leaps we are taking
    :param offset: How many minutes of offset we assume to ensure we have candle data generated after the timestamp
    :return: When to wake up from the sleep next time
    """
    delta = tick_size.to_timedelta()
    return round_datetime_up(ts, delta, offset)


def snap_to_previous_tick(
        ts: datetime.datetime,
        tick_size: CycleDuration,
        offset: datetime.timedelta = datetime.timedelta(minutes=0)) -> datetime.datetime:
    """Calculate what should the tick time for given real time.

    If `ts` matches the tick, do nothing.

    :param ts: Current timestamp
    :param tick_size: How big leaps we are taking
    :param offset: How many minutes of offset we assume to ensure we have candle data generated after the timestamp
    :return: What tick are we living in
    """
    delta = tick_size.to_timedelta()
    return round_datetime_down(ts, delta, offset)

_TICK_DURATIONS = {
    CycleDuration.cycle_1m: datetime.timedelta(minutes=1),
    CycleDuration.cycle_5m: datetime.timedelta(minutes=5),
    CycleDuration.cycle_15m: datetime.timedelta(minutes=15),
    CycleDuration.cycle_1h: datetime.timedelta(hours=1),
    CycleDuration.cycle_4h: datetime.timedelta(hours=4),
    CycleDuration.cycle_8h: datetime.timedelta(hours=8),
    CycleDuration.cycle_16h: datetime.timedelta(hours=16),
    CycleDuration.cycle_24h: datetime.timedelta(hours=24),
}

assert len(_TICK_DURATIONS) == len(CycleDuration)  # sanity check
