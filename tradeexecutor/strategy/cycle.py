"""Strategy tick definitions."""
import datetime
import enum


class CycleDuration(enum.Enum):
    """Supported strategy ticks.
    """
    cycle_8h = "8h"
    cycle_16h = "16h"
    cycle_24h = "24h"

    def to_timedelta(self) -> datetime.timedelta:
        """Get the duration of the tick."""
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
    CycleDuration.cycle_8h: datetime.timedelta(hours=8),
    CycleDuration.cycle_16h: datetime.timedelta(hours=16),
    CycleDuration.cycle_24h: datetime.timedelta(hours=24),
}
