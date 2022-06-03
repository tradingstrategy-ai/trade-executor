"""Test trigger rounding"""
import datetime

from tradeexecutor.strategy.cycle import snap_to_next_tick, CycleDuration, snap_to_previous_tick


def test_tick_8h():
    """Snap our 8h ticks correctly."""
    time = datetime.datetime(2020, 1, 1, 5, 00)
    assert snap_to_next_tick(time, CycleDuration.cycle_8h) == datetime.datetime(2020, 1, 1, 8, 00)
    assert snap_to_next_tick(time, CycleDuration.cycle_8h, offset=datetime.timedelta(minutes=10)) == datetime.datetime(2020, 1, 1, 8, 10)


def test_tick_8h_snap_previous():
    """Snap our 8h ticks correctly."""
    time = datetime.datetime(2020, 1, 1, 5, 00)
    assert snap_to_previous_tick(time, CycleDuration.cycle_8h) == datetime.datetime(2020, 1, 1, 00, 00)
    assert snap_to_previous_tick(time, CycleDuration.cycle_8h, offset=datetime.timedelta(minutes=10)) == datetime.datetime(2020, 1, 1, 00, 10)

def test_tick_8h_snap_previous_equal():
    # If we are already in the time, do not move the clok
    assert snap_to_previous_tick(datetime.datetime(2020, 1, 1, 00, 00), CycleDuration.cycle_8h) == datetime.datetime(2020, 1, 1, 00, 00)
    assert snap_to_previous_tick(datetime.datetime(2020, 1, 1, 8, 00), CycleDuration.cycle_8h) == datetime.datetime(2020, 1, 1, 8, 00)
    assert snap_to_previous_tick(datetime.datetime(2020, 1, 1, 16, 00), CycleDuration.cycle_8h) == datetime.datetime(2020, 1, 1, 16, 00)


def test_tick_at_the_correct_time():
    """If we are at the correct timestamp, do not round."""
    time = datetime.datetime(2020, 1, 1, 8, 00)
    assert snap_to_next_tick(time, CycleDuration.cycle_8h) == datetime.datetime(2020, 1, 1, 8, 00)
    assert snap_to_next_tick(time, CycleDuration.cycle_8h, offset=datetime.timedelta(minutes=10)) == datetime.datetime(2020, 1, 1, 8, 10)


def test_tick_preserve_timezone():
    """Preserve UTC timezone in tick snap."""
    time = datetime.datetime(2020, 1, 1, 5, 00, tzinfo=datetime.timezone.utc)
    res = snap_to_next_tick(time, CycleDuration.cycle_8h, offset=datetime.timedelta(minutes=10))
    assert res.tzinfo == datetime.timezone.utc
    assert res == datetime.datetime(2020, 1, 1, 8, 10, tzinfo=datetime.timezone.utc)
