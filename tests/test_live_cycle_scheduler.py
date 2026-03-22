"""Test live cycle scheduling helpers."""

import datetime

from tradeexecutor.cli.loop import (
    calculate_live_strategy_cycle_timestamp,
    calculate_since_last_cycle_end_schedule,
    load_latest_live_cycle_end,
)
from tradeexecutor.state.state import State
from tradeexecutor.strategy.cycle import CycleDuration, snap_to_previous_tick
from tradeexecutor.strategy.strategy_cycle_trigger import StrategyCycleTrigger


def test_since_last_cycle_end_runs_immediately_without_anchor() -> None:
    """Rolling live scheduling should run immediately when no prior cycle end exists.

    1. Create an empty state and a fixed live clock timestamp.
    2. Calculate the rolling schedule for the new trigger mode.
    3. Verify both the logical cycle timestamp and scheduler wake-up are immediate.
    """

    # 1. Create an empty state and a fixed live clock timestamp.
    state = State()
    now_ = datetime.datetime(2026, 3, 19, 12, 0, 5)

    # 2. Calculate the rolling schedule for the new trigger mode.
    strategy_cycle_timestamp, run_at = calculate_since_last_cycle_end_schedule(
        state,
        CycleDuration.cycle_1d,
        now_,
    )

    # 3. Verify both the logical cycle timestamp and scheduler wake-up are immediate.
    assert strategy_cycle_timestamp == now_
    assert run_at == now_


def test_since_last_cycle_end_uses_previous_end_when_not_yet_due() -> None:
    """Rolling live scheduling should wait until 24 hours after the previous cycle end.

    1. Record a completed live cycle with a non-midnight end timestamp.
    2. Calculate the next rolling schedule before the next cycle is due.
    3. Verify the logical cycle timestamp and wake-up match `last_end + cycle_duration`.
    """

    # 1. Record a completed live cycle with a non-midnight end timestamp.
    state = State()
    ended_at = datetime.datetime(2026, 3, 19, 0, 7, 11)
    state.record_cycle_end(4, now_=ended_at, live=True)
    now_ = datetime.datetime(2026, 3, 19, 18, 0, 0)

    # 2. Calculate the next rolling schedule before the next cycle is due.
    strategy_cycle_timestamp, run_at = calculate_since_last_cycle_end_schedule(
        state,
        CycleDuration.cycle_1d,
        now_,
    )

    # 3. Verify the logical cycle timestamp and wake-up match `last_end + cycle_duration`.
    expected_due_at = datetime.datetime(2026, 3, 20, 0, 7, 11)
    assert strategy_cycle_timestamp == expected_due_at
    assert run_at == expected_due_at
    assert strategy_cycle_timestamp != datetime.datetime(2026, 3, 20, 0, 0, 0)


def test_since_last_cycle_end_runs_immediately_when_overdue() -> None:
    """Rolling live scheduling should execute immediately after a missed due time.

    1. Record a completed live cycle and move the wall clock past the next due time.
    2. Calculate the rolling schedule after the due time has already passed.
    3. Verify the logical cycle timestamp stays anchored to the missed due time while wake-up is immediate.
    """

    # 1. Record a completed live cycle and move the wall clock past the next due time.
    state = State()
    ended_at = datetime.datetime(2026, 3, 17, 12, 34, 56)
    state.record_cycle_end(8, now_=ended_at, live=True)
    now_ = datetime.datetime(2026, 3, 19, 9, 0, 0)

    # 2. Calculate the rolling schedule after the due time has already passed.
    strategy_cycle_timestamp, run_at = calculate_since_last_cycle_end_schedule(
        state,
        CycleDuration.cycle_1d,
        now_,
    )

    # 3. Verify the logical cycle timestamp stays anchored to the missed due time while wake-up is immediate.
    assert strategy_cycle_timestamp == datetime.datetime(2026, 3, 18, 12, 34, 56)
    assert run_at == now_


def test_load_latest_live_cycle_end_prefers_other_data_over_uptime() -> None:
    """Rolling live scheduling should prefer the persisted per-cycle live end timestamp.

    1. Create a state where the latest per-cycle stored timestamp differs from uptime.
    2. Load the rolling cycle anchor from state.
    3. Verify the per-cycle `decision_cycle_ended_at` value wins over uptime fallback.
    """

    # 1. Create a state where the latest per-cycle stored timestamp differs from uptime.
    state = State()
    state.uptime.record_cycle_complete(5, datetime.datetime(2026, 3, 19, 13, 0, 0))
    state.other_data.save(5, "decision_cycle_ended_at", datetime.datetime(2026, 3, 19, 12, 59, 30).isoformat())

    # 2. Load the rolling cycle anchor from state.
    latest_cycle_end = load_latest_live_cycle_end(state)

    # 3. Verify the per-cycle `decision_cycle_ended_at` value wins over uptime fallback.
    assert latest_cycle_end == datetime.datetime(2026, 3, 19, 12, 59, 30)


def test_load_latest_live_cycle_end_falls_back_to_uptime() -> None:
    """Rolling live scheduling should fall back to uptime when per-cycle data is missing.

    Also verifies the fallback survives a JSON round-trip, where cycles_completed_at
    values become ISO strings instead of datetime objects. This reproduces the
    hyper-ai production crash of 2026-03-21.

    1. Create a state with only uptime completion tracking populated.
    2. Load the rolling cycle anchor from state.
    3. Verify the latest uptime completion timestamp is used.
    4. Round-trip the state through JSON and verify the fallback still returns a datetime.
    """

    # 1. Create a state with only uptime completion tracking populated.
    state = State()
    completed_at = datetime.datetime(2026, 3, 19, 12, 15, 0)
    state.uptime.record_cycle_complete(2, completed_at)

    # 2. Load the rolling cycle anchor from state.
    latest_cycle_end = load_latest_live_cycle_end(state)

    # 3. Verify the latest uptime completion timestamp is used.
    assert latest_cycle_end == completed_at

    # 4. Round-trip the state through JSON and verify the fallback still returns a datetime.
    state_reloaded = State.from_json(state.to_json())
    latest_after_roundtrip = load_latest_live_cycle_end(state_reloaded)
    assert isinstance(latest_after_roundtrip, datetime.datetime)
    assert latest_after_roundtrip == completed_at


def test_existing_live_trigger_modes_keep_wall_clock_alignment() -> None:
    """Existing live trigger modes should keep their snapped wall-clock timestamps.

    1. Create a fixed live clock timestamp for a daily cycle.
    2. Calculate logical cycle timestamps for the pre-existing live trigger modes.
    3. Verify both modes still snap to the previous wall-clock tick.
    """

    # 1. Create a fixed live clock timestamp for a daily cycle.
    now_ = datetime.datetime(2026, 3, 19, 12, 34, 56)
    expected_timestamp = snap_to_previous_tick(now_, CycleDuration.cycle_1d)

    # 2. Calculate logical cycle timestamps for the pre-existing live trigger modes.
    cycle_offset_timestamp = calculate_live_strategy_cycle_timestamp(
        StrategyCycleTrigger.cycle_offset,
        CycleDuration.cycle_1d,
        now_,
    )
    trading_pair_data_timestamp = calculate_live_strategy_cycle_timestamp(
        StrategyCycleTrigger.trading_pair_data_availability,
        CycleDuration.cycle_1d,
        now_,
    )

    # 3. Verify both modes still snap to the previous wall-clock tick.
    assert cycle_offset_timestamp == expected_timestamp
    assert trading_pair_data_timestamp == expected_timestamp


def test_since_last_cycle_end_timestamp_uses_due_time_not_current_wall_clock() -> None:
    """Rolling live timestamps should stay anchored to the due time after restart catch-up.

    1. Record a completed live cycle and advance the clock past the next due time.
    2. Calculate the logical rolling cycle timestamp for the next live cycle.
    3. Verify the timestamp uses the due time instead of the current wall clock.
    """

    # 1. Record a completed live cycle and advance the clock past the next due time.
    state = State()
    ended_at = datetime.datetime(2026, 3, 18, 23, 59, 58)
    state.record_cycle_end(9, now_=ended_at, live=True)
    now_ = datetime.datetime(2026, 3, 20, 0, 1, 1)

    # 2. Calculate the logical rolling cycle timestamp for the next live cycle.
    strategy_cycle_timestamp = calculate_live_strategy_cycle_timestamp(
        StrategyCycleTrigger.since_last_cycle_end,
        CycleDuration.cycle_1d,
        now_,
        state,
    )

    # 3. Verify the timestamp uses the due time instead of the current wall clock.
    assert strategy_cycle_timestamp == datetime.datetime(2026, 3, 19, 23, 59, 58)
    assert strategy_cycle_timestamp != now_
