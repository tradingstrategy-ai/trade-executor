"""Tests for the manifest-only HyperCore readiness helper."""

import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from tradingstrategy.vault_data_client import VaultDataDeploymentError, VaultDataVersionMismatch, VaultManifestUnavailable

from tradeexecutor.cli import loop as loop_module
from tradeexecutor.cli.loop import ExecutionLoop
from tradeexecutor.strategy.hypercore_data_availability import (
    HYPERCORE_READINESS_WINDOW,
    calculate_hypercore_slot_schedule,
    fetch_current_hypercore_snapshot,
    poll_hypercore_decision_snapshot,
    next_hypercore_poll,
)
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.strategy_cycle_trigger import StrategyCycleTrigger
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.state.state import State


def _manifest(last_scan: str | None, last_candle: str | None) -> dict:
    """Build the smallest schema-valid manifest used by these tests."""

    return {
        "schema_version": 1,
        "published_at": "2026-09-22T03:00:00Z",
        "price_file": {"key": "cleaned-vault-prices-1h.parquet", "etag": "etag"},
        "chains": {
            "9999": {
                "name": "Hypercore",
                "last_successful_price_scan_ended_at": last_scan,
                "last_candle_at": last_candle,
            }
        },
    }


def test_startup_snapshot_does_not_wait_for_decision_readiness(tmp_path: Path) -> None:
    """Build the start-up valuation universe from the current verified file.

    1. Return a valid receipt whose HyperCore data is not ready for the next slot.
    2. Race one price upload and retry with the new receipt and ETag.
    3. Fail promptly on missing data or repeated races, rather than wait for a slot.
    """
    # 1. The latest published data may still be from before the pending slot.
    client = Mock()
    older = _manifest("2026-09-21T20:00:00Z", "2026-09-21T16:00:00Z")
    newer = _manifest("2026-09-21T20:00:00Z", "2026-09-21T16:00:00Z")
    newer["price_file"]["etag"] = "new-etag"
    client.fetch_vault_scan_manifest.side_effect = [older, newer]
    destination = tmp_path / "prices.parquet"
    client.download.side_effect = [VaultDataVersionMismatch("race"), destination]

    # 2. One race rechecks the receipt, and both transfers remain ETag-pinned.
    assert fetch_current_hypercore_snapshot(client, destination) is newer
    assert client.fetch_vault_scan_manifest.call_count == 2
    assert [call.kwargs["expected_etag"] for call in client.download.call_args_list] == ["etag", "new-etag"]
    assert all(call.kwargs["destination"] == destination for call in client.download.call_args_list)

    # 3. Missing receipts and a second race fail start-up immediately.
    client.reset_mock()
    client.fetch_vault_scan_manifest.side_effect = VaultManifestUnavailable("HTTP 503")
    with pytest.raises(VaultManifestUnavailable, match="HTTP 503"):
        fetch_current_hypercore_snapshot(client, destination)
    client.reset_mock()
    client.fetch_vault_scan_manifest.side_effect = None
    client.fetch_vault_scan_manifest.return_value = older
    client.download.side_effect = VaultDataVersionMismatch("race")
    with pytest.raises(VaultDataVersionMismatch, match="changed twice"):
        fetch_current_hypercore_snapshot(client, destination)
    assert client.fetch_vault_scan_manifest.call_count == 2


def test_hypercore_startup_builds_universe_before_future_slot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Expose universe construction errors on restart, not at the next decision.

    1. Set up a pending two-day slot and a verified but not slot-ready receipt.
    2. Enter the real live start-up path without starting a scheduler.
    3. Confirm universe warm-up is reached immediately with the verified file.
    """
    # 1. Replace only the external snapshot and watchdog boundaries.
    monkeypatch.setattr(loop_module, "create_watchdog_registry", lambda mode: object())
    monkeypatch.setattr(loop_module, "start_background_watchdog", lambda registry: None)
    monkeypatch.setattr(loop_module, "register_worker", lambda *args: None)
    monkeypatch.setattr(loop_module.logger, "trade", lambda *args: None, raising=False)
    monkeypatch.setattr(loop_module, "native_datetime_utc_now", lambda: datetime.datetime(2026, 9, 23, 12))
    monkeypatch.setattr(loop_module, "create_vault_data_client", lambda client: object())
    snapshot_calls = []

    def fetch_snapshot(client: object, destination: Path) -> dict:
        """Record the start-up download without using an external dataset."""
        snapshot_calls.append(destination)
        return _manifest("2026-09-21T20:00:00Z", "2026-09-21T16:00:00Z")

    monkeypatch.setattr(loop_module, "fetch_current_hypercore_snapshot", fetch_snapshot)
    state = State()
    loop = SimpleNamespace(
        is_backtest=lambda: False,
        is_live_trading_unit_test=lambda: False,
        backtest_start=None,
        backtest_end=None,
        cycle_duration=CycleDuration.cycle_2d,
        execution_context=object(),
        run_state=object(),
        strategy_cycle_trigger=StrategyCycleTrigger.hypercore_data_available,
        trade_immediately=False,
        client=object(),
        store=SimpleNamespace(sync=lambda state: None),
        universe_options=UniverseOptions(),
        tick_offset=datetime.timedelta(0),
    )

    def warm_up() -> None:
        """Stop after the universe build entry point proves start-up reached it."""
        assert loop.universe_options.vault_price_snapshot == tmp_path / "vault-prices.parquet"
        assert loop.universe_options.end_at <= loop_module.native_datetime_utc_now()
        raise RuntimeError("universe build reached before slot readiness")

    loop.warm_up_live_trading = warm_up

    # 2. The old path would wait for the future slot before calling warm-up.
    with pytest.raises(RuntimeError, match="universe build reached before slot readiness"):
        ExecutionLoop._run_live(loop, state, tmp_path)

    # 3. The pending slot remains gated despite this immediate valuation build.
    assert snapshot_calls == [tmp_path / "vault-prices.parquet"]
    assert state.pending_data_availability_slot == datetime.datetime(2026, 9, 24)


def test_manifest_wait_polls_json_until_ready(tmp_path: Path) -> None:
    """Keep the scheduler free until a future slot's data is ready.

    1. Return without requesting data before the slot, then probe incomplete data.
    2. Advance to the next scheduled quarter-hour and probe a ready receipt.
    3. Verify only the ready receipt downloads the verified price file.
    """
    # 1. The first scheduled callback may run before midnight after warm-up.
    slot = datetime.datetime(2026, 9, 22)
    current = [slot - datetime.timedelta(hours=1)]
    client = Mock()
    client.fetch_vault_scan_manifest.side_effect = [
        _manifest(None, None),
        _manifest("2026-09-22T00:15:00Z", "2026-09-22T00:00:00Z"),
    ]
    manifest, count = poll_hypercore_decision_snapshot(client, slot, tmp_path / "prices", now=lambda: current[0])
    assert manifest is None
    assert count == 0
    assert next_hypercore_poll(slot, current[0]) == slot
    client.fetch_vault_scan_manifest.assert_not_called()

    current[0] = slot
    manifest, count = poll_hypercore_decision_snapshot(client, slot, tmp_path / "prices", now=lambda: current[0])
    assert manifest is None
    assert count == 1
    assert current[0] == slot
    client.download.assert_not_called()

    # 2. The scheduler, not this helper, controls the wait between probes.
    current[0] = next_hypercore_poll(slot, current[0])
    manifest, count = poll_hypercore_decision_snapshot(client, slot, tmp_path / "prices", now=lambda: current[0])

    # 3. Only the ready probe downloads prices; no indicators are involved.
    assert current[0] == slot + datetime.timedelta(minutes=15)
    assert manifest["price_file"]["etag"] == "etag"
    assert count == 1
    assert client.fetch_vault_scan_manifest.call_count == 2
    client.download.assert_called_once()


def test_manifest_wait_times_out_before_next_slot(tmp_path: Path) -> None:
    """Reject late JSON receipts and preserve the eight-hour deadline.

    1. Deliver a ready receipt exactly at the deadline.
    2. Check polling skips elapsed marks but never schedules past the deadline.
    3. Verify an expired slot makes no further network requests.
    """
    # 1. The network mock changes time while delivering its response.
    slot = datetime.datetime(2026, 9, 22)
    current = [slot]
    client = Mock()

    def fetch(**kwargs) -> dict:
        """Model a slow receipt that arrives too late to qualify."""
        current[0] = slot + HYPERCORE_READINESS_WINDOW
        return _manifest("2026-09-22T04:00:00Z", "2026-09-22T03:30:00Z")

    client.fetch_vault_scan_manifest.side_effect = fetch
    with pytest.raises(TimeoutError):
        poll_hypercore_decision_snapshot(client, slot, tmp_path / "prices", now=lambda: current[0])
    client.download.assert_not_called()

    # 2. Recurring jobs remain on the original deadline-aware grid.
    assert next_hypercore_poll(slot, slot + datetime.timedelta(minutes=16)) == slot + datetime.timedelta(minutes=30)
    assert next_hypercore_poll(slot, slot + datetime.timedelta(hours=7, minutes=59)) == slot + HYPERCORE_READINESS_WINDOW
    with pytest.raises(TimeoutError):
        poll_hypercore_decision_snapshot(client, slot, tmp_path / "prices", now=lambda: current[0])
    assert client.fetch_vault_scan_manifest.call_count == 1

    # 3. No second request starts once the original deadline has elapsed.
    assert client.fetch_vault_scan_manifest.call_count == 1


def test_hypercore_slot_schedule_joins_open_window() -> None:
    """Check a restart during the window keeps the midnight logical slot.

    1. Calculate a schedule from an intraday UTC timestamp.
    2. Verify fresh and restarted decisions preserve the two-day midnight grid.
    3. Skip expired unexecuted decisions without marking them completed.
    """

    # 1. Calculate a schedule from an intraday UTC timestamp.
    now = datetime.datetime(2026, 9, 22, 4, 15)
    state = State()
    slot = calculate_hypercore_slot_schedule(now, CycleDuration.cycle_2d, state)

    # 2. Fresh and restarted decisions preserve the two-day midnight grid.
    assert slot == datetime.datetime(2026, 9, 22)

    assert slot + CycleDuration.cycle_2d.to_timedelta() == datetime.datetime(2026, 9, 24)
    state.pending_data_availability_slot = slot
    assert calculate_hypercore_slot_schedule(now, CycleDuration.cycle_2d, state) == slot

    # 3. Expiry needs no operator command, including a restart exactly at 08:00.
    for restart in (slot + datetime.timedelta(hours=8), slot + datetime.timedelta(hours=12)):
        assert calculate_hypercore_slot_schedule(restart, CycleDuration.cycle_2d, state) == datetime.datetime(2026, 9, 24)
    assert state.last_cycle_at is None
    assert state.pending_data_availability_slot == slot  # Caller persists the replacement.

    # A process restarted after completing this slot must not trade it again.
    state.pending_data_availability_slot = None
    state.last_cycle_at = slot
    assert calculate_hypercore_slot_schedule(now, CycleDuration.cycle_2d, state) == datetime.datetime(2026, 9, 24)


def test_hypercore_restart_refuses_persisted_trades() -> None:
    """Keep trade reconciliation separate from harmless missed-data windows.

    1. Load real trade records and identify their decision timestamp.
    2. Attempt restart both inside and after that decision's readiness window.
    3. Verify both attempts refuse replay without changing pending state.
    """
    # 1. Reuse a serialised portfolio rather than fabricating trade behaviour.
    state = State.read_json_file(Path(__file__).parents[1] / "cli" / "show-positions-long.json")
    slot = next(iter(state.portfolio.get_all_trades())).opened_at
    state.pending_data_availability_slot = slot

    # 2. Trade protection applies regardless of whether the data window expired.
    for now in (slot, slot + datetime.timedelta(days=4)):
        with pytest.raises(RuntimeError, match="reconcile"):
            calculate_hypercore_slot_schedule(now, CycleDuration.cycle_2d, state)

    # 3. No unsafe automatic completion or state clearing occurred.
    assert state.pending_data_availability_slot == slot


def test_sparse_scan_wait_and_version_race(tmp_path: Path) -> None:
    """Wait for a four-hour scan and retry a racing publication on the same slot.

    1. Model sparse receipts and one transient network failure before 04:00.
    2. Return a ready receipt whose price ETag races with the next publication.
    3. Verify one immediate JSON recheck, private download and capped budgets.
    """
    # 1. Simulate the API and clock; no hourly samples or parquet probes occur
    # during the sixteen quarter-hour polls preceding the next four-hour scan.
    slot = datetime.datetime(2026, 9, 22)
    current = [slot]
    calls = []
    client = Mock()

    def fetch(*, request_budget: float) -> dict:
        """Model a scanner publishing its next observation at 04:00."""
        calls.append((current[0], request_budget))
        if len(calls) == 2:
            raise VaultManifestUnavailable("HTTP 503")
        if current[0] < slot + datetime.timedelta(hours=4):
            return _manifest("2026-09-21T20:30:00Z", "2026-09-21T20:00:00Z")
        manifest = _manifest("2026-09-22T04:00:00Z", "2026-09-22T03:30:00Z")
        manifest["published_at"] = "2026-09-22T04:00:00Z"
        manifest["price_file"]["etag"] = "new" if len(calls) == 18 else "old"
        return manifest

    client.fetch_vault_scan_manifest.side_effect = fetch
    client.download.side_effect = [VaultDataVersionMismatch("race"), tmp_path / "private.parquet"]

    # 2. The scheduler calls the single-probe API every fifteen minutes.
    total_polls = 0
    manifest = None
    while manifest is None:
        manifest, count = poll_hypercore_decision_snapshot(
            client, slot, tmp_path / "private.parquet", now=lambda: current[0],
        )
        total_polls += count
        if manifest is None:
            current[0] = next_hypercore_poll(slot, current[0])

    # 3. Only the ready receipts trigger price requests; both target private data.
    assert total_polls == 18
    assert client.download.call_count == 2
    assert manifest["price_file"]["etag"] == "new"
    assert calls[-1][0] == calls[-2][0] == slot + datetime.timedelta(hours=4)
    assert all(budget == 300 for _, budget in calls)
    assert client.download.call_args.kwargs["destination"] == tmp_path / "private.parquet"


def test_verified_transfer_may_finish_after_readiness_deadline(tmp_path: Path) -> None:
    """Separate JSON readiness from the duration of a large parquet download.

    1. Receive a valid receipt just before the eight-hour cutoff.
    2. Advance the clock past the cutoff while transferring matching data.
    3. Verify success, while expired receipts and deployment errors cannot retry.
    """
    # 1. A deterministic clock avoids an eight-hour integration-test wait.
    slot = datetime.datetime(2026, 9, 22)
    current = [slot + datetime.timedelta(hours=7, minutes=59)]
    client = Mock()
    client.fetch_vault_scan_manifest.return_value = _manifest("2026-09-22T02:55:00Z", "2026-09-22T00:30:00Z")

    def download(*args, **kwargs) -> Path:
        """Model a large matching parquet transfer crossing the deadline."""
        current[0] = slot + datetime.timedelta(hours=8, minutes=5)
        return tmp_path / "snapshot"

    client.download.side_effect = download
    # 2. Data transfer may finish after the readiness deadline.
    manifest, count = poll_hypercore_decision_snapshot(client, slot, tmp_path / "snapshot", now=lambda: current[0])
    assert manifest is not None
    assert count == 1
    assert client.fetch_vault_scan_manifest.call_args.kwargs["request_budget"] == 60

    # 3. No receipt may start once the original deadline has expired.
    with pytest.raises(TimeoutError):
        poll_hypercore_decision_snapshot(client, slot, tmp_path / "snapshot", now=lambda: current[0])
    assert client.fetch_vault_scan_manifest.call_count == 1

    current[0] = slot
    client.download.side_effect = VaultDataDeploymentError("Missing ETag")
    with pytest.raises(VaultDataDeploymentError, match="Missing ETag"):
        poll_hypercore_decision_snapshot(client, slot, tmp_path / "snapshot", now=lambda: current[0])
    assert client.fetch_vault_scan_manifest.call_count == 2
