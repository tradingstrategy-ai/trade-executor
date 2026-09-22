"""Tests for the manifest-only HyperCore readiness helper."""

import datetime
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest
from tradingstrategy.vault_data_client import VaultDataDeploymentError, VaultDataVersionMismatch, VaultManifestUnavailable

from tradeexecutor.strategy.hypercore_data_availability import (
    HYPERCORE_READINESS_WINDOW,
    calculate_hypercore_slot_schedule,
    poll_hypercore_decision_snapshot,
    next_hypercore_poll,
    fetch_hypercore_decision_snapshot,
)
from tradeexecutor.strategy.cycle import CycleDuration
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


def test_manifest_wait_polls_json_until_ready(tmp_path: Path) -> None:
    """Return control on incomplete data and wait only during initial warm-up.

    1. Probe incomplete data once without downloading prices or advancing time.
    2. Use the startup wrapper to wait on the same quarter-hour grid.
    3. Verify only a ready receipt downloads prices, with accurate request counts.
    """
    # 1. Fake only network/time boundaries; exercise the actual probe and wrapper.
    slot = datetime.datetime(2026, 9, 22)
    current = [slot]
    client = Mock()
    client.fetch_vault_scan_manifest.return_value = _manifest(None, None)
    manifest, count = poll_hypercore_decision_snapshot(client, slot, tmp_path / "prices", now=lambda: current[0])
    assert manifest is None
    assert count == 1
    assert current[0] == slot
    client.download.assert_not_called()

    # 2. Startup waits, while the scheduler can use the single-probe API above.
    client.reset_mock()
    client.fetch_vault_scan_manifest.side_effect = [
        _manifest(None, None),
        _manifest("2026-09-22T00:15:00Z", "2026-09-22T00:00:00Z"),
    ]

    def sleep(seconds: float) -> None:
        """Advance the deterministic clock instead of sleeping fifteen minutes."""
        current[0] += datetime.timedelta(seconds=seconds)

    manifest, count = fetch_hypercore_decision_snapshot(
        client, slot, tmp_path / "prices", now=lambda: current[0], sleep=sleep,
    )

    # 3. One sleep and one verified download suffice; no indicators are involved.
    assert current[0] == slot + datetime.timedelta(minutes=15)
    assert manifest["price_file"]["etag"] == "etag"
    assert count == 2
    assert client.fetch_vault_scan_manifest.call_count == 2
    client.download.assert_called_once()


def test_manifest_wait_times_out_before_next_slot(tmp_path: Path) -> None:
    """Reject late JSON receipts, preserve the deadline grid, and honour shutdown.

    1. Deliver a ready receipt exactly at the deadline.
    2. Check polling skips elapsed marks but never schedules past the deadline.
    3. Verify a pre-existing shutdown request prevents all network requests.
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
        fetch_hypercore_decision_snapshot(client, slot, tmp_path / "prices", now=lambda: current[0])
    client.download.assert_not_called()

    # 2. Both startup and recurring jobs share the same deadline-aware grid.
    assert next_hypercore_poll(slot, slot + datetime.timedelta(minutes=16)) == slot + datetime.timedelta(minutes=30)
    assert next_hypercore_poll(slot, slot + datetime.timedelta(hours=7, minutes=59)) == slot + HYPERCORE_READINESS_WINDOW
    with pytest.raises(TimeoutError):
        poll_hypercore_decision_snapshot(client, slot, tmp_path / "prices", now=lambda: current[0])
    assert client.fetch_vault_scan_manifest.call_count == 1

    # 3. Cancellation must not initiate another HTTP request.
    shutdown = threading.Event()
    shutdown.set()
    client.reset_mock()
    with pytest.raises(RuntimeError, match="shutdown"):
        fetch_hypercore_decision_snapshot(client, slot, tmp_path / "prices", now=lambda: slot, shutdown_event=shutdown)
    client.fetch_vault_scan_manifest.assert_not_called()


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

    def sleep(seconds: float) -> None:
        """Advance the injected clock on each poll interval."""
        current[0] += datetime.timedelta(seconds=seconds)

    client.fetch_vault_scan_manifest.side_effect = fetch
    client.download.side_effect = [VaultDataVersionMismatch("race"), tmp_path / "private.parquet"]

    # 2. Exercise the actual polling and version retry orchestration.
    manifest, count = fetch_hypercore_decision_snapshot(
        client, slot, tmp_path / "private.parquet", now=lambda: current[0], sleep=sleep,
    )

    # 3. Only the ready receipts trigger price requests; both target private data.
    assert count == 18
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
    fetch_hypercore_decision_snapshot(client, slot, tmp_path / "snapshot", now=lambda: current[0])
    assert client.fetch_vault_scan_manifest.call_args.kwargs["request_budget"] == 60

    # 3. No receipt may start once the original deadline has expired.
    with pytest.raises(TimeoutError):
        fetch_hypercore_decision_snapshot(client, slot, tmp_path / "snapshot", now=lambda: current[0])
    assert client.fetch_vault_scan_manifest.call_count == 1

    current[0] = slot
    client.download.side_effect = VaultDataDeploymentError("Missing ETag")
    with pytest.raises(VaultDataDeploymentError, match="Missing ETag"):
        fetch_hypercore_decision_snapshot(client, slot, tmp_path / "snapshot", now=lambda: current[0])
    assert client.fetch_vault_scan_manifest.call_count == 2
