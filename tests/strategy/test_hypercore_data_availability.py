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
    wait_for_hypercore_data_availability,
    fetch_hypercore_decision_snapshot,
)
from tradeexecutor.strategy.cycle import CycleDuration


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


def test_manifest_wait_polls_json_until_ready() -> None:
    """Check polling stops on a ready receipt without any parquet concern.

    1. Return an incomplete receipt on the first poll and a ready receipt on
       the second.
    2. Advance an injected clock by one polling interval between calls.
    3. Verify the helper returns the ready receipt and poll count.
    """

    # 1. Return an incomplete receipt on the first poll and a ready receipt on the second.
    slot = datetime.datetime(2026, 9, 22)
    current = [slot]
    responses = iter(
        [
            _manifest("2026-09-21T23:55:00Z", "2026-09-21T23:55:00Z"),
            _manifest("2026-09-22T02:55:00Z", "2026-09-22T00:30:00Z"),
        ]
    )

    def now() -> datetime.datetime:
        return current[0]

    def fetch() -> dict:
        result = next(responses)
        current[0] += datetime.timedelta(minutes=15)
        return result

    def sleep(seconds: float) -> None:
        current[0] += datetime.timedelta(seconds=seconds)

    # 2. Advance an injected clock by one polling interval between calls.
    manifest, poll_count = wait_for_hypercore_data_availability(fetch, slot, now=now, poll_interval=datetime.timedelta(milliseconds=1), sleep=sleep)

    # 3. Verify the helper returns the ready receipt and poll count.
    assert poll_count == 2
    assert manifest["price_file"]["etag"] == "etag"


def test_manifest_wait_times_out_before_next_slot() -> None:
    """Reject late readiness and shutdown without a stale-data fallback.

    1. Keep returning a valid but incomplete receipt.
    2. Advance the injected clock beyond the eight-hour readiness window.
    3. Verify a timeout is raised even for a ready response; shutdown skips IO.
    """

    # 1. Keep returning a valid but incomplete receipt.
    slot = datetime.datetime(2026, 9, 22)
    current = [slot]

    def now() -> datetime.datetime:
        return current[0]

    def fetch() -> dict:
        current[0] = slot + HYPERCORE_READINESS_WINDOW
        return _manifest("2026-09-22T04:00:00Z", "2026-09-22T03:30:00Z")

    def sleep(seconds: float) -> None:
        current[0] += datetime.timedelta(seconds=seconds)

    # 2. Advance the injected clock beyond the eight-hour readiness window.
    # 3. Verify a timeout is raised.
    with pytest.raises(TimeoutError):
        wait_for_hypercore_data_availability(fetch, slot, now=now, poll_interval=datetime.timedelta(milliseconds=1), sleep=sleep)

    current[0] = slot
    shutdown = threading.Event()
    shutdown.set()
    fetch_mock = Mock()  # A pre-existing stop request must prevent network IO.
    with pytest.raises(RuntimeError, match="shutdown"):
        wait_for_hypercore_data_availability(fetch_mock, slot, now=now, shutdown_event=shutdown)
    fetch_mock.assert_not_called()

    # An off-grid deadline must not sleep on to the next quarter-hour mark.
    incomplete = Mock(return_value=_manifest(None, None))
    with pytest.raises(TimeoutError):
        wait_for_hypercore_data_availability(
            incomplete, slot, now=now, sleep=sleep,
            readiness_window=datetime.timedelta(minutes=10),
        )
    assert current[0] == slot + datetime.timedelta(minutes=10)
    incomplete.assert_called_once()


def test_hypercore_slot_schedule_joins_open_window() -> None:
    """Check a restart during the window keeps the midnight logical slot.

    1. Calculate a schedule from an intraday UTC timestamp.
    2. Verify the slot is midnight and the wake-up is immediate.
    3. Verify two-day cycle duration remains the calendar anchor.
    """

    # 1. Calculate a schedule from an intraday UTC timestamp.
    now = datetime.datetime(2026, 9, 22, 4, 15)
    slot, wake_up = calculate_hypercore_slot_schedule(now, CycleDuration.cycle_2d)

    # 2. Verify the slot is midnight and the wake-up is immediate.
    assert slot == datetime.datetime(2026, 9, 22)
    assert wake_up == now

    # 3. Verify two-day cycle duration remains the calendar anchor.
    assert slot + CycleDuration.cycle_2d.to_timedelta() == datetime.datetime(2026, 9, 24)

    # A process restarted after completing this slot must not trade it again.
    next_slot, wake_up = calculate_hypercore_slot_schedule(now, CycleDuration.cycle_2d, last_completed_slot=slot)
    assert next_slot == wake_up == datetime.datetime(2026, 9, 24)


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
