"""Readiness helpers for the HyperCore manifest-triggered live cycle.

The executor polls the small vault scan manifest after a calendar-aligned
midnight slot. This module intentionally does not construct a universe or read
parquet; those operations happen once the manifest says the required snapshot
is published.
"""

import datetime
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from eth_defi.compat import native_datetime_utc_now
from tradingstrategy.vault_scan_manifest import VaultScanManifest, parse_manifest_timestamp
from tradingstrategy.vault_data_client import (
    VAULT_SCAN_MANIFEST_BUDGET,
    VaultDataClient,
    VaultDataset,
    VaultDataVersionMismatch,
    VaultManifestUnavailable,
)

from tradeexecutor.strategy.cycle import CycleDuration, snap_to_next_tick, snap_to_previous_tick


HYPERCORE_CHAIN_ID = "9999"
HYPERCORE_POLL_INTERVAL = datetime.timedelta(minutes=15)
HYPERCORE_READINESS_WINDOW = datetime.timedelta(hours=8)
logger = logging.getLogger(__name__)


def calculate_hypercore_slot_schedule(
    now: datetime.datetime,
    cycle_duration: CycleDuration,
    last_completed_slot: datetime.datetime | None = None,
) -> tuple[datetime.datetime, datetime.datetime]:
    """Return the current/future aligned slot and when to wake for it.

    A process starting during an open eight-hour window joins that slot. Once
    the window has expired it waits for the next calendar-aligned slot.

    :param now: Current naive UTC wall clock.
    :param cycle_duration: Calendar grid, normally two days for Hyper-AI.
    :param last_completed_slot: Persisted successful decision to avoid replay.
    :return: Logical slot and earliest wall-clock wake-up time.
    """

    slot = snap_to_previous_tick(now, cycle_duration)
    if last_completed_slot is not None and slot <= last_completed_slot:
        next_slot = snap_to_next_tick(last_completed_slot + datetime.timedelta(microseconds=1), cycle_duration)
        return next_slot, max(now, next_slot)
    if now < slot + HYPERCORE_READINESS_WINDOW:
        return slot, now
    next_slot = snap_to_next_tick(now, cycle_duration)
    return next_slot, next_slot


def hypercore_manifest_is_ready(manifest: VaultScanManifest, slot: datetime.datetime) -> bool:
    """Return whether HyperCore has published data for a logical slot.

    Both scan provenance and the cleaned-file maximum must reach the slot. A
    missing chain or null timestamp is simply not ready; schema validation is
    performed by the trading-strategy client before this function is called.
    """

    chain = manifest["chains"].get(HYPERCORE_CHAIN_ID)
    if chain is None:
        return False
    scan_ended_at = chain["last_successful_price_scan_ended_at"]
    last_candle_at = chain["last_candle_at"]
    if scan_ended_at is None or last_candle_at is None:
        return False
    return (
        parse_manifest_timestamp(scan_ended_at, "last_successful_price_scan_ended_at") >= slot
        and parse_manifest_timestamp(last_candle_at, "last_candle_at") >= slot
    )


def wait_for_hypercore_data_availability(
    fetch_manifest: Callable[[], VaultScanManifest],
    slot: datetime.datetime,
    *,
    now: Callable[[], datetime.datetime] = native_datetime_utc_now,
    poll_interval: datetime.timedelta = HYPERCORE_POLL_INTERVAL,
    readiness_window: datetime.timedelta = HYPERCORE_READINESS_WINDOW,
    shutdown_event: threading.Event | None = None,
    sleep: Callable[[float], None] = time.sleep,
    on_ready: Callable[[VaultScanManifest], VaultScanManifest | None] | None = None,
) -> tuple[VaultScanManifest, int]:
    """Poll the manifest until HyperCore is ready or the slot expires.

    :param fetch_manifest:
        Uncached JSON-only client operation. It must not download parquet.
    :param slot:
        Midnight-aligned logical decision timestamp.
    :param now:
        Injectable UTC clock for tests.
    :param poll_interval:
        Time between polls.
    :param readiness_window:
        Maximum time after ``slot`` in which a poll may begin.
    :param shutdown_event:
        Optional event that interrupts waiting during executor shutdown.
    :param sleep:
        Sleep function used when no shutdown event is supplied. The live
        executor keeps the default; deterministic tests can advance a fake
        clock instead of waiting in real time.
    :param on_ready:
        Optional snapshot verification after JSON readiness. Return the matched
        receipt, or ``None`` to resume polling following a publication race.
        A matching transfer may finish after the readiness window.
    :return:
        The ready manifest and number of polls made.
    :raises TimeoutError:
        If the manifest is not ready before the deadline.
    :raises RuntimeError:
        If shutdown interrupts the wait.
    """

    deadline = slot + readiness_window
    poll_count = 0
    next_poll_at = slot
    while True:
        current = now()
        if current < next_poll_at:
            remaining_to_slot = (next_poll_at - current).total_seconds()
            if shutdown_event is not None:
                if shutdown_event.wait(remaining_to_slot):
                    raise RuntimeError("HyperCore data availability wait interrupted by shutdown")
            else:
                sleep(remaining_to_slot)
            continue
        if current >= deadline:
            raise TimeoutError(f"HyperCore data was not available for slot {slot} before {deadline}")
        if shutdown_event is not None and shutdown_event.is_set():
            raise RuntimeError("HyperCore data availability wait interrupted by shutdown")
        try:
            manifest = fetch_manifest()
        except VaultManifestUnavailable as exc:
            logger.warning("HyperCore manifest unavailable for slot %s: %s", slot, exc)
            manifest = None
        poll_count += 1
        # A request that started before the deadline must not extend the
        # eight-hour window merely because the network responded late.
        if now() >= deadline:
            raise TimeoutError(f"HyperCore data was not available for slot {slot} before {deadline}")
        if manifest is not None and hypercore_manifest_is_ready(manifest, slot):
            if on_ready is not None:
                try:
                    manifest = on_ready(manifest)
                except VaultManifestUnavailable as exc:
                    logger.warning("HyperCore snapshot not yet usable for slot %s: %s", slot, exc)
                    manifest = None
            if manifest is not None:
                if shutdown_event is not None and shutdown_event.is_set():
                    raise RuntimeError("HyperCore snapshot interrupted by shutdown")
                return manifest, poll_count

        logger.info("Waiting for HyperCore observations across midnight %s; next poll follows the 15-minute grid", slot)

        # Keep polls aligned to the logical slot instead of adding an interval
        # after a slow HTTP response. A late response skips missed poll marks;
        # it never shifts the eight-hour deadline.
        next_poll_at += poll_interval
        current = now()
        if next_poll_at <= current:
            missed_intervals = (current - next_poll_at) // poll_interval + 1
            next_poll_at += missed_intervals * poll_interval
        remaining = min((next_poll_at - current).total_seconds(), (deadline - current).total_seconds())
        if remaining <= 0:
            raise TimeoutError(f"HyperCore data was not available for slot {slot} before {deadline}")
        if shutdown_event is not None:
            if shutdown_event.wait(remaining):
                raise RuntimeError("HyperCore data availability wait interrupted by shutdown")
        else:
            # The live executor supplies an Event. This fallback keeps the
            # helper usable by simple callers and unit tests.
            sleep(remaining)


def fetch_hypercore_decision_snapshot(
    client: VaultDataClient,
    slot: datetime.datetime,
    destination: Path,
    *,
    shutdown_event: threading.Event | None = None,
    now: Callable[[], datetime.datetime] = native_datetime_utc_now,
    sleep: Callable[[float], None] = time.sleep,
    request_budget: float = VAULT_SCAN_MANIFEST_BUDGET,
) -> tuple[VaultScanManifest, int]:
    """Wait for a receipt and download its price file for one live decision.

    Called by the executor before warm-up and subsequent cycles. Four-hour
    source observations need only cross midnight; no hourly completeness is
    inferred. Downloading starts only after readiness. A publication race gets
    one immediate JSON recheck, then returns to the anchored polling cadence.
    Mismatching response headers are rejected before streaming parquet bytes.

    :param client: Authenticated vault client using the uncached manifest path.
    :param slot: Logical UTC decision midnight; its deadline survives retries.
    :param destination: Private file consumed directly by the universe loader.
    :param shutdown_event: Interrupts polling on executor shutdown.
    :param now: UTC clock; injectable for deterministic scheduling tests.
    :param sleep: Waiting function for tests without a shutdown event.
    :param request_budget: Per-JSON-request seconds, capped by the slot deadline.
    :return: Matching manifest and number of JSON requests, including retries.
    :raises TimeoutError: Readiness or a version race outlasted the window.
    """
    deadline = slot + HYPERCORE_READINESS_WINDOW
    polls = 0

    def fetch() -> VaultScanManifest:
        """Cap every JSON request by the original slot's remaining window."""
        nonlocal polls
        remaining = (deadline - now()).total_seconds()
        if remaining <= 0:
            raise TimeoutError(f"HyperCore slot {slot} expired before receipt verification")
        polls += 1
        return client.fetch_vault_scan_manifest(request_budget=min(request_budget, remaining))

    def verify(manifest: VaultScanManifest) -> VaultScanManifest | None:
        """Verify one ready receipt, allowing one immediate publication-race retry."""
        for attempt in range(2):
            if now() >= deadline:
                raise TimeoutError(f"HyperCore slot {slot} expired before readiness")
            if not hypercore_manifest_is_ready(manifest, slot):
                return None
            try:
                client.download(VaultDataset.vault_prices, expected_etag=manifest["price_file"]["etag"], destination=destination)
                return manifest
            except VaultDataVersionMismatch:
                if attempt == 1:
                    raise VaultManifestUnavailable("Price publication changed twice; waiting for next manifest poll") from None
                manifest = fetch()
        raise AssertionError("Unreachable receipt verification branch")

    # Polls are JSON-only. Verification runs only on a ready receipt, and a
    # matching large transfer can finish after the JSON readiness deadline.
    manifest, _ = wait_for_hypercore_data_availability(
        fetch, slot, now=now, shutdown_event=shutdown_event, sleep=sleep, on_ready=verify,
    )
    return manifest, polls
