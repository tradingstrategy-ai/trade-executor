"""Readiness helpers for the HyperCore manifest-triggered live cycle.

The executor polls the small vault scan manifest after a calendar-aligned
midnight slot. Only a ready decision receipt triggers a decision price download; this
module never constructs a universe or calculates indicators. Start-up fetches
one current verified snapshot; decision probes return to the scheduler between
polls, leaving position valuation free to run.
"""

import datetime
import logging
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
from tradeexecutor.state.state import State


HYPERCORE_CHAIN_ID = "9999"
HYPERCORE_POLL_INTERVAL = datetime.timedelta(minutes=15)
HYPERCORE_READINESS_WINDOW = datetime.timedelta(hours=8)
logger = logging.getLogger(__name__)


def fetch_current_hypercore_snapshot(
    client: VaultDataClient,
    destination: Path,
    *,
    now: Callable[[], datetime.datetime] = native_datetime_utc_now,
) -> tuple[VaultScanManifest, datetime.datetime]:
    """Download a verified current price file for live start-up valuation.

    Called once before the live universe is built, even when the next decision
    slot is in the future. The receipt need not be ready for that slot: this
    snapshot supports start-up checks, position valuation and risk triggers,
    but never an ungated rebalance. Decision polling downloads a new verified
    snapshot before calling the strategy, unless this one is already slot-ready.

    :param client: Authenticated vault dataset client.
    :param destination: Private price file passed to start-up universe loading.
    :param now: UTC clock used by the caller to judge the receipt deadline.
    :return: Receipt identifying the downloaded price file and its reception time.
    :raises VaultDataVersionMismatch: The price file changed during both attempts.
    """
    for attempt in range(2):
        manifest = client.fetch_vault_scan_manifest()
        received_at = now()
        try:
            client.download(VaultDataset.vault_prices, expected_etag=manifest["price_file"]["etag"], destination=destination)
            return manifest, received_at
        except VaultDataVersionMismatch:
            logger.warning("HyperCore start-up price version changed (attempt %d)", attempt + 1)
    raise VaultDataVersionMismatch("HyperCore start-up price version changed twice")


def calculate_hypercore_slot_schedule(
    now: datetime.datetime,
    cycle_duration: CycleDuration,
    state: State,
) -> datetime.datetime:
    """Choose the live loop's restart slot without replaying persisted trades.

    Resume an unexpired pending decision, or skip an expired one automatically
    if it created no trades. Fresh starts may join the current open window.
    The caller saves the returned slot before polling; this helper does not
    modify state or mark a missed decision as completed.

    :param now: Current naive UTC wall clock.
    :param cycle_duration: Calendar grid, normally two days for Hyper-AI.
    :param state: Persisted pending/completed decisions and trade history.
    :return: Logical UTC decision slot.
    :raises RuntimeError: A pending decision created trades and needs reconciliation.
    """

    pending_slot = state.pending_data_availability_slot
    if pending_slot is not None:
        if any(trade.opened_at == pending_slot for trade in state.portfolio.get_all_trades()):
            raise RuntimeError(
                f"HyperCore slot {pending_slot} already created trades; reconcile the interrupted cycle before resuming"
            )
        if now < pending_slot + HYPERCORE_READINESS_WINDOW:
            return pending_slot
        next_slot = snap_to_next_tick(now + datetime.timedelta(microseconds=1), cycle_duration)
        logger.warning("Skipping expired unexecuted HyperCore slot %s; next slot %s", pending_slot, next_slot)
        return next_slot

    last_completed_slot = state.last_cycle_at
    slot = snap_to_previous_tick(now, cycle_duration)
    if last_completed_slot is not None and slot <= last_completed_slot:
        next_slot = snap_to_next_tick(last_completed_slot + datetime.timedelta(microseconds=1), cycle_duration)
        return next_slot
    if now < slot + HYPERCORE_READINESS_WINDOW:
        return slot
    next_slot = snap_to_next_tick(now, cycle_duration)
    return next_slot


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


def next_hypercore_poll(slot: datetime.datetime, now: datetime.datetime) -> datetime.datetime:
    """Return the next quarter-hour poll, capped at the original deadline.

    The live scheduler uses this to release its worker between probes.

    :param slot: Logical decision midnight.
    :param now: Current UTC time after a probe.
    :return: Next polling time, or the deadline where polling will raise.
    """
    intervals = max(0, (now - slot) // HYPERCORE_POLL_INTERVAL + 1)
    return min(slot + intervals * HYPERCORE_POLL_INTERVAL, slot + HYPERCORE_READINESS_WINDOW)


def poll_hypercore_decision_snapshot(
    client: VaultDataClient,
    slot: datetime.datetime,
    destination: Path,
    *,
    now: Callable[[], datetime.datetime] = native_datetime_utc_now,
    request_budget: float = VAULT_SCAN_MANIFEST_BUDGET,
) -> tuple[VaultScanManifest | None, int]:
    """Probe readiness once without sleeping or constructing a universe.

    The live loop schedules another invocation if this returns no receipt,
    leaving its single worker free for valuation between polls. Only a ready
    receipt triggers a verified private download. One ETag race gets an
    immediate JSON recheck; all other retryable failures return to the grid.

    :param client: Authenticated vault dataset client.
    :param slot: Logical UTC decision midnight.
    :param destination: Private price file passed directly to universe loading.
    :param now: UTC clock, injectable for deterministic tests.
    :param request_budget: JSON request budget, capped by the original deadline.
    :return: Verified receipt or None, and actual JSON request count.
    :raises TimeoutError: The original eight-hour readiness window expired.
    """
    deadline = slot + HYPERCORE_READINESS_WINDOW
    polls = 0
    for attempt in range(2):
        current = now()
        if current >= deadline:
            raise TimeoutError(f"HyperCore data was not available for slot {slot} before {deadline}")
        if current < slot:
            return None, polls
        try:
            polls += 1
            manifest = client.fetch_vault_scan_manifest(
                request_budget=min(request_budget, (deadline - current).total_seconds()),
            )
        except VaultManifestUnavailable as exc:
            logger.warning("HyperCore manifest unavailable for slot %s: %s", slot, exc)
            return None, polls
        if now() >= deadline:
            raise TimeoutError(f"HyperCore data was not available for slot {slot} before {deadline}")
        if not hypercore_manifest_is_ready(manifest, slot):
            return None, polls
        try:
            client.download(VaultDataset.vault_prices, expected_etag=manifest["price_file"]["etag"], destination=destination)
            # Readiness, not the end of the large transfer, must precede deadline.
            return manifest, polls
        except VaultDataVersionMismatch:
            logger.warning("HyperCore price version changed for slot %s (attempt %d)", slot, attempt + 1)
    return None, polls
