"""Claim untracked Hypercore vault dust back to Lagoon reserves."""

import datetime
import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from eth_defi.compat import native_datetime_utc_now
from eth_defi.hyperliquid.api import (
    HyperliquidSession,
    UserVaultEquity,
    fetch_perp_clearinghouse_state,
    fetch_spot_clearinghouse_state,
    fetch_user_vault_equities,
    fetch_user_vault_equity,
)
from eth_defi.hyperliquid.core_writer import (
    MINIMUM_VAULT_DEPOSIT,
    build_hypercore_send_asset_to_evm_call,
    build_hypercore_transfer_usd_class_call,
    build_hypercore_withdraw_from_vault_call,
    compute_spot_to_evm_withdrawal_amount,
)
from eth_defi.hyperliquid.vault import HyperliquidVault
from eth_defi.trace import assert_transaction_success_with_explanation
from hexbytes import HexBytes
from tabulate import tabulate
from web3 import Web3

from tradeexecutor.cli.bootstrap import backup_state
from tradeexecutor.ethereum.vault.hypercore_routing import (
    HYPERCORE_FOLLOW_UP_PHASE_TOLERANCE_RAW,
    HYPERCORE_MULTICALL_GAS,
    HYPERCORE_WITHDRAWAL_PHASE1_RETRY_ATTEMPTS,
    HYPERCORE_WITHDRAWAL_SAFETY_MARGIN_RAW,
    raw_to_usdc,
    usdc_to_raw,
)
from tradeexecutor.ethereum.vault.hypercore_transit_recovery import (
    BALANCE_TIMEOUT,
    BALANCE_TOLERANCE,
    CLEANUP_WAIT_RELATIVE_TOLERANCE,
    get_spot_usdc_balances,
)
from tradeexecutor.ethereum.vault.hyperliquid_cleanup import (
    HyperliquidCleanupContext,
    load_cleanup_context,
)
from tradeexecutor.state.balance_update import (
    BalanceUpdate,
    BalanceUpdateCause,
    BalanceUpdatePositionType,
)
from tradeexecutor.state.sync import BalanceEventRef
from tradeexecutor.utils.blockchain import get_block_timestamp

logger = logging.getLogger(__name__)


HYPERCORE_DUST_CLAIM_NOTE = "Hypercore dust claim"
DEFAULT_MAX_CLAIM_USDC = Decimal("25")
POLL_INTERVAL = 2.0


@dataclass(slots=True)
class HypercoreDustLiveSnapshot:
    """Live HyperCore balances relevant for dust claiming."""

    safe_address: str
    evm_usdc_balance: Decimal
    spot_total_usdc: Decimal
    spot_free_usdc: Decimal
    perp_withdrawable: Decimal
    perp_position_count: int
    vault_equities: list[UserVaultEquity]


@dataclass(slots=True)
class HypercoreDustCandidate:
    """One possible Hypercore dust claim candidate."""

    vault_address: str
    vault_name: str
    equity: Decimal
    max_withdrawable: Decimal | None
    locked_until: datetime.datetime | None
    safe_raw_claim_amount: int
    estimated_reserve_increase: Decimal
    status: str
    reason: str

    @property
    def is_claimable(self) -> bool:
        """Whether this row can be executed."""
        return self.status == "claimable"


@dataclass(slots=True)
class HypercoreDustClaimResult:
    """Result from one executed dust claim."""

    candidate: HypercoreDustCandidate
    phase1_tx_hash: str
    phase2_tx_hash: str
    phase3_tx_hash: str
    reserve_delta: Decimal
    balance_update_id: int


@dataclass(slots=True)
class HypercoreDustClaimReport:
    """Summary of a full dust claim run."""

    candidates: list[HypercoreDustCandidate]
    executed_claims: list[HypercoreDustClaimResult]
    skipped_candidates: list[HypercoreDustCandidate]
    state_saved: bool


def _normalise_vault_address(address: str) -> str:
    """Return a stable checksum address for comparing Hypercore vaults."""
    return Web3.to_checksum_address(address)


def _get_spot_usdc_balances(spot_state) -> tuple[Decimal, Decimal]:
    """Extract total and free spot USDC from HyperCore state."""
    return get_spot_usdc_balances(spot_state)


def _fetch_live_snapshot(
    context: HyperliquidCleanupContext,
) -> HypercoreDustLiveSnapshot:
    """Read live Safe balances from HyperEVM and Hyperliquid."""
    safe_address = context.lagoon_vault.safe_address
    spot_state = fetch_spot_clearinghouse_state(context.session, user=safe_address)
    perp_state = fetch_perp_clearinghouse_state(context.session, user=safe_address)
    spot_total_usdc, spot_free_usdc = _get_spot_usdc_balances(spot_state)
    return HypercoreDustLiveSnapshot(
        safe_address=safe_address,
        evm_usdc_balance=context.reserve_token.fetch_balance_of(safe_address),
        spot_total_usdc=spot_total_usdc,
        spot_free_usdc=spot_free_usdc,
        perp_withdrawable=perp_state.withdrawable,
        perp_position_count=len(perp_state.asset_positions),
        vault_equities=fetch_user_vault_equities(context.session, user=safe_address),
    )


def _get_state_hypercore_vault_addresses(
    context: HyperliquidCleanupContext,
) -> set[str]:
    """Get vault addresses that already have open or frozen state positions."""
    addresses = set()
    for position in context.state.portfolio.get_open_and_frozen_positions():
        if not position.pair.is_hyperliquid_vault():
            continue
        vault_address = position.pair.pool_address or position.pair.base.address
        addresses.add(_normalise_vault_address(vault_address))
    return addresses


def _classify_candidate(
    *,
    context: HyperliquidCleanupContext,
    equity: UserVaultEquity,
    snapshot: HypercoreDustLiveSnapshot,
    state_vault_addresses: set[str],
    max_claim_usdc: Decimal,
) -> HypercoreDustCandidate:
    """Turn one live vault equity row into an operator-facing candidate."""
    vault_address = _normalise_vault_address(equity.vault_address)
    if vault_address in state_vault_addresses:
        return HypercoreDustCandidate(
            vault_address=vault_address,
            vault_name="unknown",
            equity=equity.equity,
            max_withdrawable=None,
            locked_until=equity.locked_until,
            safe_raw_claim_amount=0,
            estimated_reserve_increase=Decimal(0),
            status="open_in_state",
            reason="Open or frozen Hypercore vault position already exists in state",
        )

    if snapshot.perp_position_count > 0:
        return HypercoreDustCandidate(
            vault_address=vault_address,
            vault_name="unknown",
            equity=equity.equity,
            max_withdrawable=None,
            locked_until=equity.locked_until,
            safe_raw_claim_amount=0,
            estimated_reserve_increase=Decimal(0),
            status="active_perp_positions",
            reason="Safe has active HyperCore perp positions; manual review required",
        )

    if (
        snapshot.perp_withdrawable > BALANCE_TOLERANCE
        or snapshot.spot_free_usdc > BALANCE_TOLERANCE
    ):
        return HypercoreDustCandidate(
            vault_address=vault_address,
            vault_name="unknown",
            equity=equity.equity,
            max_withdrawable=None,
            locked_until=equity.locked_until,
            safe_raw_claim_amount=0,
            estimated_reserve_increase=Decimal(0),
            status="preexisting_hypercore_balances",
            reason="Safe already has HyperCore spot or perp USDC; run clean-up or review manually first",
        )

    if equity.equity <= 0:
        return HypercoreDustCandidate(
            vault_address=vault_address,
            vault_name="unknown",
            equity=equity.equity,
            max_withdrawable=None,
            locked_until=equity.locked_until,
            safe_raw_claim_amount=0,
            estimated_reserve_increase=Decimal(0),
            status="manual_review",
            reason="Live vault equity is not positive",
        )

    if not context.state.portfolio.reserves:
        return HypercoreDustCandidate(
            vault_address=vault_address,
            vault_name="unknown",
            equity=equity.equity,
            max_withdrawable=None,
            locked_until=equity.locked_until,
            safe_raw_claim_amount=0,
            estimated_reserve_increase=Decimal(0),
            status="stale_state_risk",
            reason="State has no reserve position for recording a dust claim",
        )

    try:
        vault_info = HyperliquidVault(
            session=context.session,
            vault_address=vault_address,
        ).fetch_info(user=context.lagoon_vault.safe_address)
    except Exception as e:
        return HypercoreDustCandidate(
            vault_address=vault_address,
            vault_name="unknown",
            equity=equity.equity,
            max_withdrawable=None,
            locked_until=equity.locked_until,
            safe_raw_claim_amount=0,
            estimated_reserve_increase=Decimal(0),
            status="metadata_error",
            reason=f"Could not fetch Hypercore vault metadata: {e}",
        )

    vault_name = vault_info.name or "unknown"
    max_withdrawable = vault_info.max_withdrawable

    if not equity.is_lockup_expired:
        return HypercoreDustCandidate(
            vault_address=vault_address,
            vault_name=vault_name,
            equity=equity.equity,
            max_withdrawable=max_withdrawable,
            locked_until=equity.locked_until,
            safe_raw_claim_amount=0,
            estimated_reserve_increase=Decimal(0),
            status="locked",
            reason=f"Vault lock-up remains until {equity.locked_until}",
        )

    if max_withdrawable <= 0:
        return HypercoreDustCandidate(
            vault_address=vault_address,
            vault_name=vault_name,
            equity=equity.equity,
            max_withdrawable=max_withdrawable,
            locked_until=equity.locked_until,
            safe_raw_claim_amount=0,
            estimated_reserve_increase=Decimal(0),
            status="insufficient_liquidity",
            reason="Vault reports zero max_withdrawable liquidity",
        )

    requested_amount = min(equity.equity, max_withdrawable)
    safe_raw_claim_amount = (
        context.reserve_token.convert_to_raw(requested_amount)
        - HYPERCORE_WITHDRAWAL_SAFETY_MARGIN_RAW
    )
    if safe_raw_claim_amount < MINIMUM_VAULT_DEPOSIT:
        return HypercoreDustCandidate(
            vault_address=vault_address,
            vault_name=vault_name,
            equity=equity.equity,
            max_withdrawable=max_withdrawable,
            locked_until=equity.locked_until,
            safe_raw_claim_amount=max(safe_raw_claim_amount, 0),
            estimated_reserve_increase=Decimal(0),
            status="below_floor",
            reason=(
                f"Safe claim amount after withdrawal safety margin is below "
                f"Hyperliquid floor {raw_to_usdc(MINIMUM_VAULT_DEPOSIT)} USDC"
            ),
        )

    safe_claim_amount = raw_to_usdc(safe_raw_claim_amount)
    estimated_reserve_increase = compute_spot_to_evm_withdrawal_amount(
        spot_balance=safe_claim_amount,
        desired_amount=safe_claim_amount,
    )

    if safe_claim_amount > max_claim_usdc:
        return HypercoreDustCandidate(
            vault_address=vault_address,
            vault_name=vault_name,
            equity=equity.equity,
            max_withdrawable=max_withdrawable,
            locked_until=equity.locked_until,
            safe_raw_claim_amount=safe_raw_claim_amount,
            estimated_reserve_increase=estimated_reserve_increase,
            status="claim_above_operator_cap",
            reason=f"Safe claim amount {safe_claim_amount} USDC exceeds cap {max_claim_usdc} USDC",
        )

    return HypercoreDustCandidate(
        vault_address=vault_address,
        vault_name=vault_name,
        equity=equity.equity,
        max_withdrawable=max_withdrawable,
        locked_until=equity.locked_until,
        safe_raw_claim_amount=safe_raw_claim_amount,
        estimated_reserve_increase=estimated_reserve_increase,
        status="claimable",
        reason="Live vault balance has no open/frozen state position and passes dust-claim guards",
    )


def discover_hypercore_dust_candidates(
    context: HyperliquidCleanupContext,
    max_claim_usdc: Decimal = DEFAULT_MAX_CLAIM_USDC,
) -> list[HypercoreDustCandidate]:
    """Discover and classify live Hypercore vault dust candidates."""
    snapshot = _fetch_live_snapshot(context)
    state_vault_addresses = _get_state_hypercore_vault_addresses(context)
    return [
        _classify_candidate(
            context=context,
            equity=equity,
            snapshot=snapshot,
            state_vault_addresses=state_vault_addresses,
            max_claim_usdc=max_claim_usdc,
        )
        for equity in snapshot.vault_equities
    ]


def format_hypercore_dust_candidates(
    candidates: list[HypercoreDustCandidate],
) -> str:
    """Format dust candidates for terminal output."""
    rows = [
        {
            "Vault": candidate.vault_name,
            "Address": candidate.vault_address,
            "Equity": f"{candidate.equity:.6f}",
            "Max withdrawable": "-"
            if candidate.max_withdrawable is None
            else f"{candidate.max_withdrawable:.6f}",
            "Locked until": candidate.locked_until or "-",
            "Safe raw claim": candidate.safe_raw_claim_amount,
            "Est. reserve inc.": f"{candidate.estimated_reserve_increase:.6f}",
            "Status": candidate.status,
            "Reason": candidate.reason,
        }
        for candidate in candidates
    ]
    if not rows:
        return "No live Hypercore vault balances found."
    return tabulate(rows, headers="keys", tablefmt="simple")


def _get_wait_threshold(
    baseline_balance: Decimal,
    expected_increase: Decimal,
) -> tuple[Decimal, Decimal]:
    """Calculate an accepted balance threshold for async HyperCore settlement."""
    accepted_tolerance = max(
        BALANCE_TOLERANCE,
        expected_increase * CLEANUP_WAIT_RELATIVE_TOLERANCE,
    )
    return baseline_balance + expected_increase - accepted_tolerance, accepted_tolerance


def _wait_for_perp_withdrawable_balance(
    session: HyperliquidSession,
    user: str,
    baseline_balance: Decimal,
    expected_increase_raw: int,
    timeout: float = BALANCE_TIMEOUT,
    poll_interval: float = POLL_INTERVAL,
) -> Decimal:
    """Wait until HyperCore perp withdrawable USDC increases."""
    expected_increase = raw_to_usdc(expected_increase_raw)
    expected_balance, accepted_tolerance = _get_wait_threshold(
        baseline_balance,
        expected_increase,
    )
    deadline = time.time() + timeout
    while True:
        perp_state = fetch_perp_clearinghouse_state(session, user=user)
        if perp_state.withdrawable >= expected_balance:
            return perp_state.withdrawable
        if time.time() >= deadline:
            raise AssertionError(
                f"Timed out waiting for HyperCore perp withdrawable threshold {expected_balance} "
                f"for {user} (expected increase {expected_increase}, tolerance {accepted_tolerance}), "
                f"last observed balance was {perp_state.withdrawable}"
            )
        time.sleep(poll_interval)


def _wait_for_spot_free_balance(
    session: HyperliquidSession,
    user: str,
    baseline_balance: Decimal,
    expected_increase_raw: int,
    timeout: float = BALANCE_TIMEOUT,
    poll_interval: float = POLL_INTERVAL,
) -> Decimal:
    """Wait until HyperCore free spot USDC increases."""
    expected_increase = raw_to_usdc(expected_increase_raw)
    expected_balance, accepted_tolerance = _get_wait_threshold(
        baseline_balance,
        expected_increase,
    )
    deadline = time.time() + timeout
    while True:
        spot_state = fetch_spot_clearinghouse_state(session, user=user)
        _spot_total, spot_free = _get_spot_usdc_balances(spot_state)
        if spot_free >= expected_balance:
            return spot_free
        if time.time() >= deadline:
            raise AssertionError(
                f"Timed out waiting for HyperCore spot free USDC threshold {expected_balance} "
                f"for {user} (expected increase {expected_increase}, tolerance {accepted_tolerance}), "
                f"last observed balance was {spot_free}"
            )
        time.sleep(poll_interval)


def _wait_for_evm_usdc_balance(
    context: HyperliquidCleanupContext,
    baseline_balance: Decimal,
    expected_increase: Decimal,
    timeout: float = BALANCE_TIMEOUT,
    poll_interval: float = POLL_INTERVAL,
) -> Decimal:
    """Wait until HyperEVM Safe USDC increases and return the actual delta."""
    expected_balance, accepted_tolerance = _get_wait_threshold(
        baseline_balance,
        expected_increase,
    )
    deadline = time.time() + timeout
    safe_address = context.lagoon_vault.safe_address
    while True:
        balance = context.reserve_token.fetch_balance_of(safe_address)
        if balance >= expected_balance:
            return balance - baseline_balance
        if time.time() >= deadline:
            raise AssertionError(
                f"Timed out waiting for HyperEVM USDC threshold {expected_balance} "
                f"for {safe_address} (expected increase {expected_increase}, tolerance {accepted_tolerance}), "
                f"last observed balance was {balance}"
            )
        time.sleep(poll_interval)


def _broadcast_bound_call(
    context: HyperliquidCleanupContext,
    bound_func,
) -> tuple[str, dict]:
    """Broadcast a TradingStrategyModule call and return tx hash and receipt."""
    tx_hash = context.hot_wallet.transact_and_broadcast_with_contract(
        bound_func,
        gas_limit=HYPERCORE_MULTICALL_GAS,
    )
    receipt = assert_transaction_success_with_explanation(
        context.web3,
        tx_hash,
        func=bound_func,
    )
    if isinstance(tx_hash, HexBytes):
        tx_hash = tx_hash.hex()
    return tx_hash, receipt


def _get_phase1_noop_retry_raw(
    current_vault_equity: Decimal,
    previous_raw: int,
) -> int | None:
    """Return a smaller phase-1 retry amount after a suspected silent no-op."""
    retry_raw = (
        usdc_to_raw(current_vault_equity) - HYPERCORE_WITHDRAWAL_SAFETY_MARGIN_RAW
    )
    if retry_raw < MINIMUM_VAULT_DEPOSIT:
        return None
    if retry_raw >= previous_raw:
        return None
    return retry_raw


def _get_block_timestamp(
    context: HyperliquidCleanupContext, receipt: dict
) -> datetime.datetime:
    """Get a naive UTC block timestamp for a receipt."""
    block_number = receipt.get("blockNumber")
    if block_number is None:
        return native_datetime_utc_now()
    try:
        return get_block_timestamp(context.web3, block_number)
    except Exception:
        logger.warning("Could not fetch block timestamp for block %s", block_number)
        return native_datetime_utc_now()


def _record_reserve_balance_update(
    context: HyperliquidCleanupContext,
    *,
    reserve_delta: Decimal,
    tx_hash: str,
    receipt: dict,
) -> BalanceUpdate:
    """Record a reserve correction event for claimed Hypercore dust."""
    reserve_position = context.state.portfolio.get_default_reserve_position()
    old_balance = reserve_position.quantity
    event_id = context.state.portfolio.next_balance_update_id
    context.state.portfolio.next_balance_update_id += 1
    block_number = receipt.get("blockNumber")
    evt = BalanceUpdate(
        balance_update_id=event_id,
        position_type=BalanceUpdatePositionType.reserve,
        cause=BalanceUpdateCause.correction,
        asset=reserve_position.asset,
        block_mined_at=_get_block_timestamp(context, receipt),
        strategy_cycle_included_at=native_datetime_utc_now(),
        chain_id=reserve_position.asset.chain_id,
        quantity=reserve_delta,
        old_balance=old_balance,
        usd_value=float(reserve_delta),
        owner_address=context.lagoon_vault.safe_address,
        tx_hash=tx_hash,
        log_index=None,
        position_id=None,
        notes=HYPERCORE_DUST_CLAIM_NOTE,
        block_number=block_number,
    )
    reserve_position.quantity += reserve_delta
    reserve_position.balance_updates[evt.balance_update_id] = evt
    context.state.sync.accounting.balance_update_refs.append(
        BalanceEventRef.from_balance_update_event(evt)
    )
    context.state.sync.accounting.last_updated_at = native_datetime_utc_now()
    context.state.sync.accounting.last_block_scanned = block_number
    return evt


def execute_hypercore_dust_claim(
    context: HyperliquidCleanupContext,
    candidate: HypercoreDustCandidate,
) -> HypercoreDustClaimResult:
    """Execute all Hypercore claim phases for a single candidate."""
    safe_address = context.lagoon_vault.safe_address
    baseline_snapshot = _fetch_live_snapshot(context)
    baseline_perp = baseline_snapshot.perp_withdrawable
    baseline_spot = baseline_snapshot.spot_free_usdc
    baseline_evm = baseline_snapshot.evm_usdc_balance
    expected_raw = candidate.safe_raw_claim_amount

    phase1_fn = build_hypercore_withdraw_from_vault_call(
        context.lagoon_vault,
        vault_address=candidate.vault_address,
        hypercore_usdc_amount=expected_raw,
    )
    phase1_tx_hash, _phase1_receipt = _broadcast_bound_call(context, phase1_fn)
    try:
        _wait_for_perp_withdrawable_balance(
            context.session,
            safe_address,
            baseline_balance=baseline_perp,
            expected_increase_raw=expected_raw,
        )
    except AssertionError:
        fresh_equity = fetch_user_vault_equity(
            context.session,
            user=safe_address,
            vault_address=candidate.vault_address,
            bypass_cache=True,
        )
        retry_raw = None
        if fresh_equity is not None:
            retry_raw = _get_phase1_noop_retry_raw(fresh_equity.equity, expected_raw)

        if retry_raw is None or HYPERCORE_WITHDRAWAL_PHASE1_RETRY_ATTEMPTS <= 0:
            raise

        logger.warning(
            "Retrying Hypercore dust claim phase 1 for vault %s with %s raw USDC",
            candidate.vault_address,
            retry_raw,
        )
        expected_raw = retry_raw
        phase1_retry_fn = build_hypercore_withdraw_from_vault_call(
            context.lagoon_vault,
            vault_address=candidate.vault_address,
            hypercore_usdc_amount=expected_raw,
        )
        phase1_tx_hash, _phase1_receipt = _broadcast_bound_call(
            context, phase1_retry_fn
        )
        _wait_for_perp_withdrawable_balance(
            context.session,
            safe_address,
            baseline_balance=baseline_perp,
            expected_increase_raw=expected_raw,
        )

    phase2_fn = build_hypercore_transfer_usd_class_call(
        context.lagoon_vault,
        hypercore_usdc_amount=expected_raw,
        to_perp=False,
    )
    phase2_tx_hash, _phase2_receipt = _broadcast_bound_call(context, phase2_fn)
    spot_balance = _wait_for_spot_free_balance(
        context.session,
        safe_address,
        baseline_balance=baseline_spot,
        expected_increase_raw=expected_raw,
    )

    desired_amount = raw_to_usdc(expected_raw)
    phase3_amount = compute_spot_to_evm_withdrawal_amount(
        spot_balance=spot_balance,
        desired_amount=desired_amount,
    )
    phase3_raw = context.reserve_token.convert_to_raw(phase3_amount)
    if phase3_raw <= HYPERCORE_FOLLOW_UP_PHASE_TOLERANCE_RAW:
        raise RuntimeError(
            f"Hypercore dust claim cannot bridge any meaningful USDC for vault {candidate.vault_address}: "
            f"spot free balance {spot_balance} USDC, desired amount {desired_amount} USDC"
        )

    phase3_fn = build_hypercore_send_asset_to_evm_call(
        context.lagoon_vault,
        evm_usdc_amount=phase3_raw,
    )
    phase3_tx_hash, phase3_receipt = _broadcast_bound_call(context, phase3_fn)
    reserve_delta = _wait_for_evm_usdc_balance(
        context,
        baseline_balance=baseline_evm,
        expected_increase=phase3_amount,
    )
    evt = _record_reserve_balance_update(
        context,
        reserve_delta=reserve_delta,
        tx_hash=phase3_tx_hash,
        receipt=phase3_receipt,
    )
    context.store.sync(context.state)
    return HypercoreDustClaimResult(
        candidate=candidate,
        phase1_tx_hash=phase1_tx_hash,
        phase2_tx_hash=phase2_tx_hash,
        phase3_tx_hash=phase3_tx_hash,
        reserve_delta=reserve_delta,
        balance_update_id=evt.balance_update_id,
    )


def _confirm_claims(auto_approve: bool) -> None:
    """Ask for operator confirmation before broadcasting transactions."""
    if auto_approve:
        return
    confirmation = input("Execute Hypercore vault dust claims [y/n] ").strip().lower()
    if confirmation != "y":
        raise RuntimeError("Operator aborted Hypercore dust claim")


def _reload_context_state(context: HyperliquidCleanupContext) -> None:
    """Reload state before a candidate execution."""
    context.state = context.store.load()


def run_hypercore_dust_claim(
    *,
    state_file: Path,
    strategy_file: Path,
    private_key: str,
    json_rpc_hyperliquid: str,
    vault_address: str,
    vault_adapter_address: str,
    trading_strategy_api_key: str = "",
    network: str = "mainnet",
    auto_approve: bool = False,
    max_claim_usdc: Decimal = DEFAULT_MAX_CLAIM_USDC,
    cache_path: Path | None = None,
    unit_testing: bool = False,
    log_level: str = "info",
) -> HypercoreDustClaimReport:
    """Run the Hypercore vault dust claim flow."""
    context = load_cleanup_context(
        state_file=state_file,
        strategy_file=strategy_file,
        private_key=private_key,
        json_rpc_hyperliquid=json_rpc_hyperliquid,
        vault_address=vault_address,
        vault_adapter_address=vault_adapter_address,
        trading_strategy_api_key=trading_strategy_api_key,
        network=network,
        cache_path=cache_path,
        unit_testing=unit_testing,
        log_level=log_level,
    )
    candidates = discover_hypercore_dust_candidates(
        context, max_claim_usdc=max_claim_usdc
    )
    print("\nHypercore vault dust candidates")
    print(format_hypercore_dust_candidates(candidates))

    claimable = [candidate for candidate in candidates if candidate.is_claimable]
    above_cap = [
        candidate
        for candidate in candidates
        if candidate.status == "claim_above_operator_cap"
    ]
    active_perp = [
        candidate
        for candidate in candidates
        if candidate.status == "active_perp_positions"
    ]
    if active_perp:
        raise RuntimeError(
            "Refusing Hypercore dust claim because the Safe has active HyperCore perp positions. "
            "Manual review is required before vault dust can be interpreted safely."
        )

    if auto_approve and above_cap:
        raise RuntimeError(
            f"--auto-approve refused because {len(above_cap)} Hypercore dust candidate(s) "
            f"exceed --max-claim-usdc={max_claim_usdc}"
        )

    if not claimable:
        return HypercoreDustClaimReport(
            candidates=candidates,
            executed_claims=[],
            skipped_candidates=[
                candidate for candidate in candidates if not candidate.is_claimable
            ],
            state_saved=False,
        )

    _confirm_claims(auto_approve=auto_approve or unit_testing)

    backed_up = False
    executed_claims: list[HypercoreDustClaimResult] = []
    skipped_candidates: list[HypercoreDustCandidate] = []

    for candidate in claimable:
        _reload_context_state(context)
        refreshed_candidates = discover_hypercore_dust_candidates(
            context,
            max_claim_usdc=max_claim_usdc,
        )
        refreshed = next(
            (
                refreshed_candidate
                for refreshed_candidate in refreshed_candidates
                if refreshed_candidate.vault_address == candidate.vault_address
            ),
            None,
        )
        if refreshed is None or not refreshed.is_claimable:
            skipped_candidates.append(refreshed or candidate)
            continue

        if not backed_up:
            context.store, context.state = backup_state(
                context.state_file,
                backup_suffix="claim-hypercore-vault-dust-backup",
                unit_testing=unit_testing,
            )
            backed_up = True

        executed_claims.append(execute_hypercore_dust_claim(context, refreshed))

    skipped_candidates.extend(
        candidate for candidate in candidates if not candidate.is_claimable
    )
    return HypercoreDustClaimReport(
        candidates=candidates,
        executed_claims=executed_claims,
        skipped_candidates=skipped_candidates,
        state_saved=bool(executed_claims),
    )
