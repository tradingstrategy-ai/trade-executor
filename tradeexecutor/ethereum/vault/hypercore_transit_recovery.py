"""Recover stranded HyperCore spot/perp USDC back to HyperEVM reserves.

This module only handles Safe-level HyperCore transit balances. It never
withdraws live vault equity. Untracked vault dust remains the responsibility of
``hypercore_dust_claim``.
"""

import logging
import time
from dataclasses import dataclass
from decimal import Decimal

from eth_defi.hotwallet import HotWallet
from eth_defi.hyperliquid.api import (
    HyperliquidSession,
    fetch_perp_clearinghouse_state,
    fetch_spot_clearinghouse_state,
)
from eth_defi.hyperliquid.constants import HYPERCORE_BRIDGE_FEE_MARGIN
from eth_defi.hyperliquid.core_writer import (
    build_hypercore_send_asset_to_evm_call,
    build_hypercore_transfer_usd_class_call,
    compute_spot_to_evm_withdrawal_amount,
)
from eth_defi.token import TokenDetails
from eth_defi.trace import assert_transaction_success_with_explanation
from web3 import Web3


logger = logging.getLogger(__name__)

BALANCE_TOLERANCE = Decimal("0.02")
CLEANUP_WAIT_RELATIVE_TOLERANCE = Decimal("0.001")
HYPERCORE_TRANSIT_RECOVERY_DUST_USDC = Decimal("0.50")
POLL_INTERVAL = 2.0
BALANCE_TIMEOUT = 60.0
SAFE_GAS_LIMIT = 650_000


@dataclass(slots=True)
class HypercoreTransitBalanceSnapshot:
    """Live Safe balances observed from HyperEVM and HyperCore."""

    safe_address: str
    evm_usdc_balance: Decimal
    spot_total_usdc: Decimal
    spot_free_usdc: Decimal
    perp_withdrawable: Decimal
    perp_account_value: Decimal
    perp_position_count: int


@dataclass(slots=True)
class HypercoreTransitRecoveryAction:
    """One Safe-side HyperCore transit-balance recovery action."""

    action_kind: str
    amount: Decimal
    reason: str


def get_spot_usdc_balances(spot_state) -> tuple[Decimal, Decimal]:
    """Extract total and free spot USDC from HyperCore state."""
    for balance in spot_state.balances:
        if balance.coin == "USDC":
            return balance.total, balance.total - balance.hold
    return Decimal(0), Decimal(0)


def fetch_hypercore_transit_balances(
    *,
    session: HyperliquidSession,
    safe_address: str,
    reserve_token: TokenDetails,
) -> HypercoreTransitBalanceSnapshot:
    """Read live Safe transit balances from HyperEVM and HyperCore."""
    spot_state = fetch_spot_clearinghouse_state(session, user=safe_address)
    perp_state = fetch_perp_clearinghouse_state(session, user=safe_address)
    spot_total_usdc, spot_free_usdc = get_spot_usdc_balances(spot_state)
    return HypercoreTransitBalanceSnapshot(
        safe_address=safe_address,
        evm_usdc_balance=reserve_token.fetch_balance_of(safe_address),
        spot_total_usdc=spot_total_usdc,
        spot_free_usdc=spot_free_usdc,
        perp_withdrawable=perp_state.withdrawable,
        perp_account_value=perp_state.margin_summary.account_value,
        perp_position_count=len(perp_state.asset_positions),
    )


def _positive_recovery_amount(amount: Decimal) -> Decimal:
    """Clamp negative recovery amounts to zero."""
    if amount <= BALANCE_TOLERANCE:
        return Decimal(0)
    return amount


def plan_hypercore_transit_recovery_actions(
    snapshot: HypercoreTransitBalanceSnapshot,
    *,
    leave_dust: Decimal = HYPERCORE_TRANSIT_RECOVERY_DUST_USDC,
    verified_recovery_amount: Decimal | None = None,
) -> list[HypercoreTransitRecoveryAction]:
    """Plan recovery actions for Safe-level HyperCore spot/perp USDC.

    By default, leaves ``leave_dust`` in the source perp balance, then returns
    the exact perp amount which just arrived in spot to HyperEVM. This is the
    default recovery path for every eligible HyperCore vault strategy, not an
    opt-in incident tool. It preserves any pre-existing spot balance instead
    of accidentally leaving part of a newly stranded deposit behind as spot
    dust. The executor retains the protocol's 0.01 USDC bridge-fee margin only
    when there is not enough pre-existing spot headroom. If the original spot
    balance also exceeds ``leave_dust``, it is planned as a separate later
    spot-to-EVM action.

    ``verified_recovery_amount`` is the deliberately narrower incident path.
    It returns precisely an amount which an operator has independently proved
    is stranded, instead of treating all non-dust HyperCore cash as stranded.
    This exists for HyperAI trade #1486 (Citadel, 2026-07-30; PR #1593): the
    state snapshot predates the failed CoreWriter request, but the live account
    showed 48.884068 USDC newly present in perp.  In that case, move the exact
    amount from perp to spot and then from spot to HyperEVM, preserving the
    pre-incident spot/perp balances.  The caller must never infer this amount
    from a receipt alone: it is safe only after confirming that the vault
    equity did not receive the deposit.
    """
    if snapshot.perp_position_count > 0:
        raise RuntimeError(
            f"Refusing HyperCore transit recovery because Safe {snapshot.safe_address} "
            f"has {snapshot.perp_position_count} active HyperCore perp position(s). "
            "Manual review is required before stranded USDC can be interpreted safely."
        )

    if verified_recovery_amount is not None:
        if verified_recovery_amount <= BALANCE_TOLERANCE:
            raise ValueError(
                "Verified HyperCore transit recovery amount must exceed the "
                f"{BALANCE_TOLERANCE} USDC balance tolerance, got {verified_recovery_amount}"
            )

        # Do not combine partial spot and perp balances here.  An exact amount
        # must have one unambiguous source, otherwise a transient in-flight
        # transfer could be mistaken for the #1486-style stranded deposit.
        if snapshot.perp_withdrawable + BALANCE_TOLERANCE >= verified_recovery_amount:
            return [
                HypercoreTransitRecoveryAction(
                    action_kind="perp_to_spot",
                    amount=verified_recovery_amount,
                    reason=(
                        "Return operator-verified stranded USDC from HyperCore perp to spot "
                        "without consuming pre-existing perp dust"
                    ),
                ),
                HypercoreTransitRecoveryAction(
                    action_kind="spot_to_evm",
                    amount=verified_recovery_amount,
                    reason=(
                        "Return the same operator-verified stranded USDC from HyperCore spot "
                        "to the Safe on HyperEVM"
                    ),
                ),
            ]

        if snapshot.spot_free_usdc + BALANCE_TOLERANCE >= verified_recovery_amount:
            return [
                HypercoreTransitRecoveryAction(
                    action_kind="spot_to_evm",
                    amount=verified_recovery_amount,
                    reason=(
                        "Return operator-verified stranded USDC from HyperCore spot "
                        "to the Safe on HyperEVM"
                    ),
                )
            ]

        raise RuntimeError(
            f"Safe {snapshot.safe_address} does not hold the operator-verified "
            f"{verified_recovery_amount} USDC recovery amount wholly in free spot or "
            f"withdrawable perp (spot {snapshot.spot_free_usdc}, "
            f"perp {snapshot.perp_withdrawable}). Manual review is required."
        )

    actions: list[HypercoreTransitRecoveryAction] = []

    perp_recovery_amount = _positive_recovery_amount(
        snapshot.perp_withdrawable - leave_dust
    )
    if perp_recovery_amount > 0:
        actions.append(
            HypercoreTransitRecoveryAction(
                action_kind="perp_to_spot",
                amount=perp_recovery_amount,
                reason=(
                    f"Recover HyperCore perp USDC back to HyperCore spot, "
                    f"leaving {leave_dust} USDC perp dust"
                ),
            )
        )

    if perp_recovery_amount > 0:
        # HyperAI #1486 proved that a successful CoreWriter receipt can leave
        # the exact deposit amount in perp.  Bridge that same amount after the
        # perp->spot leg.  Calculating ``projected_spot - leave_dust`` here
        # would preserve a synthetic 0.50 USDC spot dust balance and return
        # less than the stranded amount whenever spot had pre-existing dust.
        # ``execute_spot_to_evm`` is still allowed to retain the protocol's
        # 0.01 USDC fee margin when the pre-existing spot headroom is smaller.
        actions.append(
            HypercoreTransitRecoveryAction(
                action_kind="spot_to_evm",
                amount=perp_recovery_amount,
                reason=(
                    "Return the exact USDC just recovered from HyperCore perp to the Safe "
                    "on HyperEVM, preserving the pre-existing spot balance"
                ),
            )
        )

    spot_recovery_amount = _positive_recovery_amount(
        snapshot.spot_free_usdc - leave_dust
    )
    if spot_recovery_amount > 0:
        actions.append(
            HypercoreTransitRecoveryAction(
                action_kind="spot_to_evm",
                amount=spot_recovery_amount,
                reason=(
                    f"Recover HyperCore spot USDC back to the Safe on HyperEVM, "
                    f"leaving {leave_dust} USDC spot dust"
                ),
            )
        )

    return actions


def get_wait_threshold(
    baseline_balance: Decimal,
    expected_increase: Decimal,
) -> tuple[Decimal, Decimal]:
    """Calculate the minimum balance increase that recovery waits should accept."""
    accepted_tolerance = max(
        BALANCE_TOLERANCE,
        expected_increase * CLEANUP_WAIT_RELATIVE_TOLERANCE,
    )
    return baseline_balance + expected_increase - accepted_tolerance, accepted_tolerance


def wait_for_spot_free_balance(
    session: HyperliquidSession,
    user: str,
    baseline_balance: Decimal,
    expected_increase: Decimal,
    timeout: float = BALANCE_TIMEOUT,
    poll_interval: float = POLL_INTERVAL,
) -> Decimal:
    """Wait until free spot USDC reaches the expected threshold."""
    expected_balance, accepted_tolerance = get_wait_threshold(
        baseline_balance=baseline_balance,
        expected_increase=expected_increase,
    )
    deadline = time.time() + timeout
    while True:
        spot_state = fetch_spot_clearinghouse_state(session, user=user)
        _spot_total, spot_free = get_spot_usdc_balances(spot_state)
        if spot_free >= expected_balance:
            return spot_free
        if time.time() >= deadline:
            raise AssertionError(
                f"Timed out waiting for HyperCore free spot USDC threshold {expected_balance} "
                f"for {user} (expected increase {expected_increase}, tolerance {accepted_tolerance}), "
                f"last observed balance was {spot_free}"
            )
        time.sleep(poll_interval)


def wait_for_evm_usdc_balance(
    token: TokenDetails,
    address: str,
    baseline_balance: Decimal,
    expected_increase: Decimal,
    timeout: float = BALANCE_TIMEOUT,
    poll_interval: float = POLL_INTERVAL,
) -> Decimal:
    """Wait until EVM USDC reaches the expected threshold."""
    expected_balance, accepted_tolerance = get_wait_threshold(
        baseline_balance=baseline_balance,
        expected_increase=expected_increase,
    )
    deadline = time.time() + timeout
    while True:
        balance = token.fetch_balance_of(address)
        if balance >= expected_balance:
            return balance
        if time.time() >= deadline:
            raise AssertionError(
                f"Timed out waiting for HyperEVM USDC threshold {expected_balance} "
                f"for {address} (expected increase {expected_increase}, tolerance {accepted_tolerance}), "
                f"last observed balance was {balance}"
            )
        time.sleep(poll_interval)


def broadcast_bound_call(
    web3: Web3,
    hot_wallet: HotWallet,
    bound_func,
    gas_limit: int = SAFE_GAS_LIMIT,
) -> str:
    """Broadcast a single Safe/module transaction and assert success."""
    tx_hash = hot_wallet.transact_and_broadcast_with_contract(
        bound_func, gas_limit=gas_limit
    )
    assert_transaction_success_with_explanation(web3, tx_hash)
    return tx_hash.hex()


def execute_perp_to_spot(
    *,
    web3: Web3,
    hot_wallet: HotWallet,
    lagoon_vault,
    session: HyperliquidSession,
    reserve_token: TokenDetails,
    amount: Decimal,
) -> str:
    """Recover stranded USDC from HyperCore perp back to spot."""
    safe_address = lagoon_vault.safe_address
    live_snapshot = fetch_hypercore_transit_balances(
        session=session,
        safe_address=safe_address,
        reserve_token=reserve_token,
    )
    assert live_snapshot.perp_withdrawable + BALANCE_TOLERANCE >= amount, (
        f"Before transferUsdClass(perp->spot), Safe {safe_address} perp withdrawable balance is "
        f"{live_snapshot.perp_withdrawable}, expected at least {amount}"
    )

    baseline_spot_free = live_snapshot.spot_free_usdc
    fn = build_hypercore_transfer_usd_class_call(
        lagoon_vault,
        hypercore_usdc_amount=reserve_token.convert_to_raw(amount),
        to_perp=False,
    )
    tx_hash = broadcast_bound_call(web3, hot_wallet, fn)
    wait_for_spot_free_balance(
        session,
        safe_address,
        baseline_balance=baseline_spot_free,
        expected_increase=amount,
    )
    return tx_hash


def execute_spot_to_evm(
    *,
    web3: Web3,
    hot_wallet: HotWallet,
    lagoon_vault,
    session: HyperliquidSession,
    reserve_token: TokenDetails,
    amount: Decimal,
) -> str:
    """Recover stranded USDC from HyperCore spot back to HyperEVM."""
    safe_address = lagoon_vault.safe_address
    live_snapshot = fetch_hypercore_transit_balances(
        session=session,
        safe_address=safe_address,
        reserve_token=reserve_token,
    )
    assert live_snapshot.spot_free_usdc + BALANCE_TOLERANCE >= amount, (
        f"Before spotSend(spot->EVM), Safe {safe_address} free spot USDC balance is "
        f"{live_snapshot.spot_free_usdc}, expected at least {amount}"
    )

    withdraw_amount = compute_spot_to_evm_withdrawal_amount(
        spot_balance=live_snapshot.spot_free_usdc,
        desired_amount=amount,
    )
    if withdraw_amount < BALANCE_TOLERANCE:
        raise RuntimeError(
            f"Spot balance {live_snapshot.spot_free_usdc} for Safe {safe_address} "
            f"is too small to cover the HyperCore bridge fee margin "
            f"({HYPERCORE_BRIDGE_FEE_MARGIN} USDC); cannot withdraw to EVM"
        )

    baseline_evm_balance = live_snapshot.evm_usdc_balance
    fn = build_hypercore_send_asset_to_evm_call(
        lagoon_vault,
        evm_usdc_amount=reserve_token.convert_to_raw(withdraw_amount),
    )
    tx_hash = broadcast_bound_call(web3, hot_wallet, fn)
    wait_for_evm_usdc_balance(
        reserve_token,
        safe_address,
        baseline_balance=baseline_evm_balance,
        expected_increase=withdraw_amount,
    )
    return tx_hash


def execute_hypercore_transit_recovery_actions(
    *,
    web3: Web3,
    hot_wallet: HotWallet,
    lagoon_vault,
    session: HyperliquidSession,
    reserve_token: TokenDetails,
    actions: list[HypercoreTransitRecoveryAction],
) -> list[str]:
    """Execute Safe-side HyperCore transit recovery actions in order."""
    executed: list[str] = []
    for action in actions:
        if action.action_kind == "perp_to_spot":
            execute_perp_to_spot(
                web3=web3,
                hot_wallet=hot_wallet,
                lagoon_vault=lagoon_vault,
                session=session,
                reserve_token=reserve_token,
                amount=action.amount,
            )
        elif action.action_kind == "spot_to_evm":
            execute_spot_to_evm(
                web3=web3,
                hot_wallet=hot_wallet,
                lagoon_vault=lagoon_vault,
                session=session,
                reserve_token=reserve_token,
                amount=action.amount,
            )
        else:
            raise NotImplementedError(
                f"Unsupported HyperCore transit recovery action: {action.action_kind}"
            )
        executed.append(action.action_kind)
    return executed
