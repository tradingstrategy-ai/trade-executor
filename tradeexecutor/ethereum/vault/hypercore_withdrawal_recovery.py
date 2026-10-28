"""Evidence-led partial withdrawal recovery used by correct-accounts.

The operator supplies the two HyperCore execution hashes. We verify those
specific actions rather than searching for a superficially matching transfer.
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal

import requests
from web3 import Web3
from eth_defi.hotwallet import HotWallet
from eth_defi.compat import native_datetime_utc_now
from eth_defi.hyperliquid.api import fetch_user_vault_equity
from eth_defi.hyperliquid.session import create_hyperliquid_session, HyperliquidSession

from tradeexecutor.ethereum.vault import hypercore_transit_recovery as transit
from tradeexecutor.ethereum.vault.hypercore_routing import RETAIN_RESERVE_ALLOCATION_ON_FAILURE_KEY, raw_to_usdc, vault_kept_requested_funds
from tradeexecutor.ethereum.lagoon.vault import LagoonVaultSyncModel
from tradeexecutor.state.store import StateStore
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import HYPERCORE_STRANDED_USDC_KEY, TradeExecution, TradeStatus

logger = logging.getLogger(__name__)

#: ``hypercore_failure_diagnosis`` prefixes written when phase 1 never reached perp.
PHASE1_PERP_WAIT_DIAGNOSES = ("HyperCore phase1_perp_wait:", "HyperCore phase1_retry_perp_wait:")


@dataclass
class PartialWithdrawalRecovery:
    """Verified sale and current custody, without any state mutation."""

    trade: TradeExecution
    snapshot: transit.HypercoreTransitBalanceSnapshot
    net_proceeds: Decimal
    sold_quantity: Decimal
    vault_equity: Decimal
    already_returned: bool
    session: HyperliquidSession

    #: Explicit operator attestation for the legacy unsynchronised-nonce failure.
    retry_before_signing: bool = False


def clear_rejected_first_stage_withdrawals(
    trades: Iterable[TradeExecution],
    safe_address: str,
    session: HyperliquidSession | None = None,
) -> list[TradeExecution]:
    """Clear stranded-perp markers from phase-1 withdrawals the vault never paid.

    Before 2026-10-28 routing recorded a timed-out phase-1 request as USDC
    stranded in perp even when the vault equity had not moved, which happens
    when HyperCore rejects the action after a successful EVM receipt. That
    marker defers ``repair`` and makes ``correct-accounts`` look for sale
    proceeds in perp. Re-read the vault equity: when the vault kept the
    request, the trade is an ordinary failed sell that moved no funds, so
    ``repair`` may make its counter trade and unfreeze the position.

    :param trades: Trades to inspect, normally every trade in the state.
    :param safe_address: Safe whose vault equity is read.
    :param session: Hyperliquid API session; created on first use when omitted.
    :return: Trades whose markers were cleared.
    """
    cleared = []
    for trade in trades:
        metadata = trade.other_data or {}
        marker = metadata.get(HYPERCORE_STRANDED_USDC_KEY)
        # Only a phase-1 perp wait can fail before any payout. Phase 2 and 3
        # write the same perp marker after a verified withdrawal, and a vault
        # NAV move must never be able to erase that sale.
        if not (
            trade.is_sell()
            and trade.pair.is_hyperliquid_vault()
            and trade.get_status() == TradeStatus.failed
            and str(metadata.get("hypercore_failure_diagnosis", "")).startswith(PHASE1_PERP_WAIT_DIAGNOSES)
            and isinstance(marker, dict)
            and marker.get("location") == "hypercore_perp"
            and "hypercore_phase1_vault_equity_usdc" in metadata
            and "hypercore_first_stage_requested_raw" in metadata
        ):
            continue
        if session is None:
            session = create_hyperliquid_session()
        before = Decimal(metadata["hypercore_phase1_vault_equity_usdc"])
        requested_raw = int(metadata["hypercore_first_stage_requested_raw"])
        equity = fetch_user_vault_equity(session, user=safe_address, vault_address=trade.pair.pool_address, bypass_cache=True)
        if equity is None or not vault_kept_requested_funds(before, equity.equity, requested_raw):
            logger.warning(
                "Trade #%s keeps its stranded marker: vault equity %s USDC against pre-phase-1 %s USDC; reconcile with correct-accounts",
                trade.trade_id, equity.equity if equity else None, before,
            )
            continue
        trade.other_data.pop(HYPERCORE_STRANDED_USDC_KEY)
        trade.other_data.pop(RETAIN_RESERVE_ALLOCATION_ON_FAILURE_KEY, None)
        trade.other_data["hypercore_rejected_withdrawal_cleared"] = {
            "equity_before": str(before),
            "equity_now": str(equity.equity),
            "requested_raw": requested_raw,
            "cleared_at": native_datetime_utc_now().isoformat(),
        }
        trade.add_note(
            f"Stranded marker cleared: vault equity {equity.equity} USDC still matches pre-phase-1 "
            f"{before} USDC, so the {raw_to_usdc(requested_raw)} USDC withdrawal never left the vault"
        )
        logger.info("Cleared rejected HyperCore withdrawal marker from trade #%s", trade.trade_id)
        cleared.append(trade)
    return cleared


def inspect_partial_withdrawal(
    state: State,
    sync_model: LagoonVaultSyncModel,
    web3: Web3,
    evidence: dict,
) -> PartialWithdrawalRecovery:
    """Validate an interrupted two-leg sell before allowing any transfer."""
    trade = state.portfolio.get_trade_by_id(evidence["trade_id"])
    if trade is None or not trade.is_sell() or not trade.pair.is_hyperliquid_vault():
        raise RuntimeError("Withdrawal evidence must identify a HyperCore sell")
    if trade.other_data.get("hypercore_second_stage_status") != "verification_ambiguous":
        raise RuntimeError("Only a saved ambiguous second-stage withdrawal is supported")
    if trade.executed_at is not None or trade.repaired_at is not None:
        raise RuntimeError("Withdrawal has already been accounted for")
    if len(trade.blockchain_transactions) != 2:
        raise RuntimeError("Expected exactly the two original withdrawal transactions")
    if web3.eth.chain_id != 999:
        raise RuntimeError("Withdrawal evidence reconciliation currently supports mainnet only")
    safe = sync_model.get_token_storage_address()
    vault = trade.pair.pool_address
    amounts = [trade.other_data["hypercore_first_stage_requested_raw"], trade.other_data["hypercore_second_stage_requested_raw"]]
    executions = []
    for tx, amount, name in zip(trade.blockchain_transactions, amounts, ("first_execution_hash", "second_execution_hash")):
        receipt = web3.eth.get_transaction_receipt(tx.tx_hash)
        if receipt["status"] != 1:
            raise RuntimeError("Stored EVM withdrawal receipt was not successful")
        response = requests.post("https://rpc.hyperliquid.xyz/explorer", json={"type": "txDetails", "hash": evidence[name]}, timeout=15)
        response.raise_for_status()
        execution = response.json()["tx"]
        action = execution["action"]
        block_time = web3.eth.get_block(receipt["blockNumber"])["timestamp"]
        if not (
            execution["hash"].lower() == evidence[name].lower()
            and execution["user"].lower() == safe.lower()
            and action["type"] == "vaultTransfer"
            and action["vaultAddress"].lower() == vault.lower()
            and action["isDeposit"] is False
            and action["usd"] == amount
            and 0 <= execution["time"] / 1000 - block_time <= 120
        ):
            raise RuntimeError("HyperCore execution does not match the recorded EVM withdrawal")
        executions.append(execution)
    first, second = executions
    if first["error"] is not None or not second["error"] or second["time"] <= first["time"]:
        raise RuntimeError("Evidence must prove first execution success and second execution rejection")
    response = requests.post("https://api.hyperliquid.xyz/info", json={
        "type": "userNonFundingLedgerUpdates", "user": safe,
        "startTime": first["time"] - 1000, "endTime": first["time"] + 1000,
    }, timeout=15)
    response.raise_for_status()
    ledger = [row["delta"] for row in response.json() if row["hash"].lower() == first["hash"].lower()]
    if len(ledger) != 1 or ledger[0]["type"] != "vaultWithdraw" or ledger[0]["vault"].lower() != vault.lower():
        raise RuntimeError("Successful first withdrawal has no unique ledger proof")
    net = Decimal(ledger[0]["netWithdrawnUsd"])
    if not 0 < net <= raw_to_usdc(amounts[0]) or Decimal(ledger[0]["requestedUsd"]) != raw_to_usdc(amounts[0]):
        raise RuntimeError("Withdrawal ledger amount disagrees with saved request")
    position = state.portfolio.get_position_by_id(trade.position_id)
    before = Decimal(trade.other_data["hypercore_phase1_vault_equity_usdc"])
    after = Decimal(trade.other_data["hypercore_first_stage_residual_usd"])
    if not 0 < after < before:
        raise RuntimeError("Missing valid first-leg equity reduction")
    sold = position.get_quantity() * (before - after) / before
    session = create_hyperliquid_session()
    equity = fetch_user_vault_equity(session, user=safe, vault_address=vault, bypass_cache=True)
    if equity is None or equity.equity <= 0:
        raise RuntimeError("Residual vault equity is missing; operator reconciliation required")
    snapshot = transit.fetch_hypercore_transit_balances(session=session, safe_address=safe, reserve_token=sync_model.vault.underlying_token)
    if snapshot.perp_position_count:
        raise RuntimeError("Active perp exposure prevents transit recovery")
    saved_safe = state.portfolio.get_default_reserve_position().quantity
    baseline = Decimal(trade.other_data["hypercore_phase1_perp_baseline_usdc"])
    audit = trade.other_data.get("hypercore_withdrawal_recovery", {})
    retry_before_signing = evidence.get("recovery_failed_before_signing") is True
    if retry_before_signing and (
        set(audit) != {"expected_return", "completed"} or audit["completed"] != {}
    ):
        raise RuntimeError("Pre-signing retry requires the legacy empty recovery checkpoint")
    returned = False
    if audit.get("completed"):
        for tx_hash in audit["completed"].values():
            if web3.eth.get_transaction_receipt(tx_hash)["status"] != 1:
                raise RuntimeError("Recovery receipt did not succeed")
        returned = (
            set(audit["completed"]) == {"perp_to_spot", "spot_to_evm"}
            and abs(snapshot.evm_usdc_balance - saved_safe - Decimal(audit["expected_return"])) <= transit.BALANCE_TOLERANCE
            and snapshot.perp_withdrawable <= transit.HYPERCORE_TRANSIT_RECOVERY_DUST_USDC + transit.BALANCE_TOLERANCE
            and snapshot.spot_total_usdc <= transit.BALANCE_TOLERANCE
        )
    stranded = (
        (not audit or retry_before_signing)
        and abs(snapshot.evm_usdc_balance - saved_safe) <= transit.BALANCE_TOLERANCE
        and abs(snapshot.perp_withdrawable - baseline - net) <= transit.BALANCE_TOLERANCE
        and snapshot.spot_total_usdc <= transit.BALANCE_TOLERANCE
    )
    if not (returned or stranded):
        raise RuntimeError("Withdrawal custody is uncertain or recovery was interrupted; operator reconciliation required")
    slot = state.pending_data_availability_slot
    if slot is None or trade.opened_at != slot or any(
        t.trade_id != trade.trade_id and t.opened_at == slot and t.get_status() not in
        (TradeStatus.success, TradeStatus.failed, TradeStatus.repaired, TradeStatus.expired)
        for t in state.portfolio.get_all_trades()
    ):
        raise RuntimeError("Repair unfinished sibling trades before withdrawal reconciliation")
    if retry_before_signing:
        logger.warning("Operator attests the previous recovery failed before signing; unchanged custody verified. This does not independently prove absence of a pending transaction.")
    return PartialWithdrawalRecovery(trade, snapshot, net, sold, equity.equity, returned, session, retry_before_signing)


def recover_partial_withdrawal(
    recovery: PartialWithdrawalRecovery,
    state: State,
    store: StateStore,
    sync_model: LagoonVaultSyncModel,
    web3: Web3,
    hot_wallet: HotWallet | None,
    dry_run: bool,
) -> None:
    """Return verified free cash and book one partial sale in memory.

    The caller performs normal accounting checks and atomically saves the
    resolved state and decision slot. Transfer checkpoints remain unresolved.
    """
    trade = recovery.trade
    actions = [] if recovery.already_returned else transit.plan_hypercore_transit_recovery_actions(recovery.snapshot)
    if not recovery.already_returned and [a.action_kind for a in actions] != ["perp_to_spot", "spot_to_evm"]:
        raise RuntimeError("Expected two-leg transit recovery")
    logger.warning("Partial withdrawal #%s: %s USDC proceeds, %s shares sold, %s vault equity retained; not safe to restart until reconciliation completes", trade.trade_id, recovery.net_proceeds, recovery.sold_quantity, recovery.vault_equity)
    for action in actions:
        logger.info("Recovery would execute %s: %s USDC", action.action_kind, action.amount)
    baseline = Decimal(trade.other_data["hypercore_phase1_perp_baseline_usdc"])
    expected_return = (
        Decimal(trade.other_data["hypercore_withdrawal_recovery"]["expected_return"])
        if recovery.already_returned else actions[-1].amount
    )
    projected_proceeds = recovery.net_proceeds - max(Decimal(0), baseline + recovery.net_proceeds - expected_return)
    if projected_proceeds <= 0:
        raise RuntimeError("Transit costs consume the verified withdrawal proceeds")
    logger.info(
        "Projected accounting: sale credit %s USDC; pre-existing transit cash reserve correction %s USDC; "
        "retained vault equity %s USDC; consume pending slot %s and advance cycle %s -> %s after final checks",
        projected_proceeds, baseline, recovery.vault_equity, state.pending_data_availability_slot, state.cycle, state.cycle + 1,
    )
    if dry_run:
        return
    if not recovery.already_returned:
        if hot_wallet is None:
            raise RuntimeError("Withdrawal recovery requires a hot wallet/private key")
        # Initialise signing before writing a checkpoint that could imply a
        # broadcast. The generic correction hook does this too, but is bypassed
        # by evidence-led withdrawal recovery.
        hot_wallet.sync_nonce(web3)
        if recovery.retry_before_signing and web3.eth.get_transaction_count(hot_wallet.address, "pending") != hot_wallet.current_nonce:
            raise RuntimeError("Pending wallet transactions prevent a pre-signing recovery retry")
        audit = {
            "expected_return": str(actions[-1].amount),
            "completed": {},
            "started_at": native_datetime_utc_now().isoformat(),
        }
        if recovery.retry_before_signing:
            audit["previous_attempt"] = trade.other_data["hypercore_withdrawal_recovery"]
            trade.add_note("Operator confirmed legacy recovery failed before signing; live custody rechecked before retry")
        trade.other_data["hypercore_withdrawal_recovery"] = audit
        store.sync(state)

        def record_action(kind: str, tx_hash: str) -> None:
            audit["completed"][kind] = tx_hash
            store.sync(state)

        transit.execute_hypercore_transit_recovery_actions(
            web3=web3, hot_wallet=hot_wallet, lagoon_vault=sync_model.vault,
            session=recovery.session, reserve_token=sync_model.vault.underlying_token,
            actions=actions, on_completed=record_action,
        )
    snapshot = transit.fetch_hypercore_transit_balances(session=recovery.session, safe_address=recovery.snapshot.safe_address, reserve_token=sync_model.vault.underlying_token)
    audit = trade.other_data["hypercore_withdrawal_recovery"]
    expected = Decimal(audit["expected_return"])
    saved_safe = state.portfolio.get_default_reserve_position().quantity
    if abs(snapshot.evm_usdc_balance - saved_safe - expected) > transit.BALANCE_TOLERANCE or snapshot.perp_withdrawable > transit.HYPERCORE_TRANSIT_RECOVERY_DUST_USDC + transit.BALANCE_TOLERANCE or snapshot.spot_total_usdc > transit.BALANCE_TOLERANCE:
        raise RuntimeError("Recovery balances did not reconcile; unresolved checkpoint retained")
    # Attribute transit shortfall conservatively to the sale. Pre-existing cash
    # is left for the ordinary reserve correction, never added to sale proceeds.
    original_total = Decimal(trade.other_data["hypercore_phase1_perp_baseline_usdc"]) + recovery.net_proceeds
    shortfall = max(Decimal(0), original_total - expected)
    proceeds = recovery.net_proceeds - shortfall
    position = state.portfolio.get_position_by_id(trade.position_id)
    if trade.failed_at is not None:
        trade.add_note(f"Reconciled execution failure recorded at {trade.failed_at}")
    trade.failed_at = None
    state.mark_trade_success(native_datetime_utc_now(), trade,
        executed_price=float(proceeds / recovery.sold_quantity),
        executed_amount=-recovery.sold_quantity, executed_reserve=proceeds,
        lp_fees=0, native_token_price=None, force=True)
    position.last_pricing_at = native_datetime_utc_now()
    position.last_token_price = float(recovery.vault_equity / position.get_quantity())
    if position.position_id in state.portfolio.frozen_positions:
        del state.portfolio.frozen_positions[position.position_id]
        state.portfolio.open_positions[position.position_id] = position
        position.unfrozen_at = native_datetime_utc_now()
    trade.other_data["hypercore_second_stage_status"] = "reconciled"
    trade.other_data.pop("retain_reserve_allocation_on_failure", None)
    trade.other_data.pop("hypercore_stranded_usdc", None)
    trade.other_data.pop("hypercore_accounting_reconciliation_required", None)
    trade.add_note("Verified first withdrawal booked as a partial fill; rejected second withdrawal was not retried")
