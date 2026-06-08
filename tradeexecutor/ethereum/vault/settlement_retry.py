"""Async vault settlement retry module.

When the executor process restarts or a new tick fires while an async
vault trade is in ``vault_settlement_pending`` state (request confirmed,
claim pending), this module polls the vault's deposit manager for
settlement status and broadcasts the claim/reclaim transaction.

Works with any vault protocol that implements the generic
:py:class:`~eth_defi.vault.deposit_redeem.VaultDepositManager` interface
(Ostium V1.5, ERC-7540 Lagoon, etc.) — no protocol-specific imports.
"""

import logging
from itertools import chain as ichain

from eth_defi.compat import native_datetime_utc_now
from eth_defi.vault.deposit_redeem import (
    AsyncVaultRequestStatus,
    DepositRedeemEventAnalysis,
)
from hexbytes import HexBytes

from tradeexecutor.ethereum.vault.vault_routing import get_vault_for_pair
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeStatus

logger = logging.getLogger(__name__)


def check_and_resolve_vault_settlements(
    state: State,
    execution_model,
    web3config=None,
) -> list[TradeExecution]:
    """Check for vault_settlement_pending trades and attempt to complete them.

    Called during live execution startup and on each tick to resolve
    async vault trades that are awaiting settlement.

    The function scans all open and pending positions for trades with
    ``vault_settlement_pending`` status. For each one it:

    1. Reads routing metadata from ``trade.other_data``.
    2. Reconstructs the protocol-specific ticket via the deposit manager.
    3. Checks settlement status via the generic enum interface.
    4. If claimable: signs and broadcasts the claim transaction.
    5. Analyses the claim result and calls ``mark_trade_success``.
    6. If reclaimable: signs and broadcasts the reclaim transaction,
       then calls ``mark_trade_failed``.

    :param state:
        Current strategy state.

    :param execution_model:
        The execution model (must have ``tx_builder``).

    :param web3config:
        Optional Web3Config for multi-chain setups.
        Falls back to ``execution_model.web3`` if None.

    :return:
        List of trades that were successfully resolved.
    """
    resolved: list[TradeExecution] = []

    # Scan both open and pending positions
    pending_trades: list[TradeExecution] = []
    all_positions = ichain(
        state.portfolio.open_positions.values(),
        state.portfolio.pending_positions.values(),
    )
    for position in all_positions:
        for trade in position.trades.values():
            if trade.get_status() == TradeStatus.vault_settlement_pending:
                pending_trades.append(trade)

    if not pending_trades:
        return resolved

    logger.info(
        "Found %d vault settlement pending trade(s), checking status",
        len(pending_trades),
    )

    for trade in pending_trades:
        try:
            _resolve_single_vault_trade(state, trade, execution_model, web3config, resolved)
        except Exception as e:
            logger.warning(
                "Failed to resolve vault settlement for trade #%d: %s",
                trade.trade_id, e,
                exc_info=True,
            )

    return resolved


def _resolve_single_vault_trade(
    state: State,
    trade: TradeExecution,
    execution_model,
    web3config,
    resolved: list[TradeExecution],
):
    """Attempt to resolve a single vault settlement pending trade."""

    from tradingstrategy.chain import ChainId

    # Read metadata
    vault_chain_id = trade.other_data.get("vault_chain_id")
    direction = trade.other_data.get("vault_direction")

    if vault_chain_id is None or direction is None:
        logger.warning(
            "Vault settlement pending trade #%d missing metadata (chain_id=%s, direction=%s), skipping",
            trade.trade_id, vault_chain_id, direction,
        )
        return

    # Get web3 connection
    if web3config is not None:
        web3 = web3config.get_connection(ChainId(vault_chain_id))
    else:
        web3 = execution_model.web3

    if web3 is None:
        logger.warning(
            "No web3 connection for chain %d, skipping vault trade #%d",
            vault_chain_id, trade.trade_id,
        )
        return

    # Get vault and deposit manager
    vault = get_vault_for_pair(web3, trade.pair)
    deposit_manager = vault.get_deposit_manager()

    # Reconstruct ticket
    if direction == "deposit":
        ticket = deposit_manager.reconstruct_deposit_ticket(trade.other_data)
    else:
        ticket = deposit_manager.reconstruct_redemption_ticket(trade.other_data)

    # STEP A: Check for existing post-request tx (idempotent handling)
    request_tx_count = trade.other_data.get("vault_request_tx_count", 1)
    has_existing_post_tx = len(trade.blockchain_transactions) > request_tx_count
    tx_already_confirmed = False
    confirmed_receipt = None
    is_reclaim_tx = False

    if has_existing_post_tx:
        existing_tx = trade.blockchain_transactions[-1]
        is_reclaim_tx = existing_tx.other.get("vault_settlement_action") == "reclaim"

        if existing_tx.tx_hash:
            try:
                confirmed_receipt = web3.eth.get_transaction_receipt(
                    HexBytes(existing_tx.tx_hash)
                )
                if confirmed_receipt and confirmed_receipt["status"] == 1:
                    tx_already_confirmed = True
                elif confirmed_receipt and confirmed_receipt["status"] == 0:
                    # Reverted — pop and try fresh
                    trade.blockchain_transactions.pop()
                    has_existing_post_tx = False
            except Exception:
                pass

        if has_existing_post_tx and not tx_already_confirmed:
            # Rebroadcast
            try:
                web3.eth.send_raw_transaction(HexBytes(existing_tx.signed_bytes))
                confirmed_receipt = web3.eth.wait_for_transaction_receipt(
                    HexBytes(existing_tx.tx_hash), timeout=120,
                )
                if confirmed_receipt["status"] == 1:
                    tx_already_confirmed = True
                else:
                    trade.blockchain_transactions.pop()
                    has_existing_post_tx = False
            except Exception as e:
                logger.warning(
                    "Vault claim rebroadcast failed for trade #%d: %s",
                    trade.trade_id, e,
                )
                return

    if not tx_already_confirmed:
        # STEP B: Check settlement status
        if direction == "deposit":
            status = deposit_manager.get_deposit_request_status(ticket)
        else:
            status = deposit_manager.get_redemption_request_status(ticket)

        if status == AsyncVaultRequestStatus.pending:
            logger.info(
                "Vault trade #%d still pending settlement (direction=%s)",
                trade.trade_id, direction,
            )
            return
        elif status == AsyncVaultRequestStatus.none:
            logger.warning(
                "Unexpected NONE status for vault trade #%d (direction=%s)",
                trade.trade_id, direction,
            )
            return

        # STEP C: Sign and broadcast claim/reclaim
        if status == AsyncVaultRequestStatus.claimable:
            if direction == "deposit":
                func = deposit_manager.finish_deposit(ticket)
            else:
                func = deposit_manager.finish_redemption(ticket)
            is_reclaim_tx = False
        elif status == AsyncVaultRequestStatus.reclaimable:
            if direction == "deposit":
                func = deposit_manager.reclaim_deposit(ticket)
            else:
                func = deposit_manager.reclaim_withdrawal(ticket)
            if func is None:
                logger.error(
                    "Protocol does not support reclaim for trade #%d",
                    trade.trade_id,
                )
                return
            is_reclaim_tx = True
        else:
            return

        # Sync nonce before signing (may be stale from prior txs)
        tx_builder = execution_model.tx_builder
        if hasattr(tx_builder, 'hot_wallet'):
            tx_builder.hot_wallet.sync_nonce(web3)

        action_label = "reclaim" if is_reclaim_tx else "claim"
        new_tx = tx_builder.sign_transaction(
            contract=vault.vault_contract,
            args_bound_func=func,
            gas_limit=1_000_000,
            asset_deltas=[],
            notes=f"Vault {action_label} for trade #{trade.trade_id} ({direction})",
        )
        new_tx.other["vault_settlement_action"] = action_label
        trade.blockchain_transactions.append(new_tx)

        # Broadcast and wait
        try:
            web3.eth.send_raw_transaction(HexBytes(new_tx.signed_bytes))
            new_tx.broadcasted_at = native_datetime_utc_now()
            confirmed_receipt = web3.eth.wait_for_transaction_receipt(
                HexBytes(new_tx.tx_hash), timeout=120,
            )
        except Exception as e:
            logger.warning(
                "Vault %s broadcast failed for trade #%d: %s",
                action_label, trade.trade_id, e,
            )
            return

        if confirmed_receipt["status"] != 1:
            logger.warning(
                "Vault %s tx reverted for trade #%d",
                action_label, trade.trade_id,
            )
            trade.blockchain_transactions.pop()
            return

    # STEP D: Analyse confirmed tx and update trade state
    ts = native_datetime_utc_now()
    tx_hash = HexBytes(confirmed_receipt["transactionHash"])

    if is_reclaim_tx:
        # Reclaim — mark trade as failed, restore reserves
        trade.vault_settlement_pending_at = None
        state.mark_trade_failed(ts, trade)
        logger.info(
            "Vault trade #%d reclaimed successfully (direction=%s)",
            trade.trade_id, direction,
        )
    else:
        # Claim — analyse result and mark success
        if direction == "deposit":
            analysis = deposit_manager.analyse_deposit(tx_hash, ticket)
        else:
            analysis = deposit_manager.analyse_redemption(tx_hash, ticket)

        if not isinstance(analysis, DepositRedeemEventAnalysis):
            logger.error(
                "Vault claim analysis failed for trade #%d: %s",
                trade.trade_id, analysis,
            )
            return

        if direction == "deposit":
            executed_reserve = analysis.denomination_amount
            executed_amount = analysis.share_count
            price = float(executed_reserve / executed_amount) if executed_amount else 0
        else:
            executed_amount = -analysis.share_count
            executed_reserve = analysis.denomination_amount
            price = float(executed_reserve / analysis.share_count) if analysis.share_count else 0

        # Clear pending status before marking success
        trade.vault_settlement_pending_at = None

        state.mark_trade_success(
            ts,
            trade,
            executed_price=price,
            executed_amount=executed_amount,
            executed_reserve=executed_reserve,
            lp_fees=0,
            native_token_price=0,
        )

        # Handle partial deposit refunds
        if direction == "deposit" and executed_reserve < trade.planned_reserve:
            refund_amount = trade.planned_reserve - executed_reserve
            state.portfolio.adjust_reserves(
                trade.reserve_currency,
                refund_amount,
                f"Vault partial deposit refund: trade #{trade.trade_id}",
            )

        logger.info(
            "Vault trade #%d settled successfully (direction=%s, amount=%s, reserve=%s, price=%s)",
            trade.trade_id, direction, executed_amount, executed_reserve, price,
        )

    resolved.append(trade)
