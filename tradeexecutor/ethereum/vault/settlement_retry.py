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
from decimal import Decimal
from itertools import chain as ichain

from eth_defi.abi import get_topic_signature_from_event
from eth_defi.compat import native_datetime_utc_now
from eth_defi.event_reader.conversion import convert_bytes32_to_address, convert_bytes32_to_uint
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.receipt import wait_for_transaction_receipt_robust
from eth_defi.vault.deposit_redeem import (
    AsyncVaultRequestStatus,
    DepositRedeemEventAnalysis,
)
from hexbytes import HexBytes
from web3 import Web3
from web3.contract.contract import ContractFunction

from tradeexecutor.ethereum.tx import HotWalletTransactionBuilder, TransactionBuilder
from tradeexecutor.ethereum.vault.settlement_estimate import refresh_vault_settlement_estimate
from tradeexecutor.ethereum.vault.vault_routing import (
    convert_vault_flow_analysis,
    get_vault_for_pair,
)
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeStatus

logger = logging.getLogger(__name__)

VAULT_SETTLEMENT_RECEIPT_TIMEOUT = 120


def _get_chain_aware_tx_builder(
    execution_model,
    web3: Web3,
    vault_chain_id: int,
) -> TransactionBuilder:
    """Get transaction builder that signs for the vault chain."""
    tx_builder = execution_model.tx_builder
    tx_builder_chain_id = getattr(tx_builder, "chain_id", None)

    if not isinstance(tx_builder_chain_id, int) or tx_builder_chain_id == vault_chain_id:
        return tx_builder

    assert hasattr(tx_builder, "hot_wallet"), f"Cannot create satellite tx builder from {tx_builder}"

    satellite_wallet = HotWallet(tx_builder.hot_wallet.account)
    satellite_wallet.sync_nonce(web3)

    from tradeexecutor.ethereum.lagoon.tx import LagoonTransactionBuilder

    if isinstance(tx_builder, LagoonTransactionBuilder):
        satellite_vaults = getattr(execution_model, "satellite_vaults", None) or {}
        satellite_vault = satellite_vaults.get(vault_chain_id)
        assert satellite_vault, (
            f"No satellite vault configured for chain {vault_chain_id}. "
            f"Available satellite chains: {list(satellite_vaults.keys())}"
        )
        return LagoonTransactionBuilder(satellite_vault, satellite_wallet, tx_builder.extra_gnosis_gas)

    return HotWalletTransactionBuilder(web3, satellite_wallet)


def _broadcast_and_wait_for_settlement_tx(
    web3: Web3,
    tx: BlockchainTransaction,
    mark_broadcasted: bool = False,
):
    """Broadcast a vault settlement transaction and wait for all RPCs to see it."""
    web3.eth.send_raw_transaction(HexBytes(tx.signed_bytes))
    if mark_broadcasted:
        tx.broadcasted_at = native_datetime_utc_now()
    return _wait_for_settlement_tx_receipt(web3, tx.tx_hash)


def _wait_for_settlement_tx_receipt(
    web3: Web3,
    tx_hash: HexBytes | str,
):
    """Wait until a vault settlement transaction receipt is visible."""
    return wait_for_transaction_receipt_robust(
        web3,
        HexBytes(tx_hash),
        timeout=VAULT_SETTLEMENT_RECEIPT_TIMEOUT,
    )


def _normalise_topic_address(topic) -> str:
    """Convert an indexed address topic to a comparable lowercase hex address."""
    return convert_bytes32_to_address(topic).lower()


def _normalise_topic_signature(topic) -> str:
    """Convert a log topic signature to a 0x-prefixed lowercase string."""
    signature = topic.hex().lower()
    if not signature.startswith("0x"):
        signature = "0x" + signature
    return signature


def _get_request_block(web3: Web3, ticket) -> int:
    """Get the block where the async vault request was opened."""
    block_number = getattr(ticket, "block_number", 0) or 0
    if block_number:
        return block_number

    receipt = web3.eth.get_transaction_receipt(HexBytes(ticket.tx_hash))
    return receipt["blockNumber"]


def _find_already_completed_claim_tx_hash(
    web3: Web3,
    vault,
    ticket,
    direction: str,
) -> HexBytes | None:
    """Find a completed vault claim after local state was lost.

    This handles a crash after the claim transaction was confirmed on-chain but
    before the JSON state file was written. In that situation the request status
    can be ``none`` because the claim consumed the request, while local state
    still shows ``vault_settlement_pending``.
    """
    from_block = _get_request_block(web3, ticket)
    to_block = web3.eth.block_number

    logs = web3.eth.get_logs({
        "address": vault.vault_contract.address,
        "fromBlock": from_block,
        "toBlock": to_block,
    })

    if direction == "deposit":
        signatures = {
            get_topic_signature_from_event(vault.vault_contract.events.Deposit),
            "0xdcbc1c05240f31ff3ad067ef1ee35ce4997762752e3a095284754544f4c709d7",
        }
        expected_to = ticket.to.lower()
        expected_raw_amount = ticket.raw_amount

        for log in logs:
            if _normalise_topic_signature(log["topics"][0]) not in signatures:
                continue
            if len(log["topics"]) < 3:
                continue
            if _normalise_topic_address(log["topics"][2]) != expected_to:
                continue
            if convert_bytes32_to_uint(log["data"][0:32]) != expected_raw_amount:
                continue
            return HexBytes(log["transactionHash"])

    else:
        signatures = {
            get_topic_signature_from_event(vault.vault_contract.events.Withdraw),
            "0xfbde797d201c681b91056529119e0b02407c7bb96a4a2c75c01fc9667232c8db",
        }
        expected_owner = ticket.owner.lower()
        expected_to = ticket.to.lower()
        expected_raw_shares = ticket.raw_shares

        for log in logs:
            if _normalise_topic_signature(log["topics"][0]) not in signatures:
                continue
            if len(log["topics"]) < 4:
                continue
            if _normalise_topic_address(log["topics"][2]) != expected_to:
                continue
            if _normalise_topic_address(log["topics"][3]) != expected_owner:
                continue
            if convert_bytes32_to_uint(log["data"][32:64]) != expected_raw_shares:
                continue
            return HexBytes(log["transactionHash"])

    return None


def _create_recovered_claim_transaction(
    web3: Web3,
    vault_chain_id: int,
    tx_hash: HexBytes,
    action_label: str,
    trade: TradeExecution,
    func: ContractFunction | None,
) -> BlockchainTransaction:
    """Create a minimal state transaction for an already-confirmed claim."""
    tx = web3.eth.get_transaction(tx_hash)
    receipt = web3.eth.get_transaction_receipt(tx_hash)
    recovered_tx = BlockchainTransaction(
        chain_id=vault_chain_id,
        from_address=tx["from"],
        contract_address=tx["to"],
        function_selector=func.fn_name if func is not None else None,
        transaction_args=func.args if func is not None else None,
        args=func.args if func is not None else None,
        tx_hash=tx_hash.hex(),
        nonce=tx["nonce"],
        details={"recovered": True},
        notes=f"Recovered vault {action_label} for trade #{trade.trade_id}",
    )
    recovered_tx.other["vault_settlement_action"] = action_label
    recovered_tx.set_confirmation_information(
        native_datetime_utc_now(),
        receipt["blockNumber"],
        receipt["blockHash"].hex(),
        receipt.get("effectiveGasPrice", 0),
        receipt["gasUsed"],
        receipt["status"] == 1,
    )
    return recovered_tx


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


def _ensure_legacy_pending_deposit_capital_allocated(state: State, trade: TradeExecution) -> None:
    """Repair old pending deposit state that missed its funding-source allocation.

    Current code debits reserves in ``state.start_execution()`` before the async
    request is broadcast, or allocates bridge capital for satellite vault
    deposits. Older/corrupt state can have the request persisted but the funding
    source still visible. If we claim such a trade without repairing first, the
    position receives shares while the same capital remains visible elsewhere.
    """
    assert trade.is_buy(), f"Only deposit trades can reserve-debit repair, got {trade}"

    planned_reserve = trade.planned_reserve

    bridge_position = state.portfolio.get_bridge_position_for_chain(trade.pair.chain_id)
    if bridge_position is not None:
        allocated = trade.bridge_currency_allocated
        if allocated == planned_reserve:
            return
        if allocated not in (None, Decimal(0)):
            raise RuntimeError(
                f"Vault deposit trade #{trade.trade_id} has inconsistent bridge allocation: "
                f"planned_reserve={planned_reserve}, bridge_currency_allocated={allocated}"
            )

        logger.warning(
            "Repairing legacy vault pending deposit bridge accounting before claim: "
            "trade #%d, bridge allocation %s %s was missing",
            trade.trade_id,
            planned_reserve,
            trade.reserve_currency.token_symbol,
        )
        bridge_position.adjust_bridge_capital_allocated(planned_reserve)
        trade.bridge_currency_allocated = planned_reserve
        return

    allocated = trade.reserve_currency_allocated
    if allocated == planned_reserve:
        return
    if allocated not in (None, Decimal(0)):
        raise RuntimeError(
            f"Vault deposit trade #{trade.trade_id} has inconsistent reserve allocation: "
            f"planned_reserve={planned_reserve}, reserve_currency_allocated={allocated}"
        )

    logger.warning(
        "Repairing legacy vault pending deposit reserve accounting before claim: "
        "trade #%d, reserve debit %s %s was missing",
        trade.trade_id,
        planned_reserve,
        trade.reserve_currency.token_symbol,
    )
    state.portfolio.adjust_reserves(
        trade.reserve_currency,
        -planned_reserve,
        f"Repair missing reserve debit for vault pending deposit trade #{trade.trade_id}",
    )
    trade.reserve_currency_allocated = planned_reserve


def _refund_partial_deposit_to_funding_source(
    state: State,
    trade: TradeExecution,
    refund_amount: Decimal,
) -> None:
    """Return an async deposit partial-fill refund to the original funding source."""
    assert refund_amount > 0, f"Refund amount must be positive, got {refund_amount}"

    if trade.bridge_currency_allocated is not None:
        bridge_position = state.portfolio.get_bridge_position_for_chain(trade.pair.chain_id)
        assert bridge_position is not None, f"No bridge position for bridge-funded vault trade #{trade.trade_id}"
        bridge_position.adjust_bridge_capital_allocated(-refund_amount)
        trade.bridge_currency_allocated -= refund_amount
        logger.info(
            "Vault partial deposit refund returned to bridge position: trade #%d, amount=%s %s",
            trade.trade_id,
            refund_amount,
            trade.reserve_currency.token_symbol,
        )
        return

    state.portfolio.adjust_reserves(
        trade.reserve_currency,
        refund_amount,
        f"Vault partial deposit refund: trade #{trade.trade_id}",
    )
    logger.info(
        "Vault partial deposit refund returned to reserves: trade #%d, amount=%s %s",
        trade.trade_id,
        refund_amount,
        trade.reserve_currency.token_symbol,
    )


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
    refresh_vault_settlement_estimate(trade, deposit_manager, ticket, direction)

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
                confirmed_receipt = _broadcast_and_wait_for_settlement_tx(web3, existing_tx)
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
            recovered_tx_hash = _find_already_completed_claim_tx_hash(
                web3,
                vault,
                ticket,
                direction,
            )
            if recovered_tx_hash is None:
                logger.warning(
                    "Unexpected NONE status for vault trade #%d (direction=%s)",
                    trade.trade_id, direction,
                )
                return

            action_label = "claim"
            if direction == "deposit":
                func = deposit_manager.finish_deposit(ticket)
            else:
                func = deposit_manager.finish_redemption(ticket)
            # Even recovered claims need the same read-after-write guard before
            # immediate event analysis on multi-RPC providers.
            confirmed_receipt = _wait_for_settlement_tx_receipt(web3, recovered_tx_hash)
            trade.blockchain_transactions.append(
                _create_recovered_claim_transaction(
                    web3,
                    vault_chain_id,
                    recovered_tx_hash,
                    action_label,
                    trade,
                    func,
                )
            )
            tx_already_confirmed = True
            is_reclaim_tx = False
            logger.info(
                "Recovered already-confirmed vault claim for trade #%d from tx %s",
                trade.trade_id,
                recovered_tx_hash.hex(),
            )

        # STEP C: Sign and broadcast claim/reclaim
        if tx_already_confirmed:
            pass
        elif status == AsyncVaultRequestStatus.claimable:
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

        if not tx_already_confirmed:
            # Sync nonce before signing (may be stale from prior txs)
            tx_builder = _get_chain_aware_tx_builder(execution_model, web3, vault_chain_id)
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
                confirmed_receipt = _broadcast_and_wait_for_settlement_tx(
                    web3,
                    new_tx,
                    mark_broadcasted=True,
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

        executed_reserve, executed_amount, price = convert_vault_flow_analysis(
            analysis,
            direction=direction,
        )
        if direction == "deposit":
            _ensure_legacy_pending_deposit_capital_allocated(state, trade)

        # Clear pending status before marking success
        trade.vault_settlement_pending_at = None

        state.mark_trade_success(
            ts,
            trade,
            executed_price=float(price),
            executed_amount=executed_amount,
            executed_reserve=executed_reserve,
            lp_fees=0,
            native_token_price=0,
        )

        # Handle partial deposit refunds
        if direction == "deposit" and executed_reserve < trade.planned_reserve:
            refund_amount = trade.planned_reserve - executed_reserve
            _refund_partial_deposit_to_funding_source(state, trade, refund_amount)

        logger.info(
            "Vault trade #%d settled successfully (direction=%s, amount=%s, reserve=%s, price=%s)",
            trade.trade_id, direction, executed_amount, executed_reserve, price,
        )

    resolved.append(trade)
