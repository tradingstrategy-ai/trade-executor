"""Read-only verification for completed Lighter custody transfers."""

import asyncio
import datetime
import enum
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable

from eth_defi.compat import native_datetime_utc_now
from eth_defi.confirmation import wait_and_broadcast_multiple_nodes
from eth_defi.hotwallet import SignedTransactionWithNonce
from eth_defi.lighter.api import wait_for_lighter_collateral
from eth_defi.lighter.lagoon import is_lighter_withdrawal_paid
from eth_defi.lighter.session import LighterSession, create_lighter_session
from eth_defi.lighter.transfer import LighterTransferRecord
from eth_defi.lighter.valuation import fetch_lighter_total_equity
from eth_defi.token import fetch_erc20_details
from eth_typing import HexAddress
from hexbytes import HexBytes
from tradingstrategy.chain import ChainId
from web3 import Web3
from web3.exceptions import TransactionNotFound

from tradeexecutor.exchange_account.lighter import LIGHTER_PROTOCOL
from tradeexecutor.exchange_account.lighter_operator import USDC_PRECISION, LighterOperatorRecord, fetch_lighter_funding_transfer
from tradeexecutor.exchange_account.state import complete_exchange_account_funding, complete_exchange_account_transfer
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.repair import RepairAborted, repair_unmined_exchange_deposit
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeStatus

logger = logging.getLogger(__name__)

#: First step when searching back for the block that consumed a nonce.
NONCE_SEARCH_FIRST_STEP_BLOCKS = 256

#: How long ``repair`` waits for a rebroadcast deposit.
UNMINED_DEPOSIT_BROADCAST_TIMEOUT = datetime.timedelta(minutes=5)

#: Chains where the ``finalized`` block tag is established for this proof.
UNMINED_DEPOSIT_CHAINS = (ChainId.ethereum.value, ChainId.anvil.value)

#: ERC-20 ``Transfer(address,address,uint256)`` event topic.
ERC_20_TRANSFER_TOPIC = Web3.keccak(text="Transfer(address,address,uint256)")


class LighterTransferVerificationError(RuntimeError):
    """Raised when recorded Lighter transfer evidence is contradictory."""


@dataclass(frozen=True, slots=True)
class LighterTransferVerification:
    """Verified public accounting result for one Lighter transfer."""

    #: Signed quantity applied to the external exchange-account position.
    executed_amount: Decimal

    #: Positive Safe reserve debit or credit observed on Ethereum.
    executed_reserve: Decimal

    #: Public evidence fields persisted with the successful transfer.
    metadata_updates: dict[str, str]


def get_lighter_relayer_payout(web3: Web3, trade: TradeExecution, hypersync_api_key: str | None = None) -> Decimal | None:
    """Return a withdrawal Lighter's relayer has paid to the Safe, if any.

    ``withdrawPendingBalance`` is permissionless, so the relayer can claim for us.
    Our own claim then reverts, although the money has arrived, and a missing or
    failed claim alone says nothing about the withdrawal. See
    :py:mod:`eth_defi.lighter.lagoon`. Our own successful claim passes this check
    too, so only call it when the trade has no successful claim.

    :param web3:
        Chain reader for ``WithdrawPending`` events.
    :param trade:
        Lighter withdrawal trade.
    :param hypersync_api_key:
        Explicit HyperSync credential; disabled or absent credentials use RPC in executor callers.
    :return:
        Paid USDC, or ``None`` if not paid yet, or if the trade predates this
        evidence and has no baseline.
    """
    metadata = trade.other_data or {}
    if trade.is_buy() or "lighter_withdrawal_from_block" not in metadata:
        return None
    amount = abs(trade.planned_quantity)
    paid = is_lighter_withdrawal_paid(
        web3,
        owner=metadata["lighter_safe_address"],
        usdc=fetch_erc20_details(web3, trade.reserve_currency.address, chain_id=trade.reserve_currency.chain_id),
        amount=amount,
        pending_before=Decimal(str(metadata["lighter_pending_balance_before_usdc"])),
        from_block=int(metadata["lighter_withdrawal_from_block"]),
        hypersync_api_key=hypersync_api_key,
    )
    return amount if paid else None


def verify_lighter_transfer(
    web3: Web3,
    trade: TradeExecution,
    session: LighterSession,
    receipts: list[dict] | None = None,
    deposit_wait_timeout: int | None = None,
    hypersync_api_key: str | None = None,
) -> LighterTransferVerification | None:
    """Verify a Lighter transfer without mutating executor state.

    :param web3:
        Chain reader for receipt and Safe balance evidence.
    :param trade:
        Recorded Lighter custody transfer.
    :param session:
        Public Lighter API session.
    :param receipts:
        Optional receipts already collected by the execution pipeline.
    :param deposit_wait_timeout:
        Optional normal-routing wait for Lighter collateral indexing.
    :param hypersync_api_key:
        Explicit HyperSync credential; disabled or absent credentials use RPC in executor callers.
    :return:
        Verified accounting values, or ``None`` while evidence is incomplete.
    """
    if trade.pair.get_exchange_account_protocol() != LIGHTER_PROTOCOL:
        raise ValueError("Expected a Lighter exchange-account transfer")
    transaction_receipts = receipts or _fetch_transaction_receipts(web3, trade)
    if not transaction_receipts or any(receipt.get("status") != 1 for receipt in transaction_receipts):
        payout = get_lighter_relayer_payout(web3, trade, hypersync_api_key=hypersync_api_key)
        if payout is not None:
            return LighterTransferVerification(
                executed_amount=-payout,
                executed_reserve=payout,
                metadata_updates={"lighter_relayer_paid_usdc": str(payout), "lighter_protocol_fee_usdc": "0"},
            )
    if not transaction_receipts:
        return None
    if any(receipt.get("status") != 1 for receipt in transaction_receipts):
        raise LighterTransferVerificationError("Lighter custody transaction failed")

    metadata = trade.other_data or {}
    if trade.is_buy():
        account_index = trade.pair.get_exchange_account_id()
        if account_index is None:
            raise LighterTransferVerificationError("Lighter transfer has no account index")
        try:
            collateral_before = Decimal(str(metadata["lighter_collateral_before_usdc"]))
            deposit_amount = Decimal(str(metadata["lighter_deposit_amount_usdc"]))
        except KeyError as error:
            raise LighterTransferVerificationError(
                "Lighter deposit has no collateral evidence metadata",
            ) from error
        expected_collateral = collateral_before + deposit_amount
        collateral = fetch_lighter_total_equity(session, int(account_index)).collateral
        if collateral < expected_collateral:
            if deposit_wait_timeout is None:
                return None
            collateral = wait_for_lighter_collateral(
                session,
                int(account_index),
                expected_collateral,
                timeout=deposit_wait_timeout,
            )
        return LighterTransferVerification(
            executed_amount=deposit_amount,
            executed_reserve=deposit_amount,
            metadata_updates={"lighter_collateral_after_usdc": str(collateral)},
        )

    try:
        claimable = Decimal(str(metadata["lighter_claimable_usdc"]))
        safe_before = Decimal(str(metadata["lighter_safe_balance_before_claim"]))
        safe_address = str(metadata["lighter_safe_address"])
    except KeyError as error:
        raise LighterTransferVerificationError(
            "Lighter withdrawal has no Safe-credit evidence metadata",
        ) from error
    usdc = fetch_erc20_details(
        web3,
        trade.reserve_currency.address,
        chain_id=trade.reserve_currency.chain_id,
    )
    safe_after = usdc.fetch_balance_of(
        safe_address,
        block_identifier=transaction_receipts[-1]["blockNumber"],
    )
    safe_credit = Decimal(str(safe_after)) - safe_before
    if safe_credit <= 0:
        return None
    if safe_credit > claimable:
        raise LighterTransferVerificationError(
            "Lighter withdrawal credited more USDC than the recorded claim",
        )
    return LighterTransferVerification(
        executed_amount=-claimable,
        executed_reserve=safe_credit,
        metadata_updates={"lighter_protocol_fee_usdc": str(claimable - safe_credit)},
    )


@dataclass(frozen=True, slots=True)
class LighterFundingVerification:
    """Verified result of one transfer between two accounts of the Safe."""

    #: USDC credited to the destination account.
    amount: Decimal

    #: Fee charged to the sending account on top of the amount.
    fee: Decimal


def get_lighter_funding_evidence(record: LighterTransferRecord) -> dict[str, int | str]:
    """Convert a transfer-history row to the evidence stored on a funding trade.

    :param record:
        Executed transfer read from the sending account's transfer history.
    :return:
        Public ``other_data`` fields read by :py:func:`verify_lighter_funding`.
    """
    return {
        "lighter_transfer_from_account": record.from_account_index,
        "lighter_transfer_to_account": record.to_account_index,
        "lighter_transfer_amount_usdc": str(record.amount),
        "lighter_transfer_fee_usdc": str(record.fee),
    }


def verify_lighter_funding(
    trade: TradeExecution,
    record: LighterTransferRecord | None = None,
) -> LighterFundingVerification | None:
    """Verify a funding transfer from its transfer-history evidence.

    The history row is authoritative. Account equity is not used, because
    the external trader can change either account while the transfer settles.

    :param trade:
        Funding trade on the destination account's position.
    :param record:
        History row fetched during reconciliation; otherwise the evidence
        stored on the trade is used.
    :return:
        Verified amounts, or ``None`` while there is no history evidence.
    """
    evidence = get_lighter_funding_evidence(record) if record is not None else trade.other_data
    if "lighter_transfer_amount_usdc" not in evidence:
        return None
    from_account = int(evidence["lighter_transfer_from_account"])
    to_account = int(evidence["lighter_transfer_to_account"])
    amount = Decimal(str(evidence["lighter_transfer_amount_usdc"]))
    fee = Decimal(str(evidence["lighter_transfer_fee_usdc"]))
    if from_account != int(trade.other_data["funding_source_account_id"]) or to_account != int(trade.pair.get_exchange_account_id()):
        raise LighterTransferVerificationError("Lighter transfer history accounts differ from the funding trade")
    if abs(amount - trade.planned_quantity) > USDC_PRECISION:
        raise LighterTransferVerificationError("Lighter transfer history amount differs from the funding trade")
    if fee < 0:
        raise LighterTransferVerificationError("Lighter transfer history reports a negative fee")
    return LighterFundingVerification(amount=amount, fee=fee)


def reconcile_verified_lighter_transfers(
    state: State,
    web3: Web3,
    *,
    mutate: bool,
    signers: dict[int, LighterOperatorRecord] | None = None,
    hypersync_api_key: str | None = None,
) -> list[TradeExecution]:
    """Inspect or apply verified unfinished Lighter transfers.

    Reserve transfers are verified from Ethereum receipts and public Lighter
    data. A transfer between accounts interrupted before its history row was
    stored needs the sending account's operator record to read authenticated
    transfer history; without it the transfer stays pending. A funding still
    planned after a halted cycle was never submitted and is expired.

    :param state:
        State containing the transfers to inspect.
    :param web3:
        Chain reader used for the verification evidence.
    :param mutate:
        Apply verified results when ``True``; otherwise only return them.
    :param signers:
        Operator records the executor holds, keyed by account index, for
        the authenticated transfer-history reads; see
        :py:func:`~tradeexecutor.exchange_account.lighter_operator.load_lighter_signers`.
    :param hypersync_api_key:
        Explicit HyperSync credential; disabled or absent credentials use RPC in executor callers.
    :return:
        Transfers whose evidence is complete.
    """
    signers = signers or {}
    pending_transfers = [
        (position, trade)
        for position in state.portfolio.get_open_and_frozen_positions()
        for trade in position.trades.values()
        if trade.is_external_account_transfer_pending()
        and trade.pair.get_exchange_account_protocol() == LIGHTER_PROTOCOL
    ]
    pending_funding = [
        trade
        for position in state.portfolio.get_open_and_frozen_positions()
        for trade in position.trades.values()
        if trade.is_internal_account_transfer_pending()
        and trade.pair.get_exchange_account_protocol() == LIGHTER_PROTOCOL
    ]

    verified_trades = []

    # Sequential execution starts each trade just before submitting it, so a
    # transfer still planned after a halted cycle was never submitted and
    # holds no reserve. Expire it so it does not block the executor.
    never_started = [trade for trade in [*pending_funding, *(trade for _, trade in pending_transfers)] if trade.get_status() == TradeStatus.planned]
    for trade in never_started:
        verified_trades.append(trade)
        if mutate:
            trade.mark_expired(native_datetime_utc_now())
    pending_funding = [trade for trade in pending_funding if trade not in never_started]
    pending_transfers = [(position, trade) for position, trade in pending_transfers if trade not in never_started]

    for trade in pending_funding:
        record = None
        tx_hash = trade.other_data.get("lighter_transfer_tx_hash")
        if "lighter_transfer_amount_usdc" not in trade.other_data:
            if not tx_hash:
                logger.warning(
                    "Lighter funding trade #%d may have been submitted but its hash was not recorded; check the sending account's transfer history manually",
                    trade.trade_id,
                )
                continue
            sender = signers.get(int(trade.other_data.get("funding_source_account_id", -1)))
            if sender is None:
                logger.warning(
                    "Lighter funding trade #%d needs the sending account's record (LIGHTER_OPERATOR_RECORD_FILE or LIGHTER_SUB_ACCOUNT_OPERATOR_RECORD_FILES) to read transfer history",
                    trade.trade_id,
                )
                continue
            record = asyncio.run(fetch_lighter_funding_transfer(sender, str(tx_hash)))
            if record is None:
                continue
        verification = verify_lighter_funding(trade, record)
        if verification is None:
            continue
        verified_trades.append(trade)
        if mutate:
            if record is not None:
                trade.other_data.update(get_lighter_funding_evidence(record))
            complete_exchange_account_funding(
                state=state,
                trade=trade,
                executed_amount=verification.amount,
                fee=verification.fee,
                executed_at=native_datetime_utc_now(),
                recovery=True,
            )

    if not pending_transfers:
        return verified_trades

    session = create_lighter_session()
    try:
        for position, trade in pending_transfers:
            verification = verify_lighter_transfer(web3, trade, session, hypersync_api_key=hypersync_api_key)
            if verification is None:
                continue
            verified_trades.append(trade)
            if mutate:
                trade.other_data.update(verification.metadata_updates)
                complete_exchange_account_transfer(
                    state=state,
                    position=position,
                    trade=trade,
                    executed_amount=verification.executed_amount,
                    executed_reserve=verification.executed_reserve,
                    executed_at=native_datetime_utc_now(),
                    recovery=True,
                )
        return verified_trades
    finally:
        session.close()


class UnminedDepositCase(enum.Enum):
    """What the chain says about a Lighter deposit with no receipt."""

    #: The deposit's nonce is free, so its stored transaction can still be mined.
    sendable = "sendable"

    #: Another transaction took the nonce, and it moved nothing out of the Safe.
    void = "void"

    #: The evidence is incomplete or contradictory, so the trade stays pending.
    undetermined = "undetermined"


@dataclass(frozen=True, slots=True)
class UnminedDepositClassification:
    """Evidence about one Lighter deposit with no receipt."""

    #: What to do with the trade.
    case: UnminedDepositCase

    #: Why, for the operator and the trade's notes.
    reason: str

    #: Transactions to rebroadcast, in nonce order, for a sendable deposit.
    unmined: tuple[BlockchainTransaction, ...] = ()


def find_nonce_consumer(
    web3: Web3,
    sender: HexAddress,
    nonce: int,
    finalized_block: int,
) -> dict | None:
    """Find the transaction that consumed a sender's nonce.

    Reads the sender's nonce at pinned blocks. A consumed nonce alone proves nothing,
    because the consumer can be the transaction in question.

    :param web3:
        Connection with recent historic state.
    :param sender:
        Transaction sender.
    :param nonce:
        Nonce to look for.
    :param finalized_block:
        Search from this block backwards.
    :return:
        The transaction, or ``None`` if it was not found by ``finalized_block``.
    """
    high = finalized_block
    if web3.eth.get_transaction_count(sender, high) <= nonce:
        return None
    step = NONCE_SEARCH_FIRST_STEP_BLOCKS
    low = max(high - step, 0)
    while web3.eth.get_transaction_count(sender, low) > nonce:
        if low == 0:
            return None
        high = low
        step *= 2
        low = max(high - step, 0)
    while high - low > 1:
        middle = (low + high) // 2
        if web3.eth.get_transaction_count(sender, middle) > nonce:
            high = middle
        else:
            low = middle
    block = web3.eth.get_block(high, full_transactions=True)
    return next(
        (tx for tx in block["transactions"] if tx["from"].lower() == sender.lower() and tx["nonce"] == nonce),
        None,
    )


def classify_unmined_lighter_deposit(
    web3: Web3,
    state: State,
    trade: TradeExecution,
    safe_address: HexAddress,
) -> UnminedDepositClassification | None:
    """Classify a Lighter deposit from the chain, without changing anything.

    See ``docs/plans/lighter-recovery-and-gas-fixes.md``. Undetermined is always safe:
    the trade keeps blocking ``start``, which is what happened before this path existed.

    :param web3:
        Ethereum connection.
    :param state:
        State holding the trade.
    :param trade:
        Pending external exchange-account deposit.
    :param safe_address:
        Safe the deposit would have spent from.
    :return:
        The classification, or ``None`` if the deposit has a receipt, so that normal
        reconciliation applies.
    """
    transactions = sorted(trade.blockchain_transactions, key=lambda tx: tx.nonce)
    by_function = {tx.wrapped_function_selector: tx for tx in transactions}
    deposit = by_function.get("deposit")
    if deposit is None or len(transactions) != 2 or "approve" not in by_function:
        return UnminedDepositClassification(UnminedDepositCase.undetermined, "Expected an approval and a deposit")

    receipts = {}
    for tx in transactions:
        try:
            receipts[tx.tx_hash] = web3.eth.get_transaction_receipt(tx.tx_hash)
        except TransactionNotFound:
            pass

    if deposit.tx_hash in receipts:
        return None

    approve = by_function["approve"]
    if approve.tx_hash in receipts and receipts[approve.tx_hash]["status"] != 1:
        return UnminedDepositClassification(UnminedDepositCase.undetermined, f"Approval {approve.tx_hash} reverted")

    unmined = tuple(tx for tx in transactions if tx.tx_hash not in receipts)
    sender = deposit.from_address
    mined_nonce = web3.eth.get_transaction_count(sender, "latest")
    expected_nonces = list(range(mined_nonce, mined_nonce + len(unmined)))
    if [tx.nonce for tx in unmined] == expected_nonces:
        return UnminedDepositClassification(UnminedDepositCase.sendable, f"Nonce {mined_nonce} is free", unmined)

    if deposit.nonce >= mined_nonce:
        return UnminedDepositClassification(
            UnminedDepositCase.undetermined,
            f"Nonces {[tx.nonce for tx in unmined]} do not start at the wallet's nonce {mined_nonce}",
        )

    finalized_block = web3.eth.get_block("finalized")["number"]
    consumer = find_nonce_consumer(web3, sender, deposit.nonce, finalized_block)
    if consumer is None:
        return UnminedDepositClassification(
            UnminedDepositCase.undetermined,
            f"Nonce {deposit.nonce} is consumed but not finalized at block {finalized_block}",
        )

    consumer_hash = HexBytes(consumer["hash"]).to_0x_hex()
    if HexBytes(consumer_hash) == HexBytes(deposit.tx_hash):
        return UnminedDepositClassification(
            UnminedDepositCase.undetermined,
            f"Deposit was mined in block {consumer['blockNumber']}, but this node has no receipt",
        )

    # A different hash can still be a fee-bumped copy of the same deposit, so read its effects.
    consumer_receipt = web3.eth.get_transaction_receipt(consumer_hash)
    reserve_address = trade.reserve_currency.address.lower()
    safe_topic = HexBytes(safe_address).rjust(32, b"\0")
    for log in consumer_receipt["logs"]:
        topics = [HexBytes(topic) for topic in log["topics"]]
        if (
            log["address"].lower() == reserve_address
            and len(topics) == 3
            and topics[0] == ERC_20_TRANSFER_TOPIC
            and topics[1] == safe_topic
        ):
            return UnminedDepositClassification(
                UnminedDepositCase.undetermined,
                f"{consumer_hash} took nonce {deposit.nonce} and moved {trade.reserve_currency.token_symbol} out of the Safe",
            )

    # Corroboration only: every recorded reserve, and this allocation, must still be there.
    usdc = fetch_erc20_details(web3, trade.reserve_currency.address, chain_id=trade.reserve_currency.chain_id)
    reserves = state.portfolio.get_reserve_position(trade.reserve_currency).quantity
    required = reserves + trade.planned_reserve
    safe_balance = usdc.fetch_balance_of(safe_address, block_identifier="latest")
    if safe_balance < required:
        return UnminedDepositClassification(
            UnminedDepositCase.undetermined,
            f"Safe holds {safe_balance}, below the {required} the state expects",
        )

    return UnminedDepositClassification(
        UnminedDepositCase.void,
        f"Nonce {deposit.nonce} was consumed by {consumer_hash} in block {consumer['blockNumber']}, which moved nothing out of the Safe",
    )


def recover_unmined_lighter_deposits(
    state: State,
    web3: Web3,
    *,
    safe_address: HexAddress,
    interactive: bool,
    save: Callable[[], None],
) -> list[TradeExecution]:
    """Resolve Lighter deposits that were broadcast but never mined.

    Such a trade blocks ``start``, and the generic repair paths refuse it because they
    cannot tell where the funds are. Run from ``repair`` only, because it can send
    transactions. A rebroadcast deposit is completed by the following
    :py:func:`reconcile_verified_lighter_transfers`.

    Pool trades are out of scope. They are funded differently, so
    ``planned_reserve`` is not what they took from the reserves.

    :param state:
        State to repair.
    :param web3:
        Ethereum connection.
    :param safe_address:
        The vault's Safe.
    :param interactive:
        Ask before each action.
    :param save:
        Persist the state, called after each resolved trade.
    :return:
        Trades acted on.
    :raise RepairAborted:
        The operator declined.
    """
    if web3.eth.chain_id not in UNMINED_DEPOSIT_CHAINS:
        return []

    recovered = []
    for position in state.portfolio.get_open_and_frozen_positions():
        for trade in list(position.trades.values()):
            if not (
                trade.is_external_account_transfer()
                and trade.is_buy()
                and trade.pair.get_exchange_account_protocol() == LIGHTER_PROTOCOL
                and trade.get_status() == TradeStatus.broadcasted
            ):
                continue

            try:
                classification = classify_unmined_lighter_deposit(web3, state, trade, safe_address)
            except Exception as error:
                # Any read failure must leave the trade alone, not pass as evidence.
                logger.error("Lighter deposit #%d stays pending, could not read the chain: %s", trade.trade_id, type(error).__name__)
                continue

            if classification is None:
                continue

            if classification.case == UnminedDepositCase.undetermined:
                logger.error("Lighter deposit #%d stays pending: %s", trade.trade_id, classification.reason)
                continue

            if classification.case == UnminedDepositCase.sendable:
                if _rebroadcast_unmined_deposit(web3, trade, classification, interactive=interactive):
                    recovered.append(trade)
                    save()
                continue

            print(f"Lighter deposit #{trade.trade_id} can never be mined: {classification.reason}")
            if interactive and input(f"Return {trade.planned_reserve} to the reserves [y/n]? ").lower() != "y":
                raise RepairAborted()
            repair_unmined_exchange_deposit(state.portfolio, trade, f"Never mined. {classification.reason}")
            logger.info("Repaired unmined Lighter deposit #%d: %s", trade.trade_id, classification.reason)
            recovered.append(trade)
            save()
    return recovered


def _rebroadcast_unmined_deposit(
    web3: Web3,
    trade: TradeExecution,
    classification: UnminedDepositClassification,
    *,
    interactive: bool,
) -> bool:
    """Send a deposit's stored transactions unchanged, and record their receipts.

    Nothing is re-signed, as the trade's hashes are the evidence for reconciliation.

    :return:
        True if the transactions were sent.
    """
    sender = trade.blockchain_transactions[0].from_address
    for tx in classification.unmined:
        if HexBytes(Web3.keccak(hexstr=tx.signed_bytes)) != HexBytes(tx.tx_hash):
            logger.error("Lighter deposit #%d stays pending: stored bytes do not match %s", trade.trade_id, tx.tx_hash)
            return False

    required = sum(tx.details["gas"] * tx.details["maxFeePerGas"] for tx in classification.unmined)
    balance = web3.eth.get_balance(sender)
    if balance < required:
        logger.error(
            "Lighter deposit #%d stays pending: %s needs %s ETH more up front. Fund it with an incoming "
            "transfer, because anything sent from this wallet consumes the deposit's nonce.",
            trade.trade_id,
            sender,
            Decimal(required - balance) / 10**18,
        )
        return False

    print(f"Lighter deposit #{trade.trade_id} was never mined and can still be sent: {classification.reason}")
    if interactive and input("Rebroadcast its stored transactions [y/n]? ").lower() != "y":
        raise RepairAborted()

    signed = [
        SignedTransactionWithNonce(
            rawTransaction=HexBytes(tx.signed_bytes),
            hash=HexBytes(tx.tx_hash),
            r=0,
            s=0,
            v=0,
            address=tx.from_address,
            nonce=tx.nonce,
            source=tx.details,
        )
        for tx in classification.unmined
    ]
    receipts = wait_and_broadcast_multiple_nodes(
        web3,
        signed,
        max_timeout=UNMINED_DEPOSIT_BROADCAST_TIMEOUT,
        node_switch_timeout=datetime.timedelta(minutes=1),
    )
    now = native_datetime_utc_now()
    for tx in classification.unmined:
        receipt = receipts[HexBytes(tx.tx_hash)]
        tx.set_confirmation_information(
            ts=now,
            block_number=receipt["blockNumber"],
            block_hash=HexBytes(receipt["blockHash"]).to_0x_hex(),
            realised_gas_units_consumed=receipt["gasUsed"],
            realised_gas_price=receipt.get("effectiveGasPrice", 0),
            status=receipt["status"] == 1,
        )
    return True


def _fetch_transaction_receipts(
    web3: Web3,
    trade: TradeExecution,
) -> list[dict] | None:
    """Read all recorded receipts, returning no result before they are complete."""
    if not trade.blockchain_transactions:
        metadata = trade.other_data or {}
        transaction_hash = metadata.get(
            "lighter_deposit_tx_hash" if trade.is_buy() else "lighter_claim_tx_hash",
        )
        if not transaction_hash:
            return None
        try:
            return [web3.eth.get_transaction_receipt(transaction_hash)]
        except TransactionNotFound:
            return None
    receipts = []
    for transaction in trade.blockchain_transactions:
        if not transaction.tx_hash:
            return None
        try:
            receipts.append(web3.eth.get_transaction_receipt(transaction.tx_hash))
        except TransactionNotFound:
            return None
    return receipts
