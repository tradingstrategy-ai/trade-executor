"""Read-only verification for completed Lighter custody transfers."""

from dataclasses import dataclass
from decimal import Decimal

from eth_defi.compat import native_datetime_utc_now
from eth_defi.lighter.api import wait_for_lighter_collateral
from eth_defi.lighter.session import LighterSession, create_lighter_session
from eth_defi.lighter.valuation import fetch_lighter_total_equity
from eth_defi.token import fetch_erc20_details
from web3 import Web3
from web3.exceptions import TransactionNotFound

from tradeexecutor.exchange_account.lighter import LIGHTER_PROTOCOL
from tradeexecutor.exchange_account.state import complete_exchange_account_transfer
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution


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


def verify_lighter_transfer(
    web3: Web3,
    trade: TradeExecution,
    session: LighterSession,
    receipts: list[dict] | None = None,
    deposit_wait_timeout: int | None = None,
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
    :return:
        Verified accounting values, or ``None`` while evidence is incomplete.
    """
    if trade.pair.get_exchange_account_protocol() != LIGHTER_PROTOCOL:
        raise ValueError("Expected a Lighter exchange-account transfer")
    transaction_receipts = receipts or _fetch_transaction_receipts(web3, trade)
    if transaction_receipts is None:
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


def reconcile_verified_lighter_transfers(
    state: State,
    web3: Web3,
    *,
    mutate: bool,
) -> list[TradeExecution]:
    """Inspect or apply verified unfinished Lighter transfers.

    :param state:
        State containing the transfers to inspect.
    :param web3:
        Chain reader used for the verification evidence.
    :param mutate:
        Apply verified results when ``True``; otherwise only return them.
    :return:
        Transfers whose evidence is complete.
    """
    pending_transfers = [
        (position, trade)
        for position in state.portfolio.get_open_and_frozen_positions()
        for trade in position.trades.values()
        if trade.is_external_account_transfer_pending()
        and trade.pair.get_exchange_account_protocol() == LIGHTER_PROTOCOL
    ]
    if not pending_transfers:
        return []

    session = create_lighter_session()
    try:
        verified_trades = []
        for position, trade in pending_transfers:
            verification = verify_lighter_transfer(web3, trade, session)
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
