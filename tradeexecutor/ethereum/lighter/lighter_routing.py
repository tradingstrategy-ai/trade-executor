"""Route Lighter exchange-account cash transfers through Lagoon."""

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from decimal import Decimal

from eth_defi.abi import get_deployed_contract
from eth_defi.compat import native_datetime_utc_now
from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonVault
from eth_defi.lighter.constants import LIGHTER_L1_CONTRACT
from eth_defi.lighter.lagoon import (
    LIGHTER_L1_WITHDRAW_GAS_LIMIT,
    LIGHTER_ROUTE_PERP,
    fetch_lighter_withdrawal_baseline,
    wait_for_lighter_withdrawal,
)
from eth_defi.lighter.session import LighterSession
from eth_defi.lighter.transfer import LighterTransferError
from eth_defi.lighter.valuation import fetch_lighter_total_equity
from eth_defi.provider.receipt import wait_for_transaction_receipt_robust
from eth_defi.token import TokenDetails, fetch_erc20_details
from hexbytes import HexBytes
from web3 import Web3
from web3.contract.contract import ContractFunction
from web3.exceptions import TransactionNotFound

from tradeexecutor.exchange_account.lighter import create_executor_lighter_session, get_lighter_master_account_index, is_lighter_sub_account_pair
from tradeexecutor.exchange_account.lighter_operator import (
    LighterOperatorRecord,
    request_lighter_funding_transfer,
    request_lighter_withdrawal,
    wait_for_lighter_funding_transfer,
    wait_for_lighter_withdrawal_claimable,
)
from tradeexecutor.exchange_account.state import (
    complete_exchange_account_funding,
    complete_exchange_account_transfer,
)
from tradeexecutor.ethereum.lighter.transfer_verification import (
    LighterTransferVerificationError,
    get_lighter_funding_evidence,
    get_lighter_relayer_payout,
    verify_lighter_funding,
    verify_lighter_transfer,
)
from tradeexecutor.ethereum.lagoon.tx import LagoonTransactionBuilder
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeFlag
from tradeexecutor.ethereum.swap import report_failure
from tradeexecutor.strategy.routing import RoutingModel, RoutingState
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


logger = logging.getLogger(__name__)

#: Fallback gas limit for the ERC-20 approval wrapped by Lagoon.
LIGHTER_APPROVAL_GAS_LIMIT = 500_000
#: Whole transaction gas limit for a Lighter deposit wrapped by Lagoon.
#:
#: A deposit cannot be estimated, because it is signed before its approval is
#: mined. LIX mainnet deposits used up to 223,281 gas, the highest being the first
#: to a new Lighter account. A node refuses a transaction unless the hot wallet
#: holds the limit times the maximum fee, so a generous limit stops a deposit the
#: wallet could afford.
LIGHTER_DEPOSIT_GAS_LIMIT = 400_000
#: Fallback gas limit for a Lighter pending withdrawal claim wrapped by Lagoon.
LIGHTER_WITHDRAWAL_CLAIM_GAS_LIMIT = 1_000_000
#: Margin on a gas estimate, for state that changes before the transaction is mined.
LIGHTER_GAS_ESTIMATE_MARGIN = 1.5


def estimate_lighter_gas_limit(
    tx_builder: LagoonTransactionBuilder,
    function: ContractFunction,
    fallback_gas_limit: int,
) -> int:
    """Return the whole transaction's gas limit for a Safe-wrapped Lighter call.

    A node refuses a transaction unless the hot wallet holds the limit times the
    maximum fee, so a fixed limit can stop a transaction the wallet could afford.

    The estimate is not capped, because signing below it could only produce a
    reverting transaction. An estimate can fail for reasons that do not stop the
    transaction, such as an RPC error, so the fixed limit remains the fallback and
    a failed estimate cannot block a custody movement.

    :param tx_builder:
        Lagoon transaction builder of the Safe making the call.
    :param function:
        The call the Safe will make.
    :param fallback_gas_limit:
        Gas limit of the wrapped call if estimation fails.
    :return:
        Gas limit to pass as ``total_gas_limit``.
    """
    try:
        return math.ceil(tx_builder.estimate_gas(function) * LIGHTER_GAS_ESTIMATE_MARGIN)
    except Exception as error:
        logger.warning(
            "Could not estimate gas for Lighter %s(), using the fixed limit: %s",
            function.fn_name,
            type(error).__name__,
        )
        return fallback_gas_limit + tx_builder.extra_gnosis_gas
#: Default maximum wait for a secure or L1 Lighter withdrawal to arrive or become claimable.
#:
#: A live sub-account L1 withdrawal took about 29 minutes, so leave margin.
DEFAULT_LIGHTER_WITHDRAWAL_TIMEOUT_SECONDS = 60 * 60
#: Public Lighter deposit observation timeout.
DEFAULT_LIGHTER_DEPOSIT_TIMEOUT_SECONDS = 15 * 60
#: Default maximum wait for a transfer between accounts to appear in history.
DEFAULT_LIGHTER_TRANSFER_TIMEOUT_SECONDS = 5 * 60


@dataclass(frozen=True, slots=True)
class LighterKeyManagement:
    """The Lighter API keys the executor signs with, and how long it waits for Lighter.

    Parsed by the CLI from owner-only operator records. The master key signs
    secure withdrawals, transfers sent from the master and public-pool mints,
    burns and withdrawals; a sub-account key signs transfers sent from that
    sub-account. Safe-side deposits and claims are guarded Ethereum
    transactions and need no key.
    """

    #: Owner-only master API signer for secure withdrawals and transfers sent from the master.
    operator_record: LighterOperatorRecord | None = None

    #: Signers of sub-accounts the executor may transfer from, keyed by account index.
    #:
    #: Empty by default: the executor then moves capital between sub-accounts
    #: through the Safe, and the sub-account keys stay with the external traders.
    sub_account_operator_records: dict[int, LighterOperatorRecord] = field(default_factory=dict)

    #: Maximum wait for a secure or L1 withdrawal to arrive or become claimable.
    withdrawal_timeout: int = DEFAULT_LIGHTER_WITHDRAWAL_TIMEOUT_SECONDS

    #: Maximum wait for a transfer between accounts to appear in history.
    transfer_timeout: int = DEFAULT_LIGHTER_TRANSFER_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        """Reject an invalid execution timeout at the CLI configuration boundary."""
        if type(self.withdrawal_timeout) is not int or self.withdrawal_timeout <= 0:
            raise ValueError("Lighter withdrawal timeout must be a positive number of seconds")
        if type(self.transfer_timeout) is not int or self.transfer_timeout <= 0:
            raise ValueError("Lighter transfer timeout must be a positive number of seconds")

    def get_operator_record(self, account_index: int) -> LighterOperatorRecord | None:
        """Return the signer of one Lighter account, if the executor holds it.

        :param account_index:
            Master or sub-account index.
        :return:
            The account's operator record, or ``None``.
        """
        if self.operator_record is not None and self.operator_record.account_index == account_index:
            return self.operator_record
        return self.sub_account_operator_records.get(account_index)


def ensure_lighter_l1_withdrawal_mined(web3: Web3, trade: TradeExecution) -> None:
    """Broadcast a checkpointed key-free L1 withdrawal request if the chain has not seen it, and wait for it.

    The request is signed and checkpointed on the trade before it is sent, so
    a crash in between leaves a request the chain never saw. Sending the
    same signed bytes again is idempotent: a nonce already used is rejected,
    a mined hash returns its receipt.

    :param web3:
        Ethereum connection.
    :param trade:
        Sub-account withdrawal trade carrying ``lighter_withdrawal_request_id``.
    :raise RuntimeError:
        If the request reverted, or was never broadcast and the signed request is not saved.
    """
    tx_hash = trade.other_data["lighter_withdrawal_request_id"]
    try:
        receipt = web3.eth.get_transaction_receipt(tx_hash)
    except TransactionNotFound:
        receipt = None
    if receipt is None:
        signed = trade.other_data.get("lighter_withdrawal_request_signed_tx")
        if not signed:
            raise RuntimeError(f"Lighter L1 withdrawal request {tx_hash} is not on chain and its signed request was not saved; check the request manually")
        logger.info("Broadcasting the saved Lighter L1 withdrawal request %s", tx_hash)
        web3.eth.send_raw_transaction(HexBytes(signed))
        receipt = wait_for_transaction_receipt_robust(web3, tx_hash)
    if receipt["status"] != 1:
        raise RuntimeError(f"Lighter L1 withdrawal request {tx_hash} reverted")
    # A request saved by an earlier release has no baseline block.
    trade.other_data.setdefault("lighter_withdrawal_from_block", receipt["blockNumber"])
    logger.info("Lighter L1 withdrawal request for account %s mined: %s", trade.pair.get_exchange_account_id(), tx_hash)


class LighterRoutingState(RoutingState):
    """Per-cycle objects needed by Lighter routing."""

    def __init__(
        self,
        universe: TradingStrategyUniverse,
        execution_details: dict,
    ):
        """Initialise routing objects for a single execution cycle.

        :param universe:
            Trading universe containing the synthetic Lighter pair.
        :param execution_details:
            Transaction builder and, during execution, Lagoon vault objects.
        """
        super().__init__(universe)
        self.tx_builder = execution_details["tx_builder"]
        self.token_cache = execution_details.get("token_cache")
        # Account-check and repair commands initialise routing without a vault.
        # Transfer preparation requires it and is only reached during execution.
        self.vault = execution_details.get("vault")


class LighterRouting(RoutingModel):
    """Prepare and settle Safe-owned Lighter cash transfers."""

    def __init__(
        self,
        reserve_token_address: str,
        *,
        lighter_contract: str = LIGHTER_L1_CONTRACT,
        key_management: LighterKeyManagement | None = None,
        session: LighterSession | None = None,
    ):
        """Create the Lighter exchange-account router.

        :param reserve_token_address:
            USDC reserve token address used by the Safe and Lighter.
        :param lighter_contract:
            Lighter L1 contract receiving Safe deposits and claims.
        :param key_management:
            CLI-parsed Lighter API keys and wait settings.
        :param session:
            Optional public Lighter API client, primarily for tests.
        """
        super().__init__({}, reserve_token_address.lower())
        self.lighter_contract = Web3.to_checksum_address(lighter_contract)
        self.key_management = key_management or LighterKeyManagement()
        self.lighter_session = session if session is not None else create_executor_lighter_session()

    def create_routing_state(
        self,
        universe: TradingStrategyUniverse,
        execution_details: object,
    ) -> LighterRoutingState:
        """Create state for one Lighter execution cycle.

        :param universe:
            Trading universe containing the Lighter exchange-account pair.
        :param execution_details:
            Transaction builder and optional Lagoon vault execution details.
        :return:
            Per-cycle Lighter routing state.
        """
        assert isinstance(execution_details, dict)
        return LighterRoutingState(universe, execution_details)

    def setup_trades(
        self,
        state: State,
        routing_state: LighterRoutingState,
        trades: list[TradeExecution],
        check_balances: bool = False,
        rebroadcast: bool = False,
    ) -> None:
        """Prepare Safe deposits and withdrawals, and submit transfers between accounts, for Lighter trades.

        :param state:
            Strategy state that owns the transfer trades.
        :param routing_state:
            Per-cycle Lighter transaction and API dependencies.
        :param trades:
            Planned Lighter exchange-account transfer trades.
        :param check_balances:
            Compatibility flag required by the routing-model interface.
        :param rebroadcast:
            Compatibility flag required by the routing-model interface.
        """
        for trade in trades:
            assert trade.pair.is_exchange_account()
            if trade.is_internal_account_transfer():
                self._execute_funding(state, trade)
                continue
            assert trade.is_external_account_transfer()
            if trade.blockchain_transactions:
                continue
            if trade.is_buy():
                if is_lighter_sub_account_pair(trade.pair):
                    raise RuntimeError(
                        "An L1 deposit always credits the Lighter master account; "
                        "fund a sub-account with an internal account transfer"
                    )
                trade.set_blockchain_transactions(
                    self._prepare_deposit(routing_state, trade)
                )
            else:
                if is_lighter_sub_account_pair(trade.pair):
                    self._prepare_sub_account_withdrawal(routing_state, trade)
                else:
                    self._prepare_withdrawal(routing_state, trade)

    def settle_trade(
        self,
        web3: Web3,
        state: State,
        trade: TradeExecution,
        receipts: dict,
        stop_on_execution_failure: bool = False,
    ) -> None:
        """Verify Lighter transfer receipts and complete state accounting.

        :param web3:
            Connected chain reader used for the Safe-side transfer result.
        :param state:
            Strategy state updated when the transfer succeeds or fails.
        :param trade:
            Lighter exchange-account transfer being settled.
        :param receipts:
            Transaction receipts collected by the execution pipeline.
        :param stop_on_execution_failure:
            Propagate a failed transaction according to the execution policy.
        """
        if trade.is_internal_account_transfer():
            # Off-chain transfer: setup_trades() stored the transfer-history
            # evidence. There are no Ethereum receipts.
            funding = verify_lighter_funding(trade)
            if funding is None:
                raise RuntimeError("Lighter account transfer has no transfer-history evidence")
            complete_exchange_account_funding(
                state=state,
                trade=trade,
                executed_amount=funding.amount,
                fee=funding.fee,
                executed_at=native_datetime_utc_now(),
            )
            return

        transaction_receipts = [
            _find_receipt(receipts, transaction)
            for transaction in trade.blockchain_transactions
        ]
        if any(receipt.get("status") != 1 for receipt in transaction_receipts) and get_lighter_relayer_payout(web3, trade) is None:
            # A reverted claim is only a failure if the money has not arrived:
            # Lighter's relayer claiming first also reverts it.
            report_failure(
                native_datetime_utc_now(),
                state,
                trade,
                stop_on_execution_failure,
            )
            return

        try:
            verification = verify_lighter_transfer(
                web3,
                trade,
                self.lighter_session,
                receipts=transaction_receipts,
                deposit_wait_timeout=DEFAULT_LIGHTER_DEPOSIT_TIMEOUT_SECONDS,
            )
        except LighterTransferVerificationError as error:
            raise RuntimeError(str(error)) from error
        if verification is None:
            raise RuntimeError("Lighter transfer receipts are mined but external settlement is incomplete")
        trade.other_data.update(verification.metadata_updates)

        position = state.portfolio.find_position_for_trade(trade)
        complete_exchange_account_transfer(
            state=state,
            position=position,
            trade=trade,
            executed_at=native_datetime_utc_now(),
            executed_amount=verification.executed_amount,
            executed_reserve=verification.executed_reserve,
        )

    def needs_sequential_trade_execution(self, trades: list[TradeExecution]) -> bool:
        """Require one transfer at a time for Safe and Lighter consistency.

        :param trades:
            Lighter custody transfers considered for the current execution batch.
        :return:
            Always ``True`` because Lighter withdrawal requests are stateful.
        """
        return True

    def get_sequential_trade_execution_reason(self, trades: list[TradeExecution]) -> str:
        """Explain why Lighter transfers execute sequentially.

        :param trades:
            Lighter custody transfers considered for the current execution batch.
        :return:
            Operator-visible explanation for sequential execution.
        """
        return "Lighter custody transfers require sequential Safe and API settlement"

    def _prepare_deposit(
        self,
        routing_state: LighterRoutingState,
        trade: TradeExecution,
    ) -> list[BlockchainTransaction]:
        """Prepare the approval and deposit calls before broadcast."""
        web3 = routing_state.tx_builder.web3
        usdc = self._fetch_reserve_token(routing_state, trade)
        zk = get_deployed_contract(web3, "lighter/ZkLighter.json", self.lighter_contract)
        amount_raw = usdc.convert_to_raw(trade.planned_reserve)
        asset_index = zk.functions.USDC_ASSET_INDEX().call()
        lighter_account_index = trade.pair.get_exchange_account_id()
        if lighter_account_index is None:
            raise RuntimeError("Lighter exchange-account pair has no account index")
        vault = self._get_lagoon_vault(routing_state)
        collateral_before = fetch_lighter_total_equity(
            self.lighter_session,
            int(lighter_account_index),
        ).collateral
        trade.other_data["lighter_collateral_before_usdc"] = str(collateral_before)
        approve_call = usdc.contract.functions.approve(self.lighter_contract, amount_raw)
        approve_tx = routing_state.tx_builder.sign_transaction(
            usdc.contract,
            approve_call,
            total_gas_limit=estimate_lighter_gas_limit(routing_state.tx_builder, approve_call, LIGHTER_APPROVAL_GAS_LIMIT),
            asset_deltas=[],
            notes=trade.notes,
        )
        deposit_tx = routing_state.tx_builder.sign_transaction(
            zk,
            zk.functions.deposit(
                vault.safe_address,
                asset_index,
                0,
                amount_raw,
            ),
            total_gas_limit=LIGHTER_DEPOSIT_GAS_LIMIT,
            asset_deltas=[],
            notes=trade.notes,
        )
        trade.other_data["lighter_deposit_amount_usdc"] = str(trade.planned_reserve)
        return [approve_tx, deposit_tx]

    def _require_master_operator(self, trade: TradeExecution) -> LighterOperatorRecord:
        """Return the master signer and check it owns the trade's account tree.

        :param trade:
            Lighter transfer about to be executed.
        :return:
            The master account operator record.
        """
        operator = self.key_management.operator_record
        if operator is None:
            raise RuntimeError(
                "Automatic Lighter transfers require an operator record passed by the CLI"
            )
        master_index = get_lighter_master_account_index(trade.pair)
        if operator.account_index != master_index:
            raise RuntimeError(
                f"Lighter operator record signs for account {operator.account_index}, "
                f"but the trade belongs to master account {master_index}"
            )
        return operator

    def _execute_funding(self, state: State, trade: TradeExecution) -> None:
        """Submit and confirm a transfer between two accounts of the Safe.

        The source is the master or a sub-account whose signer the executor
        holds; the destination is any other account of the same master.

        The transfer hash is recorded and checkpointed before the wait, with
        the trade marked broadcasted, so an interrupted run blocks start-up
        and is reconciled from transfer history instead of being resubmitted.
        A definitive Lighter rejection moves no funds: the trade is marked
        failed with ``outcome == "transfer_rejected"`` so it does not block
        the executor. A trade that was started but has no recorded hash may
        have been submitted, so reconciliation keeps it pending for a manual
        check of the sender's transfer history.

        :param state:
            Strategy state owning the trade.
        :param trade:
            Funding trade on the destination position.
        """
        # The master record proves the executor is configured for this account
        # tree even when a sub-account's own key signs the transfer.
        self._require_master_operator(trade)
        source = int(trade.other_data.get("funding_source_account_id", -1))
        destination = int(trade.pair.get_exchange_account_id())
        if source == destination:
            raise RuntimeError("Lighter account transfer source and destination must differ")
        operator = self.key_management.get_operator_record(source)
        if operator is None:
            raise RuntimeError(
                f"The executor holds no signer for Lighter account {source}; pass its record with "
                "LIGHTER_OPERATOR_RECORD_FILE for the master or LIGHTER_SUB_ACCOUNT_OPERATOR_RECORD_FILES for a sub-account"
            )
        amount = trade.planned_quantity
        tx_hash = trade.other_data.get("lighter_transfer_tx_hash")
        if not tx_hash:
            # Persist the started status first: reconciliation expires a trade
            # still planned on disk as never submitted, which must not happen
            # to a transfer Lighter may have accepted.
            self.checkpoint_state()
            try:
                request = asyncio.run(request_lighter_funding_transfer(operator, destination, amount))
            except LighterTransferError:
                state.mark_trade_failed(native_datetime_utc_now(), trade)
                trade.other_data["outcome"] = "transfer_rejected"
                self.checkpoint_state()
                raise
            tx_hash = request.tx_hash
            trade.other_data["lighter_transfer_tx_hash"] = tx_hash
            trade.mark_broadcasted(native_datetime_utc_now())
            self.checkpoint_state()

        record = asyncio.run(
            wait_for_lighter_funding_transfer(operator, str(tx_hash), self.key_management.transfer_timeout)
        )
        trade.other_data.update(get_lighter_funding_evidence(record))

    def _prepare_sub_account_withdrawal(
        self,
        routing_state: LighterRoutingState,
        trade: TradeExecution,
    ) -> None:
        """Withdraw from a sub-account to the Safe without its API key.

        The Safe module calls ``ZkLighter.withdraw(subAccountIndex, ...)``.
        Lighter binds the request to the Safe's master account and adds the
        proceeds to the Safe's onchain pending balance once the batch is
        proven. The executor waits for that balance and prepares the same
        Safe claim as a master withdrawal, so verification and accounting
        are shared. Lighter's relayer may claim first, in which case there is
        nothing left to send.

        The signed request and its hash are checkpointed before broadcast. A
        restart finds the pending trade, broadcasts the same signed request
        again if the chain has not seen it, and recovers it instead of
        sending a second request.

        :param routing_state:
            Per-cycle Lighter routing dependencies.
        :param trade:
            Withdrawal trade on a sub-account position.
        """
        vault = self._get_lagoon_vault(routing_state)
        web3 = routing_state.tx_builder.web3
        usdc = self._fetch_reserve_token(routing_state, trade)
        amount = abs(trade.planned_quantity)
        zk = get_deployed_contract(web3, "lighter/ZkLighter.json", self.lighter_contract)
        if not trade.other_data.get("lighter_withdrawal_request_id"):
            from_block, pending_before = fetch_lighter_withdrawal_baseline(web3, owner=vault.safe_address, usdc=usdc, zk_lighter=self.lighter_contract)
            # USDC has a Lighter tick size of one raw unit, so base amount equals raw USDC.
            withdraw_call = zk.functions.withdraw(
                int(trade.pair.get_exchange_account_id()),
                zk.functions.USDC_ASSET_INDEX().call(),
                LIGHTER_ROUTE_PERP,
                usdc.convert_to_raw(amount),
            )
            request_tx = routing_state.tx_builder.sign_transaction(
                zk,
                withdraw_call,
                total_gas_limit=estimate_lighter_gas_limit(routing_state.tx_builder, withdraw_call, LIGHTER_L1_WITHDRAW_GAS_LIMIT),
                asset_deltas=[],
                notes=trade.notes,
            )
            trade.other_data.update({
                "lighter_withdrawal_method": "l1_withdraw",
                "lighter_withdrawal_request_id": request_tx.tx_hash,
                "lighter_withdrawal_request_signed_tx": HexBytes(request_tx.signed_bytes).to_0x_hex(),
                "lighter_withdrawal_requested_at": int(time.time()),
                "lighter_pending_balance_before_usdc": str(pending_before),
                "lighter_withdrawal_from_block": from_block,
                "lighter_safe_balance_before_claim": str(self._get_safe_balance(routing_state, trade)),
                "lighter_safe_address": vault.safe_address,
            })
            self.checkpoint_state()
        ensure_lighter_l1_withdrawal_mined(web3, trade)

        paid = wait_for_lighter_withdrawal(
            web3,
            owner=vault.safe_address,
            usdc=usdc,
            amount=amount,
            pending_before=Decimal(trade.other_data["lighter_pending_balance_before_usdc"]),
            from_block=int(trade.other_data["lighter_withdrawal_from_block"]),
            timeout=self.key_management.withdrawal_timeout,
            zk_lighter=self.lighter_contract,
        )
        self._finish_withdrawal(routing_state, trade, amount, paid)

    def _finish_withdrawal(
        self,
        routing_state: LighterRoutingState,
        trade: TradeExecution,
        amount: Decimal,
        paid: bool,
    ) -> None:
        """Prepare the Safe claim, or record that Lighter's relayer has already paid.

        :param routing_state:
            Per-cycle Lighter routing dependencies.
        :param trade:
            Withdrawal trade whose funds have arrived or are claimable.
        :param amount:
            USDC to claim or already paid.
        :param paid:
            ``True`` if the relayer has paid, so a claim would revert.
        """
        trade.other_data["lighter_claimable_usdc"] = str(amount)
        if paid:
            trade.other_data["paid_without_transaction"] = True
            trade.mark_broadcasted(native_datetime_utc_now())
        else:
            trade.set_blockchain_transactions([self._prepare_claim(routing_state, trade, amount)])
        self.checkpoint_state()

    def _prepare_withdrawal(
        self,
        routing_state: LighterRoutingState,
        trade: TradeExecution,
    ) -> None:
        """Request, wait for, and prepare a Lighter claim transaction."""
        vault = self._get_lagoon_vault(routing_state)
        operator = self._require_master_operator(trade)
        request_id = trade.other_data.get("lighter_withdrawal_request_id")
        requested_at = int(trade.other_data.get("lighter_withdrawal_requested_at", 0))
        if not request_id:
            requested_at = int(time.time())
            from_block, pending_before = fetch_lighter_withdrawal_baseline(
                routing_state.tx_builder.web3,
                owner=vault.safe_address,
                usdc=self._fetch_reserve_token(routing_state, trade),
                zk_lighter=self.lighter_contract,
            )
            trade.other_data["lighter_withdrawal_from_block"] = from_block
            trade.other_data["lighter_pending_balance_before_usdc"] = str(pending_before)
            trade.other_data["lighter_safe_address"] = vault.safe_address
            # Persist the started status first, so a process failure after the
            # request leaves the transfer unclean for explicit recovery instead
            # of expiring it as never submitted; start-up never submits a
            # duplicate request.
            self.checkpoint_state()
            request_id = asyncio.run(
                request_lighter_withdrawal(operator, abs(trade.planned_quantity))
            )
            trade.other_data["lighter_withdrawal_request_id"] = request_id
            trade.other_data["lighter_withdrawal_requested_at"] = requested_at
            trade.other_data["lighter_safe_balance_before_claim"] = str(
                self._get_safe_balance(routing_state, trade)
            )
            # Keep the trade started until the Safe claim is ready. The normal
            # transaction broadcaster performs the single state transition to
            # broadcasted after it receives the signed claim transaction.
            self.checkpoint_state()

        claimable = asyncio.run(
            wait_for_lighter_withdrawal_claimable(
                operator,
                abs(trade.planned_quantity),
                self.key_management.withdrawal_timeout,
                requested_at,
                str(request_id),
            )
        )
        paid = get_lighter_relayer_payout(routing_state.tx_builder.web3, trade) is not None
        self._finish_withdrawal(routing_state, trade, claimable, paid)

    def _prepare_claim(
        self,
        routing_state: LighterRoutingState,
        trade: TradeExecution,
        claimable: Decimal,
    ) -> BlockchainTransaction:
        """Prepare a Safe-wrapped Lighter pending withdrawal claim."""
        web3 = routing_state.tx_builder.web3
        usdc = self._fetch_reserve_token(routing_state, trade)
        zk = get_deployed_contract(web3, "lighter/ZkLighter.json", self.lighter_contract)
        amount_raw = usdc.convert_to_raw(claimable)
        vault = self._get_lagoon_vault(routing_state)
        claim_call = zk.functions.withdrawPendingBalance(
            vault.safe_address,
            zk.functions.USDC_ASSET_INDEX().call(),
            amount_raw,
        )
        return routing_state.tx_builder.sign_transaction(
            zk,
            claim_call,
            total_gas_limit=estimate_lighter_gas_limit(routing_state.tx_builder, claim_call, LIGHTER_WITHDRAWAL_CLAIM_GAS_LIMIT),
            asset_deltas=[],
            notes=trade.notes,
        )

    def _get_safe_balance(
        self,
        routing_state: LighterRoutingState,
        trade: TradeExecution,
    ) -> Decimal:
        """Read Safe USDC before a delayed withdrawal claim.

        :param routing_state:
            Per-cycle Lighter routing dependencies.
        :param trade:
            Withdrawal trade whose reserve token is read from the Safe.
        :return:
            Current Safe reserve balance in human-readable USDC.
        """
        usdc = self._fetch_reserve_token(routing_state, trade)
        vault = self._get_lagoon_vault(routing_state)
        return Decimal(str(usdc.fetch_balance_of(vault.safe_address)))

    @staticmethod
    def _fetch_reserve_token(routing_state: LighterRoutingState, trade: TradeExecution) -> TokenDetails:
        """Return the USDC reserve token moved by a Lighter transfer."""
        return fetch_erc20_details(
            routing_state.tx_builder.web3,
            trade.reserve_currency.address,
            cache=routing_state.token_cache,
            chain_id=trade.reserve_currency.chain_id,
        )

    @staticmethod
    def _get_lagoon_vault(routing_state: LighterRoutingState) -> LagoonVault:
        """Return the Lagoon vault required to prepare a custody transfer.

        :param routing_state:
            Per-cycle routing state containing optional Lagoon execution data.
        :return:
            Lagoon vault used as the Safe custody source or destination.
        """
        if routing_state.vault is None:
            raise RuntimeError(
                "Lighter custody transfers require Lagoon vault execution details; "
                "run through Lagoon execution rather than account inspection",
            )
        return routing_state.vault


def _find_receipt(receipts: dict, transaction: BlockchainTransaction) -> dict:
    """Find a transaction receipt regardless of hash key representation."""
    expected = _normalise_transaction_hash(transaction.tx_hash)
    for tx_hash, receipt in receipts.items():
        if _normalise_transaction_hash(tx_hash) == expected:
            return receipt
    raise KeyError(f"No receipt for Lighter transaction {expected}")


def _normalise_transaction_hash(value: object) -> str:
    """Normalise transaction hashes from Web3 and state records."""
    if isinstance(value, (bytes, bytearray, HexBytes)):
        return "0x" + bytes(value).hex().lower()
    text = str(value).lower()
    return text if text.startswith("0x") else "0x" + text
