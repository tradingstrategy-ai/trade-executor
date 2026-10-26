"""Route Lighter exchange-account cash transfers through Lagoon."""

import asyncio
import logging
import time
from dataclasses import dataclass
from decimal import Decimal

from eth_defi.abi import get_deployed_contract
from eth_defi.compat import native_datetime_utc_now
from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonVault
from eth_defi.lighter.constants import LIGHTER_L1_CONTRACT
from eth_defi.lighter.lagoon import (
    LIGHTER_L1_WITHDRAW_GAS_LIMIT,
    LIGHTER_ROUTE_PERP,
    fetch_lighter_pending_balance,
    wait_for_lighter_pending_balance,
)
from eth_defi.lighter.session import LighterSession, create_lighter_session
from eth_defi.lighter.transfer import LighterTransferError
from eth_defi.lighter.valuation import fetch_lighter_total_equity
from eth_defi.provider.receipt import wait_for_transaction_receipt_robust
from eth_defi.token import TokenDetails, fetch_erc20_details
from hexbytes import HexBytes
from web3 import Web3

from tradeexecutor.exchange_account.lighter import get_lighter_master_account_index, is_lighter_sub_account_pair
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
    verify_lighter_funding,
    verify_lighter_transfer,
)
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeFlag
from tradeexecutor.ethereum.swap import report_failure
from tradeexecutor.strategy.routing import RoutingModel, RoutingState
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


logger = logging.getLogger(__name__)

#: Gas limit for the ERC-20 approval wrapped by Lagoon.
LIGHTER_APPROVAL_GAS_LIMIT = 500_000
#: Gas limit for a Lighter deposit wrapped by Lagoon.
LIGHTER_DEPOSIT_GAS_LIMIT = 1_000_000
#: Gas limit for a Lighter pending withdrawal claim wrapped by Lagoon.
LIGHTER_WITHDRAWAL_CLAIM_GAS_LIMIT = 1_000_000
#: Default maximum wait for a secure or L1 Lighter withdrawal to become claimable.
#:
#: A live sub-account L1 withdrawal took about 29 minutes, so leave margin.
DEFAULT_LIGHTER_WITHDRAWAL_TIMEOUT_SECONDS = 60 * 60
#: Public Lighter deposit observation timeout.
DEFAULT_LIGHTER_DEPOSIT_TIMEOUT_SECONDS = 15 * 60
#: Default maximum wait for a master to sub-account transfer to appear in history.
DEFAULT_LIGHTER_TRANSFER_TIMEOUT_SECONDS = 5 * 60


@dataclass(frozen=True, slots=True)
class LighterRoutingConfig:
    """Private settings parsed by the CLI for automatic Lighter transfers."""

    #: Owner-only master API signer for secure withdrawals and master to sub-account transfers.
    operator_record: LighterOperatorRecord | None = None

    #: Maximum wait for Lighter to make a secure or L1 withdrawal claimable.
    withdrawal_timeout: int = DEFAULT_LIGHTER_WITHDRAWAL_TIMEOUT_SECONDS

    #: Maximum wait for a master to sub-account transfer to appear in history.
    transfer_timeout: int = DEFAULT_LIGHTER_TRANSFER_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        """Reject an invalid execution timeout at the CLI configuration boundary."""
        if type(self.withdrawal_timeout) is not int or self.withdrawal_timeout <= 0:
            raise ValueError("Lighter withdrawal timeout must be a positive number of seconds")
        if type(self.transfer_timeout) is not int or self.transfer_timeout <= 0:
            raise ValueError("Lighter transfer timeout must be a positive number of seconds")


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
        config: LighterRoutingConfig | None = None,
        session: LighterSession | None = None,
    ):
        """Create the Lighter exchange-account router.

        :param reserve_token_address:
            USDC reserve token address used by the Safe and Lighter.
        :param lighter_contract:
            Lighter L1 contract receiving Safe deposits and claims.
        :param config:
            CLI-parsed private withdrawal configuration.
        :param session:
            Optional public Lighter API client, primarily for tests.
        """
        super().__init__({}, reserve_token_address.lower())
        self.lighter_contract = Web3.to_checksum_address(lighter_contract)
        self.config = config or LighterRoutingConfig()
        self.lighter_session = session if session is not None else create_lighter_session()

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
        """Prepare deposit or withdrawal transactions for Lighter trades.

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
            assert TradeFlag.external_account_transfer in (trade.flags or set())
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
        if any(receipt.get("status") != 1 for receipt in transaction_receipts):
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
        approve_tx = routing_state.tx_builder.sign_transaction(
            usdc.contract,
            usdc.contract.functions.approve(self.lighter_contract, amount_raw),
            gas_limit=LIGHTER_APPROVAL_GAS_LIMIT,
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
            gas_limit=LIGHTER_DEPOSIT_GAS_LIMIT,
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
        operator = self.config.operator_record
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
        """Submit and confirm a master to sub-account transfer.

        The transfer hash is recorded and checkpointed before the wait, with
        the trade marked broadcasted, so an interrupted run blocks start-up
        and is reconciled from transfer history instead of being resubmitted.
        A definitive Lighter rejection moves no funds: the trade is marked
        failed with ``outcome == "transfer_rejected"`` so it does not block
        the executor.

        :param state:
            Strategy state owning the trade.
        :param trade:
            Funding trade on the destination sub-account position.
        """
        operator = self._require_master_operator(trade)
        if not is_lighter_sub_account_pair(trade.pair):
            raise RuntimeError("Lighter account transfers fund sub-accounts only")
        if int(trade.other_data.get("funding_source_account_id", -1)) != operator.account_index:
            raise RuntimeError("Lighter account transfer must be funded by the master account")
        destination = int(trade.pair.get_exchange_account_id())
        amount = trade.planned_quantity
        tx_hash = trade.other_data.get("lighter_transfer_tx_hash")
        if not tx_hash:
            try:
                request = asyncio.run(request_lighter_funding_transfer(operator, destination, amount))
            except LighterTransferError:
                state.mark_trade_failed(native_datetime_utc_now(), trade)
                trade.other_data["outcome"] = "transfer_rejected"
                self.checkpoint_state()
                raise
            tx_hash = request.tx_hash
            trade.other_data["lighter_transfer_tx_hash"] = tx_hash
            trade.other_data["lighter_transfer_requested_fee_usdc"] = str(request.fee)
            trade.other_data["lighter_transfer_requested_at"] = int(time.time())
            trade.mark_broadcasted(native_datetime_utc_now())
            self.checkpoint_state()

        record = asyncio.run(
            wait_for_lighter_funding_transfer(operator, str(tx_hash), self.config.transfer_timeout)
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
        are shared.

        The request hash is checkpointed before broadcast. A restart finds
        the pending trade and recovers it instead of sending a second request.

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
            pending_before = fetch_lighter_pending_balance(web3, owner=vault.safe_address, usdc=usdc, zk_lighter=self.lighter_contract)
            # USDC has a Lighter tick size of one raw unit, so base amount equals raw USDC.
            request_tx = routing_state.tx_builder.sign_transaction(
                zk,
                zk.functions.withdraw(
                    int(trade.pair.get_exchange_account_id()),
                    zk.functions.USDC_ASSET_INDEX().call(),
                    LIGHTER_ROUTE_PERP,
                    usdc.convert_to_raw(amount),
                ),
                gas_limit=LIGHTER_L1_WITHDRAW_GAS_LIMIT,
                asset_deltas=[],
                notes=trade.notes,
            )
            trade.other_data.update({
                "lighter_withdrawal_method": "l1_withdraw",
                "lighter_withdrawal_request_id": request_tx.tx_hash,
                "lighter_withdrawal_requested_at": int(time.time()),
                "lighter_pending_balance_before_usdc": str(pending_before),
                "lighter_safe_balance_before_claim": str(self._get_safe_balance(routing_state, trade)),
                "lighter_safe_address": vault.safe_address,
            })
            self.checkpoint_state()
            web3.eth.send_raw_transaction(HexBytes(request_tx.signed_bytes))
            receipt = wait_for_transaction_receipt_robust(web3, request_tx.tx_hash)
            if receipt["status"] != 1:
                raise RuntimeError(f"Lighter L1 withdrawal request {request_tx.tx_hash} reverted")
            logger.info("Lighter L1 withdrawal request for account %s mined: %s", trade.pair.get_exchange_account_id(), request_tx.tx_hash)

        wait_for_lighter_pending_balance(
            web3,
            owner=vault.safe_address,
            usdc=usdc,
            minimum=Decimal(trade.other_data["lighter_pending_balance_before_usdc"]) + amount,
            timeout=self.config.withdrawal_timeout,
            zk_lighter=self.lighter_contract,
        )

        trade.other_data["lighter_claimable_usdc"] = str(amount)
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
            # A process failure before checkpointing leaves the transfer unclean
            # for explicit recovery; start-up never submits a duplicate request.
            request_id = asyncio.run(
                request_lighter_withdrawal(operator, abs(trade.planned_quantity))
            )
            trade.other_data["lighter_withdrawal_request_id"] = request_id
            trade.other_data["lighter_withdrawal_requested_at"] = requested_at
            trade.other_data["lighter_safe_balance_before_claim"] = str(
                self._get_safe_balance(routing_state, trade)
            )
            trade.other_data["lighter_safe_address"] = vault.safe_address
            # Keep the trade started until the Safe claim is ready. The normal
            # transaction broadcaster performs the single state transition to
            # broadcasted after it receives the signed claim transaction.
            self.checkpoint_state()

        claimable = asyncio.run(
            wait_for_lighter_withdrawal_claimable(
                operator,
                abs(trade.planned_quantity),
                self.config.withdrawal_timeout,
                requested_at,
                str(request_id),
            )
        )
        trade.other_data["lighter_claimable_usdc"] = str(claimable)
        trade.set_blockchain_transactions([self._prepare_claim(routing_state, trade, claimable)])
        self.checkpoint_state()

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
        return routing_state.tx_builder.sign_transaction(
            zk,
            zk.functions.withdrawPendingBalance(
                vault.safe_address,
                zk.functions.USDC_ASSET_INDEX().call(),
                amount_raw,
            ),
            gas_limit=LIGHTER_WITHDRAWAL_CLAIM_GAS_LIMIT,
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
