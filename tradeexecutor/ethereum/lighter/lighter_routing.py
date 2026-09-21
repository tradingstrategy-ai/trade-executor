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
from eth_defi.lighter.session import LighterSession, create_lighter_session
from eth_defi.lighter.valuation import fetch_lighter_total_equity
from eth_defi.token import fetch_erc20_details
from hexbytes import HexBytes
from web3 import Web3

from tradeexecutor.exchange_account.lighter_operator import (
    LighterOperatorRecord,
    request_lighter_withdrawal,
    wait_for_lighter_withdrawal_claimable,
)
from tradeexecutor.exchange_account.state import complete_exchange_account_transfer
from tradeexecutor.ethereum.lighter.transfer_verification import (
    LighterTransferVerificationError,
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
#: Default maximum wait for a secure Lighter withdrawal.
DEFAULT_LIGHTER_WITHDRAWAL_TIMEOUT_SECONDS = 30 * 60
#: Public Lighter deposit observation timeout.
DEFAULT_LIGHTER_DEPOSIT_TIMEOUT_SECONDS = 15 * 60


@dataclass(frozen=True, slots=True)
class LighterRoutingConfig:
    """Private settings parsed by the CLI for automatic Lighter transfers.

    :param operator_record:
        Owner-only delegated API signer for secure Lighter withdrawals.
    :param withdrawal_timeout:
        Maximum wait for Lighter to make a secure withdrawal claimable.
    """

    #: Owner-only delegated API signer for secure Lighter withdrawals.
    operator_record: LighterOperatorRecord | None = None

    #: Maximum wait for Lighter to make a secure withdrawal claimable.
    withdrawal_timeout: int = DEFAULT_LIGHTER_WITHDRAWAL_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        """Reject an invalid execution timeout at the CLI configuration boundary."""
        if type(self.withdrawal_timeout) is not int or self.withdrawal_timeout <= 0:
            raise ValueError("Lighter withdrawal timeout must be a positive number of seconds")


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
            assert TradeFlag.external_account_transfer in (trade.flags or set())
            if trade.is_buy():
                if trade.blockchain_transactions:
                    continue
                trade.set_blockchain_transactions(
                    self._prepare_deposit(routing_state, trade)
                )
            else:
                if trade.blockchain_transactions:
                    continue
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
        usdc = fetch_erc20_details(
            web3,
            trade.reserve_currency.address,
            cache=routing_state.token_cache,
            chain_id=trade.reserve_currency.chain_id,
        )
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

    def _prepare_withdrawal(
        self,
        routing_state: LighterRoutingState,
        trade: TradeExecution,
    ) -> None:
        """Request, wait for, and prepare a Lighter claim transaction."""
        vault = self._get_lagoon_vault(routing_state)
        operator = self.config.operator_record
        if operator is None:
            raise RuntimeError(
                "Automatic Lighter withdrawals require an operator record passed by the CLI"
            )
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
        usdc = fetch_erc20_details(
            web3,
            trade.reserve_currency.address,
            cache=routing_state.token_cache,
            chain_id=trade.reserve_currency.chain_id,
        )
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
        usdc = fetch_erc20_details(
            routing_state.tx_builder.web3,
            trade.reserve_currency.address,
            cache=routing_state.token_cache,
            chain_id=trade.reserve_currency.chain_id,
        )
        vault = self._get_lagoon_vault(routing_state)
        return Decimal(str(usdc.fetch_balance_of(vault.safe_address)))

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
