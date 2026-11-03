"""Execute Lighter public-pool deposits and withdrawals through the Ethereum satellite Safe.

A pool trade is one :py:class:`~tradeexecutor.state.trade.TradeExecution`
advanced through persisted phases stored in ``trade.other_data["lighter_pool"]``:

Buy
    Safe ``approve`` + ZkLighter ``deposit`` on Ethereum, wait for the Lighter
    collateral credit, sign and submit a pool mint, confirm the share holding.

Sell
    Sign and submit a pool burn, confirm the share holding, wait for released
    collateral, sign and submit a secure withdrawal, identify its exact
    withdrawal-history id, wait until it is claimable, then claim it into the
    Safe with ZkLighter ``withdrawPendingBalance``.

The upstream :py:class:`eth_defi.lighter.execution.LighterPoolExecutor`
signs, persists, submits once and reconciles every API operation. Its
``LighterOperation`` records are saved on the trade as JSON strings through
the executor's pre-broadcast state callback before submission. EVM
transactions are signed by the satellite transaction builder, saved on the
trade before broadcast and broadcast idempotently. A restart resumes from the
saved phase and reconciles the saved operation; it never signs a replacement.

The trade is marked ``vault_settlement_pending`` at its first checkpoint,
before any money moves, so a crash in any later phase and a bounded wait
that times out both leave a pending trade that the shared settlement sweep
resumes, see
:py:func:`tradeexecutor.ethereum.vault.settlement_retry.check_and_resolve_vault_settlements`.
A crash before that checkpoint leaves a ``started`` trade with nothing moved,
which ``repair`` handles like any other unbroadcast trade. Rejections,
identity mismatches, ambiguous withdrawal history and a second trade while
one is unresolved raise
:py:class:`~eth_defi.vault.deposit_redeem.VaultRequestInvariantViolation`,
which the sweep propagates instead of retrying.

The Safe-owned Lighter master account is dedicated to this executor: one
writer per account and API key. A buy invests all available collateral,
which includes the activation collateral left by deployment on the first buy;
a sell withdraws all available collateral. See
``docs/plans/arbitrum-multi-vault-live-plan.md`` and
``deps/web3-ethereum-defi/docs/README-lighter-core-execution.md``.
"""

import datetime
import enum
import json
import logging
import time
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Callable

from eth_defi.abi import get_deployed_contract
from eth_defi.compat import native_datetime_utc_now
from eth_defi.lighter.api import LIGHTER_COLLATERAL_TOLERANCE
from eth_defi.lighter.constants import LIGHTER_L1_CONTRACT, LIGHTER_POOL_LOCKUP, LIGHTER_USDC_ETHEREUM
from eth_defi.lighter.execution import (
    LighterOperation,
    LighterOperationStage,
    LighterPoolExecutor,
    fetch_execution_account,
    fetch_pool_state,
    fetch_withdrawals,
    prepare_pool_deposit,
    prepare_pool_redemption,
    prepare_secure_withdrawal,
    wait_for_claimable_withdrawal,
    wait_for_pool_shares,
)
from eth_defi.lighter.session import LighterSession
from eth_defi.provider.receipt import wait_for_transaction_receipt_robust
from eth_defi.token import TokenDetails, fetch_erc20_details
from eth_defi.vault.deposit_redeem import VaultRequestInvariantViolation
from hexbytes import HexBytes
from web3 import Web3
from web3.contract import Contract
from web3.exceptions import TransactionNotFound

from tradeexecutor.ethereum.lighter.lighter_pool import (
    create_lighter_pool_identity,
    create_lighter_pool_session,
    create_lighter_pool_signer,
)
from tradeexecutor.ethereum.lighter.lighter_pool_valuation import check_lighter_pool_deposit
from tradeexecutor.ethereum.lighter.lighter_routing import (
    DEFAULT_LIGHTER_DEPOSIT_TIMEOUT_SECONDS,
    LIGHTER_APPROVAL_GAS_LIMIT,
    LIGHTER_DEPOSIT_GAS_LIMIT,
    LIGHTER_WITHDRAWAL_CLAIM_GAS_LIMIT,
    LighterKeyManagement,
)
from tradeexecutor.ethereum.tx import TransactionBuilder
from tradeexecutor.exchange_account.lighter_operator import LighterOperatorRecord
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeStatus
from tradeexecutor.strategy.routing import RoutingModel, RoutingState
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


logger = logging.getLogger(__name__)

#: ``trade.other_data`` key holding the pool lifecycle evidence.
LIGHTER_POOL_TRADE_DATA = "lighter_pool"

#: Default bound for one API operation to execute on the sequencer.
DEFAULT_LIGHTER_POOL_OPERATION_TIMEOUT_SECONDS = 5 * 60

#: Default bound for collateral released by a burn to become available.
DEFAULT_LIGHTER_POOL_COLLATERAL_TIMEOUT_SECONDS = 5 * 60

#: Polling interval for Lighter reads.
DEFAULT_LIGHTER_POOL_POLL_SECONDS = 5.0

#: Wait after a pool buy before its shares can be burned: Lighter's public-pool
#: lockup plus a margin for clock skew. Measured from the executor's own
#: execution time of the buy, not Lighter's ``entry_timestamp``, whose units
#: upstream has not verified on a funded account.
LIGHTER_POOL_EXIT_LOCKUP = LIGHTER_POOL_LOCKUP + datetime.timedelta(seconds=30)


class LighterPoolPhase(enum.StrEnum):
    """Last durably established step of a pool trade."""

    #: Trade started, no external action yet.
    started = "started"
    #: Buy: Safe approve and deposit are signed and saved.
    deposit_signed = "deposit_signed"
    #: Buy: the deposit is mined on Ethereum.
    deposit_mined = "deposit_mined"
    #: Buy: Lighter credited the collateral; mint next.
    collateral_credited = "collateral_credited"
    #: Sell: burn next, or reconcile the saved burn.
    burn = "burn"
    #: Sell: shares burned; wait for collateral to become available.
    burned = "burned"
    #: Sell: secure withdrawal next, or reconcile the saved one.
    withdraw = "withdraw"
    #: Sell: exact withdrawal identified; wait until claimable.
    withdraw_submitted = "withdraw_submitted"
    #: Sell: Safe claim next, or confirm the saved claim.
    claim = "claim"
    #: All evidence recorded; accounting can complete.
    complete = "complete"


@dataclass(slots=True)
class LighterPoolContext:
    """Per-call execution dependencies on the Ethereum custody chain."""

    #: Ethereum connection.
    web3: Web3
    #: Satellite Safe transaction builder.
    tx_builder: TransactionBuilder
    #: Ethereum Safe owning the Lighter account.
    safe_address: str
    #: Native Ethereum USDC.
    usdc: TokenDetails
    #: ZkLighter L1 contract.
    zk: Contract
    #: Lighter API session.
    session: LighterSession
    #: Owner-only master account signer record.
    operator: LighterOperatorRecord


class LighterPoolRoutingState(RoutingState):
    """Per-cycle objects for Lighter pool routing.

    :param universe:
        Trading universe.
    :param execution_details:
        Transaction builder, swapped to the Ethereum satellite builder by
        :py:class:`~tradeexecutor.strategy.generic.generic_router.GenericRouting`
        while ``setup_trades()`` runs.
    """

    def __init__(self, universe: TradingStrategyUniverse, execution_details: dict):
        super().__init__(universe)
        self.tx_builder = execution_details["tx_builder"]
        self.token_cache = execution_details.get("token_cache")


def get_lighter_pool_trade_data(trade: TradeExecution) -> dict | None:
    """Return a pool trade's saved lifecycle evidence, if it has started.

    :param trade:
        Lighter pool trade.
    :return:
        Mutable evidence dict, or ``None``.
    """
    return trade.other_data.get(LIGHTER_POOL_TRADE_DATA)


class LighterPoolRouting(RoutingModel):
    """Run Lighter pool trades through the Ethereum satellite Safe and the Lighter API.

    :param reserve_token_address:
        Ethereum native USDC.
    :param key_management:
        CLI-parsed Lighter API keys and wait settings.
    :param session:
        Lighter API session shared with pricing.
    :param lighter_contract:
        ZkLighter L1 contract on Ethereum.
    :param deposit_timeout:
        Bound for a mined deposit to be credited on Lighter.
    :param operation_timeout:
        Bound for one API operation to execute.
    :param collateral_timeout:
        Bound for burned collateral to become available.
    :param poll_interval:
        Lighter polling interval.
    """

    def __init__(
        self,
        reserve_token_address: str = LIGHTER_USDC_ETHEREUM,
        *,
        key_management: LighterKeyManagement | None = None,
        session: LighterSession | None = None,
        lighter_contract: str = LIGHTER_L1_CONTRACT,
        deposit_timeout: float = DEFAULT_LIGHTER_DEPOSIT_TIMEOUT_SECONDS,
        operation_timeout: float = DEFAULT_LIGHTER_POOL_OPERATION_TIMEOUT_SECONDS,
        collateral_timeout: float = DEFAULT_LIGHTER_POOL_COLLATERAL_TIMEOUT_SECONDS,
        poll_interval: float = DEFAULT_LIGHTER_POOL_POLL_SECONDS,
    ):
        super().__init__({}, reserve_token_address.lower())
        self.key_management = key_management or LighterKeyManagement()
        self.session = session
        self.lighter_contract = Web3.to_checksum_address(lighter_contract)
        self.deposit_timeout = deposit_timeout
        self.operation_timeout = operation_timeout
        self.collateral_timeout = collateral_timeout
        self.poll_interval = poll_interval

    # ----- Routing model interface -----

    def create_routing_state(self, universe: TradingStrategyUniverse, execution_details: object) -> LighterPoolRoutingState:
        assert isinstance(execution_details, dict)
        return LighterPoolRoutingState(universe, execution_details)

    def needs_sequential_trade_execution(self, trades: list[TradeExecution]) -> bool:
        """Pool trades share one Lighter account and API-key nonce."""
        return True

    def get_sequential_trade_execution_reason(self, trades: list[TradeExecution]) -> str:
        return "Lighter pool trades share one Safe-owned account and API-key nonce"

    def check_trade_before_execution(self, trade: TradeExecution) -> str | None:
        """Expire a pool buy before cash is reserved if the pool is closed or full.

        :param trade:
            Planned trade.
        :return:
            Reason the buy cannot execute, or ``None``.
        """
        if not trade.is_buy():
            return None
        pool = fetch_pool_state(self.get_session(), trade.pair.get_lighter_pool_index())
        _reason, message = check_lighter_pool_deposit(pool)
        return message

    def setup_trades(
        self,
        state: State,
        routing_state: LighterPoolRoutingState,
        trades: list[TradeExecution],
        check_balances: bool = False,
        rebroadcast: bool = False,
    ) -> None:
        """Advance each pool trade as far as its bounded waits allow.

        :param check_balances:
            Unused; the Lighter account and Safe balances are verified by phase.
        :param rebroadcast:
            Unused; saved transactions are always rebroadcast idempotently.
        """
        for trade in trades:
            assert trade.pair.is_lighter_pool(), f"Not a Lighter pool trade: {trade}"
            self.advance_trade(state, trade, routing_state.tx_builder)

    def settle_trade(
        self,
        web3: Web3,
        state: State,
        trade: TradeExecution,
        receipts: dict,
        stop_on_execution_failure: bool = False,
    ) -> None:
        """Complete accounting from saved evidence, or leave the trade pending.

        Called directly by sequential execution after ``setup_trades()``,
        because pool phases broadcast their own transactions.

        :param web3:
            Unused; evidence was verified on the custody chain.
        :param receipts:
            Unused; see above.
        :param stop_on_execution_failure:
            Unused; failures halt with an exception.
        """
        self.finalise_trade(state, trade)

    # ----- Lifecycle -----

    def get_session(self) -> LighterSession:
        """Return the Lighter API session."""
        if self.session is None:
            self.session = create_lighter_pool_session()
        return self.session

    def create_context(self, tx_builder: TransactionBuilder) -> LighterPoolContext:
        """Bind the Ethereum satellite builder and operator record.

        :param tx_builder:
            Builder signing for the Ethereum Safe that owns the Lighter account.
        :return:
            Execution dependencies.
        """
        operator = self.key_management.operator_record
        if operator is None:
            raise RuntimeError("Lighter pool execution requires the operator record passed with --lighter-operator-record-file")
        web3 = tx_builder.web3
        assert web3.eth.chain_id == 1, f"Lighter pool custody is on Ethereum, got chain {web3.eth.chain_id}"
        safe_address = Web3.to_checksum_address(tx_builder.get_token_delivery_address())
        if safe_address.lower() != operator.safe_address.lower():
            raise VaultRequestInvariantViolation(f"Lighter operator record belongs to Safe {operator.safe_address}, but the Ethereum satellite Safe is {safe_address}")
        return LighterPoolContext(
            web3=web3,
            tx_builder=tx_builder,
            safe_address=safe_address,
            usdc=fetch_erc20_details(web3, LIGHTER_USDC_ETHEREUM),
            zk=get_deployed_contract(web3, "lighter/ZkLighter.json", self.lighter_contract),
            session=self.get_session(),
            operator=operator,
        )

    def advance_trade(self, state: State, trade: TradeExecution, tx_builder: TransactionBuilder) -> bool:
        """Run the remaining phases of one pool trade.

        :param state:
            Strategy state owning the trade.
        :param trade:
            Started, broadcasted or pending pool trade.
        :param tx_builder:
            Ethereum satellite transaction builder.
        :return:
            ``True`` when every phase completed, ``False`` when a bounded wait timed out.
        """
        ctx = self.create_context(tx_builder)
        data = trade.other_data.get(LIGHTER_POOL_TRADE_DATA)
        if data is None:
            data = {
                "direction": "buy" if trade.is_buy() else "sell",
                "phase": LighterPoolPhase.started.value,
                "pool_index": trade.pair.get_lighter_pool_index(),
                "account_index": ctx.operator.account_index,
                "safe_address": ctx.safe_address,
                "operations": {},
            }
            trade.other_data[LIGHTER_POOL_TRADE_DATA] = data
        else:
            assert data["account_index"] == ctx.operator.account_index and data["safe_address"].lower() == ctx.safe_address.lower(), (
                f"Lighter pool trade #{trade.trade_id} belongs to account {data['account_index']} and Safe {data['safe_address']}; "
                "the configured operator record differs. Halt and inspect custody."
            )
        if data["phase"] == LighterPoolPhase.started:
            competing = [
                other.trade_id
                for position in state.portfolio.get_all_positions()
                for other in position.trades.values()
                if other.trade_id != trade.trade_id
                and other.pair.is_lighter_pool()
                and other.get_status() not in {TradeStatus.failed, TradeStatus.repaired}
                and (other_data := get_lighter_pool_trade_data(other)) is not None
                and other_data["phase"] != LighterPoolPhase.complete
            ]
            if competing:
                # One writer per account: the nonce and collateral of an
                # unresolved operation must not be touched by another trade.
                raise VaultRequestInvariantViolation(f"Lighter pool trade #{trade.trade_id} cannot start while trades {competing} on account {ctx.operator.account_index} are unresolved")
        signer = create_lighter_pool_signer(ctx.operator)
        try:
            if trade.is_buy():
                self._advance_buy(state, trade, ctx, data, signer)
            else:
                self._advance_sell(state, trade, ctx, data, signer)
        except TimeoutError as error:
            # A bounded wait is not a failure: the shared sweep resumes it.
            logger.warning("Lighter pool trade #%d waits in phase %s: %s", trade.trade_id, data["phase"], error)
            self.checkpoint_state()
            return False
        finally:
            signer.close()
        self.checkpoint_state()
        return data["phase"] == LighterPoolPhase.complete

    def finalise_trade(self, state: State, trade: TradeExecution) -> None:
        """Mark a completed pool trade successful, or pending if a wait remains.

        :param state:
            Strategy state.
        :param trade:
            Pool trade after :py:meth:`advance_trade`.
        """
        data = get_lighter_pool_trade_data(trade)
        assert data is not None, f"Lighter pool trade #{trade.trade_id} has no lifecycle evidence"
        now = native_datetime_utc_now()
        if data["phase"] != LighterPoolPhase.complete:
            self._mark_pending(state, trade, data)
            logger.info("Lighter pool trade #%d remains pending in phase %s", trade.trade_id, data["phase"])
            return

        if trade.get_status() == TradeStatus.started:
            state.mark_broadcasted(now, trade)
        trade.vault_settlement_pending_at = None
        if trade.is_buy():
            executed_quantity = Decimal(data["executed_shares"])
            executed_reserve = Decimal(trade.planned_reserve)
        else:
            executed_quantity = -Decimal(data["burned_shares"])
            executed_reserve = Decimal(data["claimed_usdc"])
        assert executed_quantity != 0 and executed_reserve > 0
        state.mark_trade_success(
            now,
            trade,
            executed_price=float(executed_reserve / abs(executed_quantity)),
            executed_amount=executed_quantity,
            executed_reserve=executed_reserve,
            lp_fees=0,
            native_token_price=0,
        )
        logger.info(
            "Lighter pool trade #%d complete: %s shares for %s USDC",
            trade.trade_id,
            executed_quantity,
            executed_reserve,
        )

    def _mark_pending(self, state: State, trade: TradeExecution, data: dict) -> None:
        """Record the trade as pending so a crash or timeout is resumed by the sweep.

        Called at the first checkpoint, before any money moves, and again
        on every later checkpoint to refresh the saved phase.

        :param state:
            Strategy state.
        :param trade:
            Pool trade.
        :param data:
            Its lifecycle evidence.
        """
        if trade.get_status() == TradeStatus.started:
            state.mark_broadcasted(native_datetime_utc_now(), trade)
        if trade.get_status() != TradeStatus.vault_settlement_pending:
            state.mark_vault_settlement_pending(
                native_datetime_utc_now(),
                trade,
                {"vault_address": trade.pair.pool_address, "vault_owner": data["safe_address"], "vault_chain_id": 1},
            )
        trade.other_data["lighter_pool_phase"] = data["phase"]
        self.checkpoint_state()

    # ----- Buy -----

    def _advance_buy(self, state: State, trade: TradeExecution, ctx: LighterPoolContext, data: dict, signer) -> None:
        pool_index = data["pool_index"]
        if data["phase"] == LighterPoolPhase.started:
            account, equity = fetch_execution_account(ctx.session, ctx.operator.account_index)
            data["collateral_before"] = str(equity.collateral)
            data["shares_before"] = str(_get_held_shares(account, pool_index))
            amount_raw = ctx.usdc.convert_to_raw(Decimal(trade.planned_reserve))
            asset_index = ctx.zk.functions.USDC_ASSET_INDEX().call()
            approve = ctx.tx_builder.sign_transaction(
                ctx.usdc.contract,
                ctx.usdc.contract.functions.approve(self.lighter_contract, amount_raw),
                gas_limit=LIGHTER_APPROVAL_GAS_LIMIT,
                asset_deltas=[],
                notes=trade.notes,
            )
            deposit = ctx.tx_builder.sign_transaction(
                ctx.zk,
                ctx.zk.functions.deposit(ctx.safe_address, asset_index, 0, amount_raw),
                gas_limit=LIGHTER_DEPOSIT_GAS_LIMIT,
                asset_deltas=[],
                notes=trade.notes,
            )
            trade.set_blockchain_transactions([approve, deposit])
            data["deposit_usdc"] = str(trade.planned_reserve)
            data["phase"] = LighterPoolPhase.deposit_signed.value
            # The first checkpoint marks the trade pending and saves the
            # signed transactions before any broadcast.
            self._mark_pending(state, trade, data)

        if data["phase"] == LighterPoolPhase.deposit_signed:
            # Deposited USDC has left the Safe: a later failure must not refund it.
            trade.other_data["retain_reserve_allocation_on_failure"] = True
            self._mark_pending(state, trade, data)
            for tx in trade.blockchain_transactions:
                _broadcast_and_confirm(ctx.web3, tx)
            data["phase"] = LighterPoolPhase.deposit_mined.value
            self.checkpoint_state()

        if data["phase"] == LighterPoolPhase.deposit_mined:
            required = Decimal(data["collateral_before"]) + Decimal(data["deposit_usdc"]) - LIGHTER_COLLATERAL_TOLERANCE
            collateral = self._wait_for_collateral(ctx, required, self.deposit_timeout)
            data["collateral_credited"] = str(collateral)
            data["phase"] = LighterPoolPhase.collateral_credited.value
            self.checkpoint_state()

        if data["phase"] == LighterPoolPhase.collateral_credited:

            def prepare(executor: LighterPoolExecutor) -> LighterOperation:
                # A dedicated account invests all available collateral, which
                # includes the activation collateral on the first buy.
                _, equity = fetch_execution_account(ctx.session, ctx.operator.account_index)
                return prepare_pool_deposit(executor, pool_index, equity.available_balance)

            operation = self._run_operation(ctx, data, "mint", signer, prepare)
            expected = int(data["shares_before"]) + operation.amount
            wait_for_pool_shares(ctx.session, ctx.operator.account_index, pool_index, expected, timeout=self.operation_timeout)
            data["executed_shares"] = str(operation.amount)
            data["phase"] = LighterPoolPhase.complete.value

    # ----- Sell -----

    def _advance_sell(self, state: State, trade: TradeExecution, ctx: LighterPoolContext, data: dict, signer) -> None:
        pool_index = data["pool_index"]
        if data["phase"] == LighterPoolPhase.started:
            account, _ = fetch_execution_account(ctx.session, ctx.operator.account_index)
            held = _get_held_shares(account, pool_index)
            position = state.portfolio.get_position_by_id(trade.position_id)
            closing = abs(Decimal(trade.planned_quantity)) >= position.get_quantity()
            assert held > 0, f"Lighter account {ctx.operator.account_index} holds no shares of pool {pool_index}"
            data["shares_before"] = str(held)
            data["full_exit"] = closing
            data["phase"] = LighterPoolPhase.burn.value
            self._mark_pending(state, trade, data)

        if data["phase"] == LighterPoolPhase.burn:
            self._mark_pending(state, trade, data)
            if "burn" not in data["operations"]:
                self._wait_for_pool_lockup(state, trade)
                _, equity = fetch_execution_account(ctx.session, ctx.operator.account_index)
                data["collateral_before_burn"] = str(equity.available_balance)

            def prepare(executor: LighterPoolExecutor) -> LighterOperation:
                if data["full_exit"]:
                    return prepare_pool_redemption(executor, pool_index, None)
                pool = fetch_pool_state(ctx.session, pool_index)
                return prepare_pool_redemption(executor, pool_index, abs(Decimal(trade.planned_quantity)) * pool.share_value)

            operation = self._run_operation(ctx, data, "burn", signer, prepare)
            expected = int(data["shares_before"]) - operation.amount
            wait_for_pool_shares(ctx.session, ctx.operator.account_index, pool_index, expected, timeout=self.operation_timeout)
            data["burned_shares"] = str(operation.amount)
            data["phase"] = LighterPoolPhase.burned.value
            self.checkpoint_state()

        if data["phase"] == LighterPoolPhase.burned:
            # Any released collateral; the secure withdrawal enforces its own 1 USDC minimum.
            minimum = Decimal(data["collateral_before_burn"]) + LIGHTER_COLLATERAL_TOLERANCE
            available = self._wait_for_available(ctx, minimum, self.collateral_timeout)
            data["available_after_burn"] = str(available)
            data["phase"] = LighterPoolPhase.withdraw.value
            self.checkpoint_state()

        if data["phase"] == LighterPoolPhase.withdraw:
            if "withdraw" not in data["operations"]:
                # The history schema has no transaction hash: record every
                # existing id so exactly one new row identifies this request.
                data["withdrawal_baseline_ids"] = [str(row["id"]) for row in fetch_withdrawals(ctx.session, ctx.operator.account_index, signer.create_auth_token())]
                self.checkpoint_state()

            def prepare(executor: LighterPoolExecutor) -> LighterOperation:
                _, equity = fetch_execution_account(ctx.session, ctx.operator.account_index)
                amount = equity.available_balance.quantize(Decimal(1) / Decimal(10**ctx.usdc.decimals), rounding=ROUND_DOWN)
                data["withdrawal_usdc"] = str(amount)
                return prepare_secure_withdrawal(executor, ctx.usdc, amount)

            self._run_operation(ctx, data, "withdraw", signer, prepare)
            data["withdrawal_id"] = self._identify_withdrawal(ctx, data, signer)
            data["phase"] = LighterPoolPhase.withdraw_submitted.value
            self.checkpoint_state()

        if data["phase"] == LighterPoolPhase.withdraw_submitted:
            row = wait_for_claimable_withdrawal(
                ctx.session,
                ctx.operator.account_index,
                data["withdrawal_id"],
                signer.create_auth_token,
                timeout=self.key_management.withdrawal_timeout,
                poll_interval=max(self.poll_interval, 1.0),
            )
            data["claimable_usdc"] = str(Decimal(str(row["amount"])))
            data["phase"] = LighterPoolPhase.claim.value
            self.checkpoint_state()

        if data["phase"] == LighterPoolPhase.claim:
            claimable = Decimal(data["claimable_usdc"])
            if "claim_tx_hash" not in data:
                data["safe_balance_before_claim"] = str(ctx.usdc.fetch_balance_of(ctx.safe_address))
                claim = ctx.tx_builder.sign_transaction(
                    ctx.zk,
                    ctx.zk.functions.withdrawPendingBalance(ctx.safe_address, ctx.zk.functions.USDC_ASSET_INDEX().call(), ctx.usdc.convert_to_raw(claimable)),
                    gas_limit=LIGHTER_WITHDRAWAL_CLAIM_GAS_LIMIT,
                    asset_deltas=[],
                    notes=trade.notes,
                )
                trade.blockchain_transactions.append(claim)
                data["claim_tx_hash"] = claim.tx_hash
                self.checkpoint_state()
            claim = next(tx for tx in trade.blockchain_transactions if tx.tx_hash == data["claim_tx_hash"])
            _broadcast_and_confirm(ctx.web3, claim)
            balance = ctx.usdc.fetch_balance_of(ctx.safe_address)
            received = balance - Decimal(data["safe_balance_before_claim"])
            assert received >= claimable - LIGHTER_COLLATERAL_TOLERANCE, (
                f"Safe {ctx.safe_address} received {received} USDC from claiming {claimable}; halt and reconcile custody"
            )
            data["claimed_usdc"] = str(claimable)
            data["phase"] = LighterPoolPhase.complete.value

    # ----- Helpers -----

    def _run_operation(
        self,
        ctx: LighterPoolContext,
        data: dict,
        kind: str,
        signer,
        prepare: Callable[[LighterPoolExecutor], LighterOperation],
    ) -> LighterOperation:
        """Prepare and submit once, or reconcile the saved operation after a restart.

        :param kind:
            ``mint``, ``burn`` or ``withdraw``: the evidence slot.
        :param prepare:
            Signs and persists a new operation through the executor.
        :return:
            Confirmed operation.
        """

        def persist(operation: LighterOperation) -> None:
            data["operations"][kind] = json.dumps(operation.as_dict())
            self.checkpoint_state()

        executor = LighterPoolExecutor(ctx.session, create_lighter_pool_identity(ctx.operator), signer, persist)
        saved = data["operations"].get(kind)
        operation = LighterOperation.from_dict(json.loads(saved)) if saved else None
        if operation is not None:
            if operation.stage == LighterOperationStage.confirmed:
                return operation
            if operation.stage == LighterOperationStage.rejected:
                raise VaultRequestInvariantViolation(f"Saved Lighter {kind} operation {operation.tx_hash} was rejected; inspect the trade before retrying")
            if operation.stage == LighterOperationStage.prepared:
                # Upstream marks an operation unknown before posting it, so a
                # saved prepared operation was never sent and its nonce is
                # unused: sign a fresh one rather than resubmit the old payload.
                operation = None
        try:
            if operation is None:
                operation = prepare(executor)
                executor.submit(operation)
            return executor.reconcile(operation, timeout=self.operation_timeout, poll_interval=self.poll_interval)
        except ValueError as error:
            # Upstream raises ValueError both for a sequencer rejection, saved
            # as such before raising, and for an undecodable response that
            # leaves the outcome unknown. Only the recorded rejection is an
            # invariant failure; the uncertain outcome stays retryable.
            if operation is not None and operation.stage == LighterOperationStage.rejected:
                raise VaultRequestInvariantViolation(f"Lighter {kind} operation {operation.tx_hash} was rejected; inspect the trade before retrying") from error
            raise

    def _wait_for_pool_lockup(self, state: State, trade: TradeExecution) -> None:
        """Wait until the position's latest pool buy is past Lighter's lockup.

        Lighter rejects a burn of shares minted within the pool lockup
        (five minutes). A test or strategy that exits right after entering
        would otherwise have its burn rejected, which halts the trade.

        :param state:
            Strategy state.
        :param trade:
            Sell trade about to burn.
        """
        position = state.portfolio.get_position_by_id(trade.position_id)
        bought_at = max((t.executed_at for t in position.trades.values() if t.is_buy() and t.is_success()), default=None)
        if bought_at is None:
            return
        unlock_at = bought_at + LIGHTER_POOL_EXIT_LOCKUP
        while (remaining := (unlock_at - native_datetime_utc_now()).total_seconds()) > 0:
            logger.info("Waiting %.0f s for the Lighter pool lockup of trade #%d's shares to end at %s", remaining, trade.trade_id, unlock_at)
            time.sleep(min(remaining, 30.0))

    def _wait_for_collateral(self, ctx: LighterPoolContext, minimum: Decimal, timeout: float) -> Decimal:
        """Wait for total collateral to reach a minimum, such as a credited deposit."""
        deadline = time.monotonic() + timeout
        while True:
            _, equity = fetch_execution_account(ctx.session, ctx.operator.account_index)
            if equity.collateral >= minimum:
                return equity.collateral
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Lighter collateral is {equity.collateral}, waiting for {minimum}")
            logger.info("Waiting for Lighter collateral %s, need %s", equity.collateral, minimum)
            time.sleep(self.poll_interval)

    def _wait_for_available(self, ctx: LighterPoolContext, minimum: Decimal, timeout: float) -> Decimal:
        """Wait for available collateral released by a burn."""
        deadline = time.monotonic() + timeout
        while True:
            _, equity = fetch_execution_account(ctx.session, ctx.operator.account_index)
            if equity.available_balance >= minimum:
                return equity.available_balance
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Lighter available collateral is {equity.available_balance}, waiting for {minimum}")
            logger.info("Waiting for released Lighter collateral %s, need %s", equity.available_balance, minimum)
            time.sleep(self.poll_interval)

    def _identify_withdrawal(self, ctx: LighterPoolContext, data: dict, signer) -> str:
        """Find the single new withdrawal-history row created by the confirmed request.

        Matching by amount or time is not used. Several new rows violate the
        one-writer assumption and halt; no new row yet is indexer lag.

        :return:
            Exact withdrawal-history id.
        """
        baseline = set(data["withdrawal_baseline_ids"])
        deadline = time.monotonic() + self.operation_timeout
        while True:
            new = [str(row["id"]) for row in fetch_withdrawals(ctx.session, ctx.operator.account_index, signer.create_auth_token()) if str(row["id"]) not in baseline]
            if len(new) == 1:
                return new[0]
            if len(new) > 1:
                raise VaultRequestInvariantViolation(f"Lighter account {ctx.operator.account_index} has {len(new)} new withdrawals {new}; another writer used the account. Halt and reconcile.")
            if time.monotonic() >= deadline:
                raise TimeoutError("The confirmed secure withdrawal is not yet visible in withdrawal history")
            time.sleep(self.poll_interval)

    def resume_pending_trade(self, state: State, trade: TradeExecution, tx_builder: TransactionBuilder) -> bool:
        """Resume a pending pool trade from the shared settlement sweep.

        :param state:
            Strategy state.
        :param trade:
            ``vault_settlement_pending`` pool trade.
        :param tx_builder:
            Ethereum satellite transaction builder.
        :return:
            ``True`` if the trade completed now.
        """
        completed = self.advance_trade(state, trade, tx_builder)
        self.finalise_trade(state, trade)
        return completed


def _get_held_shares(account: dict, pool_index: int) -> int:
    """Return raw shares held in one pool from a strict account record."""
    rows = [row for row in account["shares"] if int(row["public_pool_index"]) == pool_index]
    assert len(rows) <= 1, f"Duplicate Lighter pool holding for pool {pool_index}"
    return int(rows[0]["shares_amount"]) if rows else 0


def _broadcast_and_confirm(web3: Web3, tx: BlockchainTransaction) -> None:
    """Broadcast a saved transaction unless it is mined, and record its confirmation.

    Rebroadcasting the same signed bytes is idempotent: a mined hash returns
    its receipt and a known transaction is only awaited.

    :param web3:
        Ethereum connection.
    :param tx:
        Signed, saved transaction.
    :raise VaultRequestInvariantViolation:
        If the transaction reverted.
    """
    tx_hash = HexBytes(tx.tx_hash)
    try:
        receipt = web3.eth.get_transaction_receipt(tx_hash)
    except TransactionNotFound:
        receipt = None
    if receipt is None:
        try:
            web3.eth.send_raw_transaction(HexBytes(tx.signed_bytes))
        except ValueError as error:
            # Already in the mempool from an interrupted attempt: wait for it.
            logger.warning("Rebroadcast of %s was refused (%s); waiting for its receipt", tx.tx_hash, error)
        tx.broadcasted_at = tx.broadcasted_at or native_datetime_utc_now()
        receipt = wait_for_transaction_receipt_robust(web3, tx_hash)
    block = web3.eth.get_block(receipt["blockNumber"])
    tx.set_confirmation_information(
        ts=datetime.datetime.fromtimestamp(block["timestamp"], datetime.UTC).replace(tzinfo=None),
        block_number=receipt["blockNumber"],
        block_hash=receipt["blockHash"].hex(),
        realised_gas_units_consumed=receipt["gasUsed"],
        realised_gas_price=receipt.get("effectiveGasPrice", 0),
        status=receipt["status"] == 1,
    )
    if receipt["status"] != 1:
        raise VaultRequestInvariantViolation(f"Lighter pool transaction {tx.tx_hash} reverted; halt and inspect custody")
