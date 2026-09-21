"""State accounting tests for manual external exchange-account transfers."""

from decimal import Decimal
from types import SimpleNamespace

import pytest
from eth_defi.compat import native_datetime_utc_now
from eth_defi.token import USDC_NATIVE_TOKEN
from pytest_mock import MockerFixture

from tradeexecutor.ethereum.rebroadcast import rebroadcast_all
from tradeexecutor.ethereum.lighter.lighter_routing import LighterRouting, LighterRoutingConfig
from tradeexecutor.exchange_account.lighter import create_lighter_exchange_account_pair
from tradeexecutor.exchange_account.lighter_operator import LighterOperatorRecord
from tradeexecutor.exchange_account.state import (
    ExchangeAccountTransferError,
    create_exchange_account_transfer,
    mark_exchange_account_transfer_broadcasted,
    open_exchange_account_position,
    reconcile_completed_external_account_transfers,
    record_exchange_account_transfer,
)
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.repair import repair_tx_not_generated
from tradeexecutor.state.state import State, UncleanState
from tradeexecutor.state.trade import TradeExecution, TradeStatus
from tradeexecutor.strategy.account_correction import preflight_state_for_account_correction


#: Public test-only Lighter account identifier.
LIGHTER_ACCOUNT_INDEX = 1
#: Initial Safe reserve before the exchange account receives its allocation.
INITIAL_SAFE_USDC = Decimal("20")
#: Initial capital allocated to the synthetic Lighter account.
INITIAL_LIGHTER_USDC = Decimal("10")


def _create_state() -> tuple[State, AssetIdentifier, TradingPosition]:
    """Create one reserve and one funded synthetic Lighter account position."""
    reserve_asset = AssetIdentifier(
        chain_id=1,
        address=USDC_NATIVE_TOKEN[1],
        token_symbol="USDC",
        decimals=6,
    )
    state = State()
    state.portfolio.initialise_reserves(reserve_asset, reserve_token_price=1.0)
    state.portfolio.get_reserve_position(reserve_asset).quantity = INITIAL_SAFE_USDC
    pair = create_lighter_exchange_account_pair(
        quote=reserve_asset,
        account_index=LIGHTER_ACCOUNT_INDEX,
    )
    open_exchange_account_position(
        state=state,
        strategy_cycle_at=native_datetime_utc_now(),
        pair=pair,
        reserve_currency=reserve_asset,
        reserve_amount=INITIAL_LIGHTER_USDC,
        notes="Initial Lighter account allocation",
    )
    position = next(iter(state.portfolio.open_positions.values()))
    return state, reserve_asset, position


def _create_transfer(
    state: State,
    reserve_asset: AssetIdentifier,
    position: TradingPosition,
    *,
    amount: Decimal,
    deposit: bool,
) -> TradeExecution:
    """Create and broadcast one test-only manual transfer trade."""
    trade = create_exchange_account_transfer(
        state=state,
        position=position,
        strategy_cycle_at=native_datetime_utc_now(),
        reserve_currency=reserve_asset,
        amount=amount,
        deposit=deposit,
        notes="Test external account transfer",
        metadata={"direction": "deposit" if deposit else "withdraw"},
    )
    mark_exchange_account_transfer_broadcasted(trade, native_datetime_utc_now())
    return trade


def test_record_exchange_account_transfer_preserves_custody_value() -> None:
    """Record verified Safe/Lighter transfers without changing combined value.

    1. Create a funded external-account position and a Safe reserve.
    2. Record a verified Safe-to-Lighter deposit and inspect its state effects.
    3. Record withdrawals, including the full close, and verify the position stays open.
    4. Reject an underfunded accounting mutation.
    """
    # 1. Create a funded external-account position and a Safe reserve.
    state, reserve_asset, position = _create_state()
    reserve = state.portfolio.get_reserve_position(reserve_asset)
    assert reserve.quantity == Decimal("10")
    assert position.get_quantity() == INITIAL_LIGHTER_USDC

    # 2. Record a verified Safe-to-Lighter deposit and inspect its state effects.
    deposit = _create_transfer(
        state,
        reserve_asset,
        position,
        amount=Decimal("5"),
        deposit=True,
    )
    record_exchange_account_transfer(
        state=state,
        position=position,
        trade=deposit,
        reserve_currency=reserve_asset,
        amount=Decimal("5"),
        executed_at=native_datetime_utc_now(),
    )
    assert deposit.is_success()
    assert deposit.get_action_verb() == "Deposit to external account"
    assert reserve.quantity == Decimal("5")
    assert position.get_quantity() == Decimal("15")
    assert state.portfolio.get_net_asset_value() == pytest.approx(Decimal("20"))

    # 3. Record withdrawals, including the full close, and verify the position stays open.
    withdrawal = _create_transfer(
        state,
        reserve_asset,
        position,
        amount=Decimal("5"),
        deposit=False,
    )
    record_exchange_account_transfer(
        state=state,
        position=position,
        trade=withdrawal,
        reserve_currency=reserve_asset,
        amount=Decimal("5"),
        executed_at=native_datetime_utc_now(),
    )
    full_withdrawal = _create_transfer(
        state,
        reserve_asset,
        position,
        amount=Decimal("10"),
        deposit=False,
    )
    record_exchange_account_transfer(
        state=state,
        position=position,
        trade=full_withdrawal,
        reserve_currency=reserve_asset,
        amount=Decimal("10"),
        executed_at=native_datetime_utc_now(),
    )
    assert reserve.quantity == INITIAL_SAFE_USDC
    assert position.get_quantity() == Decimal(0)
    assert position.position_id in state.portfolio.open_positions
    assert state.portfolio.get_net_asset_value() == pytest.approx(INITIAL_SAFE_USDC)

    # 4. Reject an underfunded accounting mutation.
    invalid = _create_transfer(
        state,
        reserve_asset,
        position,
        amount=Decimal("21"),
        deposit=True,
    )
    with pytest.raises(ExchangeAccountTransferError, match="negative reserve quantity"):
        record_exchange_account_transfer(
            state=state,
            position=position,
            trade=invalid,
            reserve_currency=reserve_asset,
            amount=Decimal("21"),
            executed_at=native_datetime_utc_now(),
        )


def test_pending_external_account_transfer_makes_state_unclean() -> None:
    """Block account checks and strategy startup while a manual withdrawal waits.

    1. Create planned, started, broadcasted and failed manual transfer states.
    2. Verify each transfer makes state and account-correction preflight unclean.
    3. Preserve the legacy unfinished predicate for planned and started trades.
    """
    # 1. Create planned, started, broadcasted and failed manual transfer states.
    for phase in ("planned", "started", "broadcasted", "failed"):
        state, reserve_asset, position = _create_state()
        trade = create_exchange_account_transfer(
            state=state,
            position=position,
            strategy_cycle_at=native_datetime_utc_now(),
            reserve_currency=reserve_asset,
            amount=Decimal("1"),
            deposit=False,
            notes="Test pending external account transfer",
            metadata={"direction": "withdraw"},
        )
        if phase == "started":
            trade.started_at = native_datetime_utc_now()
        elif phase == "broadcasted":
            mark_exchange_account_transfer_broadcasted(trade, native_datetime_utc_now())
        elif phase == "failed":
            trade.failed_at = native_datetime_utc_now()

        # 2. Verify each transfer makes state and account-correction preflight unclean.
        with pytest.raises(UncleanState):
            state.check_if_clean()
        with pytest.raises(UncleanState):
            preflight_state_for_account_correction(state)

        # 3. Preserve the legacy unfinished predicate for non-broadcasted trades.
        assert trade.is_unfinished() is (phase == "broadcasted")


def test_repair_does_not_rebroadcast_pending_external_account_transfer(
    mocker: MockerFixture,
) -> None:
    """Keep external account transfers out of generic transaction rebroadcasting.

    1. Create a broadcasted Lighter withdrawal that has not been reconciled.
    2. Run the generic transaction-rebroadcast selector used by repair.
    3. Verify it selects no transfer and does not submit a transaction.
    """
    # 1. Create a broadcasted Lighter withdrawal that has not been reconciled.
    state, reserve_asset, position = _create_state()
    _create_transfer(
        state,
        reserve_asset,
        position,
        amount=Decimal("1"),
        deposit=False,
    )

    # 2. Run the generic transaction-rebroadcast selector used by repair.
    execution_model = mocker.Mock()
    trades, transactions = rebroadcast_all(
        mocker.Mock(),
        state,
        execution_model,
        mocker.Mock(),
        mocker.Mock(),
    )

    # 3. Verify it selects no transfer and does not submit a transaction.
    assert trades == []
    assert transactions == []
    execution_model.execute_trades.assert_not_called()


def test_repair_does_not_create_counter_trade_for_pending_external_transfer() -> None:
    """Leave an unmined external transfer for receipt-only reconciliation.

    1. Create a planned Lighter withdrawal without a transaction receipt.
    2. Run the missing-transaction repair selector.
    3. Verify it creates no counter-trade and leaves the transfer unclean.
    """
    # 1. Create a planned Lighter withdrawal without a transaction receipt.
    state, reserve_asset, position = _create_state()
    transfer = create_exchange_account_transfer(
        state=state,
        position=position,
        strategy_cycle_at=native_datetime_utc_now(),
        reserve_currency=reserve_asset,
        amount=Decimal("1"),
        deposit=False,
        notes="Interrupted Lighter withdrawal",
        metadata={"direction": "withdraw"},
        automatic=True,
    )

    # 2. Run the missing-transaction repair selector.
    repair_trades = repair_tx_not_generated(state, interactive=False)

    # 3. Verify it creates no counter-trade and leaves the transfer unclean.
    assert repair_trades == []
    assert transfer.is_planned()
    with pytest.raises(UncleanState):
        state.check_if_clean()


def test_lighter_withdrawal_keeps_trade_started_until_safe_claim(
    mocker: MockerFixture,
) -> None:
    """Checkpoint a Lighter withdrawal without double-marking its broadcast state.

    1. Create a started automatic Lighter withdrawal.
    2. Mock only the Lighter API request, delay observation and Safe claim signing.
    3. Prepare the claim and verify the state remains started for normal broadcasting.
    """
    # 1. Create a started automatic Lighter withdrawal.
    state, reserve_asset, position = _create_state()
    trade = create_exchange_account_transfer(
        state=state,
        position=position,
        strategy_cycle_at=native_datetime_utc_now(),
        reserve_currency=reserve_asset,
        amount=Decimal("1"),
        deposit=False,
        notes="Automatic Lighter withdrawal",
        metadata={"direction": "withdraw"},
        automatic=True,
    )
    trade.started_at = native_datetime_utc_now()
    router = LighterRouting(reserve_asset.address, session=mocker.Mock())
    routing_state = SimpleNamespace(
        operator_record=mocker.Mock(),
        vault=SimpleNamespace(safe_address="0x0000000000000000000000000000000000000001"),
    )

    # 2. Mock only the Lighter API request, delay observation and Safe claim signing.
    async def request_withdrawal(*_args: object) -> str:
        return "withdrawal-request-1"

    async def wait_for_claimable(*_args: object) -> Decimal:
        return Decimal("1")

    claim_transaction = mocker.Mock()
    mocker.patch(
        "tradeexecutor.ethereum.lighter.lighter_routing.request_lighter_withdrawal",
        side_effect=request_withdrawal,
    )
    mocker.patch(
        "tradeexecutor.ethereum.lighter.lighter_routing.wait_for_lighter_withdrawal_claimable",
        side_effect=wait_for_claimable,
    )
    mocker.patch.object(router, "_get_safe_balance", return_value=Decimal("5"))
    mocker.patch.object(router, "_prepare_claim", return_value=claim_transaction)
    mocker.patch.object(router, "checkpoint_state")

    # 3. Prepare the claim and verify the state remains started for normal broadcasting.
    router._prepare_withdrawal(routing_state, trade)
    assert trade.get_status() == TradeStatus.started
    assert trade.other_data["lighter_withdrawal_request_id"] == "withdrawal-request-1"
    assert trade.blockchain_transactions == [claim_transaction]


def test_lighter_routing_uses_explicit_cli_configuration(
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Pass the private Lighter operator record without reading process environment.

    1. Set a conflicting legacy environment value and create a routing configuration.
    2. Create the production Lighter router with the explicit configuration.
    3. Verify the per-cycle routing state uses the supplied record and timeout.
    """
    # 1. Set a conflicting legacy environment value and create a routing configuration.
    monkeypatch.setenv("LIGHTER_OPERATOR_RECORD_FILE", "/does/not/exist.json")
    operator_record = LighterOperatorRecord(
        vault_address="0x0000000000000000000000000000000000000001",
        safe_address="0x0000000000000000000000000000000000000002",
        module_address="0x0000000000000000000000000000000000000003",
        account_index=LIGHTER_ACCOUNT_INDEX,
        api_key_index=4,
        api_private_key="0x" + "11" * 32,
    )
    config = LighterRoutingConfig(
        operator_record=operator_record,
        withdrawal_timeout=1800,
    )

    # 2. Create the production Lighter router with the explicit configuration.
    router = LighterRouting(
        USDC_NATIVE_TOKEN[1],
        config=config,
        session=mocker.Mock(),
    )
    routing_state = router.create_routing_state(
        SimpleNamespace(),
        {
            "tx_builder": mocker.Mock(),
            "vault": mocker.Mock(),
        },
    )

    # 3. Verify the per-cycle routing state uses the supplied record and timeout.
    assert routing_state.operator_record is operator_record
    assert router.config.withdrawal_timeout == 1800


def test_completed_external_account_transfer_is_reconciled_without_rebroadcast(
    mocker: MockerFixture,
) -> None:
    """Reconcile a mined external transfer without submitting another transaction.

    1. Create and broadcast an automatic Lighter withdrawal with one transaction.
    2. Return a successful receipt for the recorded transaction.
    3. Reconcile the existing trade and verify the state is clean.
    """
    # 1. Create and broadcast an automatic Lighter withdrawal with one transaction.
    state, reserve_asset, position = _create_state()
    withdrawal = create_exchange_account_transfer(
        state=state,
        position=position,
        strategy_cycle_at=native_datetime_utc_now(),
        reserve_currency=reserve_asset,
        amount=Decimal("1"),
        deposit=False,
        notes="Checkpointed automatic Lighter withdrawal",
        metadata={
            "direction": "withdraw",
        },
        automatic=True,
    )
    state.start_execution(
        native_datetime_utc_now(),
        withdrawal,
        underflow_check=True,
    )
    mark_exchange_account_transfer_broadcasted(
        withdrawal,
        native_datetime_utc_now(),
    )
    withdrawal.blockchain_transactions = [
        SimpleNamespace(tx_hash="0x" + "12" * 32),
    ]

    # 2. Return a successful receipt for the recorded transaction.
    web3 = SimpleNamespace(
        eth=SimpleNamespace(
            get_transaction_receipt=mocker.Mock(return_value={"status": 1}),
        ),
    )

    # 3. Reconcile the existing trade and verify the state is clean.
    reconciled = reconcile_completed_external_account_transfers(state, web3)
    assert reconciled == [withdrawal]
    web3.eth.get_transaction_receipt.assert_called_once_with("0x" + "12" * 32)
    assert withdrawal.is_success()
    assert position.get_quantity() == Decimal("9")
    assert state.portfolio.get_reserve_position(reserve_asset).quantity == Decimal("11")
    state.check_if_clean()
