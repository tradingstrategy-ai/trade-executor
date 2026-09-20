"""State accounting tests for manual external exchange-account transfers."""

from decimal import Decimal
from types import SimpleNamespace

import pytest
from eth_defi.compat import native_datetime_utc_now
from eth_defi.token import USDC_NATIVE_TOKEN
from pytest_mock import MockerFixture

from tradeexecutor.ethereum.lighter.lighter_routing import LighterRouting
from tradeexecutor.exchange_account.lighter import create_lighter_exchange_account_pair
from tradeexecutor.exchange_account.state import (
    ExchangeAccountTransferError,
    create_exchange_account_transfer,
    mark_exchange_account_transfer_broadcasted,
    open_exchange_account_position,
    record_exchange_account_transfer,
)
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State, UncleanState
from tradeexecutor.state.trade import TradeExecution, TradeStatus
from tradeexecutor.strategy.account_correction import preflight_state_for_account_correction
from tradeexecutor.strategy.generic.generic_router import GenericRouting
from tradeexecutor.strategy.pandas_trader.runner import PandasTraderRunner


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


def test_runner_resumes_only_checkpointed_lighter_withdrawals(
    mocker: MockerFixture,
) -> None:
    """Resume a saved Lighter withdrawal but leave unsafe interrupted transfers unclean.

    1. Create a started automatic withdrawal with its public Lighter request ID.
    2. Resume it through the regular rebroadcast execution path.
    3. Verify a broadcasted automatic transfer is not resubmitted.
    """
    # 1. Create a started automatic withdrawal with its public Lighter request ID.
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
            "lighter_withdrawal_request_id": "withdrawal-request-1",
        },
        automatic=True,
    )
    withdrawal.started_at = native_datetime_utc_now()
    runner = object.__new__(PandasTraderRunner)
    runner.routing_model = GenericRouting(None)
    runner.execution_model = mocker.Mock()
    runner.setup_routing = mocker.Mock(return_value=(mocker.Mock(), None, None))
    store = mocker.Mock()

    # 2. Resume it through the regular rebroadcast execution path.
    runner.resume_pending_exchange_account_transfers(mocker.Mock(), state, store)
    runner.execution_model.execute_trades.assert_called_once()
    assert runner.execution_model.execute_trades.call_args.kwargs["rebroadcast"] is True

    # 3. Verify a broadcasted automatic transfer is not resubmitted.
    state, reserve_asset, position = _create_state()
    interrupted_transfer = create_exchange_account_transfer(
        state=state,
        position=position,
        strategy_cycle_at=native_datetime_utc_now(),
        reserve_currency=reserve_asset,
        amount=Decimal("1"),
        deposit=True,
        notes="Interrupted automatic Lighter transfer",
        metadata={"direction": "deposit"},
        automatic=True,
    )
    mark_exchange_account_transfer_broadcasted(
        interrupted_transfer,
        native_datetime_utc_now(),
    )
    runner.execution_model.reset_mock()
    runner.resume_pending_exchange_account_transfers(mocker.Mock(), state, store)
    runner.execution_model.execute_trades.assert_not_called()
    with pytest.raises(UncleanState):
        state.check_if_clean()
