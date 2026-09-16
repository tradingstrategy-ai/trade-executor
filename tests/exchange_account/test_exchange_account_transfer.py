"""State accounting tests for manual external exchange-account transfers."""

from decimal import Decimal

import pytest
from eth_defi.compat import native_datetime_utc_now
from eth_defi.token import USDC_NATIVE_TOKEN

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
from tradeexecutor.state.trade import TradeExecution
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

    1. Create planned, started and broadcasted manual transfer states.
    2. Verify each transfer makes state and account-correction preflight unclean.
    3. Preserve the legacy unfinished predicate for planned and started trades.
    """
    # 1. Create planned, started and broadcasted manual transfer states.
    for phase in ("planned", "started", "broadcasted"):
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

        # 2. Verify each transfer makes state and account-correction preflight unclean.
        with pytest.raises(UncleanState):
            state.check_if_clean()
        with pytest.raises(UncleanState):
            preflight_state_for_account_correction(state)

        # 3. Preserve the legacy unfinished predicate for planned and started trades.
        assert trade.is_unfinished() is (phase == "broadcasted")
