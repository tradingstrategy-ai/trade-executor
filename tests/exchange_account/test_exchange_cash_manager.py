"""Tests for the pure external exchange cash-management policy."""

from decimal import Decimal

import pytest
from eth_defi.compat import native_datetime_utc_now
from eth_defi.lighter.valuation import LighterEquity
from eth_defi.token import USDC_NATIVE_TOKEN
from pytest_mock import MockerFixture

from tradeexecutor.exchange_account.cash_manager import (
    ExchangeCashManagementError,
    ExchangeCashManagementInput,
    ExchangeCashManager,
)
from tradeexecutor.exchange_account.lighter import (
    create_lighter_cash_management_transfer,
    create_lighter_exchange_account_pair,
    validate_lighter_cash_management_parameters,
)
from tradeexecutor.exchange_account.state import open_exchange_account_position
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeFlag


def _inputs(**overrides: str) -> ExchangeCashManagementInput:
    """Create a valid policy input with convenient test defaults."""
    values = {
        "safe_usdc": Decimal("100"),
        "exchange_available_usdc": Decimal("80"),
        "pending_redemptions_usdc": Decimal("0"),
        "safe_cash_buffer_usdc": Decimal("20"),
        "free_collateral_buffer_usdc": Decimal("5"),
        "minimum_transfer_usdc": Decimal("1"),
    }
    values.update({key: Decimal(value) for key, value in overrides.items()})
    return ExchangeCashManagementInput(**values)


def test_exchange_cash_manager_selects_safe_transfers() -> None:
    """Choose safe deposits and withdrawals without spending unavailable cash.

    1. Choose an idle Safe deposit while preserving the configured buffer.
    2. Choose a withdrawal when pending redemptions need more Safe liquidity.
    3. Return no action for pending, minimum-sized and fully buffered cash.
    4. Reject an invalid negative balance.
    """
    manager = ExchangeCashManager()

    # 1. Choose an idle Safe deposit while preserving the configured buffer.
    decision = manager.decide(_inputs())
    assert decision.direction == "deposit"
    assert decision.amount_usdc == Decimal("80")

    # 2. Choose a withdrawal when pending redemptions need more Safe liquidity.
    decision = manager.decide(
        _inputs(
            safe_usdc="10",
            exchange_available_usdc="80",
            pending_redemptions_usdc="40",
        )
    )
    assert decision.direction == "withdraw"
    assert decision.amount_usdc == Decimal("50")

    # 3. Return no action for pending, minimum-sized and fully buffered cash.
    assert not manager.decide(_inputs(), transfer_pending=True).should_transfer
    assert not manager.decide(_inputs(minimum_transfer_usdc="81")).should_transfer
    assert not manager.decide(
        _inputs(safe_usdc="20", safe_cash_buffer_usdc="20")
    ).should_transfer

    # 4. Reject an invalid negative balance.
    with pytest.raises(ExchangeCashManagementError):
        manager.decide(_inputs(safe_usdc="-1"))


def test_lighter_cash_management_trade_uses_the_normal_state_pipeline(
    mocker: MockerFixture,
) -> None:
    """Create an automatic Lighter deposit that the normal pipeline can execute.

    1. Create a Safe reserve and a zero-value Lighter exchange-account position.
    2. Mock only the public Lighter equity read and ask the strategy helper for a transfer.
    3. Start and complete the returned trade through normal state accounting.
    4. Verify reserve allocation, capital conservation and automatic-transfer flags.
    """
    # 1. Create a Safe reserve and a zero-value Lighter exchange-account position.
    reserve_asset = AssetIdentifier(
        chain_id=1,
        address=USDC_NATIVE_TOKEN[1],
        token_symbol="USDC",
        decimals=6,
    )
    state = State()
    state.portfolio.initialise_reserves(reserve_asset, reserve_token_price=1.0)
    state.portfolio.get_reserve_position(reserve_asset).quantity = Decimal("100")
    pair = create_lighter_exchange_account_pair(reserve_asset, account_index=1)
    open_exchange_account_position(
        state=state,
        strategy_cycle_at=native_datetime_utc_now(),
        pair=pair,
        reserve_currency=reserve_asset,
    )
    position = next(iter(state.portfolio.open_positions.values()))

    # 2. Mock only the public Lighter equity read and ask the strategy helper for a transfer.
    equity = LighterEquity(
        account_index=1,
        collateral=Decimal("0"),
        unrealised_pnl=Decimal("0"),
        total_asset_value=Decimal("0"),
        available_balance=Decimal("0"),
        initial_margin_requirement=Decimal("0"),
        maintenance_margin_requirement=Decimal("0"),
        position_count=0,
    )
    mocker.patch(
        "tradeexecutor.exchange_account.lighter.fetch_lighter_total_equity",
        return_value=equity,
    )
    trade = create_lighter_cash_management_transfer(
        state=state,
        position=position,
        strategy_cycle_at=native_datetime_utc_now(),
        reserve_currency=reserve_asset,
        safe_usdc=Decimal("100"),
        safe_cash_buffer_usdc=Decimal("20"),
        free_collateral_buffer_usdc=Decimal("0"),
        minimum_transfer_usdc=Decimal("1"),
        session=mocker.Mock(),
    )

    # 3. Start and complete the returned trade through normal state accounting.
    assert trade is not None
    state.start_execution(native_datetime_utc_now(), trade, underflow_check=True)
    state.mark_trade_success(
        executed_at=native_datetime_utc_now(),
        trade=trade,
        executed_price=1.0,
        executed_amount=Decimal("80"),
        executed_reserve=Decimal("80"),
        lp_fees=0,
        native_token_price=0,
        force=True,
    )

    # 4. Verify reserve allocation, capital conservation and automatic-transfer flags.
    assert trade.planned_quantity == Decimal("80")
    assert trade.planned_reserve == Decimal("80")
    assert TradeFlag.external_account_transfer in trade.flags
    assert TradeFlag.automatic_exchange_account_transfer in trade.flags
    assert trade.is_success()
    assert state.portfolio.get_reserve_position(reserve_asset).quantity == Decimal("20")
    assert position.get_quantity() == Decimal("80")
    assert state.portfolio.get_net_asset_value() == pytest.approx(Decimal("100"))


def test_exchange_cash_management_parameters_are_validated() -> None:
    """Reject unsafe automatic exchange cash-management configuration.

    1. Accept the documented default timeout and non-negative buffers.
    2. Reject a non-positive withdrawal timeout and negative transfer buffer.
    """
    # 1. Accept the documented default timeout and non-negative buffers.
    validate_lighter_cash_management_parameters({})

    # 2. Reject a non-positive withdrawal timeout and negative transfer buffer.
    with pytest.raises(ExchangeCashManagementError, match="withdrawal_timeout"):
        validate_lighter_cash_management_parameters({"lighter_withdrawal_timeout": 0})
    with pytest.raises(ExchangeCashManagementError, match="safe_cash_buffer"):
        validate_lighter_cash_management_parameters({"lighter_safe_cash_buffer_usd": -1})
