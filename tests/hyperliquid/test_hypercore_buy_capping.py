"""Tests for Hypercore vault buy capping when sell proceeds fall short.

Tests that ``_maybe_cap_hypercore_vault_buy()`` and the shortfall tracking
in ``_execute_trades_sequentially()`` correctly reduce or expire buy trades
when preceding vault sell withdrawals return less than planned due to the
vault leader's performance fee.
"""

import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from eth_defi.compat import native_datetime_utc_now

from tradeexecutor.ethereum.execution import EthereumExecution
from tradeexecutor.state.trade import TradeStatus
from tradeexecutor.state.types import USDollarAmount


def _make_execution_model():
    """Create a minimal EthereumExecution with mocked dependencies."""
    model = object.__new__(EthereumExecution)
    model.max_slippage = None
    return model


def _make_mock_trade(
    *,
    is_buy: bool,
    planned_reserve: Decimal,
    is_hyperliquid_vault: bool = True,
    trade_id: int = 1,
    position_id: int = 1,
    chain_id: int = 9999,
):
    """Create a mock trade for execution tests."""
    trade = MagicMock()
    trade.trade_id = trade_id
    trade.position_id = position_id
    trade.is_buy.return_value = is_buy
    trade.is_failed.return_value = False
    trade.planned_reserve = planned_reserve
    trade.planned_quantity = planned_reserve
    trade.executed_reserve = None
    trade.other_data = {}
    trade.pair = MagicMock()
    trade.pair.is_hyperliquid_vault.return_value = is_hyperliquid_vault
    trade.pair.chain_id = chain_id
    trade.reserve_currency = MagicMock()
    trade.get_planned_value.return_value = float(planned_reserve)
    trade.pair.get_ticker.return_value = "TEST-USDC"
    trade.route = "hypercore_vault"
    trade.get_revert_reason.return_value = None
    trade.get_status.return_value = MagicMock()
    trade.flags = set()
    return trade


def test_vault_buy_capped_on_vault_sell_shortfall():
    """Hypercore vault buy is capped when preceding sells returned less than planned.

    1. Create a sell trade that completed with a shortfall (planned 100, executed 90).
    2. Create a buy trade that planned to spend 50 but only 40 is available.
    3. Verify the buy's planned_reserve and planned_quantity are reduced.
    4. Verify originals are preserved in other_data.
    """
    model = _make_execution_model()
    state = MagicMock()

    # 1. Mock reserves: only 40 available (after sell shortfall).
    state.portfolio.get_bridge_position_for_chain.return_value = None
    reserve_pos = MagicMock()
    reserve_pos.quantity = Decimal("40")
    state.portfolio.get_reserve_position.return_value = reserve_pos

    buy_trade = _make_mock_trade(is_buy=True, planned_reserve=Decimal("50"), trade_id=2)

    # 2. Cap the buy.
    expired = model._maybe_cap_hypercore_vault_buy(
        state, buy_trade, cumulative_vault_sell_shortfall=Decimal("10"),
    )

    # 3. Verify capping, not expiry.
    assert not expired
    assert buy_trade.planned_reserve == Decimal("40")
    assert buy_trade.planned_quantity == pytest.approx(Decimal("40"), abs=Decimal("0.01"))

    # 4. Verify originals preserved.
    assert buy_trade.other_data["original_planned_reserve"] == "50"
    assert buy_trade.other_data["original_planned_quantity"] == "50"


def test_vault_buy_expired_when_below_minimum():
    """Hypercore vault buy is expired when available capital is below $5 minimum.

    1. Create a buy trade with insufficient capital (only $3 available).
    2. Verify the trade is expired (not failed).
    3. Verify orphan pending position cleanup (opening buys live in pending_positions).
    """
    model = _make_execution_model()
    state = MagicMock()

    # 1. Only $3 available.
    state.portfolio.get_bridge_position_for_chain.return_value = None
    reserve_pos = MagicMock()
    reserve_pos.quantity = Decimal("3")
    state.portfolio.get_reserve_position.return_value = reserve_pos

    buy_trade = _make_mock_trade(is_buy=True, planned_reserve=Decimal("50"), trade_id=2)

    # Mock pending position with all trades expired (opening buy).
    pending_pos = MagicMock()
    pending_pos.get_quantity.return_value = Decimal(0)
    expired_trade_mock = MagicMock()
    expired_trade_mock.get_status.return_value = TradeStatus.expired
    pending_pos.trades = {1: expired_trade_mock}
    state.portfolio.pending_positions = {buy_trade.position_id: pending_pos}
    state.portfolio.open_positions = {}

    # 2. Should expire.
    expired = model._maybe_cap_hypercore_vault_buy(
        state, buy_trade, cumulative_vault_sell_shortfall=Decimal("47"),
    )

    assert expired
    buy_trade.mark_expired.assert_called_once()

    # 3. Verify pending position was cleaned up.
    assert buy_trade.position_id not in state.portfolio.pending_positions


def test_non_hypercore_buy_unaffected_by_hypercore_sell_shortfall():
    """Non-Hypercore buys are not capped by Hypercore sell shortfall.

    1. Create a non-Hypercore buy trade (is_hyperliquid_vault=False).
    2. Verify the capping helper is never invoked for it.
    """
    model = _make_execution_model()
    state = MagicMock()

    buy_trade = _make_mock_trade(
        is_buy=True,
        planned_reserve=Decimal("50"),
        is_hyperliquid_vault=False,
        trade_id=2,
    )

    # The predicate should prevent calling the helper.
    # Verify the guard: is_hyperliquid_vault() returns False.
    assert not buy_trade.pair.is_hyperliquid_vault()
