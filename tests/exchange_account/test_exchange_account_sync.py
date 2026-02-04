"""Test exchange account sync model with mocked balance changes.

These tests verify that the ExchangeAccountSyncModel correctly:
1. Detects account value changes between syncs
2. Generates BalanceUpdate events for the difference (PnL)
3. Updates position quantity to match actual account value
"""

import datetime
from decimal import Decimal
from unittest.mock import Mock

import pytest

from tradeexecutor.state.state import State
from tradeexecutor.state.identifier import (
    TradingPairIdentifier,
    AssetIdentifier,
    TradingPairKind,
)
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeExecution, TradeType
from tradeexecutor.state.balance_update import BalanceUpdateCause, BalanceUpdatePositionType
from tradeexecutor.exchange_account.sync_model import ExchangeAccountSyncModel


@pytest.fixture
def exchange_account_pair():
    """Create exchange account pair for testing."""
    usdc = AssetIdentifier(
        chain_id=901,
        address="0x0000000000000000000000000000000000000001",
        token_symbol="USDC",
        decimals=6,
    )
    derive_account = AssetIdentifier(
        chain_id=901,
        address="0x0000000000000000000000000000000000000002",
        token_symbol="DERIVE-ACCOUNT",
        decimals=6,
    )
    return TradingPairIdentifier(
        base=derive_account,
        quote=usdc,
        pool_address="0x0000000000000000000000000000000000000003",
        exchange_address="0x0000000000000000000000000000000000000004",
        internal_id=1,
        internal_exchange_id=1,
        fee=0.0,
        kind=TradingPairKind.exchange_account,
        exchange_name="derive",
        other_data={
            "exchange_protocol": "derive",
            "exchange_subaccount_id": 1,
            "exchange_is_testnet": True,
        },
    )


@pytest.fixture
def state_with_position(exchange_account_pair):
    """Create state with an open exchange account position (100k deposit)."""
    state = State()
    opened_at = datetime.datetime(2024, 1, 1)

    position = TradingPosition(
        position_id=1,
        pair=exchange_account_pair,
        opened_at=opened_at,
        last_pricing_at=opened_at,
        last_token_price=1.0,
        last_reserve_price=1.0,
        reserve_currency=exchange_account_pair.quote,
    )

    # Add initial deposit trade (100k)
    trade = TradeExecution(
        trade_id=1,
        position_id=1,
        trade_type=TradeType.rebalance,
        pair=exchange_account_pair,
        opened_at=opened_at,
        planned_quantity=Decimal("100000.0"),
        planned_price=1.0,
        planned_reserve=Decimal("100000.0"),
        reserve_currency=exchange_account_pair.quote,
    )
    trade.started_at = opened_at
    trade.mark_broadcasted(opened_at)
    trade.mark_success(
        executed_at=opened_at,
        executed_price=1.0,
        executed_quantity=Decimal("100000.0"),
        executed_reserve=Decimal("100000.0"),
        lp_fees=0.0,
        native_token_price=1.0,
    )
    position.trades[1] = trade

    state.portfolio.open_positions[1] = position
    return state


def test_sync_detects_profit(state_with_position, exchange_account_pair):
    """Test that sync detects profit and creates positive balance update."""
    # Mock: account value increased from 100k to 105k (5k profit)
    mock_value_func = Mock(return_value=Decimal("105000.0"))

    sync_model = ExchangeAccountSyncModel(mock_value_func)

    events = sync_model.sync_positions(
        timestamp=datetime.datetime.utcnow(),
        state=state_with_position,
        strategy_universe=None,
        pricing_model=None,
    )

    assert len(events) == 1
    evt = events[0]
    assert evt.quantity == Decimal("5000.0")
    assert evt.old_balance == Decimal("100000.0")
    assert evt.usd_value == 5000.0
    assert evt.cause == BalanceUpdateCause.vault_flow
    assert evt.position_type == BalanceUpdatePositionType.open_position
    assert evt.position_id == 1
    assert "derive" in evt.notes.lower()


def test_sync_detects_loss(state_with_position, exchange_account_pair):
    """Test that sync detects loss and creates negative balance update."""
    # Mock: account value decreased from 100k to 95k (5k loss)
    mock_value_func = Mock(return_value=Decimal("95000.0"))

    sync_model = ExchangeAccountSyncModel(mock_value_func)

    events = sync_model.sync_positions(
        timestamp=datetime.datetime.utcnow(),
        state=state_with_position,
        strategy_universe=None,
        pricing_model=None,
    )

    assert len(events) == 1
    evt = events[0]
    assert evt.quantity == Decimal("-5000.0")
    assert evt.old_balance == Decimal("100000.0")
    assert evt.usd_value == -5000.0


def test_sync_no_change_no_event(state_with_position, exchange_account_pair):
    """Test that no event is created when balance unchanged."""
    # Mock: account value unchanged at 100k
    mock_value_func = Mock(return_value=Decimal("100000.0"))

    sync_model = ExchangeAccountSyncModel(mock_value_func)

    events = sync_model.sync_positions(
        timestamp=datetime.datetime.utcnow(),
        state=state_with_position,
        strategy_universe=None,
        pricing_model=None,
    )

    assert len(events) == 0


def test_sync_stores_balance_update_on_position(state_with_position, exchange_account_pair):
    """Test that balance update is stored on position."""
    mock_value_func = Mock(return_value=Decimal("110000.0"))

    sync_model = ExchangeAccountSyncModel(mock_value_func)

    sync_model.sync_positions(
        timestamp=datetime.datetime.utcnow(),
        state=state_with_position,
        strategy_universe=None,
        pricing_model=None,
    )

    position = state_with_position.portfolio.open_positions[1]
    assert len(position.balance_updates) == 1

    # Balance update should be accessible by ID
    evt = list(position.balance_updates.values())[0]
    assert evt.quantity == Decimal("10000.0")


def test_sync_updates_position_quantity(state_with_position, exchange_account_pair):
    """Test that position quantity reflects balance update after sync."""
    mock_value_func = Mock(return_value=Decimal("110000.0"))

    sync_model = ExchangeAccountSyncModel(mock_value_func)

    sync_model.sync_positions(
        timestamp=datetime.datetime.utcnow(),
        state=state_with_position,
        strategy_universe=None,
        pricing_model=None,
    )

    position = state_with_position.portfolio.open_positions[1]
    # Position quantity should now include the balance update
    assert position.get_quantity() == Decimal("110000.0")


def test_sync_allocates_unique_event_ids(state_with_position, exchange_account_pair):
    """Test that balance update IDs are allocated correctly."""
    initial_next_id = state_with_position.portfolio.next_balance_update_id

    # First sync: profit
    mock_value_func = Mock(return_value=Decimal("105000.0"))
    sync_model = ExchangeAccountSyncModel(mock_value_func)

    events1 = sync_model.sync_positions(
        timestamp=datetime.datetime.utcnow(),
        state=state_with_position,
        strategy_universe=None,
        pricing_model=None,
    )

    # Second sync: more profit
    mock_value_func.return_value = Decimal("108000.0")

    events2 = sync_model.sync_positions(
        timestamp=datetime.datetime.utcnow(),
        state=state_with_position,
        strategy_universe=None,
        pricing_model=None,
    )

    assert len(events1) == 1
    assert len(events2) == 1
    assert events1[0].balance_update_id == initial_next_id
    assert events2[0].balance_update_id == initial_next_id + 1
    assert state_with_position.portfolio.next_balance_update_id == initial_next_id + 2


def test_sync_tracks_accounting_refs(state_with_position, exchange_account_pair):
    """Test that balance update refs are tracked in accounting."""
    mock_value_func = Mock(return_value=Decimal("105000.0"))
    sync_model = ExchangeAccountSyncModel(mock_value_func)

    initial_refs = len(state_with_position.sync.accounting.balance_update_refs)

    sync_model.sync_positions(
        timestamp=datetime.datetime.utcnow(),
        state=state_with_position,
        strategy_universe=None,
        pricing_model=None,
    )

    assert len(state_with_position.sync.accounting.balance_update_refs) == initial_refs + 1
    assert state_with_position.sync.accounting.last_updated_at is not None


def test_sync_handles_api_error_gracefully(state_with_position, exchange_account_pair):
    """Test that API errors don't crash sync, just skip the position."""
    mock_value_func = Mock(side_effect=Exception("API unreachable"))

    sync_model = ExchangeAccountSyncModel(mock_value_func)

    # Should not raise, just return empty list
    events = sync_model.sync_positions(
        timestamp=datetime.datetime.utcnow(),
        state=state_with_position,
        strategy_universe=None,
        pricing_model=None,
    )

    assert len(events) == 0


def test_sync_ignores_non_exchange_account_positions(state_with_position, exchange_account_pair):
    """Test that sync only processes exchange_account positions."""
    # Add a spot position (should be ignored)
    spot_pair = TradingPairIdentifier(
        base=AssetIdentifier(chain_id=1, address="0x10", token_symbol="WETH", decimals=18),
        quote=AssetIdentifier(chain_id=1, address="0x11", token_symbol="USDC", decimals=6),
        pool_address="0x12",
        exchange_address="0x13",
        kind=TradingPairKind.spot_market_hold,
    )
    spot_position = TradingPosition(
        position_id=2,
        pair=spot_pair,
        opened_at=datetime.datetime(2024, 1, 1),
        last_pricing_at=datetime.datetime(2024, 1, 1),
        last_token_price=1.0,
        last_reserve_price=1.0,
        reserve_currency=spot_pair.quote,
    )
    state_with_position.portfolio.open_positions[2] = spot_position

    # Mock should only be called for exchange_account position
    mock_value_func = Mock(return_value=Decimal("105000.0"))
    sync_model = ExchangeAccountSyncModel(mock_value_func)

    events = sync_model.sync_positions(
        timestamp=datetime.datetime.utcnow(),
        state=state_with_position,
        strategy_universe=None,
        pricing_model=None,
    )

    # Only 1 event for exchange_account position, not for spot
    assert len(events) == 1
    assert mock_value_func.call_count == 1


def test_sync_treasury_returns_empty(state_with_position):
    """Test that sync_treasury returns empty list (not used for exchange accounts)."""
    mock_value_func = Mock()
    sync_model = ExchangeAccountSyncModel(mock_value_func)

    events = sync_model.sync_treasury(
        strategy_cycle_ts=datetime.datetime.utcnow(),
        state=state_with_position,
    )

    assert events == []


def test_has_position_sync_returns_true():
    """Test that has_position_sync returns True."""
    mock_value_func = Mock()
    sync_model = ExchangeAccountSyncModel(mock_value_func)

    assert sync_model.has_position_sync() is True
