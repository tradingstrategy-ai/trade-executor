"""Test state pruning functionality."""
import datetime
import json
from decimal import Decimal

import pytest

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.prune import prune_closed_position, prune_closed_positions
from tradeexecutor.state.state import State

from tradeexecutor.state.balance_update import BalanceUpdate, BalanceUpdateCause, BalanceUpdatePositionType
from tradingstrategy.chain import ChainId


@pytest.fixture
def mock_assets():
    """Create mock assets for testing."""
    usdc = AssetIdentifier(ChainId.ethereum.value, "0x1", "USDC", 6, 1)
    weth = AssetIdentifier(ChainId.ethereum.value, "0x2", "WETH", 18, 2)
    return usdc, weth


@pytest.fixture
def mock_trading_pair(mock_assets):
    """Create a mock trading pair."""
    usdc, weth = mock_assets
    return TradingPairIdentifier(
        weth,
        usdc,
        "0x3",
        "0x4",
        internal_id=1,
        internal_exchange_id=1
    )


def create_position_with_balance_updates(
    mock_trading_pair,
    position_id: int,
    balance_update_count: int,
    is_closed: bool = True
):
    """Create a position with specified number of balance updates."""
    position = TradingPosition(
        position_id=position_id,
        pair=mock_trading_pair,
        opened_at=datetime.datetime(2023, 1, 1),
        last_pricing_at=datetime.datetime(2023, 1, 1),
        last_token_price=Decimal("2000"),
        last_reserve_price=Decimal("1"),
        reserve_currency=mock_trading_pair.quote,
    )

    # Add balance updates
    for i in range(balance_update_count):
        balance_update = BalanceUpdate(
            balance_update_id=position_id * 10 + i,
            position_id=position_id,
            cause=BalanceUpdateCause.deposit,
            position_type=BalanceUpdatePositionType.open_position,
            asset=mock_trading_pair.base,
            block_mined_at=datetime.datetime(2023, 1, 1) + datetime.timedelta(hours=i),
            strategy_cycle_included_at=datetime.datetime(2023, 1, 1) + datetime.timedelta(hours=i),
            chain_id=ChainId.ethereum.value,
            old_balance=Decimal("0"),
            usd_value=Decimal("100"),
            quantity=Decimal("0.05"),
            owner_address="0x5",
            tx_hash=f"0x{position_id}{i}",
            log_index=1,
        )
        position.balance_updates[position_id * 10 + i] = balance_update

    # Close the position if requested
    if is_closed:
        position.closed_at = datetime.datetime(2023, 1, 2)

    return position


@pytest.fixture
def closed_position_with_balance_updates(mock_trading_pair):
    """Create a closed position with balance updates."""
    return create_position_with_balance_updates(mock_trading_pair, 1, 5, is_closed=True)


@pytest.fixture
def open_position_with_balance_updates(mock_trading_pair):
    """Create an open position with balance updates."""
    return create_position_with_balance_updates(mock_trading_pair, 2, 3, is_closed=False)


def test_prune_closed_position_success(closed_position_with_balance_updates):
    """Test successfully pruning balance updates from a closed position."""
    position = closed_position_with_balance_updates

    # Verify position has balance updates before pruning
    assert len(position.balance_updates) == 5
    assert position.is_closed()

    # Prune the position
    removed_updates = prune_closed_position(position)

    # Verify pruning results
    assert len(removed_updates) == 5
    assert len(position.balance_updates) == 0

    # Verify we got the actual balance update objects back
    assert all(isinstance(update, BalanceUpdate) for update in removed_updates.values())
    assert all(update.position_id == 1 for update in removed_updates.values())


def test_prune_open_position_fails(open_position_with_balance_updates):
    """Test that pruning an open position raises ValueError."""
    position = open_position_with_balance_updates

    # Verify position is open
    assert not position.is_closed()
    assert len(position.balance_updates) == 3

    # Attempt to prune should fail
    with pytest.raises(ValueError, match="Cannot prune open position"):
        prune_closed_position(position)

    # Position should be unchanged
    assert len(position.balance_updates) == 3


def test_prune_closed_position_no_balance_updates(mock_trading_pair):
    """Test pruning a closed position with no balance updates."""
    position = TradingPosition(
        position_id=3,
        pair=mock_trading_pair,
        opened_at=datetime.datetime(2023, 1, 1),
        last_pricing_at=datetime.datetime(2023, 1, 1),
        last_token_price=Decimal("2000"),
        last_reserve_price=Decimal("1"),
        reserve_currency=mock_trading_pair.quote,
    )
    position.closed_at = datetime.datetime(2023, 1, 2)

    # Prune position with no balance updates
    removed_updates = prune_closed_position(position)

    assert len(removed_updates) == 0
    assert len(position.balance_updates) == 0


def test_prune_closed_positions_bulk(mock_assets):
    """Test bulk pruning of all closed positions in state."""
    usdc, weth = mock_assets

    # Create state
    state = State()
    state.portfolio = Portfolio()

    # Create mock trading pair
    pair = TradingPairIdentifier(
        weth, usdc, "0x3", "0x4", internal_id=1, internal_exchange_id=1
    )

    # Add closed positions with balance updates (2, 3, 4 balance updates respectively)
    for pos_id in range(3):
        position = create_position_with_balance_updates(
            pair,
            pos_id + 1,
            pos_id + 2,
            is_closed=True
        )
        state.portfolio.closed_positions[pos_id + 1] = position

    # Add an open position that should not be pruned
    open_position = create_position_with_balance_updates(
        pair,
        999,
        1,
        is_closed=False
    )
    state.portfolio.open_positions[999] = open_position

    # Verify initial state
    assert len(state.portfolio.closed_positions) == 3
    assert len(state.portfolio.open_positions) == 1
    total_initial_balance_updates = sum(
        len(pos.balance_updates) for pos in state.portfolio.closed_positions.values()
    )
    assert total_initial_balance_updates == 2 + 3 + 4  # 9 total
    assert len(state.portfolio.open_positions[999].balance_updates) == 1

    # Prune all closed positions
    result = prune_closed_positions(state)

    # Verify results
    assert result["positions_processed"] == 3
    assert result["balance_updates_removed"] == 9
    assert "bytes_saved" in result
    assert result["bytes_saved"] > 0  # Should have calculated some bytes saved

    # Verify all closed positions have no balance updates
    for position in state.portfolio.closed_positions.values():
        assert len(position.balance_updates) == 0

    # Verify open position is unchanged
    assert len(state.portfolio.open_positions[999].balance_updates) == 1


def test_prune_closed_positions_no_closed_positions():
    """Test bulk pruning when there are no closed positions."""
    # Create empty state
    state = State()
    state.portfolio = Portfolio()

    # Prune when no closed positions exist
    result = prune_closed_positions(state)

    assert result["positions_processed"] == 0
    assert result["balance_updates_removed"] == 0
    assert result["bytes_saved"] == 0


def test_prune_closed_positions_bytes_calculation(mock_assets):
    """Test that bytes calculation is reasonable and consistent."""
    usdc, weth = mock_assets

    # Create state with one closed position with balance updates
    state = State()
    state.portfolio = Portfolio()

    pair = TradingPairIdentifier(
        weth, usdc, "0x3", "0x4", internal_id=1, internal_exchange_id=1
    )

    position = create_position_with_balance_updates(pair, 1, 3, is_closed=True)
    state.portfolio.closed_positions[1] = position

    # Prune and check bytes calculation
    result = prune_closed_positions(state)

    assert result["positions_processed"] == 1
    assert result["balance_updates_removed"] == 3
    assert result["bytes_saved"] > 0

    # Bytes saved should be reasonable (each balance update is substantial JSON)
    # Rough check: should be at least 100 bytes per update, but not crazy large
    assert result["bytes_saved"] > 300  # At least 100 bytes per update
    assert result["bytes_saved"] < 10000  # But not unreasonably large
