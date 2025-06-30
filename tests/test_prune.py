"""Test state pruning functionality."""
import datetime
from decimal import Decimal

import pytest

from tradeexecutor.state.balance_update import BalanceUpdate, BalanceUpdateCause, BalanceUpdatePositionType
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeType
from tradeexecutor.state.prune import prune_blockchain_transaction, prune_trade, prune_closed_position, prune_closed_positions

from tradingstrategy.chain import ChainId


def create_position(mock_trading_pair, position_id: int, is_closed: bool = True):
    """Create a basic position."""
    position = TradingPosition(
        position_id=position_id,
        pair=mock_trading_pair,
        opened_at=datetime.datetime(2023, 1, 1),
        last_pricing_at=datetime.datetime(2023, 1, 1),
        last_token_price=Decimal("2000"),
        last_reserve_price=Decimal("1"),
        reserve_currency=mock_trading_pair.quote,
    )

    # Close the position if requested
    if is_closed:
        position.closed_at = datetime.datetime(2023, 1, 2)

    return position


def add_balance_updates(position: TradingPosition, count: int):
    """Add balance updates to a position."""
    position_id = position.position_id
    for i in range(count):
        balance_update = BalanceUpdate(
            balance_update_id=position_id * 10 + i,
            position_id=position_id,
            cause=BalanceUpdateCause.deposit,
            position_type=BalanceUpdatePositionType.open_position,
            asset=position.pair.base,
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


def create_position_with_balance_updates(
    mock_trading_pair,
    position_id: int,
    balance_update_count: int,
    is_closed: bool = True
):
    """Create a position with specified number of balance updates."""
    position = create_position(mock_trading_pair, position_id, is_closed)
    add_balance_updates(position, balance_update_count)
    return position


def create_blockchain_transaction(tx_id: int = 1):
    """Create a blockchain transaction with prunable data."""
    tx = BlockchainTransaction()
    tx.tx_hash = f"0x{tx_id:064x}"
    tx.transaction_args = ("arg1", "arg2", 12345)
    tx.wrapped_args = ("wrapped_arg1", "wrapped_arg2")
    tx.signed_bytes = f"0x{tx_id:032x}"
    tx.signed_tx_object = f"0x{tx_id:048x}"
    tx.details = {"data": "0x123456", "gas": 21000, "gasPrice": 20000000000}
    return tx


def create_trade(mock_trading_pair, trade_id: int, position_id: int):
    """Create a basic trade."""
    return TradeExecution(
        trade_id=trade_id,
        position_id=position_id,
        trade_type=TradeType.rebalance,
        pair=mock_trading_pair,
        opened_at=datetime.datetime(2023, 1, 1),
        planned_quantity=Decimal("1.0"),
        planned_price=2000.0,
        planned_reserve=Decimal("2000"),
        reserve_currency=mock_trading_pair.quote,
    )


def add_blockchain_transactions(trade: TradeExecution, count: int):
    """Add blockchain transactions to a trade."""
    trade_id = trade.trade_id
    for i in range(count):
        tx = create_blockchain_transaction(trade_id * 10 + i)
        trade.blockchain_transactions.append(tx)


def add_trades(position: TradingPosition, mock_trading_pair, trade_count: int, blockchain_tx_per_trade: int = 2):
    """Add trades with blockchain transactions to a position."""
    position_id = position.position_id
    for i in range(trade_count):
        trade_id = position_id * 100 + i
        trade = create_trade(mock_trading_pair, trade_id, position_id)
        add_blockchain_transactions(trade, blockchain_tx_per_trade)
        position.trades[trade_id] = trade


def create_trade_with_blockchain_transactions(
    mock_trading_pair,
    trade_id: int,
    position_id: int,
    blockchain_tx_count: int = 2
):
    """Create a trade with blockchain transactions."""
    trade = create_trade(mock_trading_pair, trade_id, position_id)
    add_blockchain_transactions(trade, blockchain_tx_count)
    return trade


def create_position_with_trades(
    mock_trading_pair,
    position_id: int,
    trade_count: int,
    blockchain_tx_per_trade: int = 2,
    is_closed: bool = True
):
    """Create a position with trades that have blockchain transactions."""
    position = create_position(mock_trading_pair, position_id, is_closed)
    add_trades(position, mock_trading_pair, trade_count, blockchain_tx_per_trade)
    return position


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


@pytest.fixture
def closed_position_with_balance_updates(mock_trading_pair):
    """Create a closed position with balance updates."""
    return create_position_with_balance_updates(mock_trading_pair, 1, 5, is_closed=True)


@pytest.fixture
def open_position_with_balance_updates(mock_trading_pair):
    """Create an open position with balance updates."""
    return create_position_with_balance_updates(mock_trading_pair, 2, 3, is_closed=False)


def test_prune_blockchain_transaction():
    """Test pruning individual blockchain transaction."""
    tx = create_blockchain_transaction()

    # Verify transaction has prunable data
    assert tx.transaction_args is not None
    assert tx.wrapped_args is not None
    assert tx.signed_bytes is not None
    assert tx.signed_tx_object is not None
    assert tx.details is not None
    assert tx.details["data"] is not None

    # Prune the transaction
    prune_blockchain_transaction(tx)

    # Verify data was pruned
    assert tx.transaction_args is None
    assert tx.wrapped_args is None
    assert tx.signed_bytes is None
    assert tx.signed_tx_object is None
    assert tx.details["data"] is None
    # Other details should remain
    assert tx.details["gas"] == 21000
    assert tx.details["gasPrice"] == 20000000000


def test_prune_blockchain_transaction_no_details():
    """Test pruning blockchain transaction with None details."""
    tx = create_blockchain_transaction()
    tx.details = None

    # Should not crash when details is None
    prune_blockchain_transaction(tx)

    assert tx.transaction_args is None
    assert tx.wrapped_args is None
    assert tx.signed_bytes is None
    assert tx.signed_tx_object is None


def test_prune_trade(mock_trading_pair):
    """Test pruning individual trade execution."""
    trade = create_trade_with_blockchain_transactions(mock_trading_pair, 1, 1, 3)

    # Verify trade has blockchain transactions with prunable data
    assert len(trade.blockchain_transactions) == 3
    for tx in trade.blockchain_transactions:
        assert tx.transaction_args is not None
        assert tx.wrapped_args is not None

    # Prune the trade
    stats = prune_trade(trade)

    # Verify stats
    assert stats['blockchain_transactions_processed'] == 3
    assert stats['trades_processed'] == 1

    # Verify all blockchain transactions were pruned
    assert len(trade.blockchain_transactions) == 3  # Count should remain same
    for tx in trade.blockchain_transactions:
        assert tx.transaction_args is None
        assert tx.wrapped_args is None
        assert tx.signed_bytes is None
        assert tx.signed_tx_object is None


def test_prune_trade_no_blockchain_transactions(mock_trading_pair):
    """Test pruning trade with no blockchain transactions."""
    trade = create_trade_with_blockchain_transactions(mock_trading_pair, 1, 1, 0)

    # Verify trade has no blockchain transactions
    assert len(trade.blockchain_transactions) == 0

    # Prune the trade
    stats = prune_trade(trade)

    # Verify stats
    assert stats['blockchain_transactions_processed'] == 0
    assert stats['trades_processed'] == 1


def test_prune_closed_position_success(closed_position_with_balance_updates):
    """Test successfully pruning balance updates from a closed position."""
    position = closed_position_with_balance_updates

    # Verify position has balance updates before pruning
    assert len(position.balance_updates) == 5
    assert position.is_closed()

    # Prune the position
    stats = prune_closed_position(position)

    # Verify pruning results
    assert stats['balance_updates_removed'] == 5
    assert len(position.balance_updates) == 0


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
    stats = prune_closed_position(position)

    assert stats['balance_updates_removed'] == 0
    assert len(position.balance_updates) == 0


def test_prune_closed_position_with_trades(mock_trading_pair):
    """Test pruning closed position with trades and blockchain transactions."""
    position = create_position_with_trades(mock_trading_pair, 1, 2, 3, is_closed=True)

    # Add some balance updates too
    add_balance_updates(position, 2)

    # Verify initial state
    assert len(position.trades) == 2
    assert len(position.balance_updates) == 2
    total_blockchain_txs = sum(len(trade.blockchain_transactions) for trade in position.trades.values())
    assert total_blockchain_txs == 6  # 2 trades * 3 blockchain txs each

    # Verify blockchain transactions have prunable data
    for trade in position.trades.values():
        for tx in trade.blockchain_transactions:
            assert tx.transaction_args is not None

    # Prune the position
    stats = prune_closed_position(position)

    # Verify stats
    assert stats['positions_processed'] == 1
    assert stats['balance_updates_removed'] == 2
    assert stats['blockchain_transactions_processed'] == 6
    assert stats['trades_processed'] == 2

    # Verify balance updates were removed
    assert len(position.balance_updates) == 0

    # Verify trades remain but blockchain transactions were pruned
    assert len(position.trades) == 2
    for trade in position.trades.values():
        assert len(trade.blockchain_transactions) == 3  # Count remains
        for tx in trade.blockchain_transactions:
            assert tx.transaction_args is None
            assert tx.wrapped_args is None


def test_prune_closed_position_combined_balance_updates_and_trades(mock_trading_pair):
    """Test pruning position with both balance updates and trades with blockchain transactions."""
    position = create_position_with_trades(mock_trading_pair, 1, 2, 2, is_closed=True)

    # Add balance updates
    add_balance_updates(position, 3)

    # Verify initial state
    assert len(position.trades) == 2
    assert len(position.balance_updates) == 3
    total_blockchain_txs = sum(len(trade.blockchain_transactions) for trade in position.trades.values())
    assert total_blockchain_txs == 4  # 2 trades * 2 blockchain txs each

    # Verify blockchain transactions have prunable data before pruning
    for trade in position.trades.values():
        for tx in trade.blockchain_transactions:
            assert tx.transaction_args is not None
            assert tx.wrapped_args is not None

    # Prune the position
    stats = prune_closed_position(position)

    # Verify comprehensive stats from both balance updates and trades
    assert stats['positions_processed'] == 1
    assert stats['balance_updates_removed'] == 3
    assert stats['blockchain_transactions_processed'] == 4
    assert stats['trades_processed'] == 2

    # Verify balance updates were removed
    assert len(position.balance_updates) == 0

    # Verify trades remain but blockchain transactions were pruned
    assert len(position.trades) == 2
    for trade in position.trades.values():
        assert len(trade.blockchain_transactions) == 2  # Count remains same
        for tx in trade.blockchain_transactions:
            assert tx.transaction_args is None
            assert tx.wrapped_args is None
            assert tx.signed_bytes is None
            assert tx.signed_tx_object is None


def test_prune_closed_positions_bulk(mock_assets):
    """Test bulk pruning of all closed positions in state with both balance updates and trades."""
    usdc, weth = mock_assets

    # Create state
    state = State()
    state.portfolio = Portfolio()

    # Create mock trading pair
    pair = TradingPairIdentifier(
        weth, usdc, "0x3", "0x4", internal_id=1, internal_exchange_id=1
    )

    # Position 1: 2 balance updates, 1 trade with 2 blockchain txs
    position1 = create_position(pair, 1, is_closed=True)
    add_balance_updates(position1, 2)
    add_trades(position1, pair, 1, 2)
    state.portfolio.closed_positions[1] = position1

    # Position 2: 3 balance updates, 2 trades with 1 blockchain tx each
    position2 = create_position(pair, 2, is_closed=True)
    add_balance_updates(position2, 3)
    add_trades(position2, pair, 2, 1)
    state.portfolio.closed_positions[2] = position2

    # Position 3: 1 balance update, 1 trade with 3 blockchain txs
    position3 = create_position(pair, 3, is_closed=True)
    add_balance_updates(position3, 1)
    add_trades(position3, pair, 1, 3)
    state.portfolio.closed_positions[3] = position3

    # Add an open position that should not be pruned
    open_position = create_position_with_balance_updates(pair, 999, 1, is_closed=False)
    state.portfolio.open_positions[999] = open_position

    # Verify initial state
    assert len(state.portfolio.closed_positions) == 3
    assert len(state.portfolio.open_positions) == 1

    total_balance_updates = sum(len(pos.balance_updates) for pos in state.portfolio.closed_positions.values())
    assert total_balance_updates == 2 + 3 + 1  # 6 total

    total_trades = sum(len(pos.trades) for pos in state.portfolio.closed_positions.values())
    assert total_trades == 1 + 2 + 1  # 4 total

    total_blockchain_txs = sum(
        len(tx.blockchain_transactions)
        for pos in state.portfolio.closed_positions.values()
        for tx in pos.trades.values()
    )
    assert total_blockchain_txs == 2 + 2 + 3  # 7 total

    # Prune all closed positions
    result = prune_closed_positions(state)

    # Verify results
    assert result['positions_processed'] == 3
    assert result['balance_updates_removed'] == 6
    assert result['trades_processed'] == 4
    assert result['blockchain_transactions_processed'] == 7

    # Verify all closed positions were pruned
    for position in state.portfolio.closed_positions.values():
        assert len(position.balance_updates) == 0
        for trade in position.trades.values():
            for tx in trade.blockchain_transactions:
                assert tx.transaction_args is None
                assert tx.wrapped_args is None
                assert tx.signed_bytes is None
                assert tx.signed_tx_object is None

    # Verify open position is unchanged
    assert len(state.portfolio.open_positions[999].balance_updates) == 1


def test_prune_closed_positions_no_closed_positions():
    """Test bulk pruning when there are no closed positions."""
    # Create empty state
    state = State()
    state.portfolio = Portfolio()

    # Prune when no closed positions exist
    result = prune_closed_positions(state)

    assert result['positions_processed'] == 0
    assert result['balance_updates_removed'] == 0


def test_prune_closed_positions_stats_aggregation(mock_assets):
    """Test that stats are properly aggregated across multiple positions."""
    usdc, weth = mock_assets

    # Create state with one closed position with balance updates
    state = State()
    state.portfolio = Portfolio()

    pair = TradingPairIdentifier(
        weth, usdc, "0x3", "0x4", internal_id=1, internal_exchange_id=1
    )

    position = create_position_with_balance_updates(pair, 1, 3, is_closed=True)
    state.portfolio.closed_positions[1] = position

    # Prune and check stats
    result = prune_closed_positions(state)

    assert result['positions_processed'] == 1
    assert result['balance_updates_removed'] == 3
