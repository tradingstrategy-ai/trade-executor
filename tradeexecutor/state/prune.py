"""State pruning functionality to reduce state file size.

This module provides utilities to remove unnecessary data from closed positions
to keep state files manageable in size.
"""

from collections import Counter
from typing import TypedDict

from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution

class TradePruningStats(TypedDict):
    """Statistics from trade pruning operations."""
    blockchain_transactions_processed: int
    trades_processed: int

class PositionPruningStats(TradePruningStats):
    """Statistics from position pruning operations."""
    balance_updates_removed: int
    positions_processed: int


def prune_blockchain_transaction(tx: BlockchainTransaction):
    """Remove unneeded data from a BlockchainTransaction"""
    tx.transaction_args = None
    tx.wrapped_args = None
    tx.signed_bytes = None
    tx.signed_tx_object = None
    tx.details and tx.details.update({ "data": None }) # type: ignore


def prune_trade(trade: TradeExecution) -> TradePruningStats:
    """Remove unnecessary data from a TradeExecution.

    Prunes blockchain transaction data from a trade to reduce memory usage
    and storage size. This removes detailed transaction information that is
    not needed for historical analysis of closed trades.

    :param trade:
        TradeExecution to prune

    :return:
        TradePruningStats with statistics about what was pruned
    """

    # Initialize counter
    blockchain_transactions_processed = 0

    # Prune all of the trades's transactions
    for tx in trade.blockchain_transactions:
        prune_blockchain_transaction(tx)
        blockchain_transactions_processed += 1

    return TradePruningStats(
        blockchain_transactions_processed = blockchain_transactions_processed,
        trades_processed = 1
    )


def prune_closed_position(position: TradingPosition) -> PositionPruningStats:
    """Remove unnecessary data from a closed position to reduce memory usage.

    Prunes balance updates and blockchain transaction data from all trades
    in a closed position. This removes detailed transaction information and
    balance tracking data that is not needed for historical analysis.

    :param position:
        Trading position to prune

    :return:
        PruningStats with statistics about what was pruned

    :raise ValueError:
        If position is not closed
    """

    if not position.is_closed():
        raise ValueError(f"Cannot prune open position {position.position_id}")

    # Remove all balance updates
    balance_updates_removed = len(position.balance_updates)
    position.balance_updates.clear()

    # Initialize Counter with all TradePruningStats fields set to 0
    total_trade_stats = Counter({field: 0 for field in TradePruningStats.__annotations__})

    # Prune all of the position's trades
    for trade in position.trades.values():
        trade_stats = prune_trade(trade)
        total_trade_stats.update(trade_stats)

    return PositionPruningStats(
        **total_trade_stats,
        balance_updates_removed = balance_updates_removed,
        positions_processed = 1
    )


def prune_closed_positions(state: State) -> PositionPruningStats:
    """Remove unneeded data from all closed_positions in the portfolio

    :param state:
        Trading state to prune

    :return:
        PruningStats with statistics about what was pruned
    """
    # Initialize Counter with all PositionPruningStats fields set to 0
    total_stats = Counter({field: 0 for field in PositionPruningStats.__annotations__})

    # Prune all closed positions
    for position in state.portfolio.closed_positions.values():
        position_stats = prune_closed_position(position)
        total_stats.update(position_stats)

    return PositionPruningStats(**total_stats)
