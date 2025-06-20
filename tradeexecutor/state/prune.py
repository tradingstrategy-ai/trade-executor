"""State pruning functionality to reduce state file size.

This module provides utilities to remove unnecessary data from closed positions
to keep state files manageable in size.
"""

from typing import TypedDict

from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State


class PruningResult(TypedDict):
    """Result of pruning operations."""
    positions_processed: int
    balance_updates_removed: int


def prune_closed_position(position: TradingPosition) -> int:
    """Remove balance updates from a closed position.

    :param position:
        Trading position to prune

    :return:
        Number of balance updates removed

    :raise ValueError:
        If position is not closed
    """
    if not position.is_closed():
        raise ValueError(f"Cannot prune open position {position.position_id}")

    balance_updates_count = len(position.balance_updates)
    position.balance_updates.clear()

    return balance_updates_count


def prune_closed_positions(state: State) -> PruningResult:
    """Remove balance updates from all closed positions in state.

    :param state:
        Trading state to prune

    :return:
        PruningResult with pruning statistics:
        - positions_processed: Number of closed positions processed
        - balance_updates_removed: Total balance updates removed
    """
    positions_processed = 0
    total_balance_updates_removed = 0

    for position in state.portfolio.closed_positions.values():
        balance_updates_removed = prune_closed_position(position)
        positions_processed += 1
        total_balance_updates_removed += balance_updates_removed

    return {
        "positions_processed": positions_processed,
        "balance_updates_removed": total_balance_updates_removed,
    }
