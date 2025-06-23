"""State pruning functionality to reduce state file size.

This module provides utilities to remove unnecessary data from closed positions
to keep state files manageable in size.
"""

from collections import Counter
from typing import TypedDict

from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State


class PruningStats(TypedDict):
    """Statistics from pruning operations."""
    balance_updates_removed: int
    positions_processed: int


def prune_closed_position(position: TradingPosition) -> PruningStats:
    """Remove balance updates from a closed position.

    :param position:
        Trading position to prune

    :return:
        PruningStats with statistics about what was pruned

    :raise ValueError:
        If position is not closed
    """
    if not position.is_closed():
        raise ValueError(f"Cannot prune open position {position.position_id}")

    balance_updates_removed = len(position.balance_updates)
    position.balance_updates.clear()

    return PruningStats(
      balance_updates_removed=balance_updates_removed,
      positions_processed=1
    )


def prune_closed_positions(state: State) -> PruningStats:
    """Remove balance updates from all closed positions in state.

    :param state:
        Trading state to prune

    :return:
        AggregatePruningStats with pruning statistics:
        - positions_processed: Number of closed positions processed
        - balance_updates_removed: Total balance updates removed
    """
    # Initialize Counter with all PruningStats fields set to 0
    total_stats = Counter({field: 0 for field in PruningStats.__annotations__})

    for position in state.portfolio.closed_positions.values():
        position_stats = prune_closed_position(position)
        total_stats.update(position_stats)

    return PruningStats(**total_stats)
