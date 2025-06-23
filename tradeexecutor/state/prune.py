"""State pruning functionality to reduce state file size.

This module provides utilities to remove unnecessary data from closed positions
to keep state files manageable in size.
"""

import json
from typing import TypedDict, Any
from dataclasses_json.core import _ExtendedEncoder # type: ignore

from tradeexecutor.state.balance_update import BalanceUpdate
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State


class PruningResult(TypedDict):
    """Result of pruning operations."""
    positions_processed: int
    balance_updates_removed: int
    bytes_saved: int


def prune_closed_position(position: TradingPosition) -> dict[int, BalanceUpdate]:
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

    removed_items = position.balance_updates.copy()
    position.balance_updates.clear()

    return removed_items


def prune_closed_positions(state: State) -> PruningResult:
    """Remove balance updates from all closed positions in state.

    :param state:
        Trading state to prune

    :return:
        PruningResult with pruning statistics:
        - positions_processed: Number of closed positions processed
        - balance_updates_removed: Total balance updates removed
    """
    all_removed_balance_updates: dict[int, BalanceUpdate] = {}
    positions_processed = 0

    for position in state.portfolio.closed_positions.values():
        removed_items = prune_closed_position(position)
        all_removed_balance_updates.update(removed_items)
        positions_processed += 1

    # Calculate actual bytes using JSON serialization
    # This matches how the data is actually stored in the state file
    if all_removed_balance_updates:
        # Convert to a serializable format without relying on to_dict method
        serialized_data: dict[str, Any] = {
            str(k): v.to_dict() for k, v in all_removed_balance_updates.items() # type: ignore
        }
        bytes_saved = len(json.dumps(serialized_data, cls=_ExtendedEncoder))
    else:
        bytes_saved = 0

    return {
        "positions_processed": positions_processed,
        "balance_updates_removed": len(all_removed_balance_updates),
        "bytes_saved": bytes_saved
    }
