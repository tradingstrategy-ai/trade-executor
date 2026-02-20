"""History pruning functionality to remove early time-series data from state.

This module provides utilities to remove diagnostic statistics, visualisation
data, and uptime records before a given cutoff date. This is useful for
removing data from a vault's warming-up phase.
"""

import calendar
import datetime
from typing import TypedDict

from tradeexecutor.state.state import State


class HistoryPruningStats(TypedDict):
    """Statistics from history pruning operations."""
    portfolio_stats_removed: int
    position_stats_removed: int
    closed_position_stats_removed: int
    visualisation_messages_removed: int
    visualisation_calculations_removed: int
    visualisation_plot_points_removed: int
    uptime_checks_removed: int
    cycles_completed_removed: int


def prune_history(state: State, cutoff_date: datetime.datetime) -> HistoryPruningStats:
    """Remove time-series data from state that is earlier than the cutoff date.

    This prunes:

    - Portfolio statistics (``state.stats.portfolio``)
    - Per-position statistics (``state.stats.positions``)
    - Closed position statistics (``state.stats.closed_positions``)
    - Visualisation messages (``state.visualisation.messages``)
    - Visualisation calculations (``state.visualisation.calculations``)
    - Visualisation plot points (``state.visualisation.plots[*].points``)
    - Uptime checks (``state.uptime.uptime_checks``)
    - Cycles completed (``state.uptime.cycles_completed_at``)

    :param state:
        Trading state to prune
    :param cutoff_date:
        Naive UTC datetime. Entries with timestamps strictly before this
        date are removed.
    :return:
        Statistics about what was pruned
    """
    assert isinstance(cutoff_date, datetime.datetime)
    assert cutoff_date.tzinfo is None, "cutoff_date must be a naive UTC datetime"

    cutoff_unix = int(calendar.timegm(cutoff_date.utctimetuple()))

    # 1. Portfolio statistics
    original_portfolio_count = len(state.stats.portfolio)
    state.stats.portfolio = [
        ps for ps in state.stats.portfolio
        if ps.calculated_at >= cutoff_date
    ]
    portfolio_stats_removed = original_portfolio_count - len(state.stats.portfolio)

    # 2. Per-position statistics
    position_stats_removed = 0
    for position_id in list(state.stats.positions.keys()):
        entries = state.stats.positions[position_id]
        original_count = len(entries)
        state.stats.positions[position_id] = [
            ps for ps in entries
            if ps.calculated_at >= cutoff_date
        ]
        position_stats_removed += original_count - len(state.stats.positions[position_id])
        # Remove key entirely if no entries remain
        if not state.stats.positions[position_id]:
            del state.stats.positions[position_id]

    # 3. Closed position statistics
    closed_position_stats_removed = 0
    for position_id in list(state.stats.closed_positions.keys()):
        if state.stats.closed_positions[position_id].calculated_at < cutoff_date:
            del state.stats.closed_positions[position_id]
            closed_position_stats_removed += 1

    # 4. Visualisation messages (keyed by UNIX timestamp int)
    vis_messages_removed = 0
    for ts_key in list(state.visualisation.messages.keys()):
        if ts_key < cutoff_unix:
            del state.visualisation.messages[ts_key]
            vis_messages_removed += 1

    # 5. Visualisation calculations (keyed by UNIX timestamp int)
    vis_calculations_removed = 0
    for ts_key in list(state.visualisation.calculations.keys()):
        if ts_key < cutoff_unix:
            del state.visualisation.calculations[ts_key]
            vis_calculations_removed += 1

    # 6. Visualisation plot points (each Plot has points: Dict[int, float])
    vis_plot_points_removed = 0
    for plot in state.visualisation.plots.values():
        for ts_key in list(plot.points.keys()):
            if ts_key < cutoff_unix:
                del plot.points[ts_key]
                vis_plot_points_removed += 1

    # 7. Uptime checks (List[datetime | int])
    # Values may be datetime or UNIX int depending on serialisation path
    original_uptime_count = len(state.uptime.uptime_checks)
    state.uptime.uptime_checks = [
        ts for ts in state.uptime.uptime_checks
        if (ts >= cutoff_unix if isinstance(ts, (int, float)) else ts >= cutoff_date)
    ]
    uptime_checks_removed = original_uptime_count - len(state.uptime.uptime_checks)

    # 8. Cycles completed (Dict[int, datetime | int] -- key=cycle number, value=timestamp)
    # Values may be datetime or UNIX int depending on serialisation path
    cycles_removed = 0
    for cycle_num in list(state.uptime.cycles_completed_at.keys()):
        val = state.uptime.cycles_completed_at[cycle_num]
        if isinstance(val, (int, float)):
            is_before = val < cutoff_unix
        else:
            is_before = val < cutoff_date
        if is_before:
            del state.uptime.cycles_completed_at[cycle_num]
            cycles_removed += 1

    return HistoryPruningStats(
        portfolio_stats_removed=portfolio_stats_removed,
        position_stats_removed=position_stats_removed,
        closed_position_stats_removed=closed_position_stats_removed,
        visualisation_messages_removed=vis_messages_removed,
        visualisation_calculations_removed=vis_calculations_removed,
        visualisation_plot_points_removed=vis_plot_points_removed,
        uptime_checks_removed=uptime_checks_removed,
        cycles_completed_removed=cycles_removed,
    )
