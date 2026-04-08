"""History pruning and outlier removal for state time-series data.

This module provides utilities to remove diagnostic statistics, visualisation
data, and uptime records before a given cutoff date. This is useful for
removing data from a vault's warming-up phase.

It also provides share price outlier detection and removal using a rolling
median approach, useful when NAV calculation failures cause temporary
spurious share price drops that skew max drawdown and other statistics.
"""

import calendar
import datetime
import statistics
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

    # 9. Update state.created_at so that key metrics (CAGR, etc.)
    # use the cutoff date as the calculation window start instead of
    # the original strategy creation date.
    if cutoff_date > state.created_at:
        state.created_at = cutoff_date

    # 10. Update initial_share_price to the first remaining portfolio
    # stat's share price so that share-price-based return charts start
    # from zero instead of showing negative returns relative to $1.0.
    if state.stats.portfolio:
        first_share_price = state.stats.portfolio[0].share_price_usd
        if first_share_price is not None:
            state.initial_share_price = first_share_price

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


def filter_share_price_outliers(
    state: State,
    window_size: int = 5,
    threshold: float = 0.30,
) -> int:
    """Remove share price outlier entries from portfolio statistics.

    Uses a two-pass rolling median approach to detect entries where
    the share price deviates significantly from its neighbours. This
    handles cases where NAV calculation failures cause temporary
    spurious share price drops (e.g. 94% drop for a few hours).

    Pass 1 identifies outliers. Pass 2 recomputes medians with pass-1
    outliers excluded, catching entries adjacent to clusters that were
    masked in the first pass. The union of both passes is removed.

    After removal, ``state.initial_share_price`` is updated to match
    the first remaining portfolio entry's share price.

    :param state:
        Trading state to modify in-place
    :param window_size:
        Number of entries on each side to include in the median window
    :param threshold:
        Maximum allowed relative deviation from the rolling median
        (0.30 means 30% deviation triggers removal)
    :return:
        Number of entries removed
    """
    entries = state.stats.portfolio

    # Build index of entries with valid share prices
    # Each element is (original_index_into_entries, share_price_value)
    prices: list[tuple[int, float]] = []
    for i, e in enumerate(entries):
        if e.share_price_usd is not None:
            prices.append((i, e.share_price_usd))

    if len(prices) < 3:
        return 0

    def _find_outliers(price_list: list[tuple[int, float]], excluded: set[int] | None = None) -> set[int]:
        """Return set of original indices that are outliers."""
        outliers: set[int] = set()

        for pos, (idx, price) in enumerate(price_list):
            if excluded and idx in excluded:
                continue

            # Gather neighbour prices within the window
            neighbours: list[float] = []
            for j in range(max(0, pos - window_size), min(len(price_list), pos + window_size + 1)):
                if j == pos:
                    continue
                n_idx, n_price = price_list[j]
                if excluded and n_idx in excluded:
                    continue
                neighbours.append(n_price)

            if len(neighbours) == 0:
                continue

            med = statistics.median(neighbours)
            if med == 0:
                continue

            deviation = abs(price - med) / med
            if deviation > threshold:
                outliers.add(idx)

        return outliers

    # Pass 1: identify outliers against raw data
    first_pass = _find_outliers(prices)

    # Pass 2: recompute medians with pass-1 outliers excluded
    second_pass = _find_outliers(prices, excluded=first_pass)

    # Union: first pass catches isolated outliers, second pass catches
    # entries adjacent to clusters that were masked in the first pass
    all_outliers = first_pass | second_pass

    if not all_outliers:
        return 0

    state.stats.portfolio = [
        e for i, e in enumerate(entries)
        if i not in all_outliers
    ]

    # Update initial_share_price to match the first remaining entry
    if state.stats.portfolio:
        first_share_price = state.stats.portfolio[0].share_price_usd
        if first_share_price is not None:
            state.initial_share_price = first_share_price

    return len(all_outliers)
