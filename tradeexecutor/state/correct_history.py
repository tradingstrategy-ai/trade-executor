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


DEFAULT_SHARE_PRICE_OUTLIER_THRESHOLD = 0.20
DEFAULT_NAV_SYNC_OUTLIER_WINDOW_SIZE = 24
DEFAULT_NAV_SYNC_OUTLIER_THRESHOLD = 0.15
DEFAULT_NAV_SYNC_MIN_COMPONENT_CHANGE = 0.10


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


def detect_share_price_outliers(
    state: State,
    window_size: int = 5,
    threshold: float = DEFAULT_SHARE_PRICE_OUTLIER_THRESHOLD,
) -> set[int]:
    """Detect share price outlier entries in portfolio statistics.

    Uses a two-pass rolling median approach to detect entries where
    the share price deviates significantly from its neighbours. This
    handles cases where NAV calculation failures cause temporary
    spurious share price drops (e.g. 94% drop for a few hours).

    Pass 1 identifies outliers. Pass 2 recomputes medians with pass-1
    outliers excluded, catching entries adjacent to clusters that were
    masked in the first pass. The union of both passes is returned.

    :param state:
        Trading state (not modified)
    :param window_size:
        Number of entries on each side to include in the median window
    :param threshold:
        Maximum allowed relative deviation from the rolling median
        (0.20 means 20% deviation triggers removal)
    :return:
        Set of indices into ``state.stats.portfolio`` that are outliers
    """
    prices = _get_share_price_points(state)
    if len(prices) < 3:
        return set()

    return _detect_rolling_median_outliers(prices, window_size=window_size, threshold=threshold)


def detect_nav_sync_outliers(
    state: State,
    window_size: int = DEFAULT_NAV_SYNC_OUTLIER_WINDOW_SIZE,
    threshold: float = DEFAULT_NAV_SYNC_OUTLIER_THRESHOLD,
    min_component_change: float = DEFAULT_NAV_SYNC_MIN_COMPONENT_CHANGE,
    max_passes: int = 3,
) -> set[int]:
    """Detect clustered NAV/share price discontinuities caused by sync lag.

    This detector is intended for exchange-account vaults where reserve cash
    and external account equity can be briefly counted out of phase. The
    resulting NAV/share price points are often clustered, so a short rolling
    median window can miss them.

    A point is flagged when:

    1. Its share price deviates from a wider rolling median by ``threshold``
    2. A nearby reserve cash or open position equity component jumps by at
       least ``min_component_change`` relative to NAV

    :param state:
        Trading state (not modified)
    :param window_size:
        Number of priced entries on each side to use for the wide median and
        nearby component jump lookup.
    :param threshold:
        Maximum allowed relative deviation from the wide rolling median
        (0.15 means 15% deviation triggers removal)
    :param min_component_change:
        Minimum nearby free cash or open position equity change, relative to
        NAV, required to treat the deviation as a sync issue.
    :param max_passes:
        Maximum virtual removal passes to run. Large sync clusters can mask a
        nearby bad point until the first cluster has been removed.
    :return:
        Set of indices into ``state.stats.portfolio`` that are outliers
    """
    prices = _get_share_price_points(state)
    if len(prices) < 3:
        return set()

    outliers: set[int] = set()

    for _ in range(max_passes):
        remaining_prices = [
            (idx, price) for idx, price in prices
            if idx not in outliers
        ]
        candidates = _detect_rolling_median_outliers(remaining_prices, window_size=window_size, threshold=threshold)
        if not candidates:
            break

        position_by_index = {idx: pos for pos, (idx, _) in enumerate(remaining_prices)}
        filtered_candidates = {
            idx for idx in candidates
            if _has_nearby_component_jump(
                state,
                remaining_prices,
                position_by_index[idx],
                window_size=window_size,
                min_component_change=min_component_change,
            )
        }
        new_outliers = filtered_candidates - outliers
        if not new_outliers:
            break
        outliers.update(new_outliers)

    return outliers


def remove_portfolio_stat_indices(
    state: State,
    indices: set[int],
) -> int:
    """Remove entries from portfolio statistics.

    After removal, ``state.initial_share_price`` is updated to match
    the first remaining portfolio entry's share price.

    :param state:
        Trading state to modify in-place
    :param indices:
        Set of indices into ``state.stats.portfolio`` to remove.
    :return:
        Number of entries removed
    """
    if not indices:
        return 0

    state.stats.portfolio = [
        e for i, e in enumerate(state.stats.portfolio)
        if i not in indices
    ]

    # Update initial_share_price to match the first remaining entry
    if state.stats.portfolio:
        first_share_price = state.stats.portfolio[0].share_price_usd
        if first_share_price is not None:
            state.initial_share_price = first_share_price

    return len(indices)


def remove_share_price_outliers(
    state: State,
    outlier_indices: set[int],
) -> int:
    """Remove detected share price outlier entries from portfolio statistics.

    :param state:
        Trading state to modify in-place
    :param outlier_indices:
        Set of indices into ``state.stats.portfolio`` to remove,
        as returned by :py:func:`detect_share_price_outliers`
    :return:
        Number of entries removed
    """
    return remove_portfolio_stat_indices(state, outlier_indices)


def filter_share_price_outliers(
    state: State,
    window_size: int = 5,
    threshold: float = DEFAULT_SHARE_PRICE_OUTLIER_THRESHOLD,
) -> int:
    """Detect and remove share price outlier entries from portfolio statistics.

    Convenience wrapper that calls :py:func:`detect_share_price_outliers`
    then :py:func:`remove_share_price_outliers`.

    :param state:
        Trading state to modify in-place
    :param window_size:
        Number of entries on each side to include in the median window
    :param threshold:
        Maximum allowed relative deviation from the rolling median
        (0.20 means 20% deviation triggers removal)
    :return:
        Number of entries removed
    """
    outliers = detect_share_price_outliers(state, window_size=window_size, threshold=threshold)
    return remove_share_price_outliers(state, outliers)


def _get_share_price_points(state: State) -> list[tuple[int, float]]:
    """Build index of entries with valid share prices."""
    prices: list[tuple[int, float]] = []
    for i, e in enumerate(state.stats.portfolio):
        if e.share_price_usd is not None:
            prices.append((i, e.share_price_usd))
    return prices


def _detect_rolling_median_outliers(
    prices: list[tuple[int, float]],
    window_size: int,
    threshold: float,
) -> set[int]:
    """Detect outliers in an indexed price series using two median passes."""

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

    # Pass 1 identifies isolated outliers.
    first_pass = _find_outliers(prices)

    # Pass 2 recomputes medians with pass-1 outliers excluded, catching
    # entries adjacent to clusters that were masked in the first pass.
    second_pass = _find_outliers(prices, excluded=first_pass)

    return first_pass | second_pass


def _has_nearby_component_jump(
    state: State,
    prices: list[tuple[int, float]],
    pos: int,
    window_size: int,
    min_component_change: float,
) -> bool:
    """Check if a priced point is near a reserve/account component jump."""
    entries = state.stats.portfolio
    start = max(1, pos - window_size)
    end = min(len(prices), pos + window_size + 1)

    for point_pos in range(start, end):
        previous_idx = prices[point_pos - 1][0]
        current_idx = prices[point_pos][0]
        previous = entries[previous_idx]
        current = entries[current_idx]
        base = max(
            abs(previous.net_asset_value or previous.total_equity or 0),
            abs(current.net_asset_value or current.total_equity or 0),
            1,
        )

        free_cash_jump = _relative_component_jump(previous.free_cash, current.free_cash, base)
        open_position_jump = _relative_component_jump(previous.open_position_equity, current.open_position_equity, base)
        if max(free_cash_jump, open_position_jump) >= min_component_change:
            return True

    return False


def _relative_component_jump(previous: float | None, current: float | None, base: float) -> float:
    """Calculate a component jump as a fraction of NAV."""
    if previous is None or current is None:
        return 0
    return abs(current - previous) / base
