"""CLI command to remove early history data and share price outliers from state.

Example:

.. code-block:: shell

    docker compose run master-vault correct-history --cutoff-date 2026-01-15
    docker compose run gmx-ai correct-history --remove-share-price-outliers
    docker compose run gmx-ai correct-history --remove-nav-sync-outliers

"""

import datetime
import warnings
from pathlib import Path
from typing import Optional

from typer import Option

from . import shared_options
from .app import app
from tradeexecutor.cli.bootstrap import prepare_executor_id, backup_state
from tradeexecutor.state.correct_history import (
    DEFAULT_NAV_SYNC_MIN_COMPONENT_CHANGE,
    DEFAULT_NAV_SYNC_OUTLIER_THRESHOLD,
    DEFAULT_NAV_SYNC_OUTLIER_WINDOW_SIZE,
    DEFAULT_SHARE_PRICE_OUTLIER_THRESHOLD,
    detect_nav_sync_outliers,
    detect_share_price_outliers,
    prune_history,
    remove_portfolio_stat_indices,
)


@app.command()
def correct_history(
    id: str = shared_options.id,
    strategy_file: Path = shared_options.strategy_file,
    state_file: Optional[Path] = shared_options.state_file,
    unit_testing: bool = shared_options.unit_testing,
    cutoff_date: str | None = Option(
        None,
        "--cutoff-date",
        envvar="CUTOFF_DATE",
        help="Remove all history data before this date. Format: YYYY-MM-DD.",
    ),
    remove_share_price_outliers: bool = Option(
        False,
        "--remove-share-price-outliers",
        envvar="REMOVE_SHARE_PRICE_OUTLIERS",
        help="Detect and remove share price outlier entries from portfolio statistics using rolling median comparison.",
    ),
    outlier_window_size: int = Option(
        5,
        "--outlier-window-size",
        envvar="OUTLIER_WINDOW_SIZE",
        help="Number of entries on each side to include in the rolling median window for outlier detection.",
    ),
    outlier_threshold: float = Option(
        DEFAULT_SHARE_PRICE_OUTLIER_THRESHOLD,
        "--outlier-threshold",
        envvar="OUTLIER_THRESHOLD",
        help="Maximum allowed relative deviation from the rolling median before an entry is flagged as an outlier (0.20 = 20%).",
    ),
    remove_nav_sync_outliers: bool = Option(
        False,
        "--remove-nav-sync-outliers",
        envvar="REMOVE_NAV_SYNC_OUTLIERS",
        help="Detect and remove clustered NAV/share price discontinuities caused by reserve cash and exchange account equity syncing out of phase.",
    ),
    nav_sync_window_size: int = Option(
        DEFAULT_NAV_SYNC_OUTLIER_WINDOW_SIZE,
        "--nav-sync-window-size",
        envvar="NAV_SYNC_WINDOW_SIZE",
        help="Number of priced entries on each side to use for NAV sync outlier detection.",
    ),
    nav_sync_threshold: float = Option(
        DEFAULT_NAV_SYNC_OUTLIER_THRESHOLD,
        "--nav-sync-threshold",
        envvar="NAV_SYNC_THRESHOLD",
        help="Maximum allowed relative deviation from the wide rolling median for NAV sync outliers (0.15 = 15%).",
    ),
    nav_sync_min_component_change: float = Option(
        DEFAULT_NAV_SYNC_MIN_COMPONENT_CHANGE,
        "--nav-sync-min-component-change",
        envvar="NAV_SYNC_MIN_COMPONENT_CHANGE",
        help="Minimum nearby free cash or open position equity change, relative to NAV, required to treat a deviation as a NAV sync issue.",
    ),
):
    """Remove early history and statistics data from the state file.

    This is useful for removing data accumulated during a vault's warming-up
    phase, or for cleaning up spurious share price data points caused by
    temporary NAV calculation failures.

    Only diagnostic and statistics data is affected -- positions and
    trades are not modified.

    The command will:

    1. Create a backup of the current state file
    2. Load the state
    3. Optionally remove share price outlier entries (--remove-share-price-outliers)
    4. Optionally remove NAV sync outlier entries (--remove-nav-sync-outliers)
    5. Optionally remove statistics before a cutoff date (--cutoff-date)
    6. Save the modified state
    7. Report statistics
    """

    if cutoff_date is None and not remove_share_price_outliers and not remove_nav_sync_outliers:
        print("Error: At least one of --cutoff-date, --remove-share-price-outliers or --remove-nav-sync-outliers must be specified.")
        raise SystemExit(1)

    parsed_cutoff = None
    if cutoff_date is not None:
        try:
            parsed_cutoff = datetime.datetime.strptime(cutoff_date, "%Y-%m-%d")
        except ValueError:
            print(f"Error: Invalid cutoff date format '{cutoff_date}'. Use YYYY-MM-DD.")
            raise SystemExit(1)

    # Prepare executor ID
    id = prepare_executor_id(id, strategy_file)

    # Determine state file path
    if not state_file:
        state_file = Path(f"state/{id}.json")

    print(f"Correcting history in state file: {state_file}")

    # Create backup and load state
    print("Creating backup of state file...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="dataclasses_json")
        store, state = backup_state(str(state_file), backup_suffix="correct-history-backup", unit_testing=unit_testing)

    # Get original file size
    original_size = state_file.stat().st_size if state_file.exists() else 0

    # Outlier removal runs first so the detectors see full neighbourhood
    # context before cutoff pruning truncates the series
    if remove_share_price_outliers or remove_nav_sync_outliers:
        outlier_sources: dict[int, list[str]] = {}

        def _add_outlier_source(indices: set[int], source: str):
            for idx in indices:
                outlier_sources.setdefault(idx, []).append(source)

    if remove_share_price_outliers:
        print(f"Detecting share price outliers (window={outlier_window_size}, threshold={outlier_threshold:.0%})...")
        share_price_outlier_indices = detect_share_price_outliers(state, window_size=outlier_window_size, threshold=outlier_threshold)
        _add_outlier_source(share_price_outlier_indices, "share_price")

    if remove_nav_sync_outliers:
        print(
            f"Detecting NAV sync outliers "
            f"(window={nav_sync_window_size}, threshold={nav_sync_threshold:.0%}, "
            f"component-change={nav_sync_min_component_change:.0%})..."
        )
        nav_sync_outlier_indices = detect_nav_sync_outliers(
            state,
            window_size=nav_sync_window_size,
            threshold=nav_sync_threshold,
            min_component_change=nav_sync_min_component_change,
        )
        _add_outlier_source(nav_sync_outlier_indices, "nav_sync")

    if remove_share_price_outliers or remove_nav_sync_outliers:
        outlier_indices = set(outlier_sources.keys())

        if outlier_indices:
            print(f"\nFound {len(outlier_indices)} outlier data point(s) to remove:\n")
            print(f"  {'Timestamp':<24} {'Share price':>14} {'NAV':>14}  {'Source':<20}")
            print(f"  {'-' * 24} {'-' * 14} {'-' * 14}  {'-' * 20}")
            for idx in sorted(outlier_indices):
                ps = state.stats.portfolio[idx]
                nav_str = f"${ps.net_asset_value:,.2f}" if ps.net_asset_value is not None else "N/A"
                source_str = ",".join(outlier_sources[idx])
                print(f"  {str(ps.calculated_at):<24} ${ps.share_price_usd:>13,.6f} {nav_str:>14}  {source_str:<20}")

            if not unit_testing:
                answer = input(f"\nRemove these {len(outlier_indices)} entries? [y/N] ").strip().lower()
                if answer != "y":
                    print("Aborted.")
                    raise SystemExit(0)

            removed = remove_portfolio_stat_indices(state, outlier_indices)
            print(f"\nOutlier entries removed: {removed}")
        else:
            print("No outliers detected.")

    if parsed_cutoff is not None:
        print(f"Cutoff date: {parsed_cutoff.strftime('%Y-%m-%d')}")
        print("Removing history data before cutoff date...")
        result = prune_history(state, parsed_cutoff)

        print(f"Portfolio statistics removed: {result['portfolio_stats_removed']}")
        print(f"Position statistics removed: {result['position_stats_removed']}")
        print(f"Closed position statistics removed: {result['closed_position_stats_removed']}")
        print(f"Visualisation messages removed: {result['visualisation_messages_removed']}")
        print(f"Visualisation calculations removed: {result['visualisation_calculations_removed']}")
        print(f"Visualisation plot points removed: {result['visualisation_plot_points_removed']}")
        print(f"Uptime checks removed: {result['uptime_checks_removed']}")
        print(f"Cycles completed removed: {result['cycles_completed_removed']}")

    # Save the modified state
    print("Saving state...")
    store.sync(state)

    # Calculate bytes saved
    final_size = state_file.stat().st_size if state_file.exists() else 0
    bytes_saved = original_size - final_size

    print("Completed successfully!")
    if bytes_saved > 1024 * 1024:
        print(f"Estimated space saved: ~{bytes_saved / (1024 * 1024):.1f} MB")
    elif bytes_saved > 1024:
        print(f"Estimated space saved: ~{bytes_saved / 1024:.1f} KB")
    else:
        print(f"Estimated space saved: ~{bytes_saved} bytes")
