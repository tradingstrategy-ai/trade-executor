"""CLI command to remove early history data and share price outliers from state.

Example:

.. code-block:: shell

    docker compose run master-vault correct-history --cutoff-date 2026-01-15
    docker compose run gmx-ai correct-history --remove-share-price-outliers

"""

import datetime
import warnings
from pathlib import Path
from typing import Optional

from typer import Option

from . import shared_options
from .app import app
from tradeexecutor.cli.bootstrap import prepare_executor_id, backup_state
from tradeexecutor.state.correct_history import prune_history, filter_share_price_outliers


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
        0.30,
        "--outlier-threshold",
        envvar="OUTLIER_THRESHOLD",
        help="Maximum allowed relative deviation from the rolling median before an entry is flagged as an outlier (0.30 = 30%).",
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
    4. Optionally remove statistics before a cutoff date (--cutoff-date)
    5. Save the modified state
    6. Report statistics
    """

    if cutoff_date is None and not remove_share_price_outliers:
        print("Error: At least one of --cutoff-date or --remove-share-price-outliers must be specified.")
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

    # Outlier removal runs first so the detector sees full neighbourhood
    # context before cutoff pruning truncates the series
    if remove_share_price_outliers:
        print("Removing share price outliers...")
        outliers_removed = filter_share_price_outliers(state, window_size=outlier_window_size, threshold=outlier_threshold)
        print(f"Share price outlier entries removed: {outliers_removed}")

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
