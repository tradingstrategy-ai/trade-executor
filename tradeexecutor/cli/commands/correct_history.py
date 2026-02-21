"""CLI command to remove early history data from state.

Example:

.. code-block:: shell

    docker compose run master-vault correct-history --cutoff-date 2026-01-15

"""

import datetime
import warnings
from pathlib import Path
from typing import Optional

from typer import Option

from . import shared_options
from .app import app
from tradeexecutor.cli.bootstrap import prepare_executor_id, backup_state
from tradeexecutor.state.correct_history import prune_history


@app.command()
def correct_history(
    id: str = shared_options.id,
    strategy_file: Path = shared_options.strategy_file,
    state_file: Optional[Path] = shared_options.state_file,
    unit_testing: bool = shared_options.unit_testing,
    cutoff_date: str = Option(
        ...,
        "--cutoff-date",
        envvar="CUTOFF_DATE",
        help="Remove all history data before this date. Format: YYYY-MM-DD.",
    ),
):
    """Remove early history and statistics data from the state file.

    This is useful for removing data accumulated during a vault's warming-up
    phase. Only diagnostic and statistics data is affected -- positions and
    trades are not modified.

    The command will:

    1. Create a backup of the current state file
    2. Load the state
    3. Remove portfolio statistics, position statistics, visualisation data,
       and uptime records earlier than the cutoff date
    4. Save the pruned state
    5. Report pruning statistics
    """

    # Parse cutoff date
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

    print(f"Pruning history from state file: {state_file}")
    print(f"Cutoff date: {parsed_cutoff.strftime('%Y-%m-%d')}")

    # Create backup and load state
    print("Creating backup of state file...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="dataclasses_json")
        store, state = backup_state(str(state_file), backup_suffix="correct-history-backup", unit_testing=unit_testing)

    # Get original file size
    original_size = state_file.stat().st_size if state_file.exists() else 0

    # Perform history pruning
    print("Removing history data before cutoff date...")
    result = prune_history(state, parsed_cutoff)

    # Save the pruned state
    print("Saving pruned state...")
    store.sync(state)

    # Calculate bytes saved
    final_size = state_file.stat().st_size if state_file.exists() else 0
    bytes_saved = original_size - final_size

    # Report results
    print("History pruning completed successfully!")
    print(f"Portfolio statistics removed: {result['portfolio_stats_removed']}")
    print(f"Position statistics removed: {result['position_stats_removed']}")
    print(f"Closed position statistics removed: {result['closed_position_stats_removed']}")
    print(f"Visualisation messages removed: {result['visualisation_messages_removed']}")
    print(f"Visualisation calculations removed: {result['visualisation_calculations_removed']}")
    print(f"Visualisation plot points removed: {result['visualisation_plot_points_removed']}")
    print(f"Uptime checks removed: {result['uptime_checks_removed']}")
    print(f"Cycles completed removed: {result['cycles_completed_removed']}")

    if bytes_saved > 1024 * 1024:
        print(f"Estimated space saved: ~{bytes_saved / (1024 * 1024):.1f} MB")
    elif bytes_saved > 1024:
        print(f"Estimated space saved: ~{bytes_saved / 1024:.1f} KB")
    else:
        print(f"Estimated space saved: ~{bytes_saved} bytes")
