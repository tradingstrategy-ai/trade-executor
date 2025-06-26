"""CLI command to prune balance updates from closed positions.

Example:

.. code-block:: shell


"""

from pathlib import Path
from typing import Optional

from . import shared_options
from .app import app
from tradeexecutor.cli.bootstrap import prepare_executor_id, backup_state
from tradeexecutor.state.prune import prune_closed_positions


@app.command()
def prune_state(
    id: str = shared_options.id,
    strategy_file: Path = shared_options.strategy_file,
    state_file: Optional[Path] = shared_options.state_file,
    unit_testing: bool = shared_options.unit_testing,
):
    """Prune unnecessary data from state to reduce state file size.

    This command removes accumulated data from the state file that is no longer
    needed for trading operations.

    The command will:
    1. Create a backup of the current state file
    2. Load the state
    3. Remove unnecessary data properties from all closed positions
    4. Save the pruned state
    5. Report pruning statistics

    Only closed positions are affected - open positions are left unchanged.
    """

    # Prepare executor ID
    id = prepare_executor_id(id, strategy_file)

    # Determine state file path
    if not state_file:
        state_file = Path(f"state/{id}.json")

    print(f"Pruning state file: {state_file}")

    # Create backup and load state
    print("Creating backup of state file...")
    store, state = backup_state(str(state_file), backup_suffix="prune-backup", unit_testing=unit_testing)

    # Get original file size
    original_size = state_file.stat().st_size if state_file.exists() else 0

    # Perform pruning
    print("Pruning balance updates from closed positions...")
    result = prune_closed_positions(state)

    # Save the pruned state
    print("Saving pruned state...")
    store.sync(state)

    # Calculate bytes saved by comparing file sizes
    final_size = state_file.stat().st_size if state_file.exists() else 0
    bytes_saved = original_size - final_size

    # Report results
    print("Pruning completed successfully!")
    print(f"Positions processed: {result['positions_processed']}")
    print(f"Balance updates removed: {result['balance_updates_removed']}")
    print(f"Trades processed: {result['trades_processed']}")
    print(f"Blockchain transactions processed: {result['blockchain_transactions_processed']}")

    if bytes_saved > 1024 * 1024:
        print(f"Estimated space saved: ~{bytes_saved / (1024 * 1024):.1f} MB")
    elif bytes_saved > 1024:
        print(f"Estimated space saved: ~{bytes_saved / 1024:.1f} KB")
    else:
        print(f"Estimated space saved: ~{bytes_saved} bytes")
