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
    log_level: str = shared_options.log_level,
    unit_testing: bool = shared_options.unit_testing,
):
    """Prune unnecessary data from state to reduce state file size.

    This command removes accumulated data from the state file that is no longer
    needed for trading operations. Currently removes balance update events from
    all closed positions, which can accumulate over time (especially for
    interest-bearing positions) and make state files large.

    The command will:
    1. Create a backup of the current state file
    2. Load the state
    3. Remove balance updates from all closed positions
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

    # Count initial statistics
    initial_closed_positions = len(state.portfolio.closed_positions)
    initial_balance_updates = sum(
        len(pos.balance_updates)
        for pos in state.portfolio.closed_positions.values()
    )

    print(f"Found {initial_closed_positions} closed positions with {initial_balance_updates} total balance updates")

    if initial_closed_positions == 0:
        print("No closed positions found - nothing to prune")
        return

    if initial_balance_updates == 0:
        print("No balance updates found in closed positions - nothing to prune")
        return

    # Perform pruning
    print("Pruning balance updates from closed positions...")
    result = prune_closed_positions(state)

    # Save the pruned state
    print("Saving pruned state...")
    store.sync(state)

    # Report results
    print("Pruning completed successfully!")
    print(f"Positions processed: {result['positions_processed']}")
    print(f"Balance updates removed: {result['balance_updates_removed']}")

    estimated_bytes_saved = result["bytes_saved"]
    if estimated_bytes_saved > 1024 * 1024:
        print(f"Estimated space saved: ~{estimated_bytes_saved / (1024 * 1024):.1f} MB")
    elif estimated_bytes_saved > 1024:
        print(f"Estimated space saved: ~{estimated_bytes_saved / 1024:.1f} KB")
    else:
        print(f"Estimated space saved: ~{estimated_bytes_saved} bytes")
