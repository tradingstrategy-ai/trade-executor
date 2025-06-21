"""CLI command to prune balance updates from closed positions."""

import logging
from pathlib import Path
from typing import Optional

from . import shared_options
from .app import app
from tradeexecutor.cli.bootstrap import prepare_executor_id, backup_state
from tradeexecutor.cli.log import setup_logging
from tradeexecutor.state.prune import prune_closed_positions


logger = logging.getLogger(__name__)


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

    # Setup logging
    logger = setup_logging(log_level)

    # Prepare executor ID
    id = prepare_executor_id(id, strategy_file)

    # Determine state file path
    if not state_file:
        state_file = Path(f"state/{id}.json")

    logger.info("Pruning state for executor %s", id)
    logger.info("State file: %s", state_file)

    # Create backup and load state
    logger.info("Creating backup of state file...")
    store, state = backup_state(str(state_file), backup_suffix="prune-backup", unit_testing=unit_testing)

    # Count initial statistics
    initial_closed_positions = len(state.portfolio.closed_positions)
    initial_balance_updates = sum(
        len(pos.balance_updates)
        for pos in state.portfolio.closed_positions.values()
    )

    logger.info("Found %d closed positions with %d total balance updates",
                initial_closed_positions, initial_balance_updates)

    if initial_closed_positions == 0:
        logger.info("No closed positions found - nothing to prune")
        return

    if initial_balance_updates == 0:
        logger.info("No balance updates found in closed positions - nothing to prune")
        return

    # Perform pruning
    logger.info("Pruning balance updates from closed positions...")
    result = prune_closed_positions(state)

    # Save the pruned state
    logger.info("Saving pruned state...")
    store.sync(state)

    # Report results
    logger.info("Pruning completed successfully!")
    logger.info("Positions processed: %d", result["positions_processed"])
    logger.info("Balance updates removed: %d", result["balance_updates_removed"])

    estimated_bytes_saved = result["bytes_saved"]
    if estimated_bytes_saved > 1024 * 1024:
        logger.info("Estimated space saved: ~%.1f MB", estimated_bytes_saved / (1024 * 1024))
    elif estimated_bytes_saved > 1024:
        logger.info("Estimated space saved: ~%.1f KB", estimated_bytes_saved / 1024)
    else:
        logger.info("Estimated space saved: ~%d bytes", estimated_bytes_saved)
