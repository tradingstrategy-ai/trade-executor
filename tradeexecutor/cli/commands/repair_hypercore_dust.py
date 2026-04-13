"""Repair Hypercore dust positions from a state file."""

from pathlib import Path
from typing import Optional

import typer
from typer import Option

from . import shared_options
from .app import app
from ..bootstrap import create_state_store, prepare_executor_id
from ..double_position import check_double_position, get_duplicate_position_groups
from ..log import setup_logging
from ...state.repair import close_hypercore_dust_positions
from ...state.store import JSONFileStore


def _count_duplicate_hypercore_groups(state) -> int:
    """Count Hypercore duplicate-position groups in the current state."""
    return sum(
        1
        for positions in get_duplicate_position_groups(state)
        if positions[0].pair.is_hyperliquid_vault()
    )


@app.command()
def repair_hypercore_dust(
    id: str = shared_options.id,
    strategy_file: Optional[Path] = shared_options.optional_strategy_file,
    state_file: Optional[Path] = shared_options.state_file,
    log_level: str = shared_options.log_level,
    unit_testing: bool = shared_options.unit_testing,
    auto_approve: bool = Option(
        False,
        envvar="AUTO_APPROVE",
        help="Approve Hypercore dust cleanup without asking for confirmation.",
    ),
):
    """Close Hypercore dust positions and warn about duplicate vault entries.

    This command is intentionally local-state only. It does not attempt any
    on-chain execution; instead it creates repair trades for Hypercore vault
    positions that are already within the configured close epsilon.

    Duplicate Hypercore positions are diagnosed before and after cleanup so
    operators can see whether stale dust residuals were removed or whether a
    deeper manual repair is still needed.
    """

    logger = setup_logging(log_level=log_level)

    if not state_file:
        id = prepare_executor_id(id, strategy_file)
        assert id, "Executor id must be given if state file path is not given"
        state_file = Path(f"state/{id}.json")

    logger.info("Repairing Hypercore dust positions in state file %s", state_file)

    store = create_state_store(Path(state_file))
    assert isinstance(store, JSONFileStore)
    assert not store.is_pristine(), f"State file does not exist: {state_file}"

    state = store.load()

    duplicate_group_count_before = _count_duplicate_hypercore_groups(state)
    if duplicate_group_count_before:
        logger.warning(
            "Detected %d Hypercore duplicate position group(s) before dust cleanup",
            duplicate_group_count_before,
        )
        check_double_position(state, printer=logger.warning, crash=False)
    else:
        logger.info("No Hypercore duplicate position groups detected before dust cleanup")

    if not auto_approve and not unit_testing:
        confirmation = typer.confirm(
            "Create repair trades for all closeable Hypercore dust positions in this state file?"
        )
        if not confirmation:
            raise RuntimeError("Operator aborted Hypercore dust cleanup")

    created_trades = close_hypercore_dust_positions(state.portfolio)
    if created_trades:
        logger.info(
            "Auto-closed %d Hypercore dust position(s) with repair trades",
            len(created_trades),
        )
        for trade in created_trades:
            logger.info("Created repair trade %s for position %s", trade.trade_id, trade.position_id)
        store.sync(state)
    else:
        logger.info("No closeable Hypercore dust positions were found")

    duplicate_group_count_after = _count_duplicate_hypercore_groups(state)
    if duplicate_group_count_after:
        logger.warning(
            "Hypercore duplicate position groups remaining after dust cleanup: %d",
            duplicate_group_count_after,
        )
        check_double_position(state, printer=logger.warning, crash=False)
        raise RuntimeError(
            "Hypercore duplicate positions still remain after dust cleanup. "
            "Any remaining duplicates are not closeable dust and require manual repair."
        )

    logger.info("No Hypercore duplicate position groups remain after dust cleanup")
