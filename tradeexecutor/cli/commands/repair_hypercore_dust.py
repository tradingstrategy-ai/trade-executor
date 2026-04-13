"""Repair Hypercore dust positions from a state file."""

from pathlib import Path
from typing import Optional

import typer
from typer import Option

from . import shared_options
from .app import app
from ..bootstrap import backup_state, create_state_store, prepare_executor_id
from ..double_position import check_double_position, get_duplicate_position_groups
from ..log import setup_logging
from ...state.repair import (
    close_hypercore_dust_positions,
    find_hypercore_duplicate_clone_candidates,
    suppress_hypercore_duplicate_clone,
)
from ...state.store import JSONFileStore


def _count_duplicate_hypercore_groups(state) -> int:
    """Count Hypercore duplicate-position groups in the current state."""
    return sum(
        1
        for positions in get_duplicate_position_groups(state)
        if positions[0].pair.is_hyperliquid_vault()
    )


def _format_candidate_position(position) -> str:
    """Format one Hypercore duplicate-clone candidate position for CLI logging."""

    return (
        f"position_id={position.position_id}, "
        f"quantity={position.get_quantity(planned=False)}, "
        f"planned_quantity={position.get_quantity(planned=True)}, "
        f"expected_usd_equity={position.get_value(include_interest=False)}, "
        f"last_token_price={position.last_token_price}, "
        f"has_planned_trades={position.has_planned_trades()}, "
        f"is_about_to_close={position.is_about_to_close()}, "
        f"balance_updates={len(position.balance_updates)}, "
        f"trades={len(position.trades)}"
    )


@app.command()
def repair_hypercore_dust(
    id: str = shared_options.id,
    strategy_file: Optional[Path] = shared_options.optional_strategy_file,
    state_file: Optional[Path] = shared_options.state_file,
    log_level: str = shared_options.log_level,
    unit_testing: bool = shared_options.unit_testing,
    merge_dustless_duplicates: bool = Option(
        False,
        envvar="MERGE_DUSTLESS_DUPLICATES",
        help=(
            "Suppress a later phantom Hypercore duplicate clone when the duplicate group "
            "passes strict safety checks. This always requires an explicit y/n confirmation "
            "for each dangerous suppression."
        ),
    ),
    auto_approve: bool = Option(
        False,
        envvar="AUTO_APPROVE",
        help="Approve Hypercore dust cleanup without asking for confirmation.",
    ),
):
    """Close Hypercore dust positions and optionally suppress safe duplicate clones.

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

    store, state = backup_state(
        state_file,
        backup_suffix="repair-hypercore-dust-backup",
        unit_testing=unit_testing,
    )

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
            logger.info(
                "Created repair trade %s for position %s",
                trade.trade_id,
                trade.position_id,
            )
    else:
        logger.info("No closeable Hypercore dust positions were found")

    suppressed_positions: list = []
    duplicate_group_count_after = _count_duplicate_hypercore_groups(state)

    if duplicate_group_count_after and merge_dustless_duplicates:
        candidates, rejected_groups = find_hypercore_duplicate_clone_candidates(state.portfolio)

        if rejected_groups:
            logger.warning(
                "Hypercore duplicate groups remaining after dust cleanup are not safe for automatic suppression"
            )
            for reason in rejected_groups:
                logger.warning(reason)
            raise RuntimeError(
                "Hypercore duplicate positions still remain after dust cleanup, and at least one duplicate "
                "group is not safe for automatic suppression. The state file was not updated."
            )

        if candidates:
            if auto_approve and not unit_testing:
                raise RuntimeError(
                    "--auto-approve cannot be used with --merge-dustless-duplicates because "
                    "each dangerous duplicate suppression requires an explicit y/n confirmation."
                )

            logger.warning(
                "Detected %d Hypercore duplicate clone group(s) that are eligible for strict suppression",
                len(candidates),
            )

            for candidate in candidates:
                logger.warning(
                    "Eligible Hypercore duplicate clone suppression for vault %s at %s",
                    candidate.vault_name,
                    candidate.vault_address,
                )
                logger.warning(
                    "Keeping survivor position: %s",
                    _format_candidate_position(candidate.survivor_position),
                )
                logger.warning(
                    "Suppressing clone position: %s",
                    _format_candidate_position(candidate.clone_position),
                )

                if not unit_testing:
                    confirmation = typer.confirm(
                        f"Suppress duplicate clone position #{candidate.clone_position.position_id} "
                        f"and keep survivor #{candidate.survivor_position.position_id} for vault "
                        f"{candidate.vault_name}?"
                    )
                    if not confirmation:
                        raise RuntimeError(
                            "Operator aborted Hypercore duplicate-clone suppression. "
                            "The state file was not updated."
                        )

                suppressed_position = suppress_hypercore_duplicate_clone(
                    state.portfolio,
                    candidate,
                )
                suppressed_positions.append(suppressed_position)

            duplicate_group_count_after = _count_duplicate_hypercore_groups(state)

    if duplicate_group_count_after:
        logger.warning(
            "Hypercore duplicate position groups remaining after dust cleanup: %d",
            duplicate_group_count_after,
        )
        check_double_position(state, printer=logger.warning, crash=False)
        raise RuntimeError(
            "Hypercore duplicate positions still remain after dust cleanup. "
            "Any remaining duplicates are not closeable dust or safe clone suppressions, "
            "and require manual repair. The state file was not updated."
        )

    state_changed = bool(created_trades) or bool(suppressed_positions)
    if state_changed:
        logger.info("Saving repaired state to %s", store.path)
        store.sync(state)
    else:
        logger.info("No state changes were needed")

    logger.info("No Hypercore duplicate position groups remain after dust cleanup")
    logger.info("All ok")
