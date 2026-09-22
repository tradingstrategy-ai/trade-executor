"""Recover a timed-out HyperCore readiness wait without replaying missed trades.

Operators run this local-state command with the executor stopped. It backs up
the state and explicitly schedules the next future two-day midnight. It refuses
to abandon any slot that created trades or has outstanding transaction repair.
"""

import datetime
from pathlib import Path

from eth_defi.compat import native_datetime_utc_now

from tradeexecutor.cli.bootstrap import backup_state, create_state_store, prepare_executor_id
from tradeexecutor.cli.commands import shared_options
from tradeexecutor.cli.commands.app import app
from tradeexecutor.cli.log import setup_logging
from tradeexecutor.strategy.cycle import CycleDuration, snap_to_next_tick
from tradeexecutor.strategy.hypercore_data_availability import HYPERCORE_READINESS_WINDOW


@app.command()
def abandon_hypercore_slot(
    id: str = shared_options.id,
    strategy_file: Path | None = shared_options.optional_strategy_file,
    state_file: Path | None = shared_options.state_file,
    log_level: str = shared_options.log_level,
) -> None:
    """Abandon an expired, unexecuted Hyper-AI two-day decision slot.

    Run from the deployment's Compose shell after stopping the executor. This
    performs no network requests or trades. A backup and a state visualisation
    message retain an audit of the skipped slot. Slots with created trades must
    use the normal reconciliation process instead.

    :param id: Executor identifier used to resolve the default state path.
    :param strategy_file: Optional strategy path used to derive the identifier.
    :param state_file: Authoritative local state JSON to update.
    :param log_level: Operator-facing logging verbosity.
    """
    logger = setup_logging(log_level)
    if state_file is None:
        id = prepare_executor_id(id, strategy_file)
        state_file = Path(f"state/{id}.json")
    store = create_state_store(state_file)
    assert not store.is_pristine(), f"State file does not exist: {state_file}"
    state = store.load()
    slot = state.pending_data_availability_slot
    now = native_datetime_utc_now()
    assert slot is not None, "No pending HyperCore slot to abandon"
    assert now >= slot + HYPERCORE_READINESS_WINDOW, "The pending slot's readiness window is still open"
    state.check_if_clean()
    for trade in state.portfolio.get_all_trades():
        assert not (trade.is_pending() or trade.is_failed()), "Outstanding transaction repair; reconcile trades first"
        assert trade.opened_at != slot, "The pending slot created trades; reconcile this cycle instead of abandoning it"

    # Reuse the existing backup convention before changing the authoritative
    # state. Mark a future pending slot, not a fictitious completed decision.
    backup_state(state_file, backup_suffix="abandon-hypercore-slot")
    next_slot = snap_to_next_tick(now + datetime.timedelta(microseconds=1), CycleDuration.cycle_2d)
    state.pending_data_availability_slot = next_slot
    message = f"Abandoned unexecuted HyperCore slot {slot}; next slot {next_slot}"
    state.visualisation.add_message(now, message)
    store.sync(state)
    logger.info(message)
