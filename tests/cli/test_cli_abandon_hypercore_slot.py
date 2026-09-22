"""Black-box coverage for explicit recovery from a HyperCore data timeout."""

import datetime
from pathlib import Path

from eth_defi.compat import native_datetime_utc_now
from typer.testing import CliRunner

from tradeexecutor.cli.commands.abandon_hypercore_slot import app
# Register the real lightweight companion command to retain Typer's command
# group behaviour when this test runs alone, without importing unrelated RPC CLIs.
from tradeexecutor.cli.commands.hello import hello  # noqa: F401
from tradeexecutor.state.state import State
from tradeexecutor.strategy.cycle import CycleDuration, snap_to_next_tick


def test_abandon_expired_hypercore_slot(tmp_path: Path) -> None:
    """Recover an unexecuted slot through Typer and preserve a real backup.

    1. Write a state file containing an expired readiness slot.
    2. Invoke the registered CLI command with that state path.
    3. Check the backup, future slot and rejection of a second early abandon.
    """
    # 1. Use normal JSON state and real CLI plumbing without patched services.
    now = native_datetime_utc_now()
    state = State()
    expired = (now - datetime.timedelta(days=4)).replace(hour=0, minute=0, second=0, microsecond=0)
    state.pending_data_availability_slot = expired
    state_file = tmp_path / "hyper-ai.json"
    state.write_json_file(state_file)

    # 2. Execute the same command operators invoke from the Compose shell.
    runner = CliRunner()
    args = ["abandon-hypercore-slot", "--state-file", str(state_file), "--log-level", "disabled"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, (result.output, result.exception)

    # 3. The new pending slot is strictly in the future; no cycle was executed.
    updated = State.read_json_file(state_file)
    assert updated.pending_data_availability_slot == snap_to_next_tick(now, CycleDuration.cycle_2d)
    assert updated.last_cycle_at is None
    backup = state_file.with_suffix(".abandon-hypercore-slot-1.json")
    assert State.read_json_file(backup).pending_data_availability_slot == expired
    saved = state_file.read_bytes()
    result = runner.invoke(app, args)
    assert result.exit_code != 0
    assert "still open" in str(result.exception)
    assert state_file.read_bytes() == saved


def test_abandon_rejects_slot_with_trades(tmp_path: Path) -> None:
    """Do not treat an executed decision as a harmless missed-data window.

    1. Load an existing historical state fixture with real trade records.
    2. Set the pending slot to one of its trade decisions and invoke Typer.
    3. Verify recovery is refused without writing a backup or changing state.
    """
    # 1. Reuse a genuine state shape, including normal serialisation of trades.
    state = State.read_json_file(Path(__file__).with_name("show-positions-long.json"))
    trade = next(iter(state.portfolio.get_all_trades()))
    state.pending_data_availability_slot = trade.opened_at
    state_file = tmp_path / "hyper-ai.json"
    state.write_json_file(state_file)
    saved = state_file.read_bytes()

    # 2. The operator command must detect that this is transaction recovery.
    result = CliRunner().invoke(app, ["abandon-hypercore-slot", "--state-file", str(state_file), "--log-level", "disabled"])

    # 3. Refuse before backup/mutation; never silently skip a traded decision.
    assert result.exit_code != 0
    assert "reconcile" in str(result.exception)
    assert state_file.read_bytes() == saved
    assert not list(tmp_path.glob("*.abandon-hypercore-slot-*.json"))
