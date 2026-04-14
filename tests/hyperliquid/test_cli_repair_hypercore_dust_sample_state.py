"""Integration regression for Hypercore duplicate repair on a copied live-style state.

This test uses a copied local sample state so we can verify the full
`repair-hypercore-dust --merge-dustless-duplicates` flow against the
real duplicate-position pattern that previously failed in operator use.
"""

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tradeexecutor.cli.double_position import get_duplicate_position_groups
from tradeexecutor.cli.main import app
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeFlag

SAMPLE_STATE_FILE = Path.home() / "hyper-ai-4.json"

pytestmark = [
    pytest.mark.timeout(300),
    pytest.mark.skipif(
        not SAMPLE_STATE_FILE.exists(),
        reason="Set ~/hyper-ai-4.json to run this integration test",
    ),
]


def _get_hypercore_duplicate_groups(state: State) -> list[list]:
    """Return only Hypercore duplicate groups from the state."""

    return [
        positions
        for positions in get_duplicate_position_groups(state)
        if positions[0].pair.is_hyperliquid_vault()
    ]


def test_repair_hypercore_dust_repairs_hyper_ai_4_sample_state(
    tmp_path: Path,
) -> None:
    """Test the CLI fixes the copied `hyper-ai-4.json` duplicate groups end to end.

    1. Copy the local `~/hyper-ai-4.json` sample state to a temporary path.
    2. Run `repair-hypercore-dust --merge-dustless-duplicates` against the copied file.
    3. Verify the known stale positions are closed, the known live reopen positions stay open, and no Hypercore duplicates remain.
    """

    # 1. Copy the local `~/hyper-ai-4.json` sample state to a temporary path.
    state_file = tmp_path / "hyper-ai-4-copy.json"
    shutil.copy2(SAMPLE_STATE_FILE, state_file)

    initial_state = State.read_json_file(state_file)
    initial_duplicate_groups = _get_hypercore_duplicate_groups(initial_state)
    assert len(initial_duplicate_groups) == 5

    # 2. Run `repair-hypercore-dust --merge-dustless-duplicates` against the copied file.
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "repair-hypercore-dust",
            "--state-file",
            str(state_file),
            "--merge-dustless-duplicates",
            "--unit-testing",
            "--log-level",
            "disabled",
        ],
    )

    # 3. Verify the known stale positions are closed, the known live reopen positions stay open, and no Hypercore duplicates remain.
    assert result.exit_code == 0, result.stdout

    final_state = State.read_json_file(state_file)
    assert _get_hypercore_duplicate_groups(final_state) == []

    stale_duplicate_ids = {31, 30, 29, 28, 17}
    live_reopen_ids = {90, 89, 88, 87, 86}

    for position_id in stale_duplicate_ids:
        assert position_id in final_state.portfolio.closed_positions
        position = final_state.portfolio.closed_positions[position_id]
        last_trade = position.get_last_trade()
        assert TradeFlag.hypercore_duplicate_close in (last_trade.flags or set())
        assert "Closed as duplicate Hypercore clone" in (last_trade.notes or "")
        assert "Closed as duplicate HyperCore clone" not in (last_trade.notes or "")

    for position_id in live_reopen_ids:
        assert position_id in final_state.portfolio.open_positions
        assert position_id not in final_state.portfolio.closed_positions
