"""Tests for prune-state CLI command."""
import datetime
import os
from decimal import Decimal
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

from tradeexecutor.cli.main import app
from tradeexecutor.state.state import State
from tradeexecutor.state.balance_update import BalanceUpdate, BalanceUpdateCause, BalanceUpdatePositionType
from tradingstrategy.chain import ChainId


def test_prune_state_nothing_to_prune(mocker: Any, tmp_path: Any):
    """Test prune-state command with empty state (no positions to prune)."""
    # Create an empty state file
    state = State()
    state_file = tmp_path / "empty-state.json"
    state.write_json_file(state_file)

    # Use existing test strategy file
    strategy_path = os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "test_only", "base-ath.py")

    environment = {
        "STATE_FILE": str(state_file),
        "STRATEGY_FILE": strategy_path,
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",  # Disable logging to focus on stdout
    }

    mocker.patch.dict("os.environ", environment, clear=True)

    # Capture stdout following the standard CLI test pattern
    f = StringIO()
    with redirect_stdout(f):
        app(["prune-state"], standalone_mode=False)

    output = f.getvalue()

    # Verify expected stdout messages
    assert "No closed positions found - nothing to prune" in output
    assert f"Pruning state file: {state_file}" in output

    # Verify the state file still exists and is readable
    assert state_file.exists()
    final_state = State.read_json_file(state_file)
    assert len(final_state.portfolio.closed_positions) == 0

def test_prune_state_with_balance_updates(mocker: Any, tmp_path: Any):
    """Test prune-state command successfully prunes balance updates."""
    # Load the existing working state file
    source_state_file = Path(os.path.dirname(__file__)) / "show-positions-long.json"
    existing_state = State.read_json_file(source_state_file)

    # Add 1 balance update to the first closed position
    if len(existing_state.portfolio.closed_positions) > 0:
        position_id, position = next(iter(existing_state.portfolio.closed_positions.items()))

        # Create 1 simple balance update using the existing position's data
        balance_update = BalanceUpdate(
            balance_update_id=999,
            position_id=position_id,
            cause=BalanceUpdateCause.deposit,
            position_type=BalanceUpdatePositionType.open_position,
            asset=position.pair.base,
            block_mined_at=datetime.datetime(2023, 1, 1),
            strategy_cycle_included_at=datetime.datetime(2023, 1, 1),
            chain_id=ChainId.ethereum.value,
            old_balance=Decimal("0"),
            usd_value=100,
            quantity=Decimal("0.05"),
            owner_address="0x123",
            tx_hash="0xabc123",
            log_index=1,
        )
        position.balance_updates[999] = balance_update

    # Save the modified state using JSONFileStore
    from tradeexecutor.state.store import JSONFileStore
    state_file = tmp_path / "test-state.json"
    store = JSONFileStore(state_file)
    store.sync(existing_state)

    strategy_path = os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "test_only", "base-ath.py")

    environment = {
        "STATE_FILE": str(state_file),
        "STRATEGY_FILE": strategy_path,
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
    }

    mocker.patch.dict("os.environ", environment, clear=True)

    f = StringIO()
    with redirect_stdout(f):
        app(["prune-state"], standalone_mode=False)

    output = f.getvalue()

    # Verify successful pruning happened
    assert "Pruning completed successfully!" in output
    assert "Found 2 closed positions with 1 total balance updates" in output
    assert "Balance updates removed: 1" in output

    # Verify the balance update was removed (main functionality test)
    final_state = State.read_json_file(state_file)
    assert len(final_state.portfolio.closed_positions) == 2

    # The position should have no balance updates after pruning
    final_position = final_state.portfolio.closed_positions[position_id] # type: ignore
    assert len(final_position.balance_updates) == 0
