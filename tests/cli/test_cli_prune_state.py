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
    assert f"Pruning state file: {state_file}" in output
    assert "Positions processed: 0" in output
    assert "Balance updates removed: 0" in output
    assert "Trades processed: 0" in output
    assert "Blockchain transactions processed: 0" in output

    # Verify the state file still exists and is readable
    assert state_file.exists()
    final_state = State.read_json_file(state_file)
    assert len(final_state.portfolio.closed_positions) == 0

def test_prune_state_with_prunable_positions(mocker: Any, tmp_path: Any):
    """Test prune-state command successfully prunes positions with balance updates and trades."""
    # Load the existing working state file
    source_state_file = Path(os.path.dirname(__file__)) / "show-positions-long.json"
    existing_state = State.read_json_file(source_state_file)

    # Count existing prunable data
    existing_balance_updates = sum(len(pos.balance_updates) for pos in existing_state.portfolio.closed_positions.values())
    existing_trades = sum(len(pos.trades) for pos in existing_state.portfolio.closed_positions.values())
    existing_blockchain_txs = sum(
        len(trade.blockchain_transactions)
        for pos in existing_state.portfolio.closed_positions.values()
        for trade in pos.trades.values()
    )

    # Add balance updates if none exist (since we need to test balance update pruning)
    balance_updates_added = 0
    if existing_balance_updates == 0:
        for i, (position_id, position) in enumerate(existing_state.portfolio.closed_positions.items()):
            # Add 2 balance updates to first position, 1 to second
            balance_count = 2 if i == 0 else 1
            for j in range(balance_count):
                balance_update = BalanceUpdate(
                    balance_update_id=position_id * 10 + j,
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
                    tx_hash=f"0x{position_id}{j}",
                    log_index=1,
                )
                position.balance_updates[position_id * 10 + j] = balance_update
                balance_updates_added += 1

    # Calculate total counts (existing + added)
    total_balance_updates = existing_balance_updates + balance_updates_added
    total_trades = existing_trades
    total_blockchain_txs = existing_blockchain_txs

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

    # Verify successful pruning happened with comprehensive output
    assert "Pruning completed successfully!" in output
    assert "Positions processed: 2" in output
    assert f"Balance updates removed: {total_balance_updates}" in output
    assert f"Trades processed: {total_trades}" in output
    assert f"Blockchain transactions processed: {total_blockchain_txs}" in output

    # Verify all prunable data was removed (main functionality test)
    final_state = State.read_json_file(state_file)
    assert len(final_state.portfolio.closed_positions) == 2

    # Verify all positions have been fully pruned
    for position in final_state.portfolio.closed_positions.values():
        assert len(position.balance_updates) == 0
        for trade in position.trades.values():
            for tx in trade.blockchain_transactions:
                assert tx.transaction_args is None
                assert tx.wrapped_args is None
                assert tx.signed_bytes is None
                assert tx.signed_tx_object is None
