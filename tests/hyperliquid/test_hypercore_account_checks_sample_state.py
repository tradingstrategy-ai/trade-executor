"""Live CLI regressions for Hypercore vault account checks.

These tests use a copied local sample state so we can verify Hypercore
vault reporting and correction behaviour against realistic live-style data
without mutating the original file in ``~/Downloads``.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

WORKTREE_ROOT = Path(__file__).resolve().parents[2]

from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State


SAMPLE_STATE_FILE = Path.home() / "Downloads" / "hyper-ai-3.json"
TEST_PRIVATE_KEY = "0x111e53aed5e777996f26b4bdb89300bbc05b84743f32028c41be7193c0fe0b83"
REQUIRED_ENV_VARS = ("TRADING_STRATEGY_API_KEY", "JSON_RPC_HYPERLIQUID")

pytestmark = [
    pytest.mark.timeout(300),
    pytest.mark.skipif(
        (not SAMPLE_STATE_FILE.exists()) or (not all(os.environ.get(name) for name in REQUIRED_ENV_VARS)),
        reason="Set TRADING_STRATEGY_API_KEY, JSON_RPC_HYPERLIQUID, and ~/Downloads/hyper-ai-3.json to run this test",
    ),
]


def _extract_vault_and_module_addresses(state: State) -> tuple[str, str]:
    """Read CLI deployment addresses from the copied sample state."""

    vault_address = state.sync.deployment.address
    assert vault_address, "Copied sample state does not contain a Lagoon vault address"

    module_addresses = {
        tx.contract_address
        for position in state.portfolio.open_positions.values()
        for trade in position.trades.values()
        for tx in trade.blockchain_transactions
        if tx.contract_address
    }
    assert len(module_addresses) == 1, \
        f"Expected one Lagoon module address in the copied state, got {module_addresses}"

    return vault_address, next(iter(module_addresses))


def _get_hypercore_positions(state: State) -> list[TradingPosition]:
    """Return Hypercore vault positions from the strategy state."""

    return [
        position
        for position in state.portfolio.open_positions.values()
        if position.pair.is_hyperliquid_vault()
    ]


def _count_vault_flow_updates(position: TradingPosition) -> int:
    """Count Hypercore vault-flow balance updates on a position."""

    return sum(
        1
        for update in position.balance_updates.values()
        if getattr(update.cause, "value", update.cause) == "vault_flow"
    )


def _run_cli_command(environment: dict[str, str], command: str) -> subprocess.CompletedProcess[str]:
    """Run a CLI command in a subprocess pinned to the worktree imports."""

    pythonpath_parts = [str(WORKTREE_ROOT)]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)

    process_environment = {
        **environment,
        "PYTHONPATH": os.pathsep.join(pythonpath_parts),
    }

    return subprocess.run(
        [
            sys.executable,
            "-c",
            f"from tradeexecutor.cli.main import app; app(['{command}'], standalone_mode=False)",
        ],
        cwd=WORKTREE_ROOT,
        env=process_environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _get_process_output(process: subprocess.CompletedProcess[str]) -> str:
    """Combine subprocess streams for stable CLI assertions."""

    return "\n".join(part for part in (process.stdout, process.stderr) if part)


@pytest.fixture()
def strategy_file() -> Path:
    """Path to the live-style Hypercore strategy used by the sample state."""

    return Path(__file__).resolve().parents[2] / "strategies" / "test_only" / "hyper-ai-test.py"


@pytest.fixture()
def state_file(tmp_path: Path) -> Path:
    """Copy the local sample state to a temporary path for mutation-safe tests."""

    target = tmp_path / "hyper-ai-3-copy.json"
    shutil.copy2(SAMPLE_STATE_FILE, target)
    return target


@pytest.fixture()
def environment(state_file: Path, strategy_file: Path, tmp_path: Path) -> dict[str, str]:
    """Build an isolated CLI environment for the copied sample state."""

    state = State.read_json_file(state_file)
    vault_address, module_address = _extract_vault_and_module_addresses(state)
    cache_path = tmp_path / "cache"
    cache_path.mkdir(exist_ok=True)

    return {
        "EXECUTOR_ID": "test_hypercore_account_checks_sample_state",
        "STRATEGY_FILE": str(strategy_file),
        "STATE_FILE": str(state_file),
        "CACHE_PATH": str(cache_path),
        "ASSET_MANAGEMENT_MODE": "lagoon",
        "VAULT_ADDRESS": vault_address,
        "VAULT_ADAPTER_ADDRESS": module_address,
        "PRIVATE_KEY": TEST_PRIVATE_KEY,
        "JSON_RPC_HYPERLIQUID": os.environ["JSON_RPC_HYPERLIQUID"],
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "info",
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", str(Path.home())),
    }


def test_check_accounts_lists_hypercore_vault_positions(
    environment: dict[str, str],
    state_file: Path,
) -> None:
    """Test `check-accounts` lists Hypercore vault positions from the copied state.

    This verifies the CLI table includes API-tracked Hypercore vault positions
    instead of silently reporting only reserves.

    1. Load the copied sample state and confirm it contains open Hypercore vault positions.
    2. Run the `check-accounts` bootstrap flow against the copied state.
    3. Verify the resulting account table includes known Hypercore vault rows instead of only reserves.
    """

    # 1. Load the copied sample state and confirm it contains open Hypercore vault positions.
    initial_state = State.read_json_file(state_file)
    vault_positions = _get_hypercore_positions(initial_state)
    assert len(vault_positions) == 8, f"Expected 8 Hypercore vault positions, got {len(vault_positions)}"

    # 2. Run the `check-accounts` bootstrap flow against the copied state.
    process = _run_cli_command(environment, "check-accounts")
    output = _get_process_output(process)

    # 3. Verify the resulting account table includes known Hypercore vault rows instead of only reserves.
    assert process.returncode in (0, 1), output
    assert "Reserves" in output
    assert "pmalt" in output, output
    assert "[ Systemic Strategies ] L/S Grids" in output, output


def test_correct_accounts_syncs_hypercore_vault_positions(
    environment: dict[str, str],
    state_file: Path,
) -> None:
    """Test `correct-accounts` syncs Hypercore vault quantities from the API.

    This verifies the CLI applies Hypercore vault-flow updates to the copied
    sample state before the generic on-chain correction pass.

    1. Load the copied sample state and record Hypercore quantities and vault-flow update counts.
    2. Run `correct-accounts` through the CLI against the copied state.
    3. Reload the state and verify at least one Hypercore position receives a new vault-flow sync.
    4. Run `check-accounts` again and verify the final state is clean by exit code.
    """

    # 1. Load the copied sample state and record Hypercore quantities and vault-flow update counts.
    initial_state = State.read_json_file(state_file)
    initial_positions = _get_hypercore_positions(initial_state)
    assert len(initial_positions) == 8, f"Expected 8 Hypercore vault positions, got {len(initial_positions)}"

    initial_snapshot = {
        position.position_id: {
            "quantity": position.get_quantity(),
            "vault_flow_updates": _count_vault_flow_updates(position),
        }
        for position in initial_positions
    }
    zero_update_position_ids = [
        position_id
        for position_id, snapshot in initial_snapshot.items()
        if snapshot["vault_flow_updates"] == 0
    ]
    assert zero_update_position_ids, "Expected at least one Hypercore vault without prior vault-flow updates"

    # 2. Run `correct-accounts` through the CLI against the copied state.
    process = _run_cli_command(environment, "correct-accounts")
    output = _get_process_output(process)
    assert process.returncode == 0, output

    # 3. Reload the state and verify at least one Hypercore position receives a new vault-flow sync.
    final_state = State.read_json_file(state_file)
    final_positions = {
        position.position_id: position
        for position in _get_hypercore_positions(final_state)
    }
    assert set(initial_snapshot).issubset(final_positions), "Hypercore positions disappeared after correct-accounts"

    newly_synced_zero_update_positions = [
        position_id
        for position_id in zero_update_position_ids
        if _count_vault_flow_updates(final_positions[position_id]) > initial_snapshot[position_id]["vault_flow_updates"]
    ]
    quantity_changed_positions = [
        position_id
        for position_id, snapshot in initial_snapshot.items()
        if final_positions[position_id].get_quantity() != snapshot["quantity"]
    ]

    assert newly_synced_zero_update_positions or quantity_changed_positions, (
        "Expected correct-accounts to sync at least one Hypercore vault position "
        f"(newly synced zero-update positions: {newly_synced_zero_update_positions}, "
        f"quantity changes: {quantity_changed_positions})"
    )
    assert "Vault equity sync:" in output, output

    # 4. Run `check-accounts` again and verify the final state is clean by exit code.
    final_check_process = _run_cli_command(environment, "check-accounts")
    final_check_output = _get_process_output(final_check_process)
    assert final_check_process.returncode == 0, final_check_output
    assert "All accounts match" in final_check_output, final_check_output
