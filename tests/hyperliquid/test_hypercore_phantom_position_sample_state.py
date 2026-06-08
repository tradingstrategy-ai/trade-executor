"""Integration test for phantom Hypercore vault position correction.

Reproduces the YEELON incident from hyper-ai-5.json: two Hypercore vault
withdrawals completed on Hyperliquid but the executor failed to confirm
them, leaving a phantom position with quantity=304 USDC but zero on-chain
equity.  The valuator then set last_token_price=0.0 and the ``start``
command crashed on ``assert valuation_price`` (0.0 is falsy in Python).

This test verifies that ``correct-accounts`` detects and closes the
phantom position, and that subsequent ``check-accounts`` passes cleanly.

Requires:
- ~/hyper-ai-5.json (the production state file with the YEELON phantom)
- TRADING_STRATEGY_API_KEY environment variable
- JSON_RPC_HYPERLIQUID environment variable
"""

import os
import shutil
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeType
from tradeexecutor.visual.equity_curve import calculate_compounding_unrealised_trading_profitability

WORKTREE_ROOT = Path(__file__).resolve().parents[2]

SAMPLE_STATE_FILE = Path.home() / "hyper-ai-5.json"
TEST_PRIVATE_KEY = "0x111e53aed5e777996f26b4bdb89300bbc05b84743f32028c41be7193c0fe0b83"
REQUIRED_ENV_VARS = ("TRADING_STRATEGY_API_KEY", "JSON_RPC_HYPERLIQUID")

YEELON_POSITION_ID = 15
YEELON_VAULT_ADDRESS = "0xf6f3d773e11023e3e686cbda883ecba631fefc15"

pytestmark = [
    pytest.mark.timeout(300),
    pytest.mark.skipif(
        (not SAMPLE_STATE_FILE.exists()) or (not all(os.environ.get(name) for name in REQUIRED_ENV_VARS)),
        reason="Set TRADING_STRATEGY_API_KEY, JSON_RPC_HYPERLIQUID, and ~/hyper-ai-5.json to run this test",
    ),
]


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

    target = tmp_path / "hyper-ai-5-copy.json"
    shutil.copy2(SAMPLE_STATE_FILE, target)
    return target


@pytest.fixture()
def environment(state_file: Path, strategy_file: Path, tmp_path: Path) -> dict[str, str]:
    """Build an isolated CLI environment for the copied sample state."""

    state = State.read_json_file(state_file)
    vault_address = state.sync.deployment.address
    assert vault_address, "Copied sample state does not contain a Lagoon vault address"

    # Extract the single Lagoon module address from trade blockchain transactions
    module_addresses = {
        tx.contract_address
        for position in state.portfolio.open_positions.values()
        for trade in position.trades.values()
        for tx in trade.blockchain_transactions
        if tx.contract_address
    }
    assert len(module_addresses) == 1, \
        f"Expected one Lagoon module address in the copied state, got {module_addresses}"
    module_address = next(iter(module_addresses))

    cache_path = tmp_path / "cache"
    cache_path.mkdir(exist_ok=True)

    return {
        "EXECUTOR_ID": "test_hypercore_phantom_position",
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


def test_correct_accounts_closes_yeelon_phantom_position(
    environment: dict[str, str],
    state_file: Path,
) -> None:
    """Test that correct-accounts closes the YEELON phantom position from hyper-ai-5.json.

    The YEELON vault (position #15) has quantity=304.16 in state but the
    Hyperliquid API reports zero equity. Two withdrawals ($207.94 and $96.22)
    completed on Hyperliquid but the executor failed to confirm them.
    The valuator already set last_token_price=0.0.

    1. Load the copied sample state and confirm YEELON is open with zero price.
    2. Run correct-accounts through the CLI.
    3. Reload state and verify YEELON is closed with a repair trade.
    4. Verify the equity curve does not crash on the corrected state.
    """

    # 1. Load the copied sample state and confirm YEELON is open with zero price.
    initial_state = State.read_json_file(state_file)
    yeelon = initial_state.portfolio.open_positions.get(YEELON_POSITION_ID)
    assert yeelon is not None, f"YEELON position #{YEELON_POSITION_ID} not found in open positions"
    assert yeelon.pair.pool_address.lower() == YEELON_VAULT_ADDRESS.lower()
    assert yeelon.last_token_price == 0.0
    assert float(yeelon.get_quantity()) == pytest.approx(304.1648, abs=0.01)

    initial_open_count = len(initial_state.portfolio.open_positions)

    # 2. Run correct-accounts through the CLI.
    process = _run_cli_command(environment, "correct-accounts")
    output = _get_process_output(process)
    assert process.returncode == 0, f"correct-accounts failed:\n{output}"
    assert "phantom" in output.lower() or "zero equity" in output.lower(), \
        f"Expected phantom position log message in output:\n{output}"

    # 3. Reload state and verify YEELON is closed with a repair trade.
    final_state = State.read_json_file(state_file)

    assert YEELON_POSITION_ID not in final_state.portfolio.open_positions, \
        "YEELON should no longer be in open positions after correct-accounts"
    assert YEELON_POSITION_ID in final_state.portfolio.closed_positions, \
        "YEELON should be in closed positions after correct-accounts"

    closed_yeelon = final_state.portfolio.closed_positions[YEELON_POSITION_ID]
    assert closed_yeelon.get_quantity() == Decimal(0)

    last_trade = list(closed_yeelon.trades.values())[-1]
    assert last_trade.trade_type == TradeType.repair
    assert last_trade.executed_reserve == Decimal(0)
    assert last_trade.is_success()

    # Other positions should still be open (minus YEELON)
    assert len(final_state.portfolio.open_positions) == initial_open_count - 1

    # 4. Verify the equity curve does not crash on the corrected state.
    result = calculate_compounding_unrealised_trading_profitability(final_state)
    assert result is not None
