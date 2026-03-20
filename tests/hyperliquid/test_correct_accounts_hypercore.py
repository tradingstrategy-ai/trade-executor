"""Test correct-accounts CLI command with Hypercore vault on HyperEVM mainnet.

Uses the Trial 3 deployment (Safe-Hypercore-Writer-trials.md) which has
5 USDC deposited into HLP on mainnet. The test verifies that correct-accounts
can detect and auto-create a vault position for an existing HLP deposit.

Environment variables:
    HYPERCORE_WRITER_TEST_PRIVATE_KEY: deployer key for Trial 3 Lagoon vault
"""

import os
import tempfile
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.main import get_command

from tradeexecutor.cli.main import app
from tradeexecutor.state.state import State

pytestmark = pytest.mark.skipif(
    not os.environ.get("HYPERCORE_WRITER_TEST_PRIVATE_KEY"),
    reason="HYPERCORE_WRITER_TEST_PRIVATE_KEY not set",
)

#: Trial 3 deployment addresses (HyperEVM mainnet, chain 999)
TRIAL3_LAGOON_VAULT = "0x766089071255274ad4E5f91d2b486e0A1eCaC20C"
TRIAL3_MODULE = "0xE0B3a42c3e34Da277A5a840Bf86B6bd48E9D5c39"


@pytest.fixture()
def strategy_file() -> Path:
    """Path to the minimal Hypercore strategy module."""
    return Path(os.path.dirname(__file__)) / "../../strategies/test_only/minimal_hyperliquid_strategy.py"


@pytest.fixture()
def state_file() -> Path:
    """Create a temporary state file path."""
    return Path(tempfile.mkdtemp()) / "test-correct-accounts-hypercore.json"


@pytest.fixture()
def environment(state_file: Path, strategy_file: Path) -> dict:
    """Environment variables for correct-accounts CLI on HyperEVM mainnet."""
    return {
        "EXECUTOR_ID": "test_correct_accounts_hypercore",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "PRIVATE_KEY": os.environ["HYPERCORE_WRITER_TEST_PRIVATE_KEY"],
        "JSON_RPC_HYPERLIQUID": "https://rpc.hyperliquid.xyz/evm",
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "lagoon",
        "VAULT_ADDRESS": TRIAL3_LAGOON_VAULT,
        "VAULT_ADAPTER_ADDRESS": TRIAL3_MODULE,
        "NETWORK": "mainnet",
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "warning",
        "TRADING_STRATEGY_API_KEY": os.environ.get("TRADING_STRATEGY_API_KEY", ""),
        "PATH": os.environ.get("PATH", ""),
    }


def test_correct_accounts_picks_up_vault_position(
    environment: dict,
    state_file: Path,
):
    """Test that correct-accounts detects an existing HLP vault deposit.

    Uses the Trial 3 deployment on HyperEVM mainnet where 5 USDC was
    deposited into HLP via the Safe. The test:

    1. Inits empty state with USDC reserve
    2. Runs correct-accounts which should auto-create a vault position
    3. Verifies the position has correct attributes and non-zero equity
    """
    cli = get_command(app)

    # Step 1: Init state
    with patch.dict(os.environ, environment, clear=True):
        cli.main(args=["init"], standalone_mode=False)

    assert state_file.exists(), "State file was not created by init"

    # Step 2: Run correct-accounts
    # Exit code may be 0 (clean) or 1 (unclean due to USDC reserve mismatch
    # on the Safe — Trial 3 has leftover USDC that correct-accounts detects).
    # Either way, the vault position should be auto-created and saved.
    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["correct-accounts"])

        assert e.value.code in (0, 1), f"correct-accounts crashed with exit code {e.value.code}"

    # Step 3: Verify vault position was auto-created
    final_state = State.read_json_file(state_file)

    vault_positions = [
        pos for pos in final_state.portfolio.open_positions.values()
        if pos.is_vault() and pos.pair.other_data.get("vault_protocol") == "hypercore"
    ]

    assert len(vault_positions) == 1, \
        f"Expected 1 Hypercore vault position, got {len(vault_positions)}"

    vault_pos = vault_positions[0]

    # Verify position attributes
    assert vault_pos.pair.is_vault()
    assert vault_pos.pair.other_data.get("vault_protocol") == "hypercore"

    # Verify non-zero equity (Trial 3 deposited 5 USDC, value may change over time)
    quantity = vault_pos.get_quantity()
    assert quantity > 0, f"Expected positive vault equity, got {quantity}"

    # Should have one trade in success state (auto-created)
    assert len(vault_pos.trades) == 1
    trade = list(vault_pos.trades.values())[0]
    assert trade.is_success()
    assert "Auto-created" in (trade.notes or ""), \
        f"Trade notes should mention auto-creation, got: {trade.notes}"
