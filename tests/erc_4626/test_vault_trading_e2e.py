"""Test Lagoon vault trades."""

import os
from pathlib import Path

import pytest
from typer.main import get_command

from tradeexecutor.cli.commands.app import app
from tradeexecutor.state.state import State


JSON_RPC_BASE = os.environ.get("JSON_RPC_BASE")
TRADING_STRATEGY_API_KEY = os.environ.get("TRADING_STRATEGY_API_KEY")
pytestmark = pytest.mark.skipif((not JSON_RPC_BASE) or (not TRADING_STRATEGY_API_KEY), reason="No JSON_RPC_BASE and TRADING_STRATEGY_API_KEY environment variable")


@pytest.fixture()
def strategy_file() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.dirname(__file__)) / ".." / ".." / "strategies" / "test_only" / "base-ipor.py"


@pytest.fixture()
def state_file(tmp_path) -> Path:
    path = tmp_path / "base-ipor.json"
    return path


@pytest.fixture()
def environment(
    strategy_file,
    anvil_base_fork,
    state_file,
    hot_wallet,
    persistent_test_client,
):
    """Lagoon CLI environment with vault not yet deployed."""

    cache_path = persistent_test_client.transport.cache_path

    environment = {
        "EXECUTOR_ID": "test_vault_trading_e2e",
        "NAME": "test_vault_trading_e2e",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "JSON_RPC_BASE": anvil_base_fork.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "hot_wallet",
        "UNIT_TESTING": "true",
        # "LOG_LEVEL": "info",  # Set to info to get debug data for the test run
        "LOG_LEVEL": "disabled",
        "MAX_CYCLES": "2",  # decide_trades() needs 2 cycles to test all
        "TRADING_STRATEGY_API_KEY": TRADING_STRATEGY_API_KEY,
        "MAX_DATA_DELAY_MINUTES": str(10 * 60 * 24 * 365),  # 10 years or "disabled""
        "MIN_GAS_BALANCE": "0.01",
        "GAS_BALANCE_WARNING_LEVEL": "0.0",
        "PRIVATE_KEY": hot_wallet.private_key.hex(),
        "CACHE_PATH":  cache_path,
    }
    return environment


def test_vault_trading_start(
    environment: dict,
    mocker,
    state_file,
    web3,
):
    """Run a single cycle of Memecoin index strategy to see everything works.

    - Should attempt to open multiple positions using Enso
    """

    cli = get_command(app)
    mocker.patch.dict("os.environ", environment, clear=True)
    cli.main(args=["init"], standalone_mode=False)
    cli.main(args=["start"], standalone_mode=False)

    # Read results of 1 cycle of strategy
    state = State.read_json_file(state_file)
    reserve_position = state.portfolio.get_default_reserve_position()
    assert reserve_position.get_value() > 5.0  # Should have 100 USDC starting balance
    assert len(state.visualisation.get_messages_tail(5)) == 1
    for t in state.portfolio.get_all_trades():
        assert t.is_success(), f"Trade {t} failed: {t.get_revert_reason()}"
    assert len(state.portfolio.frozen_positions) == 0
