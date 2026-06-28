"""Test Lagoon vault trades."""

import os
from pathlib import Path

import pytest
from typer.main import get_command

from tradeexecutor.cli.main import app
from tradeexecutor.state.state import State
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverseModel
from tradeexecutor.utils.hex import hexbytes_to_hex_str


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
        "MAX_CYCLES": "3",  # decide_trades() needs 2 cycles to test all
        "TRADING_STRATEGY_API_KEY": TRADING_STRATEGY_API_KEY,
        "MAX_DATA_DELAY_MINUTES": str(10 * 60 * 24 * 365),  # 10 years or "disabled""
        "MIN_GAS_BALANCE": "0.01",
        "GAS_BALANCE_WARNING_LEVEL": "0.0",
        "PRIVATE_KEY": hexbytes_to_hex_str(hot_wallet.private_key),
        "CACHE_PATH":  cache_path,
    }
    return environment


@pytest.mark.slow_test_group
def test_vault_trading_start(
    environment: dict,
    mocker,
    state_file,
    web3,
):
    """Run a single Base vault trading cycle.

    This test exercises vault trading CLI wiring and accounting, not Trading
    Strategy remote data freshness.

    1. Patch the process environment with forked RPC and deterministic settings.
    2. Disable remote market-data freshness checks.
    3. Initialise and start the CLI.
    4. Assert the resulting state has one closed position and no frozen positions.
    """

    cli = get_command(app)

    # 1. Patch the process environment with forked RPC and deterministic settings.
    mocker.patch.dict("os.environ", environment, clear=True)

    # 2. Disable remote market-data freshness checks.
    mocker.patch.object(TradingStrategyUniverseModel, "check_data_age", return_value=None)

    # 3. Initialise and start the CLI.
    cli.main(args=["init"], standalone_mode=False)
    cli.main(args=["start"], standalone_mode=False)

    # 4. Assert the resulting state has one closed position and no frozen positions.
    state = State.read_json_file(state_file)
    assert state.cycle == 2
    reserve_position = state.portfolio.get_default_reserve_position()
    assert reserve_position.get_value() > 5.0  # Should have 100 USDC starting balance
    assert len(state.portfolio.frozen_positions) == 0
    assert len(state.portfolio.open_positions) == 0
    assert len(state.portfolio.closed_positions) == 1
