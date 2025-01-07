"""Test CLI command and strategy running with Lagoon vault."""

import os
from pathlib import Path

import pytest
from typer.main import get_command

from tradeexecutor.cli.commands.app import app
from tradeexecutor.state.state import State

JSON_RPC_BASE = os.environ.get("JSON_RPC_BASE")
TRADING_STRATEGY_API_KEY = os.environ.get("TRADING_STRATEGY_API_KEY")

pytestmark = pytest.mark.skipif(
     (not JSON_RPC_BASE or not TRADING_STRATEGY_API_KEY),
      reason="Set JSON_RPC_BASE and TRADING_STRATEGY_API_KEY needed to run this test"
)


@pytest.fixture()
def strategy_file() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.dirname(__file__)) / ".." / ".." / "strategies" / "test_only" / "base-memecoin-index.py"


@pytest.fixture()
def state_file(tmp_path) -> Path:
    path = tmp_path / "base-memecoin-index.json"
    return path


@pytest.fixture()
def deployed_vault_environment(
    strategy_file,
    anvil_base_fork,
    state_file,
    asset_manager,
    deposited_lagoon_vault,
):
    """Lagoon CLI environment with predeployed vault."""
    deploy_info = deposited_lagoon_vault

    environment = {
        "PATH": os.environ["PATH"],  # Need forge
        "EXECUTOR_ID": "deployed_vault_environment",
        "NAME": "deployed_vault_environment",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "JSON_RPC_BASE": anvil_base_fork.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "lagoon",
        "VAULT_ADDRESS": deploy_info.vault.vault_address,
        "VAULT_ADAPTER_ADDRESS": deploy_info.vault.trading_strategy_module_address,
        "UNIT_TESTING": "true",
        # "LOG_LEVEL": "info",  # Set to info to get debug data for the test run
        "LOG_LEVEL": "disabled",
        "RUN_SINGLE_CYCLE": "true",
        "TRADING_STRATEGY_API_KEY": TRADING_STRATEGY_API_KEY,
        "MAX_DATA_DELAY_MINUTES": str(10 * 60 * 24 * 365),  # 10 years or "disabled""
        "MIN_GAS_BALANCE": "0.005",
        "RAISE_ON_UNCLEAN": "true",  # For correct-accounts
        "PRIVATE_KEY": asset_manager.private_key.hex(),
    }
    return environment



@pytest.fixture()
def pre_deployment_vault_environment(
    strategy_file,
    anvil_base_fork,
    state_file,
    asset_manager,
):
    """Lagoon CLI environment with vault not yet deployed."""

    environment = {
        "PATH": os.environ["PATH"],  # Need forge
        "EXECUTOR_ID": "deployed_vault_environment",
        "NAME": "deployed_vault_environment",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "JSON_RPC_BASE": anvil_base_fork.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "lagoon",
        "UNIT_TESTING": "true",
        # "LOG_LEVEL": "info",  # Set to info to get debug data for the test run
        "LOG_LEVEL": "disabled",
        "RUN_SINGLE_CYCLE": "true",
        "TRADING_STRATEGY_API_KEY": TRADING_STRATEGY_API_KEY,
        "MAX_DATA_DELAY_MINUTES": str(10 * 60 * 24 * 365),  # 10 years or "disabled""
        "MIN_GAS_BALANCE": "0.005",
        "RAISE_ON_UNCLEAN": "true",  # For correct-accounts
        "PRIVATE_KEY": asset_manager.private_key.hex(),
        "ANY_ASSET": "true",
    }
    return environment


def test_cli_lagoon_deploy_vault(
    web3,
    anvil_base_fork,
    strategy_file,
    mocker,
    state_file,
    asset_manager,
    tmp_path: Path,
    base_usdc,
):
    """Deploy Lagoon vault."""

    multisig_owners = f"{web3.eth.accounts[2]}, {web3.eth.accounts[3]}, {web3.eth.accounts[4]}"

    environment = {
        "PATH": os.environ["PATH"],  # Need forge
        "EXECUTOR_ID": "test_cli_lagoon_deploy_vault",
        "NAME": "test_cli_lagoon_deploy_vault",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "JSON_RPC_BASE": anvil_base_fork.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "lagoon",
        "UNIT_TESTING": "true",
        # "LOG_LEVEL": "info",  # Set to info to get debug data for the test run
        "LOG_LEVEL": "disabled",
        "TRADING_STRATEGY_API_KEY": TRADING_STRATEGY_API_KEY,
        "PRIVATE_KEY": asset_manager.private_key.hex(),
        "VAULT_RECORD_FILE": str(tmp_path / "vault-record.json"),
        "FUND_NAME": "Example",
        "FUND_SYMBOL": "EXAM",
        "MULTISIG_OWNERS": multisig_owners,
        "DENOMINATION_ASSET": base_usdc.address,
        "ANY_ASSET": "true",
        "UNISWAP_V2": "true",
        "UNISWAP_V3": "true",
    }

    cli = get_command(app)
    mocker.patch.dict("os.environ", environment, clear=True)
    cli.main(args=["lagoon-deploy-vault"], standalone_mode=False)


def test_cli_lagoon_check_wallet(
    deployed_vault_environment: dict,
    mocker,
    state_file,
    web3,
):
    """Run check-wallet command."""

    cli = get_command(app)
    mocker.patch.dict("os.environ", deployed_vault_environment, clear=True)
    cli.main(args=["check-wallet"], standalone_mode=False)


def test_cli_lagoon_check_universe(
    pre_deployment_vault_environment: dict,
    mocker,
    state_file,
    web3,
):
    """Run check-universe command."""

    cli = get_command(app)
    mocker.patch.dict("os.environ", pre_deployment_vault_environment, clear=True)
    cli.main(args=["check-universe"], standalone_mode=False)


def test_cli_lagoon_perform_test_trade(
    deployed_vault_environment: dict,
    mocker,
    state_file,
    web3,
):
    """Run a single test trade using the vault."""

    cli = get_command(app)
    mocker.patch.dict("os.environ", deployed_vault_environment, clear=True)
    cli.main(args=["init"], standalone_mode=False)
    cli.main(args=["perform-test-trade", "--pair", "(base, uniswap-v2, KEYCAT, WETH, 0.0030)"], standalone_mode=False)

    state = State.read_json_file(state_file)
    keycat_trades = [t for t in state.portfolio.get_all_trades() if t.pair.base.token_symbol == "KEYCAT"]
    assert len(keycat_trades) == 2

    for t in keycat_trades:
        assert t.is_success()


def test_cli_lagoon_backtest(
    mocker,
    state_file,
    web3,
    pre_deployment_vault_environment,
):
    """Run backtest using a the vault strat."""

    cli = get_command(app)
    mocker.patch.dict("os.environ", pre_deployment_vault_environment, clear=True)
    cli.main(args=["backtest"], standalone_mode=False)


def test_cli_lagoon_start(
    deployed_vault_environment: dict,
    mocker,
    state_file,
    web3,
):
    """Run a single cycle of Memecoin index strategy to see everything works.

    - Should attempt to open multiple positions using Enso
    """

    cli = get_command(app)
    mocker.patch.dict("os.environ", deployed_vault_environment, clear=True)
    cli.main(args=["init"], standalone_mode=False)

    state = State.read_json_file(state_file)
    assert state.cycle == 1

    cli.main(args=["start"], standalone_mode=False)

    state = State.read_json_file(state_file)
    reserve_position = state.portfolio.get_default_reserve_position()
    assert reserve_position.get_value() > 5.0  # Should have 100 USDC starting balance
    assert len(state.visualisation.get_messages_tail(5)) == 1
    for t in state.portfolio.get_all_trades():
        assert t.is_success(), f"Trade {t} failed: {t.get_revert_reason()}"
    assert len(state.portfolio.frozen_positions) == 0


def test_cli_lagoon_correct_accounts(
    deployed_vault_environment: dict,
    mocker,
    state_file,
    web3,
):
    """Check correct-accounts works with Velvet sync model.

    - This test checks code runs, but does not attempt to repair any errors
    """
    cli = get_command(app)
    mocker.patch.dict("os.environ", deployed_vault_environment, clear=True)
    cli.main(args=["init"], standalone_mode=False)
    cli.main(args=["correct-accounts"], standalone_mode=False)
