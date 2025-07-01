"""Test CLI command and strategy running with Lagoon vault."""
import json
import os
from pathlib import Path

import flaky
import pytest
from typer.main import get_command
from web3 import Web3

from eth_defi.abi import get_deployed_contract
from eth_defi.safe.deployment import fetch_safe_deployment, disable_safe_module
from eth_defi.safe.simulate import simulate_safe_execution_anvil
from eth_defi.trace import assert_transaction_success_with_explanation
from tradeexecutor.cli.commands.app import app
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.state.state import State

JSON_RPC_BASE = os.environ.get("JSON_RPC_BASE")
TRADING_STRATEGY_API_KEY = os.environ.get("TRADING_STRATEGY_API_KEY")
CI = os.environ.get("CI") == "true"

pytestmark = pytest.mark.skipif(
     (not JSON_RPC_BASE or not TRADING_STRATEGY_API_KEY),
      reason="Set JSON_RPC_BASE and TRADING_STRATEGY_API_KEY needed to run this test"
)


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


@pytest.fixture()
def strategy_file() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.dirname(__file__)) / ".." / ".." / "strategies" / "test_only" / "base-memecoin-index.py"


@pytest.fixture()
def vaulted_strategy_file() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.dirname(__file__)) / ".." / ".." / "strategies" / "test_only" / "base-memecoin-index-with-vault.py"


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
    persistent_test_client,
):
    """Lagoon CLI environment with predeployed vault."""
    deploy_info = deposited_lagoon_vault

    cache_path = persistent_test_client.transport.cache_path

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
        "GENERATE_REPORT": "false",
        "CACHE_PATH": cache_path,
    }
    return environment



@pytest.fixture()
def pre_deployment_vault_environment(
    strategy_file,
    anvil_base_fork,
    state_file,
    asset_manager,
    persistent_test_client,
):
    """Lagoon CLI environment with vault not yet deployed."""

    cache_path = persistent_test_client.transport.cache_path

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
        "CACHE_PATH": cache_path,
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
    persistent_test_client,
):
    """Deploy Lagoon vault."""

    cache_path  =persistent_test_client.transport.cache_path

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
        "CACHE_PATH": cache_path,
        "GENERATE_REPORT": "false",  # Creating backtest report takes too long > 300s
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


# AssertionError: Could not read block number from Anvil after the launch anvil: at http://localhost:22353, stdout is 0 bytes, stderr is 312 bytes
@flaky.flaky
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


@pytest.mark.skipif(CI, reason="Too slow to run on Github, > 600 seconds")
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
    cli.main(args=["start"], standalone_mode=False)

    # Read results of 1 cycle of strategy
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


@pytest.mark.slow_test_group
def test_cli_lagoon_redeploy_guard(
    deployed_vault_environment: dict,
    mocker,
    state_file,
    web3,
    tmp_path,
    vaulted_strategy_file,
    logger,
):
    """Deploy a new guard smart contract for Lagoon vault.

    1. Deploy a Lagoon vault
    2. Deploy a new guard smart contract with different parameters (one vault trading enabled)
    3. Perform Gnosis tx to disable old guard
    4. Perform Gnosis tx to enable old guard
    5. Perform a test deposit/redeem on ERC-4626 vaults using the new guard
    """

    #
    # 1. Deploy a Lagoon vault
    #
    cli = get_command(app)
    mocker.patch.dict("os.environ", deployed_vault_environment, clear=True)
    cli.main(args=["init"], standalone_mode=False)
    cli.main(args=["correct-accounts"], standalone_mode=False)

    # After deployment, fetch our existing guard and Safe information
    existing_guard = deployed_vault_environment["VAULT_ADAPTER_ADDRESS"]
    guard_contract = get_deployed_contract(
        web3,
        "safe-integration/TradingStrategyModuleV0.json",
        existing_guard,
    )
    safe_address = guard_contract.functions.getGovernanceAddress().call()
    safe = fetch_safe_deployment(web3, safe_address)

    # Check that the Safe is valid and deployed
    owners = safe.retrieve_owners()
    assert len(owners) == 4

    #
    # 2. Deploy a new guard smart contract with different parameters (one vault trading enabled)
    #
    vault_record_file = tmp_path / "vault-record.json"
    environment = deployed_vault_environment.copy()
    environment["ERC_4626_VAULTS"] = "0x7bfa7c4f149e7415b73bdedfe609237e29cbf34a, 0x0d877Dc7C8Fa3aD980DfDb18B48eC9F8768359C4, 0x7a63e8fc1d0a5e9be52f05817e8c49d9e2d6efae"
    environment["GUARD_ONLY"] = "true"
    environment["UNISWAP_V2"] = "true"
    environment["UNISWAP_V3"] = "true"
    environment["AAVE"] = "true"
    environment["ANY_ASSET"] = "true"  # TODO: Temporarily needed, remove later
    environment["EXISTING_VAULT_ADDRESS"] = deployed_vault_environment["VAULT_ADDRESS"]
    environment["EXISTING_SAFE_ADDRESS"] = safe_address

    environment["VAULT_RECORD_FILE"] = vault_record_file.as_posix()
    mocker.patch.dict("os.environ", environment, clear=True)
    cli.main(args=["lagoon-deploy-vault"], standalone_mode=False)

    vault_info = json.load(open(vault_record_file))
    new_guard_address = vault_info["Trading strategy module"]

    # Check new guard points to the right Safe, the exisitng safe
    # and the new guard has all vaults whitelisted
    new_guard_contract = get_deployed_contract(
        web3,
        "safe-integration/TradingStrategyModuleV0.json",
        new_guard_address,
    )
    assert vault_info["Safe"].lower() == safe_address.lower(), "Vault data corrupted, Safe address mismatch"
    # https://github.com/tradingstrategy-ai/web3-ethereum-defi/blob/master/contracts/guard/src/GuardV0Base.sol
    assert new_guard_contract.functions.getGovernanceAddress().call() == safe_address
    assert new_guard_contract.functions.isAllowedApprovalDestination(Web3.to_checksum_address("0x7bfa7c4f149e7415b73bdedfe609237e29cbf34a")).call() == True  # Spark
    assert new_guard_contract.functions.isAllowedApprovalDestination(Web3.to_checksum_address("0x0d877Dc7C8Fa3aD980DfDb18B48eC9F8768359C4")).call() == True  # Harvest

    # ERC-7545: not supported yet
    # assert new_guard_contract.functions.isAllowedApprovalDestination(Web3.to_checksum_address("0x7a63e8fc1d0a5e9be52f05817e8c49d9e2d6efae")).call() == True  # maxAPY

    #
    # 3. Perform Gnosis tx to disable old guard
    #
    # disableModule() exposes Safe internal linked list
    func = disable_safe_module(
        web3,
        safe_address,
        guard_contract.address,
    )
    tx_hash = simulate_safe_execution_anvil(
        web3,
        safe_address,
        func,
    )
    assert_transaction_success_with_explanation(web3, tx_hash)
    modules = safe.retrieve_modules()
    assert modules == [], f"Old guard still present: {modules}"

    #
    # 4. Perform Gnosis tx to enable new guard
    #
    func = safe.contract.functions.enableModule(new_guard_address)
    tx_hash = simulate_safe_execution_anvil(
        web3,
        safe_address,
        func,
    )
    assert_transaction_success_with_explanation(web3, tx_hash)

    # Check module is enabled (twice)
    modules = safe.retrieve_modules()
    assert modules == [new_guard_address], f"Guard not updated, old guard still present: {modules}"
    safe = fetch_safe_deployment(web3, vault_info["Safe"])
    assert safe.contract.functions.isModuleEnabled(new_guard_address).call() == True
    assert safe.contract.functions.isModuleEnabled(guard_contract.address).call() == False

    # Fix us to use the strategy module where vaults are part of the universe loading
    environment["STRATEGY_FILE"] = vaulted_strategy_file.as_posix()

    # Update the trade-executor to use the new enabled guard contract
    environment["VAULT_ADAPTER_ADDRESS"] = new_guard_address
    mocker.patch.dict("os.environ", environment, clear=True)

    #
    # 5. Perform a test trade using the new guard
    #

    def _check_clean_state():
        _state = State.read_json_file(state_file)
        _trades = list(_state.portfolio.get_all_trades())
        for t in _trades:
            assert t.is_success(), f"Trade {t} failed: {t.get_revert_reason()}"

        for p in _state.portfolio.get_all_positions():
            assert not p.is_open(), f"Position {p} is still open after test trades: {list(p.trades.values())}"

        return _state, _trades

    # Check the test trade using a single vault by its human description
    cli.main(args=["perform-test-trade", "--pair", "(base, morpho, sparkUSDC, USDC)"], standalone_mode=False)
    state, trades = _check_clean_state()
    assert len(trades) == 2, f"Got trades: {trades}"

    # Check all vault deposit/redeem
    cli.main(args=["perform-test-trade", "--all-vaults"], standalone_mode=False)
    state, trades = _check_clean_state()
    assert len(trades) == 6, f"Got trades: {trades}"

    # Check there is no change in Aave trade whitelisting
    cli.main(args=["perform-test-trade", "--lending-reserve", "(base, aave-v3, USDC)"], standalone_mode=False)
    _check_clean_state()
    # Check there is no change in Uniswap v2 trade whitelisting
    cli.main(args=["perform-test-trade", "--pair", "(base, uniswap-v2, KEYCAT, WETH, 0.0030)"], standalone_mode=False)
    _check_clean_state()

