"""Test CLI command and strategy running with Lagoon vault on BNB Chain."""
import json
import os
from pathlib import Path
from pprint import pformat

import pytest
from typer.main import get_command
from web3 import Web3

from eth_defi.hotwallet import HotWallet
from eth_defi.lagoon.testing import fund_lagoon_vault
from eth_defi.provider.anvil import fork_network_anvil, AnvilLaunch
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.token import USDT_NATIVE_TOKEN, fetch_erc20_details, TokenDetails

from tradeexecutor.cli.commands.app import app
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.state.state import State

from eth_typing import HexAddress

JSON_RPC_BINANCE = os.environ.get("JSON_RPC_BINANCE")
TRADING_STRATEGY_API_KEY = os.environ.get("TRADING_STRATEGY_API_KEY")
CI = os.environ.get("CI") == "true"

pytestmark = pytest.mark.skipif(
     (not JSON_RPC_BINANCE or not TRADING_STRATEGY_API_KEY),
      reason="Set JSON_RPC_BINANCE and TRADING_STRATEGY_API_KEY needed to run this test"
)


@pytest.fixture()
def usdt_holder() -> HexAddress:
    return "0x8894E0a0c962CB723c1976a4421c95949bE2D4E3"


@pytest.fixture()
def usdt(web3) -> TokenDetails:
    return fetch_erc20_details(web3, USDT_NATIVE_TOKEN[56])


@pytest.fixture()
def anvil_bnb_fork(request, usdt_holder) -> AnvilLaunch:
    """Create a testable fork of live BNB chain.

    :return: JSON-RPC URL for Web3
    """
    assert JSON_RPC_BINANCE, "JSON_RPC_BINANCE not set"
    launch = fork_network_anvil(
        JSON_RPC_BINANCE,
        unlocked_addresses=[usdt_holder],
    )
    try:
        yield launch
    finally:
        # Wind down Anvil process after the test is complete
        launch.close()



@pytest.fixture()
def web3(anvil_bnb_fork) -> Web3:
    """Create a web3 connector.

    - By default use Anvil forked Base

    - Eanble Tenderly testnet with `JSON_RPC_TENDERLY` to debug
      otherwise impossible to debug Gnosis Safe transactions
    """

    tenderly_fork_rpc = os.environ.get("JSON_RPC_TENDERLY", None)

    if tenderly_fork_rpc:
        web3 = create_multi_provider_web3(tenderly_fork_rpc)
    else:
        web3 = create_multi_provider_web3(
            anvil_bnb_fork.json_rpc_url,
            default_http_timeout=(3, 250.0),  # multicall slow, so allow improved timeout
        )
    assert web3.eth.chain_id == 56
    return web3


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


@pytest.fixture()
def strategy_file() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.dirname(__file__)) / ".." / ".." / "strategies" / "test_only" / "bnb-local-high.py"


@pytest.fixture()
def strategy_file_with_anvil_checks() -> Path:
    return Path(os.path.dirname(__file__)) / ".." / ".." / "strategies" / "test_only" / "bnb-local-high-v2.py"


@pytest.fixture()
def state_file(tmp_path) -> Path:
    path = tmp_path / "bnb-local-high.json"
    return path


@pytest.fixture()
def hot_wallet(web3, usdt_holder) -> HotWallet:
    """A test account with USDT balance."""
    hw = HotWallet.create_for_testing(
        web3,
        test_account_n=1,
        eth_amount=10,
    )
    hw.sync_nonce(web3)
    return hw


@pytest.mark.slow_test_group
def test_cli_lagoon_deploy_binance_vault(
    web3,
    anvil_bnb_fork,
    strategy_file,
    mocker,
    state_file,
    hot_wallet,
    tmp_path: Path,
    persistent_test_client,
    usdt_holder: HexAddress,
):
    """Deploy Lagoon vault on Binance, run test trades and start executor.

    - This will also launch /charts endpoint on the web API
    """

    cache_path  =persistent_test_client.transport.cache_path

    multisig_owners = f"{web3.eth.accounts[2]}, {web3.eth.accounts[3]}, {web3.eth.accounts[4]}"

    usdt_address = USDT_NATIVE_TOKEN[56]

    vault_record_file = tmp_path / "vault-record.json"
    environment = {
        "PATH": os.environ["PATH"],  # Need forge
        "EXECUTOR_ID": "test_cli_lagoon_deploy_binance_vault",
        "NAME": "test_cli_lagoon_deploy_binance_vault",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "JSON_RPC_BINANCE": anvil_bnb_fork.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "lagoon",
        "UNIT_TESTING": "true",
        # "LOG_LEVEL": "info",  # Set to info to get debug data for the test run
        "LOG_LEVEL": "disabled",
        "TRADING_STRATEGY_API_KEY": TRADING_STRATEGY_API_KEY,
        "PRIVATE_KEY": hot_wallet.private_key.hex(),
        "VAULT_RECORD_FILE": str(vault_record_file),
        "FUND_NAME": "Example",
        "FUND_SYMBOL": "EXAM",
        "MULTISIG_OWNERS": multisig_owners,
        "DENOMINATION_ASSET": usdt_address,
        "ANY_ASSET": "true",
        "UNISWAP_V2": "true",
        "UNISWAP_V3": "true",
        "CACHE_PATH": cache_path,
        "GENERATE_REPORT": "false",  # Creating backtest report takes too long > 300s
        "RUN_SINGLE_CYCLE": "true",
    }

    cli = get_command(app)
    mocker.patch.dict("os.environ", environment, clear=True)

    # 1. Deploy vault
    cli.main(args=["lagoon-deploy-vault"], standalone_mode=False)

    # 1.b update our envinoment with the deployed vault address
    deploy_info = json.load(vault_record_file.open("rt"))
    environment.update({
        "VAULT_ADDRESS": deploy_info["Vault"],
        "VAULT_ADAPTER_ADDRESS": deploy_info["Trading strategy module"],
    })
    mocker.patch.dict("os.environ", environment, clear=True)

    # 2. Init state file
    cli.main(args=["init"], standalone_mode=False)

    # 3.a) Fund vault with some USDT
    fund_lagoon_vault(
        web3,
        deploy_info["Vault"],
        test_account_with_balance=usdt_holder,
        asset_manager=hot_wallet.address,
        trading_strategy_module_address=deploy_info["Trading strategy module"],
    )

    # 3. Perform a test trade
    cli.main(args=["perform-test-trade", "--pair", "(binance, pancakeswap-v2, WBNB, USDT, 0.0025)"], standalone_mode=False)

    # 4. Start executor and run 1s cycle
    cli.main(args=["start"], standalone_mode=False)

    # 4.b Check the 1s cycle has been run by inspecting the saved state
    # after the cycle
    state = State.read_json_file(state_file)
    reserve_position = state.portfolio.get_default_reserve_position()
    assert reserve_position.get_value() > 5.0  # Should have 100 USDC starting balance
    assert len(state.visualisation.get_messages_tail(5)) == 2
    for t in state.portfolio.get_all_trades():
        assert t.is_success(), f"Trade {t} failed: {t.get_revert_reason()}"
    assert len(state.portfolio.frozen_positions) == 0

    # 4.c) Check our chart registry was registered with run state shared with the web API
    other_data = state.other_data
    assert other_data.get_latest_stored_cycle() == 1, f"Expected cycle 1, got {other_data.get_latest_stored_cycle()}"
    assert len(other_data.data) > 0, "Expected some data in other_data, but none found"
    chart_count = other_data.load_latest("loaded_chart_count")
    assert chart_count == 24, f"Expected 1 chart, got {chart_count}: {pformat(other_data.data)}"



@pytest.mark.slow_test_group
def test_cli_lagoon_anvil_checks(
    web3,
    anvil_bnb_fork,
    strategy_file_with_anvil_checks,
    mocker,
    state_file,
    hot_wallet,
    tmp_path: Path,
    persistent_test_client,
    usdt_holder: HexAddress,
):
    """Check our Anvil check works.
    """

    cache_path  =persistent_test_client.transport.cache_path

    multisig_owners = f"{web3.eth.accounts[2]}, {web3.eth.accounts[3]}, {web3.eth.accounts[4]}"

    usdt_address = USDT_NATIVE_TOKEN[56]

    vault_record_file = tmp_path / "vault-record.json"
    environment = {
        "PATH": os.environ["PATH"],  # Need forge
        "EXECUTOR_ID": "test_cli_lagoon_deploy_binance_vault",
        "NAME": "test_cli_lagoon_deploy_binance_vault",
        "STRATEGY_FILE": strategy_file_with_anvil_checks.as_posix(),
        "JSON_RPC_BINANCE": anvil_bnb_fork.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "lagoon",
        "UNIT_TESTING": "true",
        # "LOG_LEVEL": "info",  # Set to info to get debug data for the test run
        "LOG_LEVEL": "disabled",
        "TRADING_STRATEGY_API_KEY": TRADING_STRATEGY_API_KEY,
        "PRIVATE_KEY": hot_wallet.private_key.hex(),
        "VAULT_RECORD_FILE": str(vault_record_file),
        "FUND_NAME": "Example",
        "FUND_SYMBOL": "EXAM",
        "MULTISIG_OWNERS": multisig_owners,
        "DENOMINATION_ASSET": usdt_address,
        "ANY_ASSET": "true",
        "UNISWAP_V2": "true",
        "UNISWAP_V3": "true",
        "CACHE_PATH": cache_path,
        "GENERATE_REPORT": "false",  # Creating backtest report takes too long > 300s
        "RUN_SINGLE_CYCLE": "true",
    }

    cli = get_command(app)
    mocker.patch.dict("os.environ", environment, clear=True)

    # 1. Deploy vault
    cli.main(args=["lagoon-deploy-vault"], standalone_mode=False)

    # 1.b update our envinoment with the deployed vault address
    deploy_info = json.load(vault_record_file.open("rt"))
    environment.update({
        "VAULT_ADDRESS": deploy_info["Vault"],
        "VAULT_ADAPTER_ADDRESS": deploy_info["Trading strategy module"],
    })
    mocker.patch.dict("os.environ", environment, clear=True)

    # 2. Init state file
    cli.main(args=["init"], standalone_mode=False)

    # 3.a) Fund vault with some USDT
    fund_lagoon_vault(
        web3,
        deploy_info["Vault"],
        test_account_with_balance=usdt_holder,
        asset_manager=hot_wallet.address,
        trading_strategy_module_address=deploy_info["Trading strategy module"],
    )

    # 3. Perform a test trade
    cli.main(args=["perform-test-trade", "--pair", "(binance, pancakeswap-v2, WBNB, USDT, 0.0025)"], standalone_mode=False)

    state = State.read_json_file(state_file)
    import ipdb ; ipdb.set_trace()


