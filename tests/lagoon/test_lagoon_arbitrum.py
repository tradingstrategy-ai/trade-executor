"""Test CLI command and strategy running with Lagoon vault on Arbitrum."""
import json
import os
from pathlib import Path
from pprint import pformat

import pytest
import flaky
from typer.main import get_command
from web3 import Web3

from eth_defi.hotwallet import HotWallet
from eth_defi.erc_4626.vault_protocol.lagoon.testing import fund_lagoon_vault
from eth_defi.provider.anvil import fork_network_anvil, AnvilLaunch
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.token import USDT_NATIVE_TOKEN, fetch_erc20_details, TokenDetails, USDC_WHALE, USDC_NATIVE_TOKEN

from tradeexecutor.cli.commands.app import app
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.state.state import State

from eth_typing import HexAddress

from tradingstrategy.chain import ChainId

JSON_RPC_ARBITRUM = os.environ.get("JSON_RPC_ARBITRUM")
TRADING_STRATEGY_API_KEY = os.environ.get("TRADING_STRATEGY_API_KEY")
CI = os.environ.get("CI") == "true"

pytestmark = pytest.mark.skipif(
     (not JSON_RPC_ARBITRUM or not TRADING_STRATEGY_API_KEY),
      reason="Set JSON_RPC_ARBITRUM and TRADING_STRATEGY_API_KEY needed to run this test"
)


@pytest.fixture()
def usdc_holder() -> HexAddress:
    return USDC_WHALE[ChainId.arbitrum.value]


@pytest.fixture()
def usdc(web3) -> TokenDetails:
    return fetch_erc20_details(web3, USDC_NATIVE_TOKEN[web3.eth.chain_id])


@pytest.fixture()
def anvil_bnb_fork(request, usdc_holder) -> AnvilLaunch:
    """Create a testable fork of live BNB chain.

    :return: JSON-RPC URL for Web3
    """
    assert JSON_RPC_ARBITRUM, "JSON_RPC_BINANCE not set"
    launch = fork_network_anvil(
        JSON_RPC_ARBITRUM,
        unlocked_addresses=[usdc_holder],
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
    assert web3.eth.chain_id == 42161
    return web3


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


@pytest.fixture()
def strategy_file() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.dirname(__file__)) / ".." / ".." / "strategies" / "test_only" / "master-vault.py"


@pytest.fixture()
def state_file(tmp_path) -> Path:
    path = tmp_path / "master-vault.json"
    return path


@pytest.fixture()
def hot_wallet(web3) -> HotWallet:
    """A test account with USDT balance."""
    hw = HotWallet.create_for_testing(
        web3,
        test_account_n=1,
        eth_amount=10,
    )
    hw.sync_nonce(web3)
    return hw


# Anvil keeps crapping out in perform-test-trade, so we mark this test as flaky
@flaky.flaky
@pytest.mark.slow_test_group
def test_cli_lagoon_deploy_arbitrum_vault(
    web3,
    anvil_bnb_fork,
    strategy_file,
    mocker,
    state_file,
    hot_wallet,
    tmp_path: Path,
    persistent_test_client,
    usdc_holder: HexAddress,
    usdc: TokenDetails,
):
    """Deploy Lagoon vault on Binance, run test trades and start executor.

    - This will also launch /charts endpoint on the web API
    """

    cache_path  =persistent_test_client.transport.cache_path

    multisig_owners = f"{web3.eth.accounts[2]}, {web3.eth.accounts[3]}, {web3.eth.accounts[4]}"

    vault_record_file = tmp_path / "vault-record.json"
    environment = {
        "PATH": os.environ["PATH"],  # Need forge
        "EXECUTOR_ID": "test_cli_lagoon_deploy_binance_vault",
        "NAME": "test_cli_lagoon_deploy_binance_vault",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "JSON_RPC_ARBITRUM": anvil_bnb_fork.json_rpc_url,
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
        "DENOMINATION_ASSET": usdc.address,
        "ANY_ASSET": "true",
        "UNISWAP_V2": "false",
        "UNISWAP_V3": "true",
        "SUSHISWAP": "true",
        "CACHE_PATH": cache_path,
        "RUN_SINGLE_CYCLE": "true",
        # Hyperithm USDC on Morpho
        "ERC_4626_VAULTS": "0x4b6f1c9e5d470b97181786b26da0d0945a7cf027",
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
        test_account_with_balance=usdc_holder,
        asset_manager=hot_wallet.address,
        trading_strategy_module_address=deploy_info["Trading strategy module"],
    )

    # 3. Perform a test trade on one of the whitelisted vaults
    # Address is Hyperithm USDC on Morpho
    cli.main(args=["perform-test-trade", "--pair", "(arbitrum, 0x4b6f1c9e5d470b97181786b26da0d0945a7cf027)"], standalone_mode=False)

    # 4. Start executor and run 1s cycle
    cli.main(args=["start"], standalone_mode=False)

    # 4.b Check the 1s cycle has been run by inspecting the saved state
    # after the cycle
    state = State.read_json_file(state_file)
    reserve_position = state.portfolio.get_default_reserve_position()
    assert reserve_position.get_value() > 5.0  # Should have 100 USDC starting balance
    for t in state.portfolio.get_all_trades():
        assert t.is_success(), f"Trade {t} failed: {t.get_revert_reason()}"
    assert len(state.portfolio.frozen_positions) == 0

