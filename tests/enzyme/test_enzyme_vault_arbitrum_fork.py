"""Enzyme and Arbitrum integration test.

- Uses Arbitrum mainnet fork
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from eth_account import Account
from eth_typing import HexAddress
from typer.main import get_command
from web3 import Web3

from eth_defi.enzyme.deployment import EnzymeDeployment, ARBITRUM_DEPLOYMENT
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import AnvilLaunch, launch_anvil
from eth_defi.token import TokenDetails, fetch_erc20_details
from eth_defi.trace import assert_transaction_success_with_explanation
from tradeexecutor.cli.commands.app import app

JSON_RPC_ARBITRUM = os.environ.get("JSON_RPC_ARBITRUM")
TRADING_STRATEGY_API_KEY = os.environ.get("TRADING_STRATEGY_API_KEY")

pytestmark = pytest.mark.skipif(
    (not JSON_RPC_ARBITRUM or not TRADING_STRATEGY_API_KEY),
     reason="Set JSON_RPC_ARBITRUM and TRADING_STRATEGY_API_KEY needed to run this test"
)


@pytest.fixture()
def anvil(usdt_whale) -> AnvilLaunch:
    """Launch Polygon fork."""

    anvil = launch_anvil(
        fork_url=JSON_RPC_ARBITRUM,
        unlocked_addresses=[usdt_whale],
        fork_block_number=257_845_348,
    )
    try:
        yield anvil
    finally:
        anvil.close()


@pytest.fixture()
def usdt_whale() -> HexAddress:
    """A random account picked, holds a lot of stablecoin"""
    # https://arbiscan.io/token/0xfd086bc7cd5c481dcc9c85ebe478a1c0b69fcbb9#balances
    return HexAddress("0x8f9c79B9De8b0713dCAC3E535fc5A1A92DB6EA2D")


@pytest.fixture
def usdt(web3) -> TokenDetails:
    details = fetch_erc20_details(web3, "0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9")
    return details


@pytest.fixture
def weth(web3) -> TokenDetails:
    # https://arbiscan.io/token/0x82af49447d8a07e3bd95bd0d56f35241523fbab1
    details = fetch_erc20_details(web3, "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1")
    return details


@pytest.fixture
def user_1(web3) -> Account:
    return web3.eth.accounts[3]


@pytest.fixture
def hot_wallet(
    web3,
    deployer,
    user_1,
    usdt: TokenDetails,
    usdt_whale,
) -> HotWallet:
    """Create hot wallet with a private key as we need to pass this key to forge, others commands."""
    wallet = HotWallet.create_for_testing(web3)
    tx_hash = usdt.contract.functions.transfer(wallet.address, 500 * 10**6).transact({"from": usdt_whale})
    assert_transaction_success_with_explanation(web3, tx_hash)
    return wallet


@pytest.fixture()
def enzyme(
    web3,
) -> EnzymeDeployment:
    deployment = EnzymeDeployment.fetch_deployment(web3, ARBITRUM_DEPLOYMENT)
    return deployment


@pytest.fixture()
def strategy_file() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.dirname(__file__)) / ".." / ".." / "strategies" / "test_only" / "enzyme_arbitrum_fork_uniswap_v3.py"


@pytest.fixture()
def state_file(tmp_path) -> Path:
    path = tmp_path / "test_enzyme_vault_arbitrum_fork.json"
    return path


@pytest.fixture()
def environment(
    strategy_file,
    hot_wallet,
    anvil,
    state_file,
):
    environment = {
        "EXECUTOR_ID": "test_enzyme_vault_arbitrum_fork",
        "NAME": "test_enzyme_vault_arbitrum_fork",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "PRIVATE_KEY": hot_wallet.account.key.hex(),
        "JSON_RPC_ARBITRUM": anvil.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "enzyme",
        "UNIT_TESTING": "true",
        # "LOG_LEVEL": "info",  # Set to info to get debug data for the test run
        "LOG_LEVEL": "disabled",
        "RUN_SINGLE_CYCLE": "true",
        "TRADING_STRATEGY_API_KEY": TRADING_STRATEGY_API_KEY,
        "PATH": os.environ["PATH"], # Needs Forge bin
        "MAX_DATA_DELAY_MINUTES": str(10 * 60 * 24 * 365),  # 10 years or "disabled""
    }
    return environment


@pytest.mark.slow_test_group
def test_enzyme_vault_arbitrum(
    environment: dict,
    web3: Web3,
    state_file: Path,
    deployer: HexAddress,
    tmp_path,
    usdt_whale,
    usdt: TokenDetails,
    weth: TokenDetails,
    user_1,
    enzyme,
):
    """Deploy Enzyme vault via CLI.

    - Set up Arbitrum mainnet fork

    - Deploy a new vault using CLI

    - Perform a test trade on the vault

    - Perform a strategy boot on the vault
    """

    cli = get_command(app)

    vault_record_file = tmp_path / 'vault_record.json'
    env = environment.copy()
    env["FUND_NAME"] = "Toholampi Capital"
    env["FUND_SYMBOL"] = "COW"
    env["VAULT_RECORD_FILE"] = vault_record_file.as_posix()
    env["DENOMINATION_ASSET"] = usdt.address
    env["WHITELISTED_ASSETS"] = " ".join([usdt.address, weth.address])

    with patch.dict(os.environ, env, clear=True):
        cli.main(args=["enzyme-deploy-vault"], standalone_mode=False)

    # Check tat the vault was created
    with (open(vault_record_file, "rt") as inp):
        vault_record = json.load(inp)
        comptroller_address = vault_record["comptroller"]
        comptroller = enzyme.contracts.get_deployed_contract("ComptrollerLib", comptroller_address)

        # For the following commands, we need to pass the deployed vault info
        env["VAULT_ADDRESS"] = vault_record["vault"]
        env["VAULT_ADAPTER_ADDRESS"] = vault_record["generic_adapter"]
        env["VAULT_DEPLOYMENT_BLOCK_NUMBER"] = str(vault_record["block_number"])

    # Initialise the state with this vault
    with patch.dict(os.environ, env, clear=True):
        cli.main(args=["init"], standalone_mode=False)

    # Deposit
    tx_hash = usdt.contract.functions.transfer(user_1, 500 * 10 ** 6).transact({"from": usdt_whale})
    assert_transaction_success_with_explanation(web3, tx_hash)
    tx_hash = usdt.contract.functions.approve(comptroller.address, 500 * 10 ** 6).transact({"from": user_1})
    assert_transaction_success_with_explanation(web3, tx_hash)
    tx_hash = comptroller.functions.buyShares(500 * 10 ** 6, 1).transact({"from": user_1})
    assert_transaction_success_with_explanation(web3, tx_hash)

    # Perform a test trade
    with patch.dict(os.environ, env, clear=True):
        cli.main(args=["perform-test-trade"], standalone_mode=False)

    # Perform a single cycle of the strategy
    with patch.dict(os.environ, env, clear=True):
        cli.main(args=["start"], standalone_mode=False)
