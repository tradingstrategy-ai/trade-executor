"""Test Enzyme using end-to-end integration and Uniswap v3."""
import os
import secrets
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from eth_account import Account
from eth_typing import HexAddress
from hexbytes import HexBytes

from typer.main import get_command
from web3 import Web3, HTTPProvider
from web3.contract import Contract

from eth_defi.abi import get_deployed_contract
from eth_defi.provider.anvil import AnvilLaunch, launch_anvil
from eth_defi.chain import install_chain_middleware
from eth_defi.deploy import deploy_contract
from eth_defi.enzyme.deployment import EnzymeDeployment, POLYGON_DEPLOYMENT
from eth_defi.enzyme.vault import Vault
from eth_defi.hotwallet import HotWallet
from eth_defi.token import TokenDetails, fetch_erc20_details
from eth_defi.trace import assert_transaction_success_with_explanation

from tradeexecutor.cli.main import app
from tradeexecutor.ethereum.token import translate_token_details
from tradeexecutor.monkeypatch.web3 import construct_sign_and_send_raw_middleware
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State

pytestmark = pytest.mark.skipif(not os.environ.get("JSON_RPC_POLYGON") or not os.environ.get("TRADING_STRATEGY_API_KEY"), reason="Set POLYGON_JSON_RPC and TRADING_STRATEGY_API_KEY environment variables to run this test")


logger = logging.getLogger(__name__)


@pytest.fixture()
def usdc_whale() -> HexAddress:
    """A random account picked from Polygon that holds a lot of USDC."""
    # https://polygonscan.com/token/0x2791bca1f2de4661ed88a30c99a7a9449aa84174#balances
    return HexAddress("0x72A53cDBBcc1b9efa39c834A540550e23463AAcB")


@pytest.fixture()
def anvil(usdc_whale) -> AnvilLaunch:
    """Launch Polygon fork."""
    rpc_url = os.environ["JSON_RPC_POLYGON"]

    anvil = launch_anvil(
        fork_url=rpc_url,
        unlocked_addresses=[usdc_whale],
    )
    try:
        yield anvil
    finally:
        anvil.close()


@pytest.fixture()
def web3(anvil) -> Web3:
    """Set up the Anvil Web3 connection.

    Also perform the Anvil state reset for each test.
    """
    web3 = Web3(HTTPProvider(anvil.json_rpc_url, request_kwargs={"timeout": 10}))
    # Get rid of attributeddict slow down
    web3.middleware_onion.clear()
    install_chain_middleware(web3)
    assert web3.eth.block_number > 1
    return web3


@pytest.fixture()
def start_block(web3) -> int:
    """Vault deployment start block hint.
    """
    return web3.eth.block_number


@pytest.fixture()
def deployer(web3) -> HexAddress:
    """Deployer account.

    - This account will deploy all smart contracts

    - Starts with 10,000 ETH
    """
    return web3.eth.accounts[0]


@pytest.fixture()
def user_1(web3) -> HexAddress:
    """User account.

    Do some account allocation for tests.
    """
    return web3.eth.accounts[1]


@pytest.fixture
def wmatic(web3) -> TokenDetails:
    details = fetch_erc20_details(web3, "0x0d500b1d8e8ef31e21c99d1db9a6444d3adf1270")
    return details


@pytest.fixture()
def wmatic_asset(wmatic: TokenDetails) -> AssetIdentifier:
    """WETH as a persistent id.
    """
    return translate_token_details(wmatic)


@pytest.fixture
def usdc(web3) -> TokenDetails:
    #details = fetch_erc20_details(web3, "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48")
    details = fetch_erc20_details(web3, "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
    return details


@pytest.fixture()
def usdc_asset(usdc: TokenDetails) -> AssetIdentifier:
    """USDC as a persistent id.
    """
    return translate_token_details(usdc)


@pytest.fixture()
def enzyme_deployment(
        web3,
) -> EnzymeDeployment:
    """Enzyme live deployment on Polygon."""
    deployment = EnzymeDeployment.fetch_deployment(web3, POLYGON_DEPLOYMENT)
    return deployment


@pytest.fixture()
def enzyme_vault_contract(
        web3,
        deployer: HexAddress,
        usdc: TokenDetails,
        user_1: HexAddress,
        enzyme_deployment: EnzymeDeployment,
) -> Contract:
    """Deploy a vault we use in our tests.

    - USDC nominatead

    - user_1 is the owner
    """
    comptroller_contract, vault_contract = enzyme_deployment.create_new_vault(
        user_1,
        usdc.contract,
        deployer=deployer,
    )
    return vault_contract


@pytest.fixture()
def vault_comptroller_contract(
        enzyme_vault_contract,
) -> Contract:
    """Get the comptroller for our test vault.

    - Needed to process deposits

    """
    web3 = enzyme_vault_contract.w3
    comptroller_address = enzyme_vault_contract.functions.getAccessor().call()
    comptroller = get_deployed_contract(web3, "enzyme/ComptrollerLib.json", comptroller_address)
    return comptroller


@pytest.fixture()
def generic_adapter(
        web3,
        deployer,
        enzyme_deployment,
        enzyme_vault_contract,
) -> Contract:
    """Deploy generic adapter that allows the vault to perform our trades."""
    generic_adapter = deploy_contract(
        web3,
        f"VaultSpecificGenericAdapter.json",
        deployer,
        enzyme_deployment.contracts.integration_manager.address,
        enzyme_vault_contract.address,
    )
    return generic_adapter


@pytest.fixture()
def vault(
        start_block: int,
        enzyme_deployment,
        enzyme_vault_contract,
        vault_comptroller_contract,
        generic_adapter,
) -> Vault:
    """Return the test vault.

    - USDC nominatead

    - user_1 is the owner
    """
    return Vault(enzyme_vault_contract, vault_comptroller_contract, enzyme_deployment, generic_adapter)


@pytest.fixture
def hot_wallet(
        web3,
        deployer,
        user_1,
        usdc: TokenDetails,
        usdc_whale,
        vault: Vault) -> HotWallet:
    """Create hot wallet for the signing tests.

    Top is up with some gas money and 500 USDC.
    """
    private_key = HexBytes(secrets.token_bytes(32))
    account = Account.from_key(private_key)
    wallet = HotWallet(account)
    wallet.sync_nonce(web3)
    tx_hash = web3.eth.send_transaction({"to": wallet.address, "from": user_1, "value": 15 * 10**18})
    assert_transaction_success_with_explanation(web3, tx_hash)
    tx_hash = usdc.contract.functions.transfer(wallet.address, 500 * 10**6).transact({"from": usdc_whale})
    assert_transaction_success_with_explanation(web3, tx_hash)

    # Promote the hot wallet to the asset manager
    tx_hash = vault.vault.functions.addAssetManagers([account.address]).transact({"from": user_1})
    assert_transaction_success_with_explanation(web3, tx_hash)

    # Add to the local signer chain
    web3.middleware_onion.add(construct_sign_and_send_raw_middleware(account))

    return wallet


@pytest.fixture()
def strategy_file() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.dirname(__file__)) / ".." / ".." / "strategies" / "test_only" / "enzyme_polygon_fork_uniswap_v3.py"


@pytest.fixture()
def state_file() -> Path:
    """The state file for the tests.

    Always start with an empty file.
    """
    path = Path("/tmp/test_enzyme_end_to_end_uniswap_v3.json")
    if path.exists():
        os.remove(path)
    return path


@pytest.fixture()
def environment(
    anvil: AnvilLaunch,
    vault: Vault,
    hot_wallet: HotWallet,
    state_file: Path,
    strategy_file: Path,
    start_block: int,
    ) -> dict:
    """Passed to init and start commands as environment variables"""
    # Set up the configuration for the live trader
    environment = {
        "EXECUTOR_ID": "test_enzyme_mainnet_fork_uniswap_v3",
        "NAME": "test_enzyme_live_trading_init",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "PRIVATE_KEY": hot_wallet.account.key.hex(),
        "JSON_RPC_ANVIL": anvil.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "enzyme",
        "UNIT_TESTING": "true",
        # "LOG_LEVEL": "info",  # Set to info to get debug data for the test run
        "LOG_LEVEL": "disabled",
        "VAULT_ADDRESS": vault.address,
        "VAULT_ADAPTER_ADDRESS": vault.generic_adapter.address,
        "CONFIRMATION_BLOCK_COUNT": "0",  # Needed for test backend, Anvil
        "MAX_CYCLES": "5",  # Run decide_trades() 5 times
        "VAULT_DEPLOYMENT_BLOCK_NUMBER": str(start_block),
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
    }
    return environment


def test_enzyme_uniswap_v3_test_trade(
    environment: dict,
    web3: Web3,
    start_block: int,
    state_file: Path,
    usdc: TokenDetails,
    vault: Vault,
    hot_wallet: HotWallet,
    enzyme_deployment: EnzymeDeployment):
    """Perform a test trade on Enzyme vault via CLI.

    - Trades on Polygon mainnet fork

    - Use a vault deployed by the test fixtures

    - Initialise the strategy to use this vault

    - Perform a test trade using Uniswap v3
    """

    cli = get_command(app)

    # Deposit some USDC to the vault to start
    deposit_amount = 500 * 10**6
    tx_hash = usdc.contract.functions.approve(vault.comptroller.address, deposit_amount).transact({"from": hot_wallet.address})
    assert_transaction_success_with_explanation(web3, tx_hash)
    tx_hash = vault.comptroller.functions.buyShares(deposit_amount, 1).transact({"from": hot_wallet.address})
    assert_transaction_success_with_explanation(web3, tx_hash)
    assert usdc.contract.functions.balanceOf(vault.address).call() == deposit_amount
    logger.info("Deposited %d %s at block %d", deposit_amount, usdc.address, web3.eth.block_number)

    # Check we have a deposit event
    logs = vault.comptroller.events.SharesBought.get_logs(fromBlock=start_block, toBlock=web3.eth.block_number)
    logger.info("Got logs %s", logs)
    assert len(logs) == 1

    env = environment

    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["init"])
        assert e.value.code == 0

    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["perform-test-trade"])
        assert e.value.code == 0

    assert usdc.contract.functions.balanceOf(vault.address).call() < deposit_amount, "No deposits where spent; trades likely did not happen"

    # Check the resulting state and see we made some trade for trading fee losses
    with state_file.open("rt") as inp:
        state: State = State.from_json(inp.read())

        assert len(list(state.portfolio.get_all_trades())) == 2

        reserve_value = state.portfolio.get_default_reserve_position().get_value()
        assert reserve_value < 500