"""Enzyme deployment fixtures.

- Common fixtures used in all Enzyme based tests

- We need to set up a lot of stuff to ramp up Enzyme

"""

import logging
import pytest
from eth_defi.enzyme.vault import Vault
from pytest import FixtureRequest

from eth_typing import HexAddress
from tradingstrategy.pair import PandasPairUniverse
from web3 import Web3, HTTPProvider
from web3.contract import Contract

from eth_defi.abi import get_deployed_contract
from eth_defi.anvil import AnvilLaunch, launch_anvil
from eth_defi.chain import install_chain_middleware
from eth_defi.deploy import deploy_contract
from eth_defi.enzyme.deployment import EnzymeDeployment, RateAsset

from eth_defi.token import create_token, fetch_erc20_details
from eth_defi.uniswap_v2.deployment import deploy_uniswap_v2_like, UniswapV2Deployment, deploy_trading_pair

from tradeexecutor.ethereum.token import translate_token_details
from tradeexecutor.ethereum.universe import create_pair_universe
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier


@pytest.fixture()
def anvil(request: FixtureRequest) -> AnvilLaunch:
    """Launch Anvil for the test backend.

    Run tests as `pytest --log-cli-level=info` to see Anvil console output created during the test,
    to debug any issues with Anvil itself.

    By default, Anvil is in `automining mode <https://book.getfoundry.sh/reference/anvil/>`__
    and creates a new block for each new transaction.

    .. note ::

        It could be possible to have a persitent Anvil instance over different tests with
        `fixture(scope="module")`. However we have spotted some hangs in Anvil
        (HTTP read timeout) and this is currently cured by letting Anvil reset itself.
    """

    # Peak into pytest logging level to help with Anvil output
    log_cli_level = request.config.getoption("--log-cli-level")
    log_level = None
    if log_cli_level:
        log_cli_level = logging.getLevelName(log_cli_level.upper())
        if log_cli_level <= logging.INFO:
            log_level = log_cli_level

    # London hardfork will enable EIP-1559 style gas fees
    anvil = launch_anvil(
        hardfork="london",
        gas_limit=15_000_000,  # Max 5M gas per block, or per transaction in test automining
        port=20001,
    )
    try:
        # Make the initial snapshot ("zero state") to which we revert between tests
        # web3 = Web3(HTTPProvider(anvil.json_rpc_url))
        # snapshot_id = make_anvil_custom_rpc_request(web3, "evm_snapshot")
        # assert snapshot_id == "0x0"
        yield anvil
    finally:
        anvil.close(log_level=log_level)


@pytest.fixture()
def web3(anvil: AnvilLaunch) -> Web3:
    """Set up the Anvil Web3 connection.

    Also perform the Anvil state reset for each test.
    """
    web3 = Web3(HTTPProvider(anvil.json_rpc_url, request_kwargs={"timeout": 2}))

    # Get rid of attributeddict slow down
    web3.middleware_onion.clear()

    install_chain_middleware(web3)

    return web3


@pytest.fixture()
def deployer(web3) -> HexAddress:
    """Deployer account.

    - This account will deploy all smart contracts

    - Starts with 10,000 ETH
    """
    return web3.eth.accounts[0]


@pytest.fixture()
def uniswap_v2(web3: Web3, deployer: HexAddress) -> UniswapV2Deployment:
    """Deploy Uniswap, WETH token."""
    assert web3.eth.get_balance(deployer) > 0
    deployment = deploy_uniswap_v2_like(web3, deployer, give_weth=500)  # Will also deploy WETH9 and give the deployer this many WETH tokens
    return deployment


@pytest.fixture()
def user_1(web3) -> HexAddress:
    """User account.

    Do some account allocation for tests.
    """
    return web3.eth.accounts[1]


@pytest.fixture()
def user_2(web3) -> HexAddress:
    """User account.

    Do some account allocation for tests.
    """
    return web3.eth.accounts[2]


@pytest.fixture()
def user_3(web3) -> HexAddress:
    """User account.

    Do some account allocation for tests.
    """
    return web3.eth.accounts[3]


@pytest.fixture
def weth(uniswap_v2) -> Contract:
    return uniswap_v2.weth


@pytest.fixture()
def weth_asset(weth: Contract) -> AssetIdentifier:
    """WETH as a persistent id.
    """
    details = fetch_erc20_details(weth.w3, weth.address)
    return translate_token_details(details)


@pytest.fixture()
def usdc(web3, deployer) -> Contract:
    """Mock USDC token.

    All initial $100M goes to `deployer`
    """
    token = create_token(web3, deployer, "USD Coin", "USDC", 100_000_000 * 10**6, decimals=6)
    return token


@pytest.fixture()
def usdc_asset(usdc: Contract) -> AssetIdentifier:
    """USDC as a persistent id.
    """
    details = fetch_erc20_details(usdc.w3, usdc.address)
    return translate_token_details(details)


@pytest.fixture()
def weth_usdc_uniswap_pair(web3, deployer, uniswap_v2, usdc, weth) -> HexAddress:
    """Create Uniswap v2 pool for WETH-USDC.

    - Add 200k initial liquidity at 1600 ETH/USDC
    """

    deposit = 200_000  # USDC
    price = 1600

    pair = deploy_trading_pair(
        web3,
        deployer,
        uniswap_v2,
        usdc,
        weth,
        deposit * 10**6,
        (deposit // price) * 10**18,
    )

    return pair


@pytest.fixture()
def mln(web3, deployer) -> Contract:
    """Mock MLN token."""
    token = create_token(web3, deployer, "Melon", "MLN", 5_000_000 * 10**18)
    return token


@pytest.fixture()
def weth_usd_mock_chainlink_aggregator(web3, deployer) -> Contract:
    """Fake ETH/USDC Chainlink price feed.

    Start with 1 ETH = 1600 USD.
    """
    aggregator = deploy_contract(
        web3,
        "MockChainlinkAggregator.json",
        deployer,
    )
    aggregator.functions.setValue(1600 * 10**18).transact({"from": deployer})
    return aggregator


@pytest.fixture()
def usdc_usd_mock_chainlink_aggregator(web3, deployer) -> Contract:
    """Fake ETH/USDC Chainlink price feed.

    Start with 1 USDC = 1 USD.
    """
    aggregator = deploy_contract(
        web3,
        "MockChainlinkAggregator.json",
        deployer,
    )
    aggregator.functions.setValue(1 * 10**6).transact({"from": deployer})
    return aggregator


@pytest.fixture()
def enzyme_deployment(
        web3,
        deployer,
        mln,
        weth,
        usdc,
        usdc_usd_mock_chainlink_aggregator,
        weth_usd_mock_chainlink_aggregator,
) -> EnzymeDeployment:
    """Enzyme deployment on a test VM fixture.

    - Comes with mocked USDC Chainlink price feed
    """

    deployment = EnzymeDeployment.deploy_core(
        web3,
        deployer,
        mln,
        weth,
    )

    # Create a vault for user 1
    # where we nominate everything in USDC
    deployment.add_primitive(
        usdc,
        usdc_usd_mock_chainlink_aggregator,
        RateAsset.USD,
    )

    deployment.contracts.value_interpreter.functions.setEthUsdAggregator(weth_usd_mock_chainlink_aggregator.address).transact({"from": deployer})

    return deployment


@pytest.fixture()
def enzyme_vault_contract(
        web3,
        deployer,
        usdc,
        user_1,
        enzyme_deployment,
) -> Contract:
    """Create an example vault.

    - USDC nominatead

    - user_1 is the owner
    """
    comptroller_contract, vault_contract = enzyme_deployment.create_new_vault(
        user_1,
        usdc,
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
def weth_usdc_trading_pair(uniswap_v2, weth_usdc_uniswap_pair, usdc_asset, weth_asset) -> TradingPairIdentifier:
    return TradingPairIdentifier(weth_asset, usdc_asset, weth_usdc_uniswap_pair, uniswap_v2.factory.address, fee=0.0030)


@pytest.fixture()
def pair_universe(web3, weth_usdc_trading_pair) -> PandasPairUniverse:
    return create_pair_universe(web3, None, [weth_usdc_trading_pair])
