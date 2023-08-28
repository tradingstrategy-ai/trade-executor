"""Enzyme deployment fixtures.

- Common fixtures used in all Enzyme based tests

- We need to set up a lot of stuff to ramp up Enzyme

"""

import logging
import pytest
import secrets
from eth_defi.enzyme.vault import Vault
from pytest import FixtureRequest

from eth_typing import HexAddress
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.exchange import Exchange, ExchangeType
from tradingstrategy.chain import ChainId
from web3 import Web3, HTTPProvider
from web3.contract import Contract
from hexbytes import HexBytes
from eth_account import Account

from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import AnvilLaunch, launch_anvil
from eth_defi.chain import install_chain_middleware
from eth_defi.deploy import deploy_contract
from eth_defi.trace import assert_transaction_success_with_explanation

from eth_defi.token import create_token, fetch_erc20_details
from eth_defi.uniswap_v2.deployment import deploy_uniswap_v2_like, UniswapV2Deployment, deploy_trading_pair

from tradeexecutor.ethereum.token import translate_token_details
from tradeexecutor.ethereum.universe import create_pair_universe
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier


logger = logging.getLogger(__name__)


@pytest.fixture()
def anvil() -> AnvilLaunch:
    anvil = launch_anvil()
    try:
        yield anvil
    finally:
        anvil.close()


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
    logger.info("Uni v2 factory deployed at %s", deployment.factory.address)
    return deployment


@pytest.fixture()
def user_1(web3) -> HexAddress:
    """User account.

    Do some account allocation for tests.
    """
    return web3.eth.accounts[1]


# Create tokens

# WETH
@pytest.fixture
def weth(uniswap_v2) -> Contract:
    return uniswap_v2.weth


@pytest.fixture()
def weth_asset(weth: Contract) -> AssetIdentifier:
    """WETH as a persistent id.
    """
    details = fetch_erc20_details(weth.w3, weth.address)
    return translate_token_details(details)

# USDC
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


# BOB
@pytest.fixture()
def bob(web3, deployer) -> Contract:
    """Mock BOB token. 
    
    All initial $50M goes to `deployer`
    """
    token = create_token(web3, deployer, "Bob Token", "BOB", 50_000_000 * 10**18, decimals=18)
    return token


@pytest.fixture()
def bob_asset(bob: Contract) -> AssetIdentifier:
    """BOB as a persistent id.
    """
    details = fetch_erc20_details(bob.w3, bob.address)
    return translate_token_details(details)


# PEPE
@pytest.fixture()
def pepe(web3, deployer) -> Contract:
    """Mock PEPE token. 
    
    All initial $25M goes to `deployer`
    """
    token = create_token(web3, deployer, "Pepe Token", "PEPE", 25_000_000 * 10**18, decimals=18)
    return token


@pytest.fixture()
def pepe_asset(pepe: Contract) -> AssetIdentifier:
    """PEPE as a persistent id.
    """
    details = fetch_erc20_details(pepe.w3, pepe.address)
    return translate_token_details(details)


# Hot wallet

@pytest.fixture
def hot_wallet(web3, deployer, user_1, usdc: Contract) -> HotWallet:
    """Create hot wallet for the signing tests.

    Top is up with some gas money and 500 USDC.
    """
    private_key = HexBytes(secrets.token_bytes(32))
    account = Account.from_key(private_key)
    wallet = HotWallet(account)
    wallet.sync_nonce(web3)
    tx_hash = web3.eth.send_transaction({"to": wallet.address, "from": user_1, "value": 15 * 10**18})
    assert_transaction_success_with_explanation(web3, tx_hash)
    tx_hash = usdc.functions.transfer(wallet.address, 500 * 10**6).transact({"from": deployer})
    assert_transaction_success_with_explanation(web3, tx_hash)
    return wallet


# Trading pairs


# WETH-USDC
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

    logger.info("%s-%s pair is at %s", weth.address, usdc.address, pair)

    return pair

# BOB-USDC
@pytest.fixture()
def bob_usdc_uniswap_pair(web3, deployer, uniswap_v2, usdc, bob) -> HexAddress:
    """Create Uniswap v2 pool for BOB-USDC.

    - Add 200k initial liquidity at 1000 BOB/USDC
    """

    deposit = 200_000  # USDC
    price = 1000

    pair = deploy_trading_pair(
        web3,
        deployer,
        uniswap_v2,
        usdc,
        bob,
        deposit * 10**6,
        (deposit // price) * 10**18,
    )

    logger.info("%s-%s pair is at %s", bob.address, usdc.address, pair)

    return pair


# PEPE-USDC
@pytest.fixture()
def pepe_usdc_uniswap_pair(web3, deployer, uniswap_v2, usdc, pepe) -> HexAddress:
    """Create Uniswap v2 pool for PEPE-USDC.

    - Add 200k initial liquidity at 300 PEPE/USDC
    """

    deposit = 200_000  # USDC
    price = 300

    pair = deploy_trading_pair(
        web3,
        deployer,
        uniswap_v2,
        usdc,
        pepe,
        deposit * 10**6,
        (deposit // price) * 10**18,
    )

    logger.info("%s-%s pair is at %s", pepe.address, usdc.address, pair)

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


# Trading pair identifiers

@pytest.fixture
def weth_usdc_trading_pair(uniswap_v2, weth_usdc_uniswap_pair, usdc_asset, weth_asset) -> TradingPairIdentifier:
    return TradingPairIdentifier(weth_asset, usdc_asset, weth_usdc_uniswap_pair, uniswap_v2.factory.address, fee=0.0030)


@pytest.fixture()
def pepe_usdc_trading_pair(uniswap_v2, pepe_usdc_uniswap_pair, usdc_asset, pepe_asset) -> TradingPairIdentifier:
    return TradingPairIdentifier(pepe_asset, usdc_asset, pepe_usdc_uniswap_pair, uniswap_v2.factory.address, fee=0.0030)


@pytest.fixture()
def bob_usdc_trading_pair(uniswap_v2, bob_usdc_uniswap_pair, usdc_asset, bob_asset) -> TradingPairIdentifier:
    return TradingPairIdentifier(bob_asset, usdc_asset, bob_usdc_uniswap_pair, uniswap_v2.factory.address, fee=0.0030)


# Exchange


@pytest.fixture()
def uniswap_v2_exchange(uniswap_v2: UniswapV2Deployment) -> Exchange:
    return Exchange(
        chain_id=ChainId.anvil,
        chain_slug="tester",
        exchange_id=int(uniswap_v2.factory.address, 16),
        exchange_slug="UniswapV2MockClient",
        address=uniswap_v2.factory.address,
        exchange_type=ExchangeType.uniswap_v2,
        pair_count=99999,
    )


# pair universes


@pytest.fixture()
def pair_universe(web3, weth_usdc_trading_pair) -> PandasPairUniverse:
    return create_pair_universe(web3, None, [weth_usdc_trading_pair])


@pytest.fixture()
def pairs(weth_usdc_trading_pair, pepe_usdc_trading_pair, bob_usdc_trading_pair) -> list[TradingPairIdentifier]:
    return [weth_usdc_trading_pair, pepe_usdc_trading_pair, bob_usdc_trading_pair]


@pytest.fixture()
def multipair_universe(web3, uniswap_v2_exchange, pairs) -> PandasPairUniverse:
    return create_pair_universe(web3, uniswap_v2_exchange, pairs)
