"""Test using on-chain pricing method."""
import datetime
import secrets

import pytest
from eth_typing import HexAddress
from hexbytes import HexBytes
from web3 import EthereumTesterProvider, Web3
from web3.contract import Contract

from eth_hentai.token import create_token
from eth_hentai.uniswap_v2 import UniswapV2Deployment, deploy_trading_pair, deploy_uniswap_v2_like

from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.pair import PairUniverse, PandasPairUniverse


from tradeexecutor.ethereum.uniswap_v2_live_pricing import UniswapV2LivePricing
from tradeexecutor.ethereum.universe import create_exchange_universe, create_pair_universe
from tradeexecutor.state.state import State, AssetIdentifier, TradingPairIdentifier, Portfolio
from tradeexecutor.cli.log import setup_pytest_logging


@pytest.fixture
def tester_provider():
    # https://web3py.readthedocs.io/en/stable/examples.html#contract-unit-tests-in-python
    return EthereumTesterProvider()


@pytest.fixture
def eth_tester(tester_provider):
    # https://web3py.readthedocs.io/en/stable/examples.html#contract-unit-tests-in-python
    return tester_provider.ethereum_tester


@pytest.fixture
def web3(tester_provider):
    """Set up a local unit testing blockchain."""
    # https://web3py.readthedocs.io/en/stable/examples.html#contract-unit-tests-in-python
    return Web3(tester_provider)


@pytest.fixture
def chain_id(web3) -> int:
    """The test chain id (67)."""
    return web3.eth.chain_id


@pytest.fixture()
def deployer(web3) -> HexAddress:
    """Deploy account.

    Do some account allocation for tests.
    """
    return web3.eth.accounts[0]


@pytest.fixture()
def hot_wallet_private_key(web3) -> HexBytes:
    """Generate a private key"""
    return HexBytes(secrets.token_bytes(32))


@pytest.fixture
def usdc_token(web3, deployer: HexAddress) -> Contract:
    """Create USDC with 10M supply."""
    token = create_token(web3, deployer, "Fake USDC coin", "USDC", 10_000_000 * 10**6, 6)
    return token


@pytest.fixture()
def uniswap_v2(web3, deployer) -> UniswapV2Deployment:
    """Uniswap v2 deployment."""
    deployment = deploy_uniswap_v2_like(web3, deployer)
    return deployment


@pytest.fixture
def weth_token(uniswap_v2: UniswapV2Deployment) -> Contract:
    """Mock some assets"""
    return uniswap_v2.weth


@pytest.fixture
def asset_usdc(usdc_token, chain_id) -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(chain_id, usdc_token.address, usdc_token.functions.symbol().call(), usdc_token.functions.decimals().call())


@pytest.fixture
def asset_weth(weth_token, chain_id) -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(chain_id, weth_token.address, weth_token.functions.symbol().call(), weth_token.functions.decimals().call())


@pytest.fixture
def weth_usdc_uniswap_trading_pair(web3, deployer, uniswap_v2, weth_token, usdc_token) -> HexAddress:
    """AAVE-USDC pool with 1.7M liquidity."""
    pair_address = deploy_trading_pair(
        web3,
        deployer,
        uniswap_v2,
        weth_token,
        usdc_token,
        1000 * 10**18,  # 1000 ETH liquidity
        1_700_000 * 10**6,  # 1.7M USDC liquidity
    )
    return pair_address


@pytest.fixture()
def exchange_universe(web3, uniswap_v2: UniswapV2Deployment) -> ExchangeUniverse:
    """We trade on one Uniswap v2 deployment on tester."""
    return create_exchange_universe(web3, [uniswap_v2])


@pytest.fixture
def weth_usdc_pair(weth_usdc_uniswap_trading_pair, asset_usdc, asset_weth) -> TradingPairIdentifier:
    return TradingPairIdentifier(asset_weth, asset_usdc, weth_usdc_uniswap_trading_pair)


@pytest.fixture()
def pair_universe(web3, exchange_universe: ExchangeUniverse, weth_usdc_pair) -> PandasPairUniverse:
    exchange = next(iter(exchange_universe.exchanges.values()))  # Get the first exchange from the universe
    return create_pair_universe(web3, exchange, [weth_usdc_pair])


def test_uniswap_simple_buy_price(
        uniswap_v2: UniswapV2Deployment,
        exchange_universe: ExchangeUniverse,
        pair_universe: PandasPairUniverse,
):
    pricing_method = UniswapV2LivePricing(uniswap_v2, pair_universe)
    exchange = next(iter(exchange_universe.exchanges.values()))  # Get the first exchange from the universe
    weth_usdc = pair_universe.get_one_pair_from_pandas_universe(exchange.exchange_id, "WETH", "USDC")
    assert pricing_method.get_simple_buy_price(datetime.datetime.utcnow(), weth_usdc.pair_id) == pytest.approx(1705.12)


def test_uniswap_simple_sell_price(
        uniswap_v2: UniswapV2Deployment,
        exchange_universe: ExchangeUniverse,
        pair_universe: PandasPairUniverse,
):
    pricing_method = UniswapV2LivePricing(uniswap_v2, pair_universe)
    exchange = next(iter(exchange_universe.exchanges.values()))  # Get the first exchange from the universe
    weth_usdc = pair_universe.get_one_pair_from_pandas_universe(exchange.exchange_id, "WETH", "USDC")
    assert pricing_method.get_simple_sell_price(datetime.datetime.utcnow(), weth_usdc.pair_id) == pytest.approx(1694.89)
