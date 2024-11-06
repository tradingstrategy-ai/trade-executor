import pytest
import secrets
import datetime
from decimal import Decimal

from web3 import EthereumTesterProvider, Web3
from web3.contract import Contract
from eth_typing import HexAddress
from hexbytes import HexBytes

from eth_defi.token import create_token
from eth_defi.uniswap_v3.deployment import (
    UniswapV3Deployment,
    deploy_uniswap_v3,
    deploy_pool,
    add_liquidity
)
from eth_defi.uniswap_v3.utils import get_default_tick_range
from eth_defi.uniswap_v3.pool import fetch_pool_details

from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_routing import UniswapV3Routing
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_live_pricing import UniswapV3LivePricing
from tradeexecutor.ethereum.universe import create_exchange_universe, create_pair_universe
from tradeexecutor.strategy.trading_strategy_universe import translate_trading_pair
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier

from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.exchange import ExchangeUniverse


#: How much values we allow to drift.
#: A hack fix receiving different decimal values on Github CI than on a local
APPROX_REL = 0.0001
APPROX_REL_DECIMAL = Decimal(APPROX_REL)

WETH_USDC_FEE = 0.003
AAVE_USDC_FEE = 0.003
AAVE_WETH_FEE = 0.003

WETH_USDC_FEE_RAW = 3000
AAVE_USDC_FEE_RAW = 3000
AAVE_WETH_FEE_RAW = 3000

@pytest.fixture
def chain_id(web3) -> int:
    """The test chain id (67)."""
    return web3.eth.chain_id


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


@pytest.fixture
def aave_token(web3, deployer: HexAddress) -> Contract:
    """Create AAVE with 10M supply."""
    token = create_token(web3, deployer, "Fake Aave coin", "AAVE", 10_000_000 * 10**18, 18)
    return token


@pytest.fixture()
def uniswap_v3(web3, deployer) -> UniswapV3Deployment:
    """Uniswap v2 deployment."""
    deployment = deploy_uniswap_v3(web3, deployer)
    return deployment


@pytest.fixture
def weth_token(uniswap_v3: UniswapV3Deployment) -> Contract:
    """Mock some assets"""
    return uniswap_v3.weth


@pytest.fixture
def asset_usdc(usdc_token, chain_id) -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(chain_id, usdc_token.address, usdc_token.functions.symbol().call(), usdc_token.functions.decimals().call())


@pytest.fixture
def asset_weth(weth_token, chain_id) -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(chain_id, weth_token.address, weth_token.functions.symbol().call(), weth_token.functions.decimals().call())


@pytest.fixture
def asset_aave(aave_token, chain_id) -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(chain_id, aave_token.address, aave_token.functions.symbol().call(), aave_token.functions.decimals().call())


@pytest.fixture
def aave_usdc_uniswap_trading_pair(web3, deployer, uniswap_v3, aave_token, usdc_token) -> HexAddress:
    """AAVE-USDC pool with 200k liquidity. Fee of 0.1%"""
    min_tick, max_tick = get_default_tick_range(AAVE_USDC_FEE_RAW)
    
    pool_contract = deploy_pool(
        web3,
        deployer,
        deployment=uniswap_v3,
        token0=aave_token,
        token1=usdc_token,
        fee=AAVE_USDC_FEE_RAW
    )
    
    add_liquidity(
        web3,
        deployer,
        deployment=uniswap_v3,
        pool=pool_contract,
        amount0=1000 * 10**18,  # 1000 AAVE liquidity
        amount1=200_000 * 10**6,  # 200k USDC liquidity
        lower_tick=min_tick,
        upper_tick=max_tick
    )
    return pool_contract.address


@pytest.fixture
def weth_usdc_uniswap_trading_pair(web3, deployer, uniswap_v3, weth_token, usdc_token) -> HexAddress:
    """ETH-USDC pool with 1.7M liquidity."""
    min_tick, max_tick = get_default_tick_range(WETH_USDC_FEE_RAW)
    
    pool_contract = deploy_pool(
        web3,
        deployer,
        deployment=uniswap_v3,
        token0=weth_token,
        token1=usdc_token,
        fee=WETH_USDC_FEE_RAW
    )
    
    add_liquidity(
        web3,
        deployer,
        deployment=uniswap_v3,
        pool=pool_contract,
        amount0=1000 * 10**18,  # 1000 ETH liquidity
        amount1=1_700_000 * 10**6,  # 1.7M USDC liquidity
        lower_tick=min_tick,
        upper_tick=max_tick
    )
    return pool_contract.address


@pytest.fixture
def aave_weth_uniswap_trading_pair(web3, deployer, uniswap_v3, aave_token, weth_token) -> HexAddress:
    """AAVE-ETH pool.

    Price is 1:5 AAVE:ETH
    """
    """ETH-USDC pool with 1.7M liquidity."""
    min_tick, max_tick = get_default_tick_range(AAVE_WETH_FEE_RAW)
    
    pool_contract = deploy_pool(
        web3,
        deployer,
        deployment=uniswap_v3,
        token0=weth_token,
        token1=aave_token,
        fee=AAVE_WETH_FEE_RAW
    )
    
    add_liquidity(
        web3,
        deployer,
        deployment=uniswap_v3,
        pool=pool_contract,
        amount0=1000 * 10**18,  # 1000 ETH liquidity
        amount1=5000 * 10**18,  # 5000 AAVE liquidity
        lower_tick=min_tick,
        upper_tick=max_tick
    )
    return pool_contract.address
    

@pytest.fixture
def weth_usdc_pair(uniswap_v3, weth_usdc_uniswap_trading_pair, asset_usdc, asset_weth) -> TradingPairIdentifier:
    return TradingPairIdentifier(
        asset_weth, 
        asset_usdc, 
        weth_usdc_uniswap_trading_pair, 
        uniswap_v3.factory.address,
        fee = WETH_USDC_FEE
    )


@pytest.fixture
def aave_usdc_pair(uniswap_v3, aave_usdc_uniswap_trading_pair, asset_usdc, asset_aave) -> TradingPairIdentifier:
    return TradingPairIdentifier(
        asset_aave, 
        asset_usdc, 
        aave_usdc_uniswap_trading_pair, 
        uniswap_v3.factory.address,
        fee = AAVE_USDC_FEE
    )
    

@pytest.fixture
def aave_weth_pair(uniswap_v3, aave_weth_uniswap_trading_pair, asset_aave, asset_weth) -> TradingPairIdentifier:
    return TradingPairIdentifier(
        asset_aave,
        asset_weth,
        aave_weth_uniswap_trading_pair,
        uniswap_v3.factory.address,
        fee=AAVE_WETH_FEE,
    )


@pytest.fixture()
def exchange_universe(web3, uniswap_v3: UniswapV3Deployment) -> ExchangeUniverse:
    """We trade on one Uniswap v2 deployment on tester."""
    return create_exchange_universe(web3, [uniswap_v3])


@pytest.fixture()
def pair_universe(web3, exchange_universe: ExchangeUniverse, weth_usdc_pair, aave_weth_pair) -> PandasPairUniverse:
    exchange = next(iter(exchange_universe.exchanges.values()))  # Get the first exchange from the universe
    return create_pair_universe(web3, exchange, [weth_usdc_pair, aave_weth_pair])


@pytest.fixture()
def routing_model(uniswap_v3: UniswapV3Deployment, asset_usdc, asset_weth, weth_usdc_pair) -> UniswapV3Routing:

    # Allowed exchanges as factory -> router pairs
    address_map = {
        "factory": uniswap_v3.factory.address,
        "router": uniswap_v3.swap_router.address,
        "position_manager": uniswap_v3.position_manager.address,
        "quoter": uniswap_v3.quoter.address
        # "router02":"0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45",
        # "quoterV2":"0x61fFE014bA17989E743c5F6cB21bF9697530B21e"
    } # TODO create address_map class

    # Three way ETH quoted trades are routed thru WETH/USDC pool
    allowed_intermediary_pairs = {
        asset_weth.address: weth_usdc_pair.pool_address
    }

    return UniswapV3Routing(
        address_map,
        allowed_intermediary_pairs,
        reserve_token_address=asset_usdc.address,
    )



def test_uniswap_v3_two_leg_buy_price_no_price_impact(
        web3: Web3,
        exchange_universe,
        pair_universe: PandasPairUniverse,
        routing_model: UniswapV3Routing,
):
    """Two-leg buy trade."""
    pricing_method = UniswapV3LivePricing(web3, pair_universe, routing_model)

    exchange = next(iter(exchange_universe.exchanges.values()))  # Get the first exchange from the universe
    weth_usdc = pair_universe.get_one_pair_from_pandas_universe(exchange.exchange_id, "WETH", "USDC")
    pair = translate_trading_pair(weth_usdc)

    # Get price for "infinite" small trade amount
    price_structure = pricing_method.get_buy_price(datetime.datetime.utcnow(), pair, None)
    assert price_structure.price == pytest.approx(1705.1154460381174, rel=APPROX_REL)
    assert price_structure.lp_fee == [0.0003000000000000003] # TODO address floating point errors
    assert price_structure.get_total_lp_fees() == pytest.approx(0.0003000000000000003, rel=APPROX_REL)

    mid_price = pricing_method.get_mid_price(datetime.datetime.utcnow(), pair)
    assert price_structure.price > mid_price
    assert mid_price == pytest.approx(1699.9232380190588, rel=APPROX_REL)


def test_uniswap_v3_two_leg_buy_price_with_price_impact(
        web3: Web3,
        exchange_universe,
        pair_universe: PandasPairUniverse,
        routing_model: UniswapV3Routing,
):
    """Two-leg buy trade w/signficant price impact."""
    pricing_method = UniswapV3LivePricing(web3, pair_universe, routing_model)

    exchange = next(iter(exchange_universe.exchanges.values()))  # Get the first exchange from the universe
    weth_usdc = pair_universe.get_one_pair_from_pandas_universe(exchange.exchange_id, "WETH", "USDC")
    pair = translate_trading_pair(weth_usdc)

    price_structure = pricing_method.get_buy_price(datetime.datetime.utcnow(), pair, Decimal(50_000))
    assert price_structure.price == pytest.approx(1755.115346038114, rel=APPROX_REL)

    mid_price = pricing_method.get_mid_price(datetime.datetime.utcnow(), pair)
    assert price_structure.price > mid_price
    assert mid_price == pytest.approx(1699.9232380190588, rel=APPROX_REL)
    
    assert price_structure.get_total_lp_fees() == pytest.approx(150.0, rel=APPROX_REL)


def test_uniswap_v3_two_leg_sell_price_no_price_impact(
        web3: Web3,
        exchange_universe,
        pair_universe: PandasPairUniverse,
        routing_model: UniswapV3Routing,
):
    pricing_method = UniswapV3LivePricing(web3, pair_universe, routing_model)

    exchange = next(iter(exchange_universe.exchanges.values()))  # Get the first exchange from the universe
    weth_usdc = pair_universe.get_one_pair_from_pandas_universe(exchange.exchange_id, "WETH", "USDC")
    pair = translate_trading_pair(weth_usdc)

    # Get price for "infinite" small trade amount
    price_structure = pricing_method.get_sell_price(datetime.datetime.utcnow(), pair, None)
    assert price_structure.price == pytest.approx(1694.73103, rel=APPROX_REL)

    mid_price = pricing_method.get_mid_price(datetime.datetime.utcnow(), pair)
    assert price_structure.price < mid_price
    assert mid_price == pytest.approx(1699.9232380190588, rel=APPROX_REL)
    
    assert price_structure.get_total_lp_fees() == pytest.approx(0.0003000000000000003, rel=APPROX_REL)


def test_uniswap_v3_two_leg_sell_price_with_price_impact(
        web3: Web3,
        exchange_universe,
        pair_universe: PandasPairUniverse,
        routing_model: UniswapV3Routing,
):
    """Two-leg buy trade w/signficant price impact."""
    pricing_method = UniswapV3LivePricing(web3, pair_universe, routing_model)

    exchange = next(iter(exchange_universe.exchanges.values()))  # Get the first exchange from the universe
    weth_usdc = pair_universe.get_one_pair_from_pandas_universe(exchange.exchange_id, "WETH", "USDC")
    pair = translate_trading_pair(weth_usdc)

    # Sell 50 ETH
    price_structure = pricing_method.get_sell_price(datetime.datetime.utcnow(), pair, Decimal(50))
    assert price_structure.price == pytest.approx(1614.42110776, rel=APPROX_REL)
    
    assert price_structure.get_total_lp_fees() == pytest.approx(0.15000000000000013, rel=APPROX_REL)

    mid_price = pricing_method.get_mid_price(datetime.datetime.utcnow(), pair)
    assert price_structure.price < mid_price
    assert mid_price == pytest.approx(1699.9232380190588, rel=APPROX_REL)


def test_uniswap_v3_three_leg_buy_price_with_price_impact(
    web3: Web3,
    uniswap_v3,
    exchange_universe,
    pair_universe: PandasPairUniverse,
    routing_model: UniswapV3Routing,
):
    """Three leg trade w/signficant price impact.

    ETH price is 1700 USD.
    AAVE price should be 1/5 = 340 USD.
    """
    pricing_method = UniswapV3LivePricing(web3, pair_universe, routing_model)

    exchange = next(iter(exchange_universe.exchanges.values()))  # Get the first exchange from the universe

    #
    # Do some setup checks before attempting to buy
    #

    aave_weth = pair_universe.get_one_pair_from_pandas_universe(exchange.exchange_id, "AAVE", "WETH")
    pair = translate_trading_pair(aave_weth)
    assert pair, "Pair missing?"

    weth_usdc = pair_universe.get_one_pair_from_pandas_universe(exchange.exchange_id, "WETH", "USDC")
    pair2 = translate_trading_pair(weth_usdc)
    assert pair2, "Pair missing?"

    aave = pair.base.address
    weth = pair.quote.address
    usdc = pair2.quote.address

    #print("AAVE is", aave)
    #print("WETH is", weth)
    #print("USDC is", usdc)
    #print("AAVE-WETH pool at", pair.pool_address)
    #print("WETH-USDC pool at", pair2.pool_address)

    pool = fetch_pool_details(web3, aave_weth.address)
    assert (
        (pool.token0.address.lower() == aave and pool.token1.address.lower() == weth)
        or
        (pool.token0.address.lower() == weth and pool.token1.address.lower() == aave)
    )

    pool = fetch_pool_details(web3, weth_usdc.address)
    assert (
        (pool.token0.address.lower() == weth and pool.token1.address.lower() == usdc)
        or
        (pool.token0.address.lower() == usdc and pool.token1.address.lower() == weth)
    )

    # Get price for 20_000 USDC
    price_structure = pricing_method.get_buy_price(datetime.datetime.utcnow(), pair, Decimal(20_000))
    assert price_structure.price == pytest.approx(350.0612529665224, rel=APPROX_REL)
    
    # makes sense 20_000 * 0.003 * 2 = 120
    assert price_structure.get_total_lp_fees() == pytest.approx(119.81999999999937, rel=APPROX_REL)

    mid_price = pricing_method.get_mid_price(datetime.datetime.utcnow(), pair)
    assert price_structure.price > mid_price
    assert mid_price == pytest.approx(339.9994284591918, rel=APPROX_REL)


def test_uniswap_v3_three_leg_sell_price_with_price_impact(
    web3: Web3,
    uniswap_v3,
    exchange_universe,
    pair_universe: PandasPairUniverse,
    routing_model: UniswapV3Routing,
):
    """Three leg sell w/signficant price impact.

    ETH price is 1700 USD.
    AAVE price should be 1/5 = 340 USD.
    """
    pricing_method = UniswapV3LivePricing(web3, pair_universe, routing_model)

    exchange = next(iter(exchange_universe.exchanges.values()))  # Get the first exchange from the universe

    aave_weth = pair_universe.get_one_pair_from_pandas_universe(exchange.exchange_id, "AAVE", "WETH")
    pair = translate_trading_pair(aave_weth)

    # Get price for 500 AAVE
    price_structure = pricing_method.get_buy_price(datetime.datetime.utcnow(), pair, Decimal(500))
    assert price_structure.price == pytest.approx(342.2495177609055, rel=APPROX_REL)
    
    # makes sense 500 * 0.003 * 2 = 3
    assert price_structure.get_total_lp_fees() == pytest.approx(2.9954999999999843, rel=APPROX_REL)

    mid_price = pricing_method.get_mid_price(datetime.datetime.utcnow(), pair)
    assert price_structure.price > mid_price
    assert mid_price == pytest.approx(339.9994284591918, rel=APPROX_REL)


def test_uniswap_v3_usd_tvl(
    web3: Web3,
    uniswap_v3,
    exchange_universe,
    pair_universe: PandasPairUniverse,
    routing_model: UniswapV3Routing,
):
    """Get USD TVL of a pool.
    """

    pricing_model = UniswapV3LivePricing(web3, pair_universe, routing_model)

    exchange = next(iter(exchange_universe.exchanges.values()))  # Get the first exchange from the universe
    aave_weth = translate_trading_pair(
        pair_universe.get_one_pair_from_pandas_universe(exchange.exchange_id, "AAVE", "WETH")
    )

    token_tvl = pricing_model.get_usd_tvl(None, aave_weth)
    assert token_tvl > 0

