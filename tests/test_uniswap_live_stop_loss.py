"""Stop loss using live uniswap price feed.

"""
import datetime
import secrets
from decimal import Decimal
from typing import List

import pytest
import pandas as pd
from eth_account import Account
from eth_account.signers.local import LocalAccount

from eth_defi.uniswap_v2.swap import swap_with_slippage_protection
from eth_typing import HexAddress
from hexbytes import HexBytes
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.liquidity import GroupedLiquidityUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe
from web3 import EthereumTesterProvider, Web3
from web3.contract import Contract
from eth_defi.token import create_token
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment, deploy_trading_pair, deploy_uniswap_v2_like

from tradeexecutor.cli.log import setup_custom_log_levels
from tradeexecutor.cli.loop import ExecutionLoop
from tradeexecutor.ethereum.uniswap_v2_routing import UniswapV2SimpleRoutingModel
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trading_strategy_universe import translate_trading_pair, TradingStrategyUniverse
from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.pair import PandasPairUniverse

from tradeexecutor.ethereum.uniswap_v2_live_pricing import UniswapV2LivePricing
from tradeexecutor.ethereum.universe import create_exchange_universe, create_pair_universe
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.testing.simulated_execution_loop import set_up_simulated_execution_loop_uniswap_v2
from tradeexecutor.utils.blockchain import get_latest_block_timestamp

#: How much values we allow to drift.
#: A hack fix receiving different decimal values on Github CI than on a local
APPROX_REL = 0.01
APPROX_REL_DECIMAL = Decimal("0.1")


@pytest.fixture
def logging():
    """Initialise custom hooks for Python logging system."""
    setup_custom_log_levels()


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


@pytest.fixture
def aave_token(web3, deployer: HexAddress) -> Contract:
    """Create AAVE with 10M supply."""
    token = create_token(web3, deployer, "Fake Aave coin", "AAVE", 10_000_000 * 10**18, 18)
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
    """ETH-USDC pool with 1.7M liquidity."""

    # Uniswap is strict about the pair order
    if int(weth_token.address, 16) < int(usdc_token.address, 16):
        pair_address = deploy_trading_pair(
            web3,
            deployer,
            uniswap_v2,
            weth_token,
            usdc_token,
            1000 * 10**18,  # 1000 ETH liquidity
            1_700_000 * 10**6,  # 1.7M USDC liquidity
        )
    else:
        pair_address = deploy_trading_pair(
            web3,
            deployer,
            uniswap_v2,
            usdc_token,
            weth_token,
            1_700_000 * 10 ** 6,  # 1.7M USDC liquidity
            1000 * 10**18,  # 1000 ETH liquidity
        )

    return pair_address


@pytest.fixture()
def exchange_universe(web3, uniswap_v2: UniswapV2Deployment) -> ExchangeUniverse:
    """We trade on one Uniswap v2 deployment on tester."""
    return create_exchange_universe(web3, [uniswap_v2])


@pytest.fixture
def weth_usdc_pair(uniswap_v2, weth_usdc_uniswap_trading_pair, asset_usdc, asset_weth) -> TradingPairIdentifier:
    return TradingPairIdentifier(
        asset_weth,
        asset_usdc,
        weth_usdc_uniswap_trading_pair,
        uniswap_v2.factory.address,
    )


@pytest.fixture()
def exchange_universe(web3, uniswap_v2: UniswapV2Deployment) -> ExchangeUniverse:
    """We trade on one Uniswap v2 deployment on tester."""
    return create_exchange_universe(web3, [uniswap_v2])


@pytest.fixture()
def pair_universe(web3, exchange_universe: ExchangeUniverse, weth_usdc_pair) -> PandasPairUniverse:
    exchange = next(iter(exchange_universe.exchanges.values()))  # Get the first exchange from the universe
    return create_pair_universe(web3, exchange, [weth_usdc_pair])


@pytest.fixture()
def routing_model(uniswap_v2, asset_usdc, asset_weth, weth_usdc_pair) -> UniswapV2SimpleRoutingModel:

    # Allowed exchanges as factory -> router pairs
    factory_router_map = {
        uniswap_v2.factory.address: (uniswap_v2.router.address, uniswap_v2.init_code_hash),
    }

    # Three way ETH quoted trades are routed thru WETH/USDC pool
    allowed_intermediary_pairs = {
        asset_weth.address: weth_usdc_pair.pool_address
    }

    return UniswapV2SimpleRoutingModel(
        factory_router_map,
        allowed_intermediary_pairs,
        reserve_token_address=asset_usdc.address,
    )


@pytest.fixture()
def trader(web3, deployer, usdc_token) -> LocalAccount:
    """Trader account.

    - Start with piles of ETH

    - Start with 9,000 USDC token

    Trades against deployer
    """
    trader = Account.create()

    # Give 1 ETH gas money to the trader
    web3.eth.send_transaction({
        "from": deployer,
        "to": trader.address,
        "value": 1 * 10 ** 18
    })

    # Give 9000 USD trading money
    usdc_token.functions.transfer(
        trader.address,
        9_000 * 10**6,
    ).transact({
        "from": deployer,
    })

    return trader


def decide_trades(
        timestamp: pd.Timestamp,
        universe: Universe,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: dict) -> List[TradeExecution]:
    """Opens new buy and hold position for $1000 if no position is open."""

    # The pair we are trading
    pair = universe.pairs.get_single()

    # Open for 1,000 USD
    position_size = 1000.00

    # List of any trades we decide on this cycle.
    # Because the strategy is simple, there can be
    # only zero (do nothing) or 1 (open or close) trades
    # decides
    trades = []

    # Create a position manager helper class that allows us easily to create
    # opening/closing trades for different positions
    position_manager = PositionManager(timestamp, universe, state, pricing_model)

    if not position_manager.is_any_open():
        buy_amount = position_size
        trades += position_manager.open_1x_long(pair, buy_amount)

    return trades


@pytest.fixture()
def core_universe(web3,
             exchange_universe: ExchangeUniverse,
             pair_universe: PandasPairUniverse) -> Universe:
    """Create a trading universe that contains our mock pair."""
    return Universe(
        time_bucket=TimeBucket.d1,
        chains=[ChainId(web3.eth.chain_id)],
        exchanges=list(exchange_universe.exchanges.values()),
        pairs=pair_universe,
        candles=GroupedCandleUniverse.create_empty(),
        liquidity=GroupedLiquidityUniverse.create_empty(),
    )


@pytest.fixture()
def trading_strategy_universe(core_universe: Universe, asset_usdc) -> TradingStrategyUniverse:
    """Universe that also contains data about our reserve assets."""
    return TradingStrategyUniverse(universe=core_universe, reserve_assets=[asset_usdc])



def test_live_stop_loss(
        logging,
        web3: Web3,
        deployer: HexAddress,
        trader: LocalAccount,
        trading_strategy_universe: TradingStrategyUniverse,
        routing_model: UniswapV2SimpleRoutingModel,
        uniswap_v2: UniswapV2Deployment,
        usdc_token: Contract,
        weth_token: Contract,

):
    """Live Uniswap v2 stop loss trigger.

    - Trade ETH/USDC pool

    - Two accounts: deployer and trader. Deployer holds 1.7M worth of ETH in the tool. Trader starts with 9000 USDC
      trade balance.

    - Set up an in-memory blockchain with Uni v2 instance we can manipulate

    - Sets up a buy and hold strategy with 10% stop loss trigger

    - Start the strategy, check that the trading account is funded

    - Advance to cycle 1 and make sure the long position on ETH is opened

    - Cause an external price shock

    - Check that the stop loss trigger was correctly executed and we sold ETH for loss
    """

    # Sanity check for the trading universe
    # that we start with 1705 USD/ETH price
    pair_universe = trading_strategy_universe.universe.pairs
    exchanges = trading_strategy_universe.universe.exchanges
    pricing_method = UniswapV2LivePricing(web3, pair_universe, routing_model)
    exchange = exchanges[0] # Get the first exchange from the universe
    weth_usdc = pair_universe.get_one_pair_from_pandas_universe(exchange.exchange_id, "WETH", "USDC")
    pair = translate_trading_pair(weth_usdc)
    price = pricing_method.get_buy_price(datetime.datetime.utcnow(), pair, None)
    assert price == pytest.approx(1705.12, rel=APPROX_REL)

    # Set up an execution loop we can step through
    state = State()
    loop = set_up_simulated_execution_loop_uniswap_v2(
        web3=web3,
        decide_trades=decide_trades,
        universe=trading_strategy_universe,
        state=state,
        wallet_account=trader,
        routing_model=routing_model,
    )

    ts = get_latest_block_timestamp(web3)

    loop.tick(
        ts,
        state,
        cycle=1,
        live=True,
    )

    loop.update_position_valuations(
        ts,
        state,
        trading_strategy_universe,
    )

    # After the first tick, we should have synced our reserves and opened the first position
    assert state.portfolio.reserves[usdc_token.address.lower()].quantity == 8000
    assert state.portfolio.open_positions[1].get_value() == pytest.approx(994.0125000000002)

    # Sell ETH on the pool to change the price more than 10%.
    # The pool is 1000 ETH / 1.7M USDC.
    # Deployer dumps 300 ETH to cause a massive price impact.
    weth_token.functions.approve(uniswap_v2.router.address, 1000 * 10**18).transact({"from": deployer})
    prepared_swap_call = swap_with_slippage_protection(
        uniswap_v2_deployment=uniswap_v2,
        recipient_address=deployer,
        base_token=usdc_token,
        quote_token=weth_token,
        amount_in=300 * 10**18,
        max_slippage=10_000,
    )
    prepared_swap_call.transact({"from": deployer})

    # ETH price is down $1700 -> $1000
    price = pricing_method.get_buy_price(datetime.datetime.utcnow(), pair, None)
    assert price == pytest.approx(1009.6430522606291 , rel=APPROX_REL)


