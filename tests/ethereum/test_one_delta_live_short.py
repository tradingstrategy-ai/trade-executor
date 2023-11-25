"""Test live short-only strategy on 1delta using forked Polygon"""
import datetime
import os
import shutil
import logging
from decimal import Decimal
from typing import List

import pytest
import pandas as pd
from eth_account import Account
from eth_account.signers.local import LocalAccount
from eth_typing import HexAddress, HexStr
from hexbytes import HexBytes
from web3 import EthereumTesterProvider, Web3
from web3.contract import Contract

from eth_defi.token import fetch_erc20_details
from eth_defi.uniswap_v3.deployment import UniswapV3Deployment, fetch_deployment as fetch_uniswap_v3_deployment
from eth_defi.hotwallet import HotWallet
from eth_defi.aave_v3.deployment import AaveV3Deployment, fetch_deployment as fetch_aave_deployment
from eth_defi.one_delta.deployment import OneDeltaDeployment
from eth_defi.one_delta.deployment import fetch_deployment as fetch_1delta_deployment
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.provider.anvil import fork_network_anvil, mine
from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.liquidity import GroupedLiquidityUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe
from tradingstrategy.lending import LendingProtocolType

from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_execution import UniswapV3ExecutionModel
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_routing import UniswapV3SimpleRoutingModel
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_live_pricing import UniswapV3LivePricing
from tradeexecutor.ethereum.one_delta.one_delta_execution import OneDeltaExecutionModel
from tradeexecutor.ethereum.one_delta.one_delta_live_pricing import OneDeltaLivePricing
from tradeexecutor.ethereum.one_delta.one_delta_routing import OneDeltaSimpleRoutingModel
from tradeexecutor.state.state import State, UncleanState
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.cycle import snap_to_previous_tick, snap_to_next_tick
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.execution_context import python_script_execution_context
from tradeexecutor.strategy.universe_model import default_universe_options
from tradeexecutor.strategy.trading_strategy_universe import translate_trading_pair, TradingStrategyUniverse, load_partial_data
from tradeexecutor.strategy.execution_context import ExecutionMode
from tradeexecutor.strategy.run_state import RunState
from tradeexecutor.ethereum.universe import create_exchange_universe, create_pair_universe
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.testing.simulated_execution_loop import set_up_simulated_execution_loop_one_delta
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradeexecutor.utils.blockchain import get_latest_block_timestamp
from tradeexecutor.visual.image_output import open_plotly_figure_in_browser, open_bytes_in_browser

pytestmark = pytest.mark.skipif(
    (os.environ.get("JSON_RPC_POLYGON") is None) or (shutil.which("anvil") is None),
    reason="Set JSON_RPC_POLYGON env install anvil command to run these tests",
)


#: How much values we allow to drift.
#: A hack fix receiving different decimal values on Github CI than on a local
APPROX_REL = 0.001
APPROX_REL_DECIMAL = Decimal("0.001")

WETH_USDC_FEE = 0.003
AAVE_USDC_FEE = 0.003

WETH_USDC_FEE_RAW = 3000
AAVE_USDC_FEE_RAW = 3000


@pytest.fixture(scope="module")
def large_usdc_holder() -> HexAddress:
    """A random account picked from Polygon that holds a lot of USDC.

    This account is unlocked on Anvil, so you have access to good USDC stash.

    `To find large holder accounts, use <https://polygonscan.com/token/0x2791bca1f2de4661ed88a30c99a7a9449aa84174#balances>`_.
    """
    # Binance Hot Wallet 6
    return HexAddress(HexStr("0xe7804c37c13166fF0b37F5aE0BB07A3aEbb6e245"))


@pytest.fixture
def anvil_polygon_chain_fork(request, large_usdc_holder) -> str:
    """Create a testable fork of live Polygon.

    :return: JSON-RPC URL for Web3
    """
    mainnet_rpc = os.environ["JSON_RPC_POLYGON"]
    launch = fork_network_anvil(
        mainnet_rpc,
        unlocked_addresses=[large_usdc_holder],
        fork_block_number=49_000_000,
    )
    try:
        yield launch.json_rpc_url
    finally:
        # Wind down Anvil process after the test is complete
        launch.close(log_level=logging.ERROR)


@pytest.fixture
def web3(anvil_polygon_chain_fork: str):
    """Set up a Web3 provider instance with a lot of workarounds for flaky nodes."""
    return create_multi_provider_web3(anvil_polygon_chain_fork)


@pytest.fixture
def chain_id(web3) -> int:
    """The fork chain id."""
    return web3.eth.chain_id


@pytest.fixture(scope="module")
def user_1() -> LocalAccount:
    """Create a test account."""
    return Account.create()


@pytest.fixture
def hot_wallet(web3, user_1, usdc, large_usdc_holder) -> HotWallet:
    """Hot wallet."""
    assert isinstance(user_1, LocalAccount)
    wallet = HotWallet(user_1)
    wallet.sync_nonce(web3)

    # give hot wallet some native token and USDC
    web3.eth.send_transaction(
        {
            "from": large_usdc_holder,
            "to": wallet.address,
            "value": 100 * 10**18,
        }
    )

    usdc.contract.functions.transfer(
        wallet.address,
        10_000 * 10**6,
    ).transact({"from": large_usdc_holder})

    wallet.sync_nonce(web3)

    # mine a few blocks
    for i in range(1, 5):
        mine(web3)

    return wallet


@pytest.fixture
def uniswap_v3_deployment(web3) -> UniswapV3Deployment:
    """Uniswap v3 deployment."""
    return fetch_uniswap_v3_deployment(
        web3,
        "0x1F98431c8aD98523631AE4a59f267346ea31F984",
        "0xE592427A0AEce92De3Edee1F18E0157C05861564",
        "0xC36442b4a4522E871399CD717aBDD847Ab11FE88",
        "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6",
    )


@pytest.fixture
def aave_v3_deployment(web3):
    return fetch_aave_deployment(
        web3,
        pool_address="0x794a61358D6845594F94dc1DB02A252b5b4814aD",
        data_provider_address="0x69FA688f1Dc47d4B5d8029D5a35FB7a548310654",
        oracle_address="0xb023e699F5a33916Ea823A16485e259257cA8Bd1",
    )


@pytest.fixture
def one_delta_deployment(web3) -> OneDeltaDeployment:
    return fetch_1delta_deployment(
        web3,
        flash_aggregator_address="0x74E95F3Ec71372756a01eB9317864e3fdde1AC53",
        broker_proxy_address="0x74E95F3Ec71372756a01eB9317864e3fdde1AC53",
    )


@pytest.fixture
def usdc(web3):
    """Get USDC on Polygon."""
    return fetch_erc20_details(web3, "0x2791bca1f2de4661ed88a30c99a7a9449aa84174")


@pytest.fixture
def ausdc(web3):
    """Get aPolUSDC on Polygon."""
    return fetch_erc20_details(web3, "0x625E7708f30cA75bfd92586e17077590C60eb4cD", contract_name="aave_v3/AToken.json")


@pytest.fixture
def weth(web3):
    """Get WETH on Polygon."""
    return fetch_erc20_details(web3, "0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619")


@pytest.fixture
def vweth(web3):
    """Get vPolWETH on Polygon."""
    return fetch_erc20_details(web3, "0x0c84331e39d6658Cd6e6b9ba04736cC4c4734351", contract_name="aave_v3/VariableDebtToken.json")


@pytest.fixture
def price_helper(uniswap_v3_deployment):
    return UniswapV3PriceHelper(uniswap_v3_deployment)


@pytest.fixture
def asset_usdc(usdc, chain_id) -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(
        chain_id,
        usdc.contract.address,
        usdc.symbol,
        usdc.decimals,
    )


@pytest.fixture
def asset_weth(weth, chain_id) -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(
        chain_id,
        weth.contract.address,
        weth.symbol,
        weth.decimals,
    )


@pytest.fixture()
def asset_ausdc(ausdc, asset_usdc, chain_id) -> AssetIdentifier:
    """USDC collateral token."""
    return AssetIdentifier(
        chain_id,
        ausdc.contract.address,
        ausdc.symbol,
        ausdc.decimals,
        underlying=asset_usdc,
        type=AssetType.collateral,
        liquidation_threshold=0.85,  # From Aave UI
    )


@pytest.fixture
def asset_vweth(vweth, asset_weth, chain_id) -> AssetIdentifier:
    """Variable debt token."""
    return AssetIdentifier(
        chain_id,
        vweth.contract.address,
        vweth.symbol,
        vweth.decimals,
        underlying=asset_weth,
        type=AssetType.borrowed,
    )


@pytest.fixture
def weth_usdc_spot_pair(uniswap_v3_deployment, asset_usdc, asset_weth) -> TradingPairIdentifier:
    return TradingPairIdentifier(
        asset_weth,
        asset_usdc,
        "0x0e44ceb592acfc5d3f09d996302eb4c499ff8c10",
        uniswap_v3_deployment.factory.address,
        fee=WETH_USDC_FEE,
    )


@pytest.fixture
def weth_usdc_shorting_pair(uniswap_v3_deployment, asset_ausdc, asset_vweth, weth_usdc_spot_pair) -> TradingPairIdentifier:
    return TradingPairIdentifier(
        asset_vweth,
        asset_ausdc,
        "0x1",  # TODO
        uniswap_v3_deployment.factory.address,
        # fee=WETH_USDC_FEE,
        kind=TradingPairKind.lending_protocol_short,
        underlying_spot_pair=weth_usdc_spot_pair,
    )


@pytest.fixture
def start_ts() -> datetime.datetime:
    """Timestamp of action started"""
    return datetime.datetime(2022, 1, 1, tzinfo=None)


@pytest.fixture
def supported_reserves(asset_usdc) -> List[AssetIdentifier]:
    """The reserve currencies we support."""
    return [asset_usdc]


@pytest.fixture
def state(web3, hot_wallet, asset_usdc, usdc) -> State:
    """State used in the tests."""
    state = State()

    events = sync_reserves(
        web3, datetime.datetime.utcnow(), hot_wallet.address, [], [asset_usdc]
    )
    assert len(events) > 0
    apply_sync_events(state, events)
    reserve_currency, exchange_rate = state.portfolio.get_default_reserve_asset()
    assert reserve_currency == asset_usdc
    return state


@pytest.fixture()
def routing_model(
    one_delta_deployment,
    aave_v3_deployment,
    uniswap_v3_deployment,
    asset_usdc,
    asset_weth,
) -> OneDeltaSimpleRoutingModel:
    return OneDeltaSimpleRoutingModel(
        address_map={
            "one_delta_broker_proxy": one_delta_deployment.broker_proxy.address,
            "aave_v3_pool": aave_v3_deployment.pool.address,
            "aave_v3_data_provider": aave_v3_deployment.data_provider.address,
            "aave_v3_oracle": aave_v3_deployment.oracle.address,
            "factory": uniswap_v3_deployment.factory.address,
            "router": uniswap_v3_deployment.swap_router.address,
            "position_manager": uniswap_v3_deployment.position_manager.address,
            "quoter": uniswap_v3_deployment.quoter.address
        },
        allowed_intermediary_pairs={},
        reserve_token_address=asset_usdc.address.lower(),
    )


@pytest.fixture()
def exchange_universe(web3, uniswap_v3_deployment: UniswapV3Deployment) -> ExchangeUniverse:
    """We trade on one uniswap v3 deployment on tester."""
    return create_exchange_universe(web3, [uniswap_v3_deployment])


@pytest.fixture()
def pair_universe(web3, exchange_universe: ExchangeUniverse, weth_usdc_spot_pair) -> PandasPairUniverse:
    """We trade on two trading pairs."""
    exchange = next(iter(exchange_universe.exchanges.values()))
    return create_pair_universe(web3, exchange, [weth_usdc_spot_pair])


@pytest.fixture()
def trading_strategy_universe(chain_id, exchange_universe, pair_universe, asset_usdc, persistent_test_client) -> TradingStrategyUniverse:
    """Universe that also contains data about our reserve assets."""
    # data_universe = Universe(
    #     time_bucket=TimeBucket.d1,
    #     chains=[ChainId(chain_id)],
    #     exchanges=list(exchange_universe.exchanges.values()),
    #     pairs=pair_universe,
    #     candles=GroupedCandleUniverse.create_empty_qstrader(),
    #     liquidity=GroupedLiquidityUniverse.create_empty(),
    # )

    # return TradingStrategyUniverse(data_universe=data_universe, reserve_assets=[asset_usdc])

    pairs = [
        (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005),
    ]

    reverses = [
        (ChainId.polygon, LendingProtocolType.aave_v3, "WETH"),
        (ChainId.polygon, LendingProtocolType.aave_v3, "USDC"),
    ]

    dataset = load_partial_data(
        persistent_test_client,
        execution_context=python_script_execution_context,
        time_bucket=TimeBucket.d1,
        pairs=pairs,
        universe_options=default_universe_options,
        start_at=pd.Timestamp("2023-10-01"),
        end_at=pd.Timestamp("2023-10-30"),
        lending_reserves=reverses,
    )

    # Convert loaded data to a trading pair universe
    return TradingStrategyUniverse.create_single_pair_universe(dataset)


def decide_trades(
    timestamp: pd.Timestamp,
    strategy_universe: TradingStrategyUniverse,
    state: State,
    pricing_model: PricingModel,
    cycle_debug_data: dict
) -> List[TradeExecution]:
    """Opens new buy and hold position for $1000 if no position is open."""
    
    pair = strategy_universe.universe.pairs.get_single()

    # Open for 1,000 USD
    position_size = 1000.00

    trades = []

    position_manager = PositionManager(timestamp, strategy_universe, state, pricing_model)

    if not position_manager.is_any_short_position_open():
        trades += position_manager.open_short(
            pair,
            position_size,
            leverage=2,
        )
    else:
        trades += position_manager.close_all()

    return trades



def test_one_delta_live_strategy_short(
    logger,
    web3: Web3,
    hot_wallet: HotWallet,
    trading_strategy_universe: TradingStrategyUniverse,
    routing_model: OneDeltaSimpleRoutingModel,
    uniswap_v3_deployment: UniswapV3Deployment,
    usdc: Contract,
    weth: Contract,
    weth_usdc_spot_pair,
):
    """Live 1delta trade.

    - Trade ETH/USDC pool

    # TODO

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
    # that we start with 1635 USD/ETH price
    pair_universe = trading_strategy_universe.data_universe.pairs
    exchanges = trading_strategy_universe.data_universe.exchange_universe
    pricing_method = OneDeltaLivePricing(web3, pair_universe, routing_model)

    weth_usdc = pair_universe.get_single()
    pair = translate_trading_pair(weth_usdc)

    # assert pair == weth_usdc_spot_pair

    # Check that our preflight checks pass
    routing_model.perform_preflight_checks_and_logging(pair_universe)

    price_structure = pricing_method.get_buy_price(datetime.datetime.utcnow(), pair, None)
    # assert price_structure.price == pytest.approx(1635.9293, rel=APPROX_REL)
    assert price_structure.price == pytest.approx(1631.0085715155444, rel=APPROX_REL)

    # Set up an execution loop we can step through
    state = State()
    loop = set_up_simulated_execution_loop_one_delta(
        web3=web3,
        decide_trades=decide_trades,
        universe=trading_strategy_universe,
        state=state,
        # wallet_account=hot_wallet.account,
        hot_wallet=hot_wallet,
        routing_model=routing_model,
    )
    loop.runner.run_state = RunState()  # Needed for visualisations

    ts = get_latest_block_timestamp(web3)

    loop.tick(
        ts,
        loop.cycle_duration,
        state,
        cycle=1,
        live=True,
    )

    loop.update_position_valuations(
        ts,
        state,
        trading_strategy_universe,
        ExecutionMode.real_trading
    )

    assert len(state.portfolio.open_positions) == 1

    # After the first tick, we should have synced our reserves and opened the first position
    mid_price = pricing_method.get_mid_price(ts, pair)
    assert mid_price == pytest.approx(1630.1912407577722, rel=APPROX_REL)

    usdc_id = f"{web3.eth.chain_id}-{usdc.address.lower()}"
    assert state.portfolio.reserves[usdc_id].quantity == 9000
    assert state.portfolio.open_positions[1].get_quantity() == Decimal('1.261256429210282326')
    assert state.portfolio.open_positions[1].get_value() == pytest.approx(944.0010729999999, rel=APPROX_REL)

    # trade another cycle
    ts = get_latest_block_timestamp(web3)
    strategy_cycle_timestamp = snap_to_next_tick(ts, loop.cycle_duration)

    loop.tick(
        ts,
        loop.cycle_duration,
        state,
        cycle=2,
        live=True,
        strategy_cycle_timestamp=strategy_cycle_timestamp,
    )

    loop.update_position_valuations(
        ts,
        state,
        trading_strategy_universe,
        ExecutionMode.real_trading
    )

    assert len(state.portfolio.open_positions) == 0
