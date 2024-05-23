"""Stop loss using live uniswap price feed.

"""
import datetime
import secrets
import os
from decimal import Decimal
from typing import List

import pytest
import pandas as pd
from eth_account import Account
from eth_account.signers.local import LocalAccount

from eth_defi.uniswap_v3.swap import swap_with_slippage_protection
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
from eth_defi.uniswap_v3.deployment import UniswapV3Deployment, deploy_pool, deploy_uniswap_v3, add_liquidity
from eth_defi.uniswap_v3.utils import get_default_tick_range

from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_execution import UniswapV3Execution
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_routing import UniswapV3Routing
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_live_pricing import UniswapV3LivePricing
from tradeexecutor.state.state import State, UncleanState
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.cycle import snap_to_previous_tick, snap_to_next_tick
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trading_strategy_universe import translate_trading_pair, TradingStrategyUniverse
from tradeexecutor.strategy.execution_context import ExecutionMode
from tradeexecutor.strategy.run_state import RunState
from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.pair import PandasPairUniverse

from tradeexecutor.ethereum.universe import create_exchange_universe, create_pair_universe
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.testing.simulated_execution_loop import set_up_simulated_execution_loop_uniswap_v3
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradeexecutor.utils.blockchain import get_latest_block_timestamp
from tradeexecutor.visual.image_output import open_plotly_figure_in_browser, open_bytes_in_browser
from tradeexecutor.statistics.summary import calculate_summary_statistics


#: How much values we allow to drift.
#: A hack fix receiving different decimal values on Github CI than on a local
APPROX_REL = 0.001
APPROX_REL_DECIMAL = Decimal("0.001")

WETH_USDC_FEE = 0.003
WETH_USDC_FEE_RAW = 3000

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
def uniswap_v3(web3, deployer) -> UniswapV3Deployment:
    """uniswap v3 deployment."""
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


@pytest.fixture()
def exchange_universe(web3, uniswap_v3: UniswapV3Deployment) -> ExchangeUniverse:
    """We trade on one uniswap v3 deployment on tester."""
    return create_exchange_universe(web3, [uniswap_v3])


@pytest.fixture
def weth_usdc_pair(uniswap_v3, weth_usdc_uniswap_trading_pair, asset_usdc, asset_weth) -> TradingPairIdentifier:
    return TradingPairIdentifier(
        asset_weth,
        asset_usdc,
        weth_usdc_uniswap_trading_pair,
        uniswap_v3.factory.address,
        fee=WETH_USDC_FEE,
        internal_id = 1,
    )


@pytest.fixture()
def exchange_universe(web3, uniswap_v3: UniswapV3Deployment) -> ExchangeUniverse:
    """We trade on one uniswap v3 deployment on tester."""
    return create_exchange_universe(web3, [uniswap_v3])


@pytest.fixture()
def pair_universe(web3, exchange_universe: ExchangeUniverse, weth_usdc_pair) -> PandasPairUniverse:
    exchange = next(iter(exchange_universe.exchanges.values()))  # Get the first exchange from the universe
    return create_pair_universe(web3, exchange, [weth_usdc_pair])


@pytest.fixture()
def routing_model(uniswap_v3: UniswapV3Deployment, asset_usdc, asset_weth, weth_usdc_pair) -> UniswapV3Routing:

    # Allowed exchanges as factory -> router pairs
    address_map = {
        "factory": uniswap_v3.factory.address,
        "router": uniswap_v3.swap_router.address,
        "position_manager": uniswap_v3.position_manager.address,
        "quoter": uniswap_v3.quoter.address
    }

    # Three way ETH quoted trades are routed thru WETH/USDC pool
    allowed_intermediary_pairs = {
        asset_weth.address: weth_usdc_pair.pool_address
    }

    return UniswapV3Routing(
        address_map,
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
    
    pair = universe.pairs.get_single()

    # Open for 1,000 USD
    position_size = 1000.00

    trades = []

    position_manager = PositionManager(timestamp, universe, state, pricing_model)

    if not position_manager.is_any_open():
        buy_amount = position_size
        trades += position_manager.open_spot(
            pair,
            buy_amount,
            stop_loss_pct=0.95,  # Use 5% stop loss
        )
    else:
        position_manager.close_all()

    return trades


def decide_trades_no_stop_loss(
        timestamp: pd.Timestamp,
        universe: Universe,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: dict) -> List[TradeExecution]:
    """Stop loss free trading logic."""

    pair = universe.pairs.get_single()

    # Open for 1,000 USD
    position_size = 1000.00

    trades = []

    position_manager = PositionManager(timestamp, universe, state, pricing_model)

    if not position_manager.is_any_open():
        buy_amount = position_size
        trades += position_manager.open_spot(
            pair,
            buy_amount,
        )
    else:
        trades += position_manager.close_all()

    return trades


@pytest.fixture()
def core_universe(web3,
             exchange_universe: ExchangeUniverse,
             pair_universe: PandasPairUniverse,
             weth_usdc_pair: TradingPairIdentifier
             ) -> Universe:
    """Create a trading universe that contains our mock pair."""

    time_bucket = TimeBucket.d1

    df = generate_ohlcv_candles(
        start=datetime.datetime(2023, 1, 1),
        end=datetime.datetime.now(),
        bucket=time_bucket,
        pair_id = weth_usdc_pair.internal_id,
        exchange_id = None,
        daily_drift = (0.98, 1.02),  # sideways
        high_drift=1.01,
        low_drift=0.99,
    )

    candle_universe = GroupedCandleUniverse(df, time_bucket)
    
    return Universe(
        time_bucket=TimeBucket.d1,
        chains=[ChainId(web3.eth.chain_id)],
        exchanges=list(exchange_universe.exchanges.values()),
        pairs=pair_universe,
        candles=candle_universe,
        liquidity=GroupedLiquidityUniverse.create_empty(),
    )


@pytest.fixture()
def trading_strategy_universe(core_universe: Universe, asset_usdc) -> TradingStrategyUniverse:
    """Universe that also contains data about our reserve assets."""
    return TradingStrategyUniverse(data_universe=core_universe, reserve_assets=[asset_usdc])



def test_live_stop_loss(
    logger,
    web3: Web3,
    deployer: HexAddress,
    trader: LocalAccount,
    trading_strategy_universe: TradingStrategyUniverse,
    routing_model: UniswapV3Routing,
    uniswap_v3: UniswapV3Deployment,
    usdc_token: Contract,
    weth_token: Contract,

):
    """Live uniswap v3 stop loss trigger.

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
    pair_universe = trading_strategy_universe.data_universe.pairs
    exchanges = trading_strategy_universe.data_universe.exchange_universe
    pricing_method = UniswapV3LivePricing(web3, pair_universe, routing_model)
    exchange = exchanges.get_single()
    weth_usdc = pair_universe.get_one_pair_from_pandas_universe(exchange.exchange_id, "WETH", "USDC")
    pair = translate_trading_pair(weth_usdc)

    # Check that our preflight checks pass
    routing_model.perform_preflight_checks_and_logging(pair_universe)

    price_structure = pricing_method.get_buy_price(datetime.datetime.utcnow(), pair, None)
    assert price_structure.price == pytest.approx(1705.12, rel=APPROX_REL)

    # Set up an execution loop we can step through
    state = State()
    loop = set_up_simulated_execution_loop_uniswap_v3(
        web3=web3,
        decide_trades=decide_trades,
        universe=trading_strategy_universe,
        state=state,
        wallet_account=trader,
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

    # After the first tick, we should have synced our reserves and opened the first position
    mid_price = pricing_method.get_mid_price(ts, pair)
    assert mid_price == pytest.approx(1701.9176812836754, rel=APPROX_REL)

    usdc_id = f"{web3.eth.chain_id}-{usdc_token.address.lower()}"
    assert state.portfolio.reserves[usdc_id].quantity == 8000
    assert state.portfolio.open_positions[1].get_quantity() == Decimal('0.586126842081438205')
    assert state.portfolio.open_positions[1].get_value() == pytest.approx(994.010747, rel=APPROX_REL)

    # Sell ETH on the pool to change the price more than 10%.
    # The pool is 1000 ETH / 1.7M USDC.
    # Deployer dumps 300 ETH to cause a massive price impact.
    weth_token.functions.approve(uniswap_v3.swap_router.address, 1000 * 10**18).transact({"from": deployer})
    prepared_swap_call = swap_with_slippage_protection(
        uniswap_v3_deployment=uniswap_v3,
        recipient_address=deployer,
        base_token=usdc_token,
        quote_token=weth_token,
        amount_in=300 * 10**18,
        max_slippage=10_000,
        pool_fees=[WETH_USDC_FEE_RAW]
    )
    prepared_swap_call.transact({"from": deployer})

    # ETH price is down $1700 -> $1000
    price_structure = pricing_method.get_buy_price(datetime.datetime.utcnow(), pair, None)
    assert price_structure.price == pytest.approx(1011.2548284709007, rel=APPROX_REL)

    ts = get_latest_block_timestamp(web3)

    # Trigger stop loss
    trades = loop.check_position_triggers(
        ts,
        state,
        trading_strategy_universe,
    )

    # Check state data looks sane
    assert len(trades) == 1, "No stop loss triggered"
    t = trades[0]
    assert t.is_stop_loss()
    assert len(state.portfolio.closed_positions) == 1
    assert len(state.portfolio.open_positions) == 0
    assert state.portfolio.closed_positions[1].is_stop_loss()

    # We are ~500 USD on loss after stop loss trigger
    usdc_id = f"{web3.eth.chain_id}-{usdc_token.address.lower()}"
    assert state.portfolio.reserves[usdc_id].quantity == pytest.approx(Decimal('8588.907521'))


def test_live_stop_loss_missing(
        logger,
        web3: Web3,
        deployer: HexAddress,
        trader: LocalAccount,
        trading_strategy_universe: TradingStrategyUniverse,
        routing_model: UniswapV3Routing,
        uniswap_v3: UniswapV3Deployment,
        usdc_token: Contract,
        weth_token: Contract,

):
    """Stop loss code does not crash/trigger if stop losses for trades are not set.
    """

    # Set up an execution loop we can step through
    state = State()
    loop = set_up_simulated_execution_loop_uniswap_v3(
        web3=web3,
        decide_trades=decide_trades_no_stop_loss,
        universe=trading_strategy_universe,
        state=state,
        wallet_account=trader,
        routing_model=routing_model,
    )
    loop.runner.run_state = RunState()

    
    
    ts = get_latest_block_timestamp(web3)

    loop.tick(
        ts,
        loop.cycle_duration,
        state,
        cycle=1,
        live=True,
    )

    # Sell ETH on the pool to change the price more than 10%.
    # The pool is 1000 ETH / 1.7M USDC.
    # Deployer dumps 300 ETH to cause a massive price impact.
    weth_token.functions.approve(uniswap_v3.swap_router.address, 1000 * 10**18).transact({"from": deployer})
    prepared_swap_call = swap_with_slippage_protection(
        uniswap_v3_deployment=uniswap_v3,
        recipient_address=deployer,
        base_token=usdc_token,
        quote_token=weth_token,
        amount_in=300 * 10**18,
        max_slippage=10_000,
        pool_fees=[WETH_USDC_FEE_RAW]
    )
    prepared_swap_call.transact({"from": deployer})

    ts = get_latest_block_timestamp(web3)

    # Trigger stop loss
    trades = loop.check_position_triggers(
        ts,
        state,
        trading_strategy_universe,
    )

    # Check state data looks sane
    assert len(trades) == 0, "Stop loss unexpectedly triggered"


@pytest.mark.skip(reason="Currently unsupported")
def test_broadcast_failed_and_repair_state(
        logger,
        web3: Web3,
        deployer: HexAddress,
        trader: LocalAccount,
        trading_strategy_universe: TradingStrategyUniverse,
        routing_model: UniswapV3Routing,
        uniswap_v3: UniswapV3Deployment,
        usdc_token: Contract,
        weth_token: Contract,

):
    """Check that we can recover from the situation where the transaction broadcast has failed.

    TODO: This test case should be moved to its own test module,
    but is here because a lot of shared fixtures.
    """

    # Set up an execution loop we can step through
    state = State()
    loop = set_up_simulated_execution_loop_uniswap_v3(
        web3=web3,
        decide_trades=decide_trades_no_stop_loss,
        universe=trading_strategy_universe,
        state=state,
        wallet_account=trader,
        routing_model=routing_model,
    )

    ts = get_latest_block_timestamp(web3)

    # Make transaction confirmation step to skip,
    execution_model = loop.execution_model
    assert isinstance(execution_model, UniswapV3Execution)

    # Set confirmation timeout to negative
    # to signal we are testing broadcast problems
    execution_model.confirmation_timeout = datetime.timedelta(seconds=-1)

    strategy_cycle_timestamp = snap_to_previous_tick(ts, loop.cycle_duration)

    loop.tick(
        ts,
        loop.cycle_duration,
        state,
        cycle=1,
        live=True,
        strategy_cycle_timestamp=strategy_cycle_timestamp,
    )

    # Reset confirmation timeout to the normal value
    execution_model.confirmation_timeout = datetime.timedelta(seconds=30)

    # We are stuck with broadcasted but not confirmed trades
    with pytest.raises(UncleanState):
        state.check_if_clean()

    t = state.portfolio.open_positions[1].trades[1]
    assert t.is_unfinished()

    # Attempt repair
    trades = loop.runner.repair_state(state)

    # We repaired one trade
    assert len(trades) == 1

    # State is clean now
    t = state.portfolio.open_positions[1].trades[1]
    assert t.is_success()
    assert t.is_repaired()

    state.check_if_clean()


def test_refresh_visualisations(
        logger,
        web3: Web3,
        deployer: HexAddress,
        trader: LocalAccount,
        trading_strategy_universe: TradingStrategyUniverse,
        routing_model: UniswapV3Routing,
        uniswap_v3: UniswapV3Deployment,
        usdc_token: Contract,
        weth_token: Contract,
):
    """Check that we can refresh visualisations."""

     # Set up an execution loop we can step through
    state = State()
    loop = set_up_simulated_execution_loop_uniswap_v3(
        web3=web3,
        decide_trades=decide_trades_no_stop_loss,
        universe=trading_strategy_universe,
        state=state,
        wallet_account=trader,
        routing_model=routing_model,
    )

    loop.runner.run_state = RunState()  # needed for visualisations

    ts = get_latest_block_timestamp(web3)  # will not show trades due to the timestamp

    # Make transaction confirmation step to skip,
    execution_model = loop.execution_model
    assert isinstance(execution_model, UniswapV3Execution)

    # Set confirmation timeout to negative
    # to signal we are testing broadcast problems
    execution_model.confirmation_timeout = datetime.timedelta(seconds=-1)

    strategy_cycle_timestamp = snap_to_previous_tick(ts, loop.cycle_duration)

    loop.tick(
        ts,
        loop.cycle_duration,
        state,
        cycle=1,
        live=True,
        strategy_cycle_timestamp=strategy_cycle_timestamp,
    )

    loop.runner.refresh_visualisations(state, trading_strategy_universe)

    strategy_cycle_timestamp = snap_to_next_tick(ts, loop.cycle_duration)

    loop.tick(
        ts,
        loop.cycle_duration,
        state,
        cycle=2,
        live=True,
        strategy_cycle_timestamp=strategy_cycle_timestamp,
    )

    loop.runner.refresh_visualisations(state, trading_strategy_universe)

    small_image = loop.runner.run_state.visualisation.small_image
    small_image_dark = loop.runner.run_state.visualisation.small_image_dark
    large_image = loop.runner.run_state.visualisation.large_image
    large_image_dark = loop.runner.run_state.visualisation.large_image_dark

    small_image_png = loop.runner.run_state.visualisation.small_image_png
    large_image_png = loop.runner.run_state.visualisation.large_image_png

    if os.environ.get('SHOW_IMAGE'):
        open_bytes_in_browser(small_image, format="svg")
        open_bytes_in_browser(small_image_dark, format="svg")
        open_bytes_in_browser(large_image, format="svg")
        open_bytes_in_browser(large_image_dark, format="svg")
        open_bytes_in_browser(small_image_png)
        open_bytes_in_browser(large_image_png)


def test_metadata_stats(
    logger,
    web3: Web3,
    deployer: HexAddress,
    trader: LocalAccount,
    trading_strategy_universe: TradingStrategyUniverse,
    routing_model: UniswapV3Routing,
    uniswap_v3: UniswapV3Deployment,
    usdc_token: Contract,
    weth_token: Contract,
):
    """This tests the calculation and json export of the stats that go in the metadata endpoint .i.e. StrategySummaryStatistics"""
    # Set up an execution loop we can step through
    state = State()
    loop = set_up_simulated_execution_loop_uniswap_v3(
        web3=web3,
        decide_trades=decide_trades_no_stop_loss,
        universe=trading_strategy_universe,
        state=state,
        wallet_account=trader,
        routing_model=routing_model,
    )

    loop.runner.run_state = RunState()  # needed for visualisations

    ts = get_latest_block_timestamp(web3)  # will not show trades due to the timestamp
    
    # Make transaction confirmation step to skip,
    execution_model = loop.execution_model
    assert isinstance(execution_model, UniswapV3Execution)

    # Set confirmation timeout to negative
    # to signal we are testing broadcast problems
    execution_model.confirmation_timeout = datetime.timedelta(seconds=-1)

    strategy_cycle_timestamp = snap_to_previous_tick(ts, loop.cycle_duration)

    loop.tick(
        ts,
        loop.cycle_duration,
        state,
        cycle=1,
        live=True,
        strategy_cycle_timestamp=strategy_cycle_timestamp,
    )

    strategy_cycle_timestamp = snap_to_next_tick(ts, loop.cycle_duration)

    loop.tick(
        ts,
        loop.cycle_duration,
        state,
        cycle=2,
        live=True,
        strategy_cycle_timestamp=strategy_cycle_timestamp,
    )
    
    ts = get_latest_block_timestamp(web3)
    strategy_cycle_timestamp = snap_to_next_tick(ts, loop.cycle_duration)

    loop.tick(
        ts,
        loop.cycle_duration,
        state,
        cycle=3,
        live=True,
        strategy_cycle_timestamp=strategy_cycle_timestamp,
    )
    
    # now there should be 2 trades completed
    # test long short table
    stats = calculate_summary_statistics(
        state,
        loop.execution_context.mode,
        key_metrics_backtest_cut_off=datetime.timedelta(seconds=0),
        cycle_duration=loop.cycle_duration,
    )
    loop.runner.run_state.summary_statistics = stats
    loop.runner.run_state.make_exportable_copy().to_json()