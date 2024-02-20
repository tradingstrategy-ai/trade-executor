"""Test live short-only strategy on 1delta using forked Polygon"""
import datetime
import os
import shutil
from decimal import Decimal
from pathlib import Path

import pytest
import flaky
import pandas as pd
from web3 import Web3
from web3.contract import Contract
from eth_typing import HexAddress, HexStr
from eth_account.signers.local import LocalAccount

from eth_defi.uniswap_v3.deployment import UniswapV3Deployment
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import fork_network_anvil, sleep, make_anvil_custom_rpc_request, mine
from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.lending import LendingProtocolType

from tradeexecutor.ethereum.one_delta.one_delta_live_pricing import OneDeltaLivePricing
from tradeexecutor.ethereum.one_delta.one_delta_routing import OneDeltaRouting
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.cycle import snap_to_next_tick
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.execution_context import python_script_execution_context, unit_test_execution_context
from tradeexecutor.strategy.universe_model import default_universe_options
from tradeexecutor.strategy.trading_strategy_universe import translate_trading_pair, TradingStrategyUniverse, load_partial_data
from tradeexecutor.strategy.execution_context import ExecutionMode
from tradeexecutor.strategy.run_state import RunState
from tradeexecutor.ethereum.universe import create_exchange_universe, create_pair_universe
from tradeexecutor.testing.simulated_execution_loop import set_up_simulated_execution_loop_one_delta
from tradeexecutor.utils.blockchain import get_latest_block_timestamp
from tradeexecutor.strategy.account_correction import check_accounts

from tradeexecutor.testing.ethereumtrader_one_delta import OneDeltaTestTrader
from tradeexecutor.testing.unit_test_trader import UnitTestTrader
from tradeexecutor.state.trade import TradeType, TradeExecution, TradeFlag


pytestmark = pytest.mark.skipif(
    any([
        os.environ.get("JSON_RPC_POLYGON") is None,
        os.environ.get("PRIVATE_KEY") is None,
        shutil.which("anvil") is None,
    ]),
    reason="Set JSON_RPC_POLYGON and PRIVATE_KEY env and install anvil command to run these tests",
)


#: How much values we allow to drift.
#: A hack fix receiving different decimal values on Github CI than on a local
APPROX_REL = 0.001
APPROX_REL_DECIMAL = Decimal("0.001")


@pytest.fixture
def hot_wallet(web3) -> HotWallet:
    """Hot wallet for the tests."""
    # return HexAddress(HexStr("0x5582df1f68731726d0CAF015893ad36Eb153b8D5"))
    return HotWallet.from_private_key(os.environ["PRIVATE_KEY"])


@pytest.fixture
def anvil_polygon_chain_fork(request, large_usdc_holder) -> str:
    """Create a testable fork of live Polygon.

    :return: JSON-RPC URL for Web3
    """
    mainnet_rpc = os.environ["JSON_RPC_POLYGON"]
    launch = fork_network_anvil(
        mainnet_rpc,
        unlocked_addresses=[large_usdc_holder],
        # position opened at 53151697, problem trade at 53244031
        fork_block_number=53244030 - 100,
    )
    try:
        yield launch.json_rpc_url
    finally:
        # Wind down Anvil process after the test is complete
        # launch.close(log_level=logging.ERROR)
        launch.close()


@pytest.fixture()
def exchange_universe(web3, uniswap_v3_deployment: UniswapV3Deployment) -> ExchangeUniverse:
    """We trade on one uniswap v3 deployment on tester."""
    return create_exchange_universe(web3, [uniswap_v3_deployment])


@pytest.fixture()
def pair_universe(web3, exchange_universe: ExchangeUniverse, weth_usdc_spot_pair) -> PandasPairUniverse:
    exchange = next(iter(exchange_universe.exchanges.values()))
    return create_pair_universe(web3, exchange, [weth_usdc_spot_pair])


@pytest.fixture()
def trading_strategy_universe(chain_id, exchange_universe, pair_universe, asset_usdc, persistent_test_client) -> TradingStrategyUniverse:

    pairs = [
        (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005),
    ]

    reverses = [
        (ChainId.polygon, LendingProtocolType.aave_v3, "WETH"),
        (ChainId.polygon, LendingProtocolType.aave_v3, "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"),
    ]

    dataset = load_partial_data(
        persistent_test_client,
        execution_context=unit_test_execution_context,
        time_bucket=TimeBucket.d1,
        pairs=pairs,
        universe_options=default_universe_options,
        start_at=pd.Timestamp("2023-12-01"),
        end_at=pd.Timestamp("2023-12-30"),
        lending_reserves=reverses,
    )

    # Convert loaded data to a trading pair universe
    return TradingStrategyUniverse.create_single_pair_universe(dataset)


@pytest.fixture
def state() -> State:
    """A state dump with some failed trades we need to unwind.

    Taken as a snapshot from alpha version trade execution run.
    """
    f = Path(__file__).parent / "1delta-failed-trade-state.json"
    return State.from_json(f.read_text())


def test_one_delta_live_strategy_short(
    logger,
    web3: Web3,
    hot_wallet: HotWallet,
    trading_strategy_universe: TradingStrategyUniverse,
    one_delta_routing_model: OneDeltaRouting,
    one_delta_deployment,
    aave_v3_deployment,
    uniswap_v3_deployment: UniswapV3Deployment,
    usdc: Contract,
    weth: Contract,
    vweth: Contract,
    ausdc: Contract,
    state: State,
    asset_usdc,
):
    """Live 1delta trade.

    """

    def decide_trades(
        timestamp: pd.Timestamp,
        strategy_universe: TradingStrategyUniverse,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: dict
    ) -> list[TradeExecution]:
        """Opens a 2x short position and reduce to half in next trade cycle."""
        
        pair = strategy_universe.universe.pairs.get_single()

        trades = []

        position_manager = PositionManager(timestamp, strategy_universe, state, pricing_model)

        if position_manager.is_any_short_position_open():
            trades += position_manager.close_all()

        return trades

    routing_model = one_delta_routing_model

    # Sanity check for the trading universe
    # that we start with 1631 USD/ETH price
    pair_universe = trading_strategy_universe.data_universe.pairs
    pricing_model = OneDeltaLivePricing(web3, pair_universe, routing_model)

    weth_usdc = pair_universe.get_single()
    pair = translate_trading_pair(weth_usdc)

    # Check that our preflight checks pass
    routing_model.perform_preflight_checks_and_logging(pair_universe)

    price_structure = pricing_model.get_buy_price(datetime.datetime.utcnow(), pair, None)
    # assert price_structure.price == pytest.approx(2329.37032348289, rel=APPROX_REL)

    # accounting check at current block, we should have 1 open short position

    assert len(state.portfolio.open_positions) == 1
    assert usdc.contract.functions.balanceOf(hot_wallet.address).call() == pytest.approx(105794365)
    assert ausdc.contract.functions.balanceOf(hot_wallet.address).call() == pytest.approx(1270109356)
    assert vweth.contract.functions.balanceOf(hot_wallet.address).call() == pytest.approx(363544866326492058)

    # Set up an execution loop we can step through
    # state = State()
    loop = set_up_simulated_execution_loop_one_delta(
        web3=web3,
        decide_trades=decide_trades,
        universe=trading_strategy_universe,
        state=state,
        wallet_account=hot_wallet.account,
        routing_model=routing_model,
    )
    loop.runner.run_state = RunState()

    mine(web3)

    assert len(state.portfolio.open_positions) == 1
    print(state.portfolio.open_positions)
    position = state.portfolio.open_positions[15]

    ts = get_latest_block_timestamp(web3)
    print(ts)

    print(web3.eth.get_block("latest")["number"])

    test_trader = OneDeltaTestTrader(
        one_delta_deployment,
        aave_v3_deployment,
        uniswap_v3_deployment,
        state,
        pair_universe,
        loop.execution_model.tx_builder,
    )

    # trader = UnitTestTrader(state)

    # print(test_trader)

    # position, trade = trader.close_short(
    #     weth_usdc,
    #     100, # TODO
    #     eth_price,
    #     2,
    # )

    # assert trade.is_leverage()
    # assert trade.get_status() == TradeStatus.planned

    pair = position.pair
    assert pair.kind.is_shorting()
    assert pair.base.underlying is not None, f"Lacks underlying asset: {pair.base}"
    assert pair.quote.underlying is not None, f"Lacks underlying asset: {pair.quote}"

    
    
    quantity = Decimal(position.get_quantity())

    # TODO: Hardcoded USD exchange rate
    price_structure = pricing_model.get_buy_price(ts, pair.underlying_spot_pair, Decimal(1))

    position2, trade, _ = state.trade_short(
        ts,
        closing=True,
        pair=pair,
        borrowed_asset_price=price_structure.price,
        trade_type=TradeType.rebalance,
        reserve_currency=pair.get_pricing_pair().quote,
        planned_mid_price=price_structure.mid_price,
        collateral_asset_price=1.0,
        notes="",
        position=position,
        flags={TradeFlag.close},
    )

    assert position == position2

    s, f = test_trader.execute_trades_simple(one_delta_routing_model, [trade])
    print(s, f)

    # loop.tick(
    #     ts,
    #     loop.cycle_duration,
    #     state,
    #     cycle=257,
    #     live=True,
    # )

    # loop.update_position_valuations(
    #     ts,
    #     state,
    #     trading_strategy_universe,
    #     ExecutionMode.real_trading
    # )

    # loop.runner.check_accounts(trading_strategy_universe, state)


    # print(web3.eth.get_block("latest")["number"])
    # ts = get_latest_block_timestamp(web3)
    # assert web3.eth.get_block("latest")["number"] == 53244031
    # print(ts)


    # # mid_price = pricing_method.get_mid_price(ts, pair)
    # # assert mid_price == pytest.approx(2327.9468067781563)

    # assert len(state.portfolio.open_positions) == 0
    # assert len(state.portfolio.frozen_positions) == 1
    # short_position = state.portfolio.frozen_positions[0]

    # for t in short_position.trades.values():
    #     print(t)
