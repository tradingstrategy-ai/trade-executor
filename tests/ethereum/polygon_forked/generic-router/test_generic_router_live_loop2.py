"""Test live spot and short strategy using Uniwwap v2 and 1delta using forked Polygon"""
import datetime
import os
import shutil
from decimal import Decimal
from logging import Logger
from typing import List

import pytest
import pandas as pd
from web3 import Web3
from web3.contract import Contract
from eth_account.signers.local import LocalAccount

from eth_defi.provider.anvil import mine
from eth_defi.uniswap_v3.deployment import UniswapV3Deployment
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import fork_network_anvil, mine
from eth_defi.uniswap_v2.fees import estimate_sell_price, estimate_buy_price, estimate_buy_price_decimals, estimate_sell_price_decimals
from eth_defi.uniswap_v3.price import estimate_buy_received_amount, estimate_sell_received_amount, get_onchain_price

from tradeexecutor.ethereum.one_delta.one_delta_routing import OneDeltaRouting
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing
from tradeexecutor.strategy.generic.generic_router import GenericRouting
from tradeexecutor.strategy.generic.generic_valuation import GenericValuation
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.strategy.execution_context import ExecutionMode
from tradeexecutor.strategy.run_state import RunState
from tradeexecutor.testing.simulated_execution_loop import set_up_simulated_execution_loop_one_delta, set_up_simulated_ethereum_generic_execution
from tradeexecutor.utils.blockchain import get_latest_block_timestamp
from tradingstrategy.chain import ChainId
from tradingstrategy.utils.time import to_int_unix_timestamp
from tradeexecutor.utils.blockchain import get_block_timestamp

pytestmark = pytest.mark.skipif(
    (os.environ.get("JSON_RPC_POLYGON") is None) or (shutil.which("anvil") is None),
    reason="Set JSON_RPC_POLYGON env install anvil command to run these tests",
)

@pytest.fixture
def anvil_polygon_chain_fork(request, large_usdc_holder):
    """Create a testable fork of live Polygon.

    :return: JSON-RPC URL for Web3
    """
    mainnet_rpc = os.environ["JSON_RPC_POLYGON"]
    launch = fork_network_anvil(
        mainnet_rpc,
        unlocked_addresses=[large_usdc_holder],
        fork_block_number=60_000_000,
    )
    try:
        yield launch.json_rpc_url
    finally:
        # Wind down Anvil process after the test is complete
        # launch.close(log_level=logging.ERROR)
        launch.close()


@pytest.fixture
def hot_wallet(web3, user_1, usdc, large_usdc_holder) -> HotWallet:
    """Hot wallet used for fork tets.

    - Starts with MATIC and $10k USDC balance
    """
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
        300_000 * 10**6,
    ).transact({"from": large_usdc_holder})

    wallet.sync_nonce(web3)

    # mine a few blocks
    for i in range(1, 5):
        mine(web3)

    return wallet


def decide_trades(
    timestamp: pd.Timestamp,
    strategy_universe: TradingStrategyUniverse,
    state: State,
    pricing_model: PricingModel,
    cycle_debug_data: dict
) -> List[TradeExecution]:
    # Every second day buy spot,
    # every second day short

    trades = []
    position_manager = PositionManager(timestamp, strategy_universe, state, pricing_model)
    cycle = cycle_debug_data["cycle"]
    pairs = strategy_universe.data_universe.pairs
    spot_btc = pairs.get_pair_by_human_description((ChainId.polygon, "quickswap", "WBTC", "WETH", 0.003))

    if position_manager.is_any_open():
        trades += position_manager.close_all()

    if cycle == 1:
        trades += position_manager.open_spot(spot_btc, 200_000.0)

    return trades


def test_generic_router_spot_and_short_strategy(
    logger: Logger,
    web3: Web3,
    hot_wallet: HotWallet,
    strategy_universe: TradingStrategyUniverse,
    # uniswap_v2_deployment: UniswapV3Deployment,
    usdc: Contract,
    weth: Contract,
    wbtc: Contract,
    wbtc_weth_spot_pair,
    generic_routing_model: GenericRouting,
    generic_pricing_model: GenericPricing,
    generic_valuation_model: GenericValuation,
    quickswap_deployment,
    uniswap_v3_deployment,
):
    """See generic manager goes through backtesting loop correctly.

    - Uses Polygon mainnet fork

    - We do not care PnL because we are just hitting simulated buy/sell
      against the current live prices at the time of runnign the test
    """

    # Set up an execution loop we can step through
    state = State()
    
    # loop = set_up_simulated_ethereum_generic_execution(
    #     web3=web3,
    #     decide_trades=decide_trades,
    #     universe=strategy_universe,
    #     state=state,
    #     routing_model=generic_routing_model,
    #     pricing_model=generic_pricing_model,
    #     valuation_model=generic_valuation_model,
    #     hot_wallet=hot_wallet,
    # )

    # #     state = State()
    # portfolio = state.portfolio

    # #
    # # Cycle #1, open short
    # #
    # ts = get_latest_block_timestamp(web3)
    # loop.tick(
    #     ts,
    #     loop.cycle_duration,
    #     state,
    #     cycle=1,
    #     live=True,
    # )
    # assert len(portfolio.open_positions) == 1
    # position = portfolio.open_positions[1]
    # assert position.get_value() == pytest.approx(200_000)

    price = estimate_buy_price_decimals(
        quickswap_deployment,
        base_token_address=wbtc.address,
        quote_token_address=usdc.address,
        quantity=Decimal(1),
        intermediate_token_address=weth.address,
    )

    price2 = estimate_sell_price_decimals(
        quickswap_deployment,
        base_token_address=wbtc.address,
        quote_token_address=usdc.address,
        quantity=Decimal(1),
        intermediate_token_address=weth.address,
    )

    price3 = estimate_buy_price(
        quickswap_deployment,
        base_token=wbtc,
        quote_token=usdc,
        quantity=Decimal(1),
        intermediate_token=weth,
    )

    print("Timestamp", get_block_timestamp(web3, 60_000_000))

    mid = (price + price2) / 2
    print("Current buy price (how much USDC to pay for 1 BTC):", price)
    print("Current sell price (how much USDC we get after selling 1 BTC):", price2)
    print("Assume mid price", mid)


    price1 = estimate_sell_received_amount(
        uniswap=uniswap_v3_deployment,
        base_token_address=wbtc.address,
        quote_token_address=usdc.address,
        quantity=1 * 10**6,
        target_pair_fee=500,
        # intermediate_token_address=weth.address,
    )
    print(price1)
    # print("Impact", (price - mid) * 100 / price)

    # import ipdb; ipdb.set_trace()

    # ts = get_latest_block_timestamp(web3)
    # for cycle in range(10):
    #     loop.tick(
    #         ts,
    #         loop.cycle_duration,
    #         state,
    #         cycle=cycle,
    #         live=True,
    #     )

    #     loop.update_position_valuations(
    #         ts,
    #         state,
    #         strategy_universe,
    #         ExecutionMode.real_trading
    #     )
    #     ts += datetime.timedelta(days=1)
    #     mine(web3, to_int_unix_timestamp(ts))

    #     loop.runner.check_accounts(strategy_universe, state)  # Check that on-chain balances reflect what we expect

