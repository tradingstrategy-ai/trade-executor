"""Test live spot and short strategy using Uniwwap v2 and 1delta using forked Polygon"""
import datetime
import os
import shutil
from logging import Logger
from typing import List

import pytest
import pandas as pd
from web3 import Web3
from web3.contract import Contract

from eth_defi.provider.anvil import mine
from eth_defi.uniswap_v3.deployment import UniswapV3Deployment
from eth_defi.hotwallet import HotWallet

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

pytestmark = pytest.mark.skipif(
    (os.environ.get("JSON_RPC_POLYGON") is None) or (shutil.which("anvil") is None),
    reason="Set JSON_RPC_POLYGON env install anvil command to run these tests",
)


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
    spot_eth = pairs.get_pair_by_human_description((ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005))

    if position_manager.is_any_open():
        trades += position_manager.close_all()

    if cycle % 2 == 0:
        # Spot day
        trades += position_manager.open_spot(spot_eth, 100.0)
    else:
        # Short day
        trades += position_manager.open_short(spot_eth, 150.0)

    return trades


def test_generic_router_spot_and_short_strategy(
    logger: Logger,
    web3: Web3,
    hot_wallet: HotWallet,
    strategy_universe: TradingStrategyUniverse,
    uniswap_v3_deployment: UniswapV3Deployment,
    one_delta_routing_model: OneDeltaRouting,
    usdc: Contract,
    weth: Contract,
    weth_usdc_spot_pair,
    generic_routing_model: GenericRouting,
    generic_pricing_model: GenericPricing,
    generic_valuation_model: GenericValuation,
):
    """See generic manager goes through backtesting loop correctly.

    - Uses Polygon mainnet fork

    - We do not care PnL because we are just hitting simulated buy/sell
      against the current live prices at the time of runnign the test
    """

    # Set up an execution loop we can step through
    state = State()
    loop = set_up_simulated_ethereum_generic_execution(
        web3=web3,
        decide_trades=decide_trades,
        universe=strategy_universe,
        state=state,
        routing_model=generic_routing_model,
        pricing_model=generic_pricing_model,
        valuation_model=generic_valuation_model,
        hot_wallet=hot_wallet,
    )

    ts = get_latest_block_timestamp(web3)
    for cycle in range(10):
        loop.tick(
            ts,
            loop.cycle_duration,
            state,
            cycle=cycle,
            live=True,
        )

        loop.update_position_valuations(
            ts,
            state,
            strategy_universe,
            ExecutionMode.real_trading
        )
        ts += datetime.timedelta(days=1)
        mine(web3, to_int_unix_timestamp(ts))

        loop.runner.check_accounts(strategy_universe, state)  # Check that on-chain balances reflect what we expect


def test_generic_router_spot_and_short_strategy_manual_tick(
    logger: Logger,
    web3: Web3,
    hot_wallet: HotWallet,
    strategy_universe: TradingStrategyUniverse,
    uniswap_v3_deployment: UniswapV3Deployment,
    one_delta_routing_model: OneDeltaRouting,
    usdc: Contract,
    weth: Contract,
    weth_usdc_spot_pair,
    generic_routing_model: GenericRouting,
    generic_pricing_model: GenericPricing,
    generic_valuation_model: GenericValuation,
):
    """Manually tick through the spot and short strategy and verify intermediate results in each step.

    See generic manager goes through backtesting loop correctly.

    - Uses Polygon mainnet fork

    - We do not care PnL because we are just hitting simulated buy/sell
      against the current live prices at the time of runnign the test

    - Anvil is forked at Polygon block 49_000_000
    """

    # Set up an execution loop we can step through
    state = State()
    portfolio = state.portfolio

    loop = set_up_simulated_ethereum_generic_execution(
        web3=web3,
        decide_trades=decide_trades,
        universe=strategy_universe,
        state=state,
        routing_model=generic_routing_model,
        pricing_model=generic_pricing_model,
        valuation_model=generic_valuation_model,
        hot_wallet=hot_wallet,
    )

    #
    # Cycle #1, open short
    #
    ts = get_latest_block_timestamp(web3)
    loop.tick(
        ts,
        loop.cycle_duration,
        state,
        cycle=1,
        live=True,
    )
    assert len(portfolio.open_positions) == 1
    position = portfolio.open_positions[1]
    assert position.pair.is_short()
    assert position.get_value() == pytest.approx(150.00924046277555)

    loop.runner.check_accounts(strategy_universe, state)  # Check that on-chain balances reflect what we expect

    # Add some time so that we manage to gain interest
    ts += datetime.timedelta(days=180)
    mine(web3, to_int_unix_timestamp(ts))

    # Sync interest
    interest_events = loop.sync_model.sync_interests(
        ts,
        state,
        strategy_universe,
        generic_pricing_model,
    )
    logger.info("Generated %d sync interest events", len(interest_events))

    # Update valuations
    ts += datetime.timedelta(days=1)
    loop.update_position_valuations(
        ts,
        state,
        strategy_universe,
        ExecutionMode.real_trading
    )
    assert len(portfolio.open_positions) == 1
    
    tolerance = 1e-5
    assert position.loan.get_borrow_interest() == pytest.approx(1.4861359925126336, abs=tolerance)
    tolerance = 1e-4
    assert position.loan.get_collateral_interest() == pytest.approx(13.711178, abs=tolerance)

    loop.runner.check_accounts(strategy_universe, state)  # Check that on-chain balances reflect what we expect

    #
    # Cycle #2, close short open spot
    #
    ts += datetime.timedelta(days=1)
    mine(web3, to_int_unix_timestamp(ts))
    loop.tick(
        ts,
        loop.cycle_duration,
        state,
        cycle=2,
        live=True,
    )
    assert len(portfolio.open_positions) == 1
    assert len(portfolio.closed_positions) == 1
    position = portfolio.open_positions[2]
    assert position.pair.is_spot()
    assert position.get_value() == pytest.approx(99.99910513591374)

    loop.runner.check_accounts(strategy_universe, state)  # Check that on-chain balances reflect what we expect

    # Update valuations
    ts += datetime.timedelta(days=1)
    loop.update_position_valuations(
        ts,
        state,
        strategy_universe,
        ExecutionMode.real_trading
    )
    assert len(portfolio.open_positions) == 1

    loop.runner.check_accounts(strategy_universe, state)  # Check that on-chain balances reflect what we expect

    #
    # Cycle #3, close spot, open short
    #
    ts += datetime.timedelta(days=1)
    mine(web3, to_int_unix_timestamp(ts))
    loop.tick(
        ts,
        loop.cycle_duration,
        state,
        cycle=3,
        live=True,
    )
    assert len(portfolio.open_positions) == 1
    assert len(portfolio.closed_positions) == 2
    position = portfolio.open_positions[3]
    assert position.pair.is_short()
    assert position.get_value() == pytest.approx(150.00835642808374)

    loop.runner.check_accounts(strategy_universe, state)  # Check that on-chain balances reflect what we expect

    # Update valuations
    ts += datetime.timedelta(days=1)
    loop.update_position_valuations(
        ts,
        state,
        strategy_universe,
        ExecutionMode.real_trading
    )
    assert len(portfolio.open_positions) == 1

    loop.runner.check_accounts(strategy_universe, state)  # Check that on-chain balances reflect what we expect


