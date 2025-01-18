"""Test live credit supply only strategy on Aave v3 using forked Polygon"""
import datetime
import os
import shutil
from decimal import Decimal
from typing import List

import pytest
import pandas as pd
from web3 import Web3
from web3.contract import Contract
import flaky

from eth_defi.uniswap_v3.deployment import UniswapV3Deployment
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import fork_network_anvil, mine
from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.lending import LendingProtocolType

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
from tradeexecutor.testing.simulated_execution_loop import set_up_simulated_ethereum_generic_execution
from tradeexecutor.utils.blockchain import get_latest_block_timestamp
from tradeexecutor.strategy.account_correction import check_accounts


pytestmark = pytest.mark.skipif(
    (os.environ.get("JSON_RPC_ETHEREUM") is None) or (shutil.which("anvil") is None),
    reason="Set JSON_RPC_ETHEREUM env install anvil command to run these tests",
)


#: How much values we allow to drift.
#: A hack fix receiving different decimal values on Github CI than on a local
APPROX_REL = 0.001
APPROX_REL_DECIMAL = Decimal("0.001")


def test_aave_v3_live_credit_supply_open_only(
    logger,
    web3: Web3,
    hot_wallet: HotWallet,
    strategy_universe: TradingStrategyUniverse,
    generic_routing_model,
    generic_pricing_model,
    generic_valuation_model,
    usdc: Contract,
    # state: State,
):
    """Live Aave v3 trade.

    - Sets up a simple strategy that open a credit supply position

    - Start the strategy, check that the trading account is funded

    - Advance to cycle 1 and make sure the credit supply position is opened

    - Advance to cycle 2 and make sure the credit supply position is still open and interest is accured
    """

    def decide_trades(
        timestamp: pd.Timestamp,
        strategy_universe: TradingStrategyUniverse,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: dict
    ) -> List[TradeExecution]:
        """Opens a credit supply."""

        # Open for 1,000 USD
        position_size = 1000.00

        trades = []

        position_manager = PositionManager(timestamp, strategy_universe, state, pricing_model)

        if not position_manager.is_any_credit_supply_position_open():
            trades += position_manager.open_credit_supply_position_for_reserves(position_size)

        return trades

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
    loop.runner.run_state = RunState()  # Needed for visualisations
    loop.runner.accounting_checks = True

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
        strategy_universe,
        ExecutionMode.simulated_trading
    )

    loop.runner.check_accounts(strategy_universe, state)

    assert len(state.portfolio.open_positions) == 1

    # After the first tick, we should have synced our reserves and opened the first position
    usdc_id = f"{web3.eth.chain_id}-{usdc.address.lower()}"
    assert state.portfolio.reserves[usdc_id].quantity == 9000

    position = state.portfolio.open_positions[1]
    assert position.get_quantity() == pytest.approx(Decimal(1000))
    assert position.get_value() == pytest.approx(1000)
    old_col_value = position.loan.get_collateral_value()
    assert old_col_value == pytest.approx(1000)
    assert position.loan.get_collateral_interest() == 0
    
    for i in range(100):
        mine(web3)

    # trade another cycle to accure interest
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
        strategy_universe,
        ExecutionMode.simulated_trading
    )

    loop.runner.check_accounts(strategy_universe, state)

    assert len(state.portfolio.open_positions) == 1
    position = state.portfolio.open_positions[1]
    assert position.get_quantity() == pytest.approx(Decimal(1000))
    assert position.get_value() == pytest.approx(1000)
    assert position.loan.get_collateral_value() == pytest.approx(1000.000308)
    assert position.loan.get_collateral_value() > old_col_value
    assert position.loan.get_collateral_interest() > 0


def test_aave_v3_live_credit_supply_open_and_increase(
    logger,
    web3: Web3,
    hot_wallet: HotWallet,
    strategy_universe: TradingStrategyUniverse,
    generic_routing_model,
    generic_pricing_model,
    generic_valuation_model,
    usdc: Contract,
):
    """Live Aave v3 trade.

    - Sets up a simple strategy that open a credit supply position

    - Start the strategy, check that the trading account is funded

    - Advance to cycle 1 and make sure the credit supply position is opened

    - Advance to cycle 2 and make sure the credit supply position is increased
    """

    def decide_trades(
        timestamp: pd.Timestamp,
        strategy_universe: TradingStrategyUniverse,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: dict
    ) -> List[TradeExecution]:
        """Opens a credit supply."""

        # Open for 1,000 USD
        position_size = 1000.00

        trades = []

        position_manager = PositionManager(timestamp, strategy_universe, state, pricing_model)

        if not position_manager.is_any_credit_supply_position_open():
            trades += position_manager.open_credit_supply_position_for_reserves(position_size)
        else:
            position = position_manager.get_current_credit_supply_position()
            trades += position_manager.adjust_credit_supply_position(position, position_size * 2)

        return trades

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
    loop.runner.run_state = RunState()  # Needed for visualisations
    loop.runner.accounting_checks = True

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
        strategy_universe,
        ExecutionMode.simulated_trading
    )

    loop.runner.check_accounts(strategy_universe, state)

    assert len(state.portfolio.open_positions) == 1

    # After the first tick, we should have synced our reserves and opened the first position
    usdc_id = f"{web3.eth.chain_id}-{usdc.address.lower()}"
    assert state.portfolio.reserves[usdc_id].quantity == 9000

    position = state.portfolio.open_positions[1]
    assert position.get_quantity() == pytest.approx(Decimal(1000))
    assert position.get_value() == pytest.approx(1000)
    old_col_value = position.loan.get_collateral_value()
    assert old_col_value == pytest.approx(1000)
    assert position.loan.get_collateral_interest() == 0
    
    for i in range(100):
        mine(web3)

    # trade another cycle to accure interest
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
        strategy_universe,
        ExecutionMode.simulated_trading
    )

    loop.runner.check_accounts(strategy_universe, state)

    assert len(state.portfolio.open_positions) == 1
    position = state.portfolio.open_positions[1]
    assert position.get_quantity() == pytest.approx(Decimal(2000))
    assert position.get_value() == pytest.approx(2000)
    assert position.loan.get_collateral_value() == pytest.approx(2000)
    assert position.loan.get_collateral_value() > old_col_value
    assert position.loan.get_collateral_interest() > 0


def test_aave_v3_live_credit_supply_open_and_reduce(
    logger,
    web3: Web3,
    hot_wallet: HotWallet,
    strategy_universe: TradingStrategyUniverse,
    generic_routing_model,
    generic_pricing_model,
    generic_valuation_model,
    usdc: Contract,
):
    """Live Aave v3 trade.

    - Sets up a simple strategy that open a credit supply position

    - Start the strategy, check that the trading account is funded

    - Advance to cycle 1 and make sure the credit supply position is opened

    - Advance to cycle 2 and make sure the credit supply position is reduced
    """

    def decide_trades(
        timestamp: pd.Timestamp,
        strategy_universe: TradingStrategyUniverse,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: dict
    ) -> List[TradeExecution]:
        """Opens a credit supply."""

        # Open for 1,000 USD
        position_size = 1000.00

        trades = []

        position_manager = PositionManager(timestamp, strategy_universe, state, pricing_model)

        if not position_manager.is_any_credit_supply_position_open():
            trades += position_manager.open_credit_supply_position_for_reserves(position_size)
        else:
            position = position_manager.get_current_credit_supply_position()
            trades += position_manager.adjust_credit_supply_position(position, 800)

        return trades

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
    loop.runner.run_state = RunState()  # Needed for visualisations
    loop.runner.accounting_checks = True

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
        strategy_universe,
        ExecutionMode.simulated_trading
    )

    loop.runner.check_accounts(strategy_universe, state)

    assert len(state.portfolio.open_positions) == 1

    # After the first tick, we should have synced our reserves and opened the first position
    usdc_id = f"{web3.eth.chain_id}-{usdc.address.lower()}"
    assert state.portfolio.reserves[usdc_id].quantity == 9000

    position = state.portfolio.open_positions[1]
    assert position.get_quantity() == pytest.approx(Decimal(1000))
    assert position.get_value() == pytest.approx(1000)
    old_col_value = position.loan.get_collateral_value()
    assert old_col_value == pytest.approx(1000)
    assert position.loan.get_collateral_interest() == 0
    
    for i in range(100):
        mine(web3)

    # trade another cycle to accure interest
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
        strategy_universe,
        ExecutionMode.simulated_trading
    )

    loop.runner.check_accounts(strategy_universe, state)

    assert len(state.portfolio.open_positions) == 1
    position = state.portfolio.open_positions[1]
    assert position.get_quantity() == pytest.approx(Decimal(800))
    assert position.get_value() == pytest.approx(800)
    assert position.loan.get_collateral_value() == pytest.approx(800)
    assert position.loan.get_collateral_interest() > 0
