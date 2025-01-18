"""Test live credit supply only strategy on 1delta using forked Polygon"""
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
from tradeexecutor.testing.simulated_execution_loop import set_up_simulated_execution_loop_one_delta, set_up_simulated_ethereum_generic_execution
from tradeexecutor.utils.blockchain import get_latest_block_timestamp
from tradeexecutor.strategy.account_correction import check_accounts


pytestmark = pytest.mark.skipif(
    (os.environ.get("JSON_RPC_POLYGON") is None) or (shutil.which("anvil") is None),
    reason="Set JSON_RPC_POLYGON env install anvil command to run these tests",
)


#: How much values we allow to drift.
#: A hack fix receiving different decimal values on Github CI than on a local
APPROX_REL = 0.001
APPROX_REL_DECIMAL = Decimal("0.001")


def test_one_delta_live_credit_supply_open_only(
    logger,
    web3: Web3,
    hot_wallet: HotWallet,
    trading_strategy_universe: TradingStrategyUniverse,
    one_delta_routing_model: OneDeltaRouting,
    uniswap_v3_deployment: UniswapV3Deployment,
    usdc: Contract,
    weth: Contract,
    asset_usdc,
):
    """Live 1delta trade.

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
        
        pair = strategy_universe.universe.pairs.get_single()

        # Open for 1,000 USD
        position_size = 1000.00

        trades = []

        position_manager = PositionManager(timestamp, strategy_universe, state, pricing_model)

        if not position_manager.is_any_credit_supply_position_open():
            trades += position_manager.open_credit_supply_position_for_reserves(position_size)

        return trades

    routing_model = one_delta_routing_model

    # Sanity check for the trading universe
    pair_universe = trading_strategy_universe.data_universe.pairs
    routing_model.perform_preflight_checks_and_logging(pair_universe)

    # Set up an execution loop we can step through
    state = State()
    loop = set_up_simulated_execution_loop_one_delta(
        web3=web3,
        decide_trades=decide_trades,
        universe=trading_strategy_universe,
        state=state,
        wallet_account=hot_wallet.account,
        routing_model=routing_model,
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
        trading_strategy_universe,
        ExecutionMode.simulated_trading
    )

    loop.runner.check_accounts(trading_strategy_universe, state)

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
        trading_strategy_universe,
        ExecutionMode.real_trading
    )

    loop.runner.check_accounts(trading_strategy_universe, state)

    assert len(state.portfolio.open_positions) == 1
    position = state.portfolio.open_positions[1]
    assert position.get_quantity() == pytest.approx(Decimal(1000))
    assert position.get_value() == pytest.approx(1000)
    assert position.loan.get_collateral_value() == pytest.approx(1000.000308)
    assert position.loan.get_collateral_value() > old_col_value
    assert position.loan.get_collateral_interest() > 0
    assert position.loan.collateral.interest_rate_at_open == pytest.approx(0.09283768887858043)
    assert position.loan.collateral.last_interest_rate == pytest.approx(0.09283768887858043)


def test_one_delta_live_credit_supply_open_and_close(
    logger,
    web3: Web3,
    hot_wallet: HotWallet,
    trading_strategy_universe: TradingStrategyUniverse,
    one_delta_routing_model: OneDeltaRouting,
    uniswap_v3_deployment: UniswapV3Deployment,
    usdc: Contract,
    weth: Contract,
    asset_usdc,
):
    """Live 1delta trade.

    - Sets up a simple strategy that open and close a credit supply position

    - Start the strategy, check that the trading account is funded

    - Advance to cycle 1 and make sure the credit supply position is opened

    - Advance to cycle 2 and make sure the credit supply position is closed
    """

    def decide_trades(
        timestamp: pd.Timestamp,
        strategy_universe: TradingStrategyUniverse,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: dict
    ) -> List[TradeExecution]:
        """Opens a credit supply position and close it."""
        
        pair = strategy_universe.universe.pairs.get_single()

        # Open for 1,000 USD
        position_size = 1000.00

        trades = []

        position_manager = PositionManager(timestamp, strategy_universe, state, pricing_model)

        if not position_manager.is_any_credit_supply_position_open():
            trades += position_manager.open_credit_supply_position_for_reserves(position_size)
        else:
            trades += position_manager.close_all()

        return trades

    routing_model = one_delta_routing_model

    # Sanity check for the trading universe
    pair_universe = trading_strategy_universe.data_universe.pairs
    routing_model.perform_preflight_checks_and_logging(pair_universe)

    # Set up an execution loop we can step through
    state = State()
    loop = set_up_simulated_execution_loop_one_delta(
        web3=web3,
        decide_trades=decide_trades,
        universe=trading_strategy_universe,
        state=state,
        wallet_account=hot_wallet.account,
        routing_model=routing_model,
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
        trading_strategy_universe,
        ExecutionMode.real_trading
    )

    loop.runner.check_accounts(trading_strategy_universe, state)

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

    # trade another cycle to close the position
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

    loop.runner.check_accounts(trading_strategy_universe, state)

    assert len(state.portfolio.open_positions) == 0
    assert state.portfolio.reserves[usdc_id].quantity == pytest.approx(Decimal(10000.000303))


# test_one_delta_live_credit_supply.py::test_one_delta_live_credit_supply_mixed_with_spot - AssertionError: assert Decimal('19000') == 9000
#  +  where Decimal('19000') = <ReservePosition <USDC at 0x2791bca1f2de4661ed88a30c99a7a9449aa84174> at 19000>.quantity
# FAILED tests/ethereum/polygon_forked/generic-router/test_generic_router.py::test_generic_routing_open_position_across_markets - AssertionError: assert Decimal('20000') == Decimal('10000')
@flaky.flaky
def test_one_delta_live_credit_supply_mixed_with_spot(
    logger,
    web3: Web3,
    hot_wallet: HotWallet,
    trading_strategy_universe: TradingStrategyUniverse,
    one_delta_routing_model: OneDeltaRouting,
    generic_routing_model,
    generic_pricing_model,
    generic_valuation_model,
    uniswap_v3_deployment: UniswapV3Deployment,
    usdc: Contract,
    weth: Contract,
    asset_usdc,
):
    """Live 1delta trade mixed with spot position using generic router

    - Sets up a simple strategy that open and close a credit supply position

    - Start the strategy, check that the trading account is funded

    - Advance to cycle 1 and make sure the credit supply position is opened

    - Advance to cycle 2 and make sure the credit supply position is closed
        and we open a new spot position in the same cycle
    """

    def decide_trades(
        timestamp: pd.Timestamp,
        strategy_universe: TradingStrategyUniverse,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: dict
    ) -> List[TradeExecution]:
        """Opens a credit supply position and close it."""
        
        pair = strategy_universe.universe.pairs.get_single()

        # Open for 1,000 USD
        position_size = 1000.00

        trades = []

        position_manager = PositionManager(timestamp, strategy_universe, state, pricing_model)

        if not position_manager.is_any_credit_supply_position_open():
            trades += position_manager.open_credit_supply_position_for_reserves(position_size)
        else:
            # close current credit position
            current_pos = position_manager.get_current_credit_supply_position()
            new_trades = position_manager.close_credit_supply_position(current_pos)
            trades.extend(new_trades)

            credit_cash = current_pos.get_quantity()
            amount = float(credit_cash) * 0.99

            # open a spot position
            new_trades = position_manager.open_spot(pair, value=amount)
            trades.extend(new_trades)

        return trades

    routing_model = generic_routing_model

    # Sanity check for the trading universe
    pair_universe = trading_strategy_universe.data_universe.pairs
    routing_model.perform_preflight_checks_and_logging(pair_universe)

    # Set up an execution loop we can step through
    state = State()
    loop = set_up_simulated_ethereum_generic_execution(
        web3=web3,
        decide_trades=decide_trades,
        universe=trading_strategy_universe,
        state=state,
        routing_model=generic_routing_model,
        pricing_model=generic_pricing_model,
        valuation_model=generic_valuation_model,
        hot_wallet=hot_wallet,
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

    loop.runner.check_accounts(trading_strategy_universe, state)

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

    # trade another cycle to close the position
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

    loop.runner.check_accounts(trading_strategy_universe, state)

    assert len(state.portfolio.open_positions) == 1
    spot_position = state.portfolio.open_positions[2]
    assert spot_position.portfolio_value_at_open == pytest.approx(10000.0003)


def test_one_delta_live_credit_supply_open_and_increase(
    logger,
    web3: Web3,
    hot_wallet: HotWallet,
    trading_strategy_universe: TradingStrategyUniverse,
    one_delta_routing_model: OneDeltaRouting,
    uniswap_v3_deployment: UniswapV3Deployment,
    usdc: Contract, 
    weth: Contract,
    asset_usdc,
):
    """Live 1delta trade.

    - Sets up a simple strategy that open and close a credit supply position

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
        """Opens a credit supply position and increase it."""
        
        pair = strategy_universe.universe.pairs.get_single()

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

    routing_model = one_delta_routing_model

    # Sanity check for the trading universe
    pair_universe = trading_strategy_universe.data_universe.pairs
    routing_model.perform_preflight_checks_and_logging(pair_universe)

    # Set up an execution loop we can step through
    state = State()
    loop = set_up_simulated_execution_loop_one_delta(
        web3=web3,
        decide_trades=decide_trades,
        universe=trading_strategy_universe,
        state=state,
        wallet_account=hot_wallet.account,
        routing_model=routing_model,
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
        trading_strategy_universe,
        ExecutionMode.real_trading
    )

    loop.runner.check_accounts(trading_strategy_universe, state)

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

    # trade another cycle to close the position
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

    loop.runner.check_accounts(trading_strategy_universe, state)

    assert len(state.portfolio.open_positions) == 1
    position = state.portfolio.open_positions[1]
    assert position.get_quantity() == pytest.approx(Decimal(2000))
    assert position.get_value() == pytest.approx(2000)
    assert position.loan.get_collateral_value() == pytest.approx(2000)


def test_one_delta_live_credit_supply_open_and_reduce(
    logger,
    web3: Web3,
    hot_wallet: HotWallet,
    trading_strategy_universe: TradingStrategyUniverse,
    one_delta_routing_model: OneDeltaRouting,
    uniswap_v3_deployment: UniswapV3Deployment,
    usdc: Contract, 
    weth: Contract,
    asset_usdc,
):
    """Live 1delta trade.

    - Sets up a simple strategy that open and close a credit supply position

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
        """Opens a credit supply position and reduce it."""
        
        pair = strategy_universe.universe.pairs.get_single()

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

    routing_model = one_delta_routing_model

    # Sanity check for the trading universe
    pair_universe = trading_strategy_universe.data_universe.pairs
    routing_model.perform_preflight_checks_and_logging(pair_universe)

    # Set up an execution loop we can step through
    state = State()
    loop = set_up_simulated_execution_loop_one_delta(
        web3=web3,
        decide_trades=decide_trades,
        universe=trading_strategy_universe,
        state=state,
        wallet_account=hot_wallet.account,
        routing_model=routing_model,
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
        trading_strategy_universe,
        ExecutionMode.real_trading
    )

    loop.runner.check_accounts(trading_strategy_universe, state)

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

    # trade another cycle to close the position
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

    loop.runner.check_accounts(trading_strategy_universe, state)

    assert len(state.portfolio.open_positions) == 1
    position = state.portfolio.open_positions[1]
    assert position.get_quantity() == pytest.approx(Decimal(800))
    assert position.get_value() == pytest.approx(800)
    assert position.loan.get_collateral_value() == pytest.approx(800)
