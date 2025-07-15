"""Test live short-only strategy on 1delta using forked Polygon"""
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

from tradeexecutor.ethereum.one_delta.one_delta_live_pricing import OneDeltaLivePricing
from tradeexecutor.ethereum.one_delta.one_delta_routing import OneDeltaRouting
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.identifier import AssetWithTrackedValue
from tradeexecutor.strategy.cycle import snap_to_next_tick
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trading_strategy_universe import translate_trading_pair, TradingStrategyUniverse, load_partial_data
from tradeexecutor.strategy.execution_context import ExecutionMode
from tradeexecutor.strategy.run_state import RunState
from tradeexecutor.ethereum.universe import create_exchange_universe, create_pair_universe
from tradeexecutor.testing.simulated_execution_loop import set_up_simulated_execution_loop_one_delta
from tradeexecutor.utils.blockchain import get_latest_block_timestamp
from tradeexecutor.strategy.account_correction import check_accounts


CI = os.environ.get("CI") == "true"


pytestmark = pytest.mark.skipif(
    (os.environ.get("JSON_RPC_POLYGON") is None) or (shutil.which("anvil") is None),
    reason="Set JSON_RPC_POLYGON env install anvil command to run these tests",
)


#: How much values we allow to drift.
#: A hack fix receiving different decimal values on Github CI than on a local
APPROX_REL = 0.001
APPROX_REL_DECIMAL = Decimal("0.001")

#   File "/home/runner/work/trade-executor/trade-executor/tradeexecutor/statistics/core.py", line 59, in calculate_position_statistics
#     profitability=position.get_total_profit_percent(),
#   File "/home/runner/work/trade-executor/trade-executor/tradeexecutor/state/position.py", line 1213, in get_total_profit_percent
#     profit = -self.get_total_profit_usd()
#   File "/home/runner/work/trade-executor/trade-executor/tradeexecutor/state/position.py", line 1195, in get_total_profit_usd
#     realised_profit = self.get_realised_profit_usd() or 0
#   File "/home/runner/work/trade-executor/trade-executor/tradeexecutor/state/position.py", line 1161, in get_realised_profit_usd
#     trade_profit = (self.get_average_sell() - self.get_average_buy()) * float(self.get_buy_quantity())
# TypeError: unsupported operand type(s) for -: 'float' and 'NoneType'
# ------------------------------ Captured log call -------------------------------
# ERROR    eth_defi.revert_reason:revert_reason.py:155 Transaction succeeded, when we tried to fetch its revert reason.
# Hash: 0x2f02ec63264fde03ce565b0204527ecb101566dc3ac3291618e836ad6b2b15ff, tx block num: 49000021, current block number: 49000021
# Transaction result:
# HexBytes('0x')
# - Maybe the chain tip is unstable
# - Maybe transaction failed due to slippage
# - Maybe someone is frontrunning you and it does not happen with eth_call replay
#
# ERROR    eth_defi.revert_reason:revert_reason.py:155 Transaction succeeded, when we tried to fetch its revert reason.
# Hash: 0x2f02ec63264fde03ce565b0204527ecb101566dc3ac3291618e836ad6b2b15ff, tx block num: 49000021, current block number: 49000021
# Transaction result:
# HexBytes('0x')
# - Maybe the chain tip is unstable
# - Maybe transaction failed due to slippage
# - Maybe someone is frontrunning you and it does not happen with eth_call replay
#
# ERROR    tradeexecutor.ethereum.swap:swap.py:55 Trade <Close short short #2
#    1.226751532789447690 WETH at 1630.9089983902218 USD, broadcasted phase
#    collateral consumption: -2000.800439012997474114586382 aPolUSDC, collateral allocation: -997.980466987002525885413618 aPolUSDC
#    reserve: 0
#    > failed and freezing the position: <could not extract the revert reason>
# WARNING  tradeexecutor.state.freeze:freeze.py:34 Freezing position for a failed trade: <Close short short #2
#    1.226751532789447690 WETH at 1630.9089983902218 USD, failed phase
#    collateral consumption: -2000.800439012997474114586382 aPolUSDC, collateral allocation: -997.980466987002525885413618 aPolUSDC
#    reserve: 0
#    >
@pytest.mark.skipif(CI, reason="Too flaky on Github")
def test_one_delta_live_strategy_short_open_and_close(
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

    - Trade ETH/USDC 0.3% pool

    - Sets up a simple strategy that open a 2x short position then close in next cycle

    - Start the strategy, check that the trading account is funded

    - Advance to cycle 1 and make sure the short position on ETH is opened

    - Advance to cycle 2 and make sure the short position is closed
    """

    def decide_trades(
        timestamp: pd.Timestamp,
        strategy_universe: TradingStrategyUniverse,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: dict
    ) -> List[TradeExecution]:
        """Opens a 2x short position and closes in next trade cycle."""
        
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

    routing_model = one_delta_routing_model

    # Sanity check for the trading universe
    # that we start with 1631 USD/ETH price
    pair_universe = trading_strategy_universe.data_universe.pairs
    pricing_method = OneDeltaLivePricing(web3, pair_universe, routing_model)

    weth_usdc = pair_universe.get_single()
    pair = translate_trading_pair(weth_usdc)

    # Check that our preflight checks pass
    routing_model.perform_preflight_checks_and_logging(pair_universe)

    price_structure = pricing_method.get_buy_price(datetime.datetime.utcnow(), pair, None)
    assert price_structure.price == pytest.approx(2239.420956551886, rel=APPROX_REL)

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

    mine(web3, increase_timestamp=3600)
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
        ExecutionMode.simulated_trading,
        interest=False,
    )

    loop.runner.check_accounts(trading_strategy_universe, state)

    assert len(state.portfolio.open_positions) == 1

    # After the first tick, we should have synced our reserves and opened the first position
    mid_price = pricing_method.get_mid_price(ts, pair)
    assert mid_price == pytest.approx(2238.0298724242684, rel=APPROX_REL)

    usdc_id = f"{web3.eth.chain_id}-{usdc.address.lower()}"
    assert state.portfolio.reserves[usdc_id].quantity == 9000
    assert state.portfolio.open_positions[1].get_quantity() == pytest.approx(Decimal(-0.893495022670441332))
    assert state.portfolio.open_positions[1].get_value() == pytest.approx(1000.0140651703407, rel=APPROX_REL)

    # mine a few block before running next tick
    mine(web3, increase_timestamp=3600)

    # trade another cycle to close the short position
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
        ExecutionMode.simulated_trading,
        interest=False,
    )

    loop.runner.check_accounts(trading_strategy_universe, state)

    assert len(state.portfolio.open_positions) == 0
    assert len(state.portfolio.closed_positions) == 1
    # assert state.portfolio.reserves[usdc_id].quantity == 10000


@pytest.mark.skipif(CI, reason="Too flaky on Github")
def test_one_delta_live_strategy_short_open_accrue_interests(
    logger,
    web3: Web3,
    hot_wallet: HotWallet,
    trading_strategy_universe: TradingStrategyUniverse,
    uniswap_v3_deployment: UniswapV3Deployment,
    one_delta_routing_model: OneDeltaRouting,
    usdc: Contract,
    weth: Contract,
    weth_usdc_spot_pair,
    mocker,
):
    """Live 1delta trade.

    - Trade ETH/USDC 0.3% pool

    - Sets up a simple strategy that open a 2x short position, and keep it running in next cycles

    - Start the strategy, check that the trading account is funded

    - Advance to cycle 1 and make sure the short position on ETH is opened

    - Advance to cycle 2-3 cycle and check the interest
    """

    def decide_trades(
        timestamp: pd.Timestamp,
        strategy_universe: TradingStrategyUniverse,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: dict
    ) -> List[TradeExecution]:
        """Opens a 2x short position and keep it opened"""
        
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

        return trades

    get_sell_price_mock = mocker.patch(
        "tradeexecutor.ethereum.one_delta.one_delta_live_pricing.OneDeltaLivePricing.get_sell_price",
        side_effect=[
            # invoke from our test
            mocker.Mock(price=2000.0, mid_price=2000.0),
            # 1st tick()
            mocker.Mock(price=2000.0, mid_price=2000.0),
            mocker.Mock(price=2000.0, mid_price=2000.0),
            # 2nd tick()
            mocker.Mock(price=1950.0, mid_price=1950.0),
            mocker.Mock(price=1950.0, mid_price=1950.0),
            mocker.Mock(price=1950.0, mid_price=1950.0),
            # 3rd tick()
            mocker.Mock(price=1800.0, mid_price=1800.0),
            mocker.Mock(price=1800.0, mid_price=1800.0),
            mocker.Mock(price=1800.0, mid_price=1800.0),
        ]
    )
    revalue = mocker.spy(AssetWithTrackedValue, "revalue")

    routing_model = one_delta_routing_model

    # Sanity check for the trading universe
    # that we start with 1631 USD/ETH price
    pair_universe = trading_strategy_universe.data_universe.pairs
    pricing_method = OneDeltaLivePricing(web3, pair_universe, routing_model)

    weth_usdc = pair_universe.get_single()
    pair = translate_trading_pair(weth_usdc)

    # Check that our preflight checks pass
    routing_model.perform_preflight_checks_and_logging(pair_universe)

    assert pricing_method.get_sell_price(datetime.datetime.utcnow(), pair, None).price == pytest.approx(2000)
    assert get_sell_price_mock.call_count == 1

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
        ExecutionMode.simulated_trading,
        interest=False,
    )

    assert len(state.portfolio.open_positions) == 1

    # After the first tick, we should have synced our reserves and opened the first position
    usdc_id = f"{web3.eth.chain_id}-{usdc.address.lower()}"
    assert state.portfolio.reserves[usdc_id].quantity == 9000

    # assert state.portfolio.open_positions[1].get_quantity() == pytest.approx(Decimal(-1))
    # assert state.portfolio.open_positions[1].get_value() == pytest.approx(1237, rel=APPROX_REL)

    # sync time should be initialized
    first_sync_at = state.sync.interest.last_sync_at
    assert first_sync_at

    # there shouldn't be any accrued interest yet
    loan = state.portfolio.open_positions[1].loan
    assert loan.get_collateral_interest() == pytest.approx(0)
    assert loan.get_borrow_interest() == pytest.approx(0)

    # loan isn't revalued yet so price is still at the point opening the loan
    assert loan.borrowed.last_usd_price == pytest.approx(2000)
    assert revalue.call_count == 1
    assert get_sell_price_mock.call_count == 3

    # mine a few block before running next tick
    for i in range(1, 5):
        mine(web3)

    # trade another cycle to close the short position
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
        ExecutionMode.simulated_trading,
        interest=False,
    )

    # position should still be open
    assert len(state.portfolio.open_positions) == 1

    # sync time should be updated
    assert state.sync.interest.last_sync_at > first_sync_at

    # there should be accrued interest now
    position = state.portfolio.open_positions[1]
    loan = position.loan
    assert loan.get_collateral_interest() > 0
    # TODO: this shouldn't be 0
    # assert loan.get_borrow_interest() > 0

    # loan should be revalued already
    assert loan.borrowed.last_usd_price == pytest.approx(1950) 
    assert revalue.call_count == 3
    assert get_sell_price_mock.call_count == 6
    assert position.get_current_price() == pytest.approx(1950)

    # mine a few block before running next tick
    for i in range(1, 5):
        mine(web3)

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

    loop.update_position_valuations(
        ts,
        state,
        trading_strategy_universe,
        ExecutionMode.simulated_trading,
        interest=False,
    )

    # # there should be accrued interest now
    position = state.portfolio.open_positions[1]
    loan = position.loan

    # loan should be revalued again
    assert loan.borrowed.last_usd_price == pytest.approx(1800) 
    assert revalue.call_count == 5
    assert position.get_current_price() == pytest.approx(1800)

    # TODO: there should be 4 interest update events (2 per cycle), but currently only 2 since vWETH isn't accrued yet
    events = list(position.balance_updates.values())
    assert len(events) == 2
    # assert len([event for event in events if event.asset.token_symbol == "variableDebtPolWETH"]) == 2
    assert len([event for event in events if event.asset.token_symbol == "aPolUSDC"]) == 2


def test_one_delta_live_strategy_short_increase(
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

    - Trade ETH/USDC 0.3% pool

    - Sets up a simple strategy that open a 2x short position then increase size in next cycle

    - Start the strategy, check that the trading account is funded

    - Advance to cycle 1 and make sure the short position on ETH is opened

    - Advance to cycle 2 and make sure the short position is increased
    """

    def decide_trades(
        timestamp: pd.Timestamp,
        strategy_universe: TradingStrategyUniverse,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: dict
    ) -> List[TradeExecution]:
        """Opens a 2x short position and reduce to half in next trade cycle."""
        
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
            position = position_manager.get_current_short_position()
            trades += position_manager.adjust_short(
                position,
                position_size * 2,
            )

        return trades

    routing_model = one_delta_routing_model

    # Sanity check for the trading universe
    # that we start with 1631 USD/ETH price
    pair_universe = trading_strategy_universe.data_universe.pairs
    pricing_method = OneDeltaLivePricing(web3, pair_universe, routing_model)

    weth_usdc = pair_universe.get_single()
    pair = translate_trading_pair(weth_usdc)

    # Check that our preflight checks pass
    routing_model.perform_preflight_checks_and_logging(pair_universe)

    price_structure = pricing_method.get_buy_price(datetime.datetime.utcnow(), pair, None)
    assert price_structure.price == pytest.approx(2239.420956551886, rel=APPROX_REL)

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
        ExecutionMode.simulated_trading,
        interest=False,
    )

    loop.runner.check_accounts(trading_strategy_universe, state)

    assert len(state.portfolio.open_positions) == 1

    # After the first tick, we should have synced our reserves and opened the first position
    mid_price = pricing_method.get_mid_price(ts, pair)
    assert mid_price == pytest.approx(2238.0298724242684, rel=APPROX_REL)

    usdc_id = f"{web3.eth.chain_id}-{usdc.address.lower()}"
    assert state.portfolio.reserves[usdc_id].quantity == 9000
    assert state.portfolio.open_positions[1].get_quantity() == pytest.approx(Decimal(-0.893495022670441332))
    assert state.portfolio.open_positions[1].get_value() == pytest.approx(1000.0140651703407, rel=APPROX_REL)

    # mine a few block before running next tick
    for i in range(1, 10):
        mine(web3)

    # trade another cycle to close the short position
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
        ExecutionMode.simulated_trading,
        interest=False,
    )

    loop.runner.check_accounts(trading_strategy_universe, state)

    # position should still be open
    assert len(state.portfolio.open_positions) == 1

    # check the position size get increased and reserve should be reduced
    assert state.portfolio.reserves[usdc_id].quantity == pytest.approx(Decimal(8000.236073000000033061951399))
    assert state.portfolio.open_positions[1].get_quantity() == pytest.approx(Decimal(-1.786568284806183555))
    assert state.portfolio.open_positions[1].get_value() == pytest.approx(1999.778098480197, rel=APPROX_REL)


@flaky.flaky()
def test_one_delta_live_strategy_short_reduce(
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

    - Trade ETH/USDC 0.3% pool

    - Sets up a simple strategy that open a 2x short position then reduce size in next cycle

    - Start the strategy, check that the trading account is funded

    - Advance to cycle 1 and make sure the short position on ETH is opened

    - Advance to cycle 2 and make sure the short position is reduced to half
    """

    def decide_trades(
        timestamp: pd.Timestamp,
        strategy_universe: TradingStrategyUniverse,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: dict
    ) -> List[TradeExecution]:
        """Opens a 2x short position and reduce to half in next trade cycle."""
        
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
            position = position_manager.get_current_short_position()
            trades += position_manager.adjust_short(
                position,
                position_size / 2,
            )

        return trades

    routing_model = one_delta_routing_model

    # Sanity check for the trading universe
    # that we start with 1631 USD/ETH price
    pair_universe = trading_strategy_universe.data_universe.pairs
    pricing_method = OneDeltaLivePricing(web3, pair_universe, routing_model)

    weth_usdc = pair_universe.get_single()
    pair = translate_trading_pair(weth_usdc)

    # Check that our preflight checks pass
    routing_model.perform_preflight_checks_and_logging(pair_universe)

    price_structure = pricing_method.get_buy_price(datetime.datetime.utcnow(), pair, None)
    assert price_structure.price == pytest.approx(2239.4654972670164, rel=APPROX_REL)

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
        ExecutionMode.simulated_trading,
        interest=False,
    )

    loop.runner.check_accounts(trading_strategy_universe, state)

    assert len(state.portfolio.open_positions) == 1

    # After the first tick, we should have synced our reserves and opened the first position
    mid_price = pricing_method.get_mid_price(ts, pair)
    assert mid_price == pytest.approx(2238.0298724242684, rel=APPROX_REL)

    usdc_id = f"{web3.eth.chain_id}-{usdc.address.lower()}"
    assert state.portfolio.reserves[usdc_id].quantity == 9000
    assert state.portfolio.open_positions[1].get_quantity() == pytest.approx(Decimal(-0.893495022670441332))
    assert state.portfolio.open_positions[1].get_value() == pytest.approx(1000.0140651703407, rel=APPROX_REL)

    # mine a few block before running next tick
    mine(web3, increase_timestamp=3600)

    # trade another cycle to close the short position
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
        ExecutionMode.simulated_trading,
        interest=False,
    )

    loop.runner.check_accounts(trading_strategy_universe, state)

    # position should still be open
    assert len(state.portfolio.open_positions) == 1

    # check the position size get reduced and reserve should be increased
    assert state.portfolio.reserves[usdc_id].quantity == pytest.approx(Decimal(9500.267716))
    assert state.portfolio.open_positions[1].get_quantity() == pytest.approx(Decimal(-0.446379690265763032))
    assert state.portfolio.open_positions[1].get_value() == pytest.approx(499.02187110825594, rel=APPROX_REL)
