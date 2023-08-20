"""Run a backtest where we open a credit supply position, using real oracle data.

"""
import os
import logging

import pytest

import pandas as pd

from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_partial_data
from tradeexecutor.strategy.execution_context import ExecutionContext, unit_test_execution_context
from tradingstrategy.client import Client
import datetime

from tradingstrategy.chain import ChainId
from tradingstrategy.lending import LendingProtocolType
from tradingstrategy.timebucket import TimeBucket
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.universe_model import UniverseOptions, default_universe_options

from tradeexecutor.state.trade import TradeExecution
from typing import List, Dict
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.state.state import State
from tradingstrategy.universe import Universe


# https://docs.pytest.org/en/latest/how-to/skipping.html#skip-all-test-functions-of-a-class-or-module
pytestmark = pytest.mark.skipif(os.environ.get("TRADING_STRATEGY_API_KEY") is None, reason="Set TRADING_STRATEGY_API_KEY environment variable to run this test")


def decide_trades(
        timestamp: pd.Timestamp,
        universe: Universe,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: Dict) -> List[TradeExecution]:
    """A simple strategy that puts all in to our lending reserve."""
    position_manager = PositionManager(timestamp, universe, state, pricing_model)
    cash = state.portfolio.get_current_cash()

    trades = []

    if not position_manager.is_any_open():
        trades += position_manager.open_credit_supply_position_for_reserves(cash)

    return trades


def create_trading_universe(
        ts: datetime.datetime,
        client: Client,
        execution_context: ExecutionContext,
        universe_options: UniverseOptions,
) -> TradingStrategyUniverse:

    pairs = [
        (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005)
    ]

    reverses = [
        (ChainId.polygon, LendingProtocolType.aave_v3, "USDC")
    ]

    dataset = load_partial_data(
        client,
        execution_context=unit_test_execution_context,
        time_bucket=TimeBucket.d1,
        pairs=pairs,
        universe_options=default_universe_options,
        start_at=universe_options.start_at,
        end_at=universe_options.end_at,
        lending_reserves=reverses,
    )

    strategy_universe = TradingStrategyUniverse.create_single_pair_universe(dataset)
    return strategy_universe


def test_backtest_open_credit_supply_real_data(
        logger: logging.Logger,
        persistent_test_client: Client,
    ):
    """Run the strategy backtest using inline decide_trades function.

    - How much interest we can get on USDC on Polygon in one month
    """

    # Run the test
    state, universe, debug_dump = run_backtest_inline(
        start_at=datetime.datetime(2023, 1, 1),
        end_at=datetime.datetime(2023, 2, 1),
        client=persistent_test_client,
        cycle_duration=CycleDuration.cycle_1d,  # Override to use 24h cycles despite what strategy file says
        decide_trades=decide_trades,
        create_trading_universe=create_trading_universe,
        initial_deposit=10_000,
        reserve_currency=ReserveCurrency.usdc,
        trade_routing=TradeRouting.uniswap_v3_usdc_poly,
    )

    portfolio = state.portfolio
    assert len(portfolio.open_positions) == 1
    credit_position = portfolio.open_positions[1]

    # Backtest creates one event before each tick
    assert len(credit_position.balance_updates) == 30

