"""Test DecideTradesV3 function signature..

"""
import os
import logging
import datetime
from _decimal import Decimal
from typing import List, Dict

import pytest
import pandas as pd

from tradeexecutor.strategy.strategy_module import StrategyParameters
from tradingstrategy.client import Client
from tradingstrategy.chain import ChainId
from tradingstrategy.lending import LendingProtocolType
from tradingstrategy.timebucket import TimeBucket

from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_partial_data
from tradeexecutor.strategy.execution_context import ExecutionContext, unit_test_execution_context
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.universe_model import UniverseOptions, default_universe_options
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.state.state import State


# https://docs.pytest.org/en/latest/how-to/skipping.html#skip-all-test-functions-of-a-class-or-module
pytestmark = pytest.mark.skipif(os.environ.get("TRADING_STRATEGY_API_KEY") is None, reason="Set TRADING_STRATEGY_API_KEY environment variable to run this test")


@pytest.fixture()
def interest_rate_mock(mocker):
    """Patch backtest sync to use fixed interest rate for every call"""
    interest_mock = mocker.patch(
        "tradeexecutor.strategy.interest.estimate_interest"
    )

    interest_mock.return_value = Decimal("1")


def create_trading_universe(
    ts: datetime.datetime,
    client: Client,
    execution_context: ExecutionContext,
    universe_options: UniverseOptions,
) -> TradingStrategyUniverse:

    assert universe_options.start_at
    assert execution_context.engine_version in ("0.3", "0.4",)  # We changed the decide_trades() signature

    pairs = [
        (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005)
    ]

    reverses = [
        (ChainId.polygon, LendingProtocolType.aave_v3, "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
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


def test_decide_trades_v03(
    logger: logging.Logger,
    persistent_test_client: Client,
    interest_rate_mock,
    ):
    """Run the strategy backtest using inline decide_trades function.

    - How much interest we can get on USDC on Polygon in one month
    """

    def decide_trades(
            timestamp: pd.Timestamp,
            parameters: StrategyParameters,
            strategy_universe: TradingStrategyUniverse,
            state: State,
            pricing_model: PricingModel) -> List[TradeExecution]:
        """A simple strategy that puts all in to our lending reserve."""
        assert parameters.test_val == 111
        position_manager = PositionManager(timestamp, strategy_universe, state, pricing_model)
        cash = state.portfolio.get_cash()

        trades = []

        if not position_manager.is_any_open():
            trades += position_manager.open_credit_supply_position_for_reserves(cash)

        return trades

    parameters = StrategyParameters({
        "test_val": 111,
        "initial_cash": 10_000,
        "cycle_duration": CycleDuration.cycle_1d,
    })

    # Run the test
    state, universe, debug_dump = run_backtest_inline(
        start_at=datetime.datetime(2023, 1, 1),
        end_at=datetime.datetime(2023, 2, 1),
        client=persistent_test_client,
        decide_trades=decide_trades,
        create_trading_universe=create_trading_universe,
        reserve_currency=ReserveCurrency.usdc,
        trade_routing=TradeRouting.uniswap_v3_usdc_poly,
        engine_version="0.4",
        parameters=parameters,
    )

    portfolio = state.portfolio
    assert len(portfolio.open_positions) == 1
    credit_position = portfolio.open_positions[1]
    assert credit_position.is_credit_supply()

    # Backtest creates one event before each tick
    assert len(credit_position.balance_updates) == 30

    interest = credit_position.loan.collateral_interest
    assert interest.opening_amount == Decimal("10000.00")
    assert credit_position.get_accrued_interest() == pytest.approx(5.777347790050936)
    assert credit_position.get_quantity() == pytest.approx(Decimal(10005.77734779005093525703117))
    assert credit_position.get_value() == pytest.approx(10005.77734779005093525703117)
    assert portfolio.calculate_total_equity() == pytest.approx(10005.77734779005093525703117)
