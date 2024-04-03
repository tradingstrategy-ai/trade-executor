"""Run a backtest where we open a credit supply position, using a fixed mocked interest rate.

"""
import os
import logging
import datetime
import itertools
from _decimal import Decimal
from unittest.mock import patch
from typing import List, Dict

import pytest
import pandas as pd

from tradingstrategy.client import Client
from tradingstrategy.chain import ChainId
from tradingstrategy.lending import LendingProtocolType
from tradingstrategy.timebucket import TimeBucket

from tradeexecutor.analysis.trade_analyser import build_trade_analysis
from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.state.balance_update import BalanceUpdateCause, BalanceUpdate
from tradeexecutor.state.identifier import AssetIdentifier
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
    assert execution_context.engine_version == "0.3"  # We changed the decide_trades() signature

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


def test_backtest_open_only_credit_supply_mock_data(
        logger: logging.Logger,
        persistent_test_client: Client,
        interest_rate_mock,
    ):
    """Run the strategy backtest using inline decide_trades function.

    - How much interest we can get on USDC on Polygon in one month
    """

    def decide_trades(
            timestamp: pd.Timestamp,
            strategy_universe: TradingStrategyUniverse,
            state: State,
            pricing_model: PricingModel,
            cycle_debug_data: Dict) -> List[TradeExecution]:
        """A simple strategy that puts all in to our lending reserve."""
        position_manager = PositionManager(timestamp, strategy_universe, state, pricing_model)
        cash = state.portfolio.get_cash()

        trades = []

        if not position_manager.is_any_open():
            trades += position_manager.open_credit_supply_position_for_reserves(cash)

        return trades

    # Run the test
    state, universe, debug_dump = run_backtest_inline(
        start_at=datetime.datetime(2023, 1, 1),
        end_at=datetime.datetime(2023, 2, 1),
        client=persistent_test_client,
        cycle_duration=CycleDuration.cycle_1d,
        decide_trades=decide_trades,
        create_trading_universe=create_trading_universe,
        initial_deposit=10_000,
        reserve_currency=ReserveCurrency.usdc,
        trade_routing=TradeRouting.uniswap_v3_usdc_poly,
        engine_version="0.3",
    )

    portfolio = state.portfolio
    assert len(portfolio.open_positions) == 1
    credit_position = portfolio.open_positions[1]
    assert credit_position.is_credit_supply()

    # Backtest creates one event before each tick
    assert len(credit_position.balance_updates) == 30

    interest = credit_position.loan.collateral_interest
    assert interest.opening_amount == Decimal("10000.00")
    assert credit_position.get_accrued_interest() == pytest.approx(5.8392619598023785)
    assert credit_position.get_quantity() == pytest.approx(Decimal(10005.8392619598023785))
    assert credit_position.get_value() == pytest.approx(10005.8392619598023785)
    assert portfolio.get_total_equity() == pytest.approx(10005.8392619598023785)

    analysis = build_trade_analysis(state.portfolio)
    summary = analysis.calculate_summary_statistics(state=state)

    assert summary.time_in_market == 1
    assert summary.time_in_market_volatile == 0
    assert summary.total_interest_paid_usd == 0
    assert summary.total_claimed_interest == 0


def test_backtest_open_and_close_credit_supply_mock_data(
        logger: logging.Logger,
        persistent_test_client: Client,
        interest_rate_mock,
    ):
    """Run the strategy backtest using inline decide_trades function.

    - Close position at half way the test run to see if
      we account the gained interest correctly.
    """

    def decide_trades(
            timestamp: pd.Timestamp,
            strategy_universe: TradingStrategyUniverse,
            state: State,
            pricing_model: PricingModel,
            cycle_debug_data: Dict) -> List[TradeExecution]:
        """A simple strategy that puts all in to our lending reserve."""
        position_manager = PositionManager(timestamp, strategy_universe, state, pricing_model)
        cash = state.portfolio.get_cash()

        trades = []

        if timestamp < pd.Timestamp("2023-01-15"):
            if not position_manager.is_any_open():
                trades += position_manager.open_credit_supply_position_for_reserves(cash)
            else:
                pos = position_manager.get_current_position()
        else:
            if position_manager.is_any_open():
                trades += position_manager.close_all()

        return trades

    # Run the test
    state, universe, debug_dump = run_backtest_inline(
        start_at=datetime.datetime(2023, 1, 1),
        end_at=datetime.datetime(2023, 2, 1),
        client=persistent_test_client,
        cycle_duration=CycleDuration.cycle_1d,
        decide_trades=decide_trades,
        create_trading_universe=create_trading_universe,
        initial_deposit=10_000,
        reserve_currency=ReserveCurrency.usdc,
        trade_routing=TradeRouting.uniswap_v3_usdc_poly,
        engine_version="0.3",
    )

    portfolio = state.portfolio
    assert len(portfolio.open_positions) == 0
    assert len(portfolio.frozen_positions) == 0
    assert len(portfolio.closed_positions) == 1

    # Examine closed credit position
    credit_position = portfolio.closed_positions[1]
    assert credit_position.is_credit_supply()
    assert len(credit_position.balance_updates) == 14

    assert credit_position.trades[2].planned_reserve == pytest.approx(Decimal(10002.67867669078453209820816))
    assert credit_position.trades[2].executed_reserve == pytest.approx(Decimal(10002.67867669078453209820816))
    assert credit_position.trades[2].get_claimed_interest() == pytest.approx(2.67867669078453209820816)


    ausdc = universe.get_credit_supply_pair().base

    interest = credit_position.loan.collateral_interest
    assert interest.opening_amount == Decimal("10000.00")
    b: BalanceUpdate
    interest_events = [b for b in credit_position.balance_updates.values() if b.cause == BalanceUpdateCause.interest]
    assert len(interest_events) > 0, "No interest added on the position"
    assert interest_events[0].asset == ausdc

    assert credit_position.calculate_accrued_interest_quantity(ausdc) == Decimal('2.67867669078453209820816')  # Non-denormalised interest
    assert interest.last_token_amount == Decimal('10002.67867669078453209820816')
    assert interest.last_accrued_interest == Decimal('2.67867669078453209820816')

    assert credit_position.get_quantity() == Decimal('0')  # Closed positions do not have quantity left
    assert credit_position.get_value() == pytest.approx(0)  # Closed positions do not have value left
    assert credit_position.get_claimed_interest() == pytest.approx(2.67867669078453209820816)  # Interest was claimed during closing
    assert credit_position.get_accrued_interest() == pytest.approx(0)  # Denormalised interest

    assert portfolio.get_cash() == pytest.approx(10002.67867669078453209820816)
    assert portfolio.get_net_asset_value() == pytest.approx(10002.67867669078453209820816)
    assert portfolio.get_total_equity() == pytest.approx(10002.67867669078453209820816)

    analysis = build_trade_analysis(state.portfolio)
    summary = analysis.calculate_summary_statistics(state=state)

    assert summary.total_claimed_interest == pytest.approx(2.678676690784532)
    assert summary.total_interest_paid_usd == 0
    assert summary.delta_neutral == 1
    assert summary.total_positions == 1

    assert summary.time_in_market == pytest.approx(0.45161290322580644)
    assert summary.time_in_market_volatile == 0
