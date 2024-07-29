
import datetime
import logging
import os
import random
from decimal import Decimal
from pathlib import Path
from typing import List

import pytest
import pandas as pd

from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel
from tradeexecutor.backtest.backtest_runner import run_backtest, setup_backtest_for_universe, run_backtest_inline
from tradeexecutor.backtest.backtest_sync import BacktestSyncModel
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.cli.loop import ExecutionTestHook
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.statistics.summary import calculate_summary_statistics
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.execution_context import ExecutionMode, unit_test_execution_context, ExecutionContext
from tradeexecutor.strategy.pandas_trader.indicator import DiskIndicatorStorage, IndicatorSet
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code
from tradeexecutor.strategy.universe_model import UniverseOptions, default_universe_options
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange, generate_simple_routing_model
from tradeexecutor.testing.synthetic_lending_data import generate_lending_reserve, generate_lending_universe
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradeexecutor.testing.synthetic_universe_data import create_synthetic_single_pair_universe
from tradeexecutor.visual.equity_curve import calculate_investment_flow, calculate_realised_profitability, calculate_deposit_adjusted_returns, \
    calculate_compounding_realised_trading_profitability, calculate_size_relative_realised_trading_returns, calculate_cumulative_daily_returns, calculate_non_cumulative_daily_returns
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe



class DepositSimulator(ExecutionTestHook):
    """Do FizzBuzz deposits/redemptions simulation."""

    def __init__(self):
        self.deposit_callbacks_done = 0

    def on_before_cycle(
            self,
            cycle: int,
            cycle_st: datetime.datetime,
            state: State,
            sync_model: BacktestSyncModel

    ):
        # Assume the deposit was made 5 minutes before the strategy cycle
        fund_event_ts = cycle_st - datetime.timedelta(minutes=5)

        # Make sure we have some money in the bank on the first day
        if cycle == 1:
            sync_model.simulate_funding(fund_event_ts, Decimal(15))

        if cycle % 3 == 0:
            sync_model.simulate_funding(fund_event_ts, Decimal(100))

        if cycle % 5 == 0:
            sync_model.simulate_funding(fund_event_ts, Decimal(-90))

        self.deposit_callbacks_done += 1


@pytest.fixture(scope="module")
def strategy_universe() -> TradingStrategyUniverse:
    """Set up a mock universe."""

    time_bucket = TimeBucket.d1

    start_at = datetime.datetime(2020, 1, 1)
    candle_end_at = datetime.datetime(2020, 6, 1)

    # Set up fake assets
    chain_id = ChainId.ethereum
    mock_exchange = generate_exchange(
        exchange_id=random.randint(1, 1000),
        chain_id=chain_id,
        address=generate_random_ethereum_address(),
    )
    usdc = AssetIdentifier(chain_id.value, generate_random_ethereum_address(), "USDC", 6, 1)
    weth = AssetIdentifier(chain_id.value, generate_random_ethereum_address(), "WETH", 18, 2)
    weth_usdc = TradingPairIdentifier(
        weth,
        usdc,
        generate_random_ethereum_address(),
        mock_exchange.address,
        internal_id=random.randint(1, 1000),
        internal_exchange_id=mock_exchange.exchange_id,
        fee=0.0030,
    )

    usdc_reserve = generate_lending_reserve(usdc, chain_id, 1)
    weth_reserve = generate_lending_reserve(weth, chain_id, 2)

    _, lending_candle_universe = generate_lending_universe(
        time_bucket,
        start_at,
        candle_end_at,
        reserves=[usdc_reserve, weth_reserve],
        aprs={
            "supply": 2,
            "variable": 5,
        }
    )

    candles = generate_ohlcv_candles(
        time_bucket,
        start_at,
        candle_end_at,
        start_price=1800,
        pair_id=weth_usdc.internal_id,
        exchange_id=mock_exchange.exchange_id,
    )

    return create_synthetic_single_pair_universe(
        candles=candles,
        chain_id=chain_id,
        exchange=mock_exchange,
        time_bucket=time_bucket,
        pair=weth_usdc,
        lending_candles=lending_candle_universe,
    )


def decide_trades(input: StrategyInput) -> List[TradeExecution]:
    """A simple strategy that puts all in to our lending reserve."""

    position_manager = input.get_position_manager()
    pair = input.get_default_pair()
    cash = input.state.portfolio.get_cash()

    trades = position_manager.open_credit_supply_position_for_reserves(
        cash * 0.98,
    )

    return trades


def test_increase_credit(
    tmp_path,
    strategy_universe,
):
    """Runs a strategy that opens and increases a credit position over time."""

    def create_indicators(timestamp: datetime.datetime, parameters: StrategyParameters, strategy_universe: TradingStrategyUniverse, execution_context: ExecutionContext):
        # No indicators needed
        return IndicatorSet()

    class MyParameters:
        initial_cash = 10_000
        cycle_duration = CycleDuration.cycle_1d

    indicator_storage = DiskIndicatorStorage(tmp_path, strategy_universe.get_cache_key())

    deposit_simulator = DepositSimulator()

    # Run the test
    result = run_backtest_inline(
        client=None,
        decide_trades=decide_trades,
        create_indicators=create_indicators,
        universe=strategy_universe,
        engine_version="0.5",
        parameters=StrategyParameters.from_class(MyParameters),
        mode=ExecutionMode.unit_testing,
        indicator_storage=indicator_storage,
        execution_test_hook=deposit_simulator,
    )
