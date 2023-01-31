"""Check that we can still read and manipulate legacy state files.

New variables and features get added to state files constantly.
Check that we can read old files.
"""
import datetime
import logging
import os
from pathlib import Path

import pytest

import pandas as pd

from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel
from tradeexecutor.backtest.backtest_runner import run_backtest, setup_backtest_for_universe
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.statistics import Statistics, calculate_naive_profitability
from tradeexecutor.statistics.core import calculate_statistics
from tradeexecutor.statistics.summary import calculate_summary_statistics
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.execution_context import ExecutionMode
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange, generate_simple_routing_model
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


@pytest.fixture(scope="module")
def state() -> State:
    """Read a random old data dump."""
    f = os.path.join(os.path.dirname(__file__), "legacy-state-dump.json")
    return State.from_json(open(f, "rt").read())


def test_legacy_get_statistics_as_dataframe(state: State):
    """We can convert any of our statistics to dataframes"""

    stats: Statistics = state.stats

    assert len(stats.portfolio) > 0, "Portfolio progress over the history statistics were not available"

    # Create time series of portfolio "total_equity" over its lifetime
    s = stats.get_portfolio_statistics_dataframe("total_equity")

    assert isinstance(s.index, pd.DatetimeIndex)
    assert s.index[0] == pd.Timestamp('2023-01-23 00:00:00')
    assert s.index[-1] == pd.Timestamp('2023-01-30 00:00:00')

    # First value by date
    assert s.loc[pd.Timestamp('2023-01-23 00:00:00')] == pytest.approx(2.380726, rel=0.05)

    # Last value by index
    assert s.iloc[-1] == pytest.approx(2.404366, rel=0.05)

    # Number of positions
    assert len(s) == 8


def test_legacy_calculate_profitability_90_days(state: State):
    """Calculate strategy profitability for last 90 days"""

    stats: Statistics = state.stats

    # Create time series of portfolio "total_equity" over its lifetime
    s = stats.get_portfolio_statistics_dataframe("total_equity")
    profitability, time_window = calculate_naive_profitability(s, look_back=pd.Timedelta(days=90))

    # Calculate last 90 days
    assert profitability == pytest.approx(0.009929744120070886, rel=0.05)


def test_legacy_calculate_all_summary_statistics(state: State):
    """Calculate all summary statistics.

    Used on the summary card etc.
    """

    # Set "last 90 days" to the end of backtest data
    now_ = pd.Timestamp(datetime.datetime(2021, 12, 31, 0, 0))

    summary = calculate_summary_statistics(
        state,
        ExecutionMode.unit_testing_trading,
        now_=now_,
    )

    assert summary.calculated_at
    assert not summary.enough_data

    datapoints = summary.performance_chart_90_days
    assert len(datapoints) == 7


def test_legacy_calculate_all_statistics(state: State):
    """Calculate state embedded statistics,"""
    portfolio = state.portfolio
    clock = datetime.datetime(2023, 1, 31)

    # Calculate statistics in both modes
    execution_mode = ExecutionMode.real_trading
    new_stats = calculate_statistics(clock, portfolio, execution_mode)

    execution_mode = ExecutionMode.backtesting
    new_stats = calculate_statistics(clock, portfolio, execution_mode)



