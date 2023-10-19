"""Check that we can still read and manipulate legacy state files.

New variables and features get added to state files constantly.
Check that we can read old files.
"""
import datetime
import os

import pytest

import pandas as pd

from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.monkeypatch.dataclasses_json import patch_dataclasses_json
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.statistics import Statistics, calculate_naive_profitability
from tradeexecutor.statistics.core import calculate_statistics
from tradeexecutor.statistics.summary import calculate_summary_statistics
from tradeexecutor.strategy.execution_context import ExecutionMode
from tradeexecutor.strategy.reverse_universe import reverse_trading_universe_from_state
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


@pytest.fixture(scope="module")
def state() -> State:
    """Read a random old data dump."""
    f = os.path.join(os.path.dirname(__file__), "legacy-state-dump.json")
    return State.from_json(open(f, "rt").read())


@pytest.fixture(scope="module")
def state2() -> State:
    """Newer legacy data dump from a different strategy."""
    f = os.path.join(os.path.dirname(__file__), "legacy-state-dump-2.json")
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
    assert profitability == pytest.approx(0.00964009122330515, rel=0.05)


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
        legacy_workarounds=True,
    )

    assert summary.calculated_at


def test_legacy_calculate_all_statistics(state: State):
    """Calculate state embedded statistics.

    Check we do not get any exceptions like NoneError of DivisionByZero.
    """
    portfolio = state.portfolio
    clock = datetime.datetime(2023, 1, 31)

    # Calculate statistics in both modes
    execution_mode = ExecutionMode.real_trading
    calculate_statistics(clock, portfolio, execution_mode)

    execution_mode = ExecutionMode.backtesting
    calculate_statistics(clock, portfolio, execution_mode)


def test_legacy_visualisation(state: State):
    """See that legacy visualisation data can be accessed.
    """
    assert state.visualisation.get_total_points() == 150


def test_empty_state_calculate_all_statistics():
    """Calculate all statistics on an empty state

    Check we do not get any exceptions like NoneError of DivisionByZero.
    """

    state = State()

    portfolio = state.portfolio
    clock = datetime.datetime(2023, 1, 31)


    # Calculate statistics in both modes
    execution_mode = ExecutionMode.real_trading
    new_stats = calculate_statistics(clock, portfolio, execution_mode)

    execution_mode = ExecutionMode.backtesting
    new_stats = calculate_statistics(clock, portfolio, execution_mode)


def test_reverse_trading_universe_from_state(
        state: State,
        persistent_test_client: Client,
):
    """See that we can load the pair and candle data for a historical state."""

    client = persistent_test_client
    universe = reverse_trading_universe_from_state(
        state,
        client,
        TimeBucket.d1,
    )

    assert universe.reserve_assets == {AssetIdentifier(chain_id=137, address='0x2791bca1f2de4661ed88a30c99a7a9449aa84174', token_symbol='USDC', decimals=6, internal_id=None, info_url=None)}
    assert len(universe.data_universe.exchanges) == 1
    assert universe.data_universe.pairs.get_count() == 1
    assert universe.data_universe.pairs.get_count() == 1
    start, end = universe.data_universe.candles.get_timestamp_range()
    assert start == pd.Timestamp('2023-01-17 00:00:00')
    assert end == pd.Timestamp('2023-02-03 00:00:00')


def test_legacy_calculate_all_statistics_dump_2(state2: State):
    """Calculate state embedded statistics.

    Use a newer legacy data dump.
    """
    portfolio = state2.portfolio
    clock = datetime.datetime(2023, 10, 10)

    # Calculate statistics in both modes
    execution_mode = ExecutionMode.real_trading
    calculate_statistics(clock, portfolio, execution_mode)

    execution_mode = ExecutionMode.backtesting
    calculate_statistics(clock, portfolio, execution_mode)
