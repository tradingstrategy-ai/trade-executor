"""Run DeFi bluechip momentum strategy for some backtesting cycles to see the code does not break."""
import datetime
import logging
import os

from pathlib import Path

import pandas as pd
import pytest

from tradeexecutor.analysis.alpha_model_analyser import create_alpha_model_timeline_all_assets, render_alpha_model_plotly_table
from tradeexecutor.backtest.backtest_runner import run_backtest, setup_backtest
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.statistics.summary import calculate_summary_statistics
from tradeexecutor.strategy.execution_context import ExecutionMode
from tradeexecutor.analysis.trade_analyser import build_trade_analysis

# https://docs.pytest.org/en/latest/how-to/skipping.html#skip-all-test-functions-of-a-class-or-module
pytestmark = pytest.mark.skipif(os.environ.get("TRADING_STRATEGY_API_KEY") is None, reason="Set TRADING_STRATEGY_API_KEY environment variable to run this test")


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


@pytest.fixture()
def strategy_path() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "test_only", "defi-bluechip-momentum.py"))


@pytest.mark.skipif(os.environ.get("SKIP_SLOW_TEST"), reason="Slow tests skipping enabled")
def test_defi_bluechip(
    strategy_path,
    logger: logging.Logger,
    persistent_test_client,
    ):
    """Check the strategy does not crash."""

    client = persistent_test_client

    end_at = datetime.datetime(2022, 6, 1)

    # Run backtest over 6 months, daily
    setup = setup_backtest(
        strategy_path,
        start_at=datetime.datetime(2021, 6, 1),
        end_at=end_at,
        initial_deposit=10_000,
    )

    state, universe, debug_dump = run_backtest(setup, client)

    # Do some sanity checks for the data
    assert len(debug_dump) > 25
    assert len(list(state.portfolio.get_all_trades())) > 25

    summary = calculate_summary_statistics(
        state,
        ExecutionMode.unit_testing_trading,
        now_=end_at,
    )

    assert summary.last_trade_at > datetime.datetime(2022, 1, 1)

    # We don't lose all the money
    assert summary.current_value > 1000

    # Create a table where we have per asset column of taken positions,
    # see that these functions do not crash
    df = create_alpha_model_timeline_all_assets(state, universe)
    figure, table = render_alpha_model_plotly_table(df)
    assert figure is not None
    assert table is not None

    # See summary calculations do not crash
    analysis = build_trade_analysis(state.portfolio)
    summary = analysis.calculate_summary_statistics()
    df = summary.to_dataframe()
    assert isinstance(df, pd.DataFrame)
