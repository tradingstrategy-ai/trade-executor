"""Run Pancake momentum v2 strategy some backtesting cycles to see the code does not break."""
import datetime
import logging
import os

import pandas as pd

from pathlib import Path

import pytest

from tradeexecutor.analysis.trade_analyser import build_trade_analysis
from tradeexecutor.backtest.backtest_runner import run_backtest, setup_backtest
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.strategy.cycle import CycleDuration
from tradingstrategy.timebucket import TimeBucket


CI = os.environ.get("CI")

# https://docs.pytest.org/en/latest/how-to/skipping.html#skip-all-test-functions-of-a-class-or-module
pytestmark = pytest.mark.skipif(os.environ.get("TRADING_STRATEGY_API_KEY") is None, reason="Set TRADING_STRATEGY_API_KEY environment variable to run this test")


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


@pytest.fixture()
def strategy_path() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "test_only", "pancake-momentum-weekly.py"))


@pytest.mark.slow_test_group
@pytest.mark.skipif(os.environ.get("SKIP_SLOW_TEST") or CI, reason="Slow tests skipping enabled, also skipped on CI")
def test_pancake_momentum_v2(
    strategy_path,
    logger: logging.Logger,
    persistent_test_client,
    ):
    """Check the strategy does not crash."""

    client = persistent_test_client

    # Override the default strategy settings to speed up
    # the tests
    # module_overrides = {
    #    "momentum_lookback_period": pd.Timedelta(days=7),
    #    "candle_time_bucket": TimeBucket.d7,
    #    "trading_strategy_cycle": CycleDuration.cycle_7d,
    #}

    # Run backtest over 6 months, daily
    setup = setup_backtest(
        strategy_path,
        start_at=datetime.datetime(2021, 6, 1),
        end_at=datetime.datetime(2021, 7, 1),
        initial_deposit=10_000,
    )

    state, universe, debug_dump = run_backtest(setup, client)



