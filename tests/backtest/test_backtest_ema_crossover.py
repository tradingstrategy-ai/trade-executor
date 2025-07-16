"""Test EMA cross-over strategy.

To run:

.. code-block:: shell

    export TRADING_STRATEGY_API_KEY="secret-token:tradingstrategy-6ce98...."
    export BNB_CHAIN_JSON_RPC="https://bsc-dataseed.binance.org/"
    pytest --log-cli-level=info -s -k test_bnb_chain_16h_momentum

"""
import datetime
import logging
import os
import pandas as pd

from pathlib import Path

import pytest
from tradeexecutor.backtest.backtest_runner import run_backtest, setup_backtest
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.strategy.cycle import CycleDuration
from tradingstrategy.timebucket import TimeBucket


# https://docs.pytest.org/en/latest/how-to/skipping.html#skip-all-test-functions-of-a-class-or-module
pytestmark = pytest.mark.skipif(os.environ.get("TRADING_STRATEGY_API_KEY") is None, reason="Set TRADING_STRATEGY_API_KEY environment variable to run this test")


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


@pytest.fixture()
def strategy_path() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.join(os.path.dirname(__file__), "../..", "strategies", "ema-crossover-long-only-no-stop-loss.py"))


@pytest.mark.skip(reason="Uses too much RAM, legacy test, disabled")
def test_ema_crossover_real_data(
    strategy_path,
    logger: logging.Logger,
    persistent_test_client,
    ):
    """Check that EMA crossover strategy against real data."""

    client = persistent_test_client

    # Run backtest over 6 months, daily
    setup = setup_backtest(
        strategy_path,
        start_at=datetime.datetime(2021, 6, 1),
        end_at=datetime.datetime(2022, 1, 1),
        initial_deposit=10_000,
        cycle_duration=CycleDuration.cycle_1d,  # Override to use daily cycles to speed up the test
        candle_time_frame=TimeBucket.d1,  # Override to use daily data to speed up the test
    )

    state, universe, debug_dump = run_backtest(setup, client)

    start, end = state.get_strategy_start_and_end()
    assert start == pd.Timestamp('2021-06-01 00:00:00')
    assert end == pd.Timestamp('2021-12-31 00:00:00')

    assert len(debug_dump) == 215

    # TODO: Not sure if we have any meaningful results to verify

"""The tests commented out for now, since the automated backtesting range was removed
due to errors in live strategies"""

@pytest.mark.skip(reason="The logic is broken and breaks live trading startup")
def test_start_end_automation(
    strategy_path,
    logger: logging.Logger,
    persistent_test_client,
    ):
    """Check that EMA crossover strategy against real data."""

#     client = persistent_test_client

#     # Run backtest over 6 months, daily
#     setup = setup_backtest(
#         strategy_path,
#         initial_deposit=10_000,
#         cycle_duration=CycleDuration.cycle_30d,  # Override to use monthly cycles to speed up the test
#         candle_time_frame=TimeBucket.d30,  # Override to use monthly data to speed up the test
#     )

#     state, universe, debug_dump = run_backtest(setup, client)

#     start, end = state.get_strategy_start_and_end()

#     assert start == pd.Timestamp('2021-04-01 00:00:00')
#     assert end >= pd.Timestamp('2023-06-20 00:00:00')  # end is dynamic

#     assert len(debug_dump) >= 28  # since end is dynamic, we can't know the exact number of cycles


@pytest.mark.skip(reason="ExecutionLoop.set_backtest_start_and_end is bad code")
def test_minimum_lookback_data_range(
    strategy_path,
    logger: logging.Logger,
    persistent_test_client,
    ):
    """Check that EMA crossover strategy against real data."""

#     client = persistent_test_client

#     # Run backtest over 6 months, daily
#     setup = setup_backtest(
#         strategy_path,
#         initial_deposit=10_000,
#         cycle_duration=CycleDuration.cycle_1d,  # Override to use monthly cycles to speed up the test
#         candle_time_frame=TimeBucket.d1,  # Override to use monthly data to speed up the test
#         minimum_data_lookback_range=datetime.timedelta(days=23),
#     )

#     state, universe, debug_dump = run_backtest(setup, client)

#     start, end = state.get_strategy_start_and_end()

#     # both start and end are dynamic
#     assert start >= pd.Timestamp('2023-07-03 00:00:00')
#     assert end >= pd.Timestamp('2023-07-26 00:00:00')
#     assert end - start == pd.Timedelta('23 days 00:00:00')

#     assert len(debug_dump) == 24
    