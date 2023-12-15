"""See tgat long-short-momentum.py does not crash."""
import datetime
import logging
import os

from pathlib import Path

import pytest

from tradeexecutor.analysis.trade_analyser import build_trade_analysis
from tradeexecutor.backtest.backtest_runner import run_backtest, setup_backtest
from tradeexecutor.backtest.tearsheet import export_backtest_report
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.state.state import State
from tradeexecutor.strategy.universe_model import UniverseOptions


# https://docs.pytest.org/en/latest/how-to/skipping.html#skip-all-test-functions-of-a-class-or-module
pytestmark = pytest.mark.skipif(os.environ.get("TRADING_STRATEGY_API_KEY") is None, reason="Set TRADING_STRATEGY_API_KEY environment variable to run this test")


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


@pytest.fixture()
def strategy_path() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "test_only", "polygon-eth-spot-short-bb.py"))


def test_sport_short_strategy(
    strategy_path,
    logger: logging.Logger,
    persistent_test_client,
    ):
    """Check the strategy does not crash.

    - Run spot + short strategy for few cycles

    - Serialise output

    - Write stats

    See that any of the steps do not crash.
    """

    client = persistent_test_client

    # Run backtest over 6 months, daily
    setup = setup_backtest(
        strategy_path,
        client=persistent_test_client,
    )

    assert setup.universe is not None

    try:
        state, universe, debug_dump = run_backtest(setup, client)
    except Exception as e:
        logger.error("Backtest failed:\n%s", e)
        logger.exception(e)
        raise e

    # See our state can be serialised and deserialised
    dumped_state = state.to_json_safe()
    state2 = State.read_json_blob(dumped_state)
    assert state.name == state2.name

    analysis = build_trade_analysis(state.portfolio)
    summary = analysis.calculate_summary_statistics(universe.data_universe.time_bucket, state)
    df = summary.to_dataframe()
    str(df)

    # See reporter does not crash
    export_backtest_report(
        state,
        universe,
        output_notebook=Path("/dev/null"),
        output_html=Path("/dev/null"),
    )

