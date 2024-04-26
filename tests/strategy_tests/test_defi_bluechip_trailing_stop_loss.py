"""Run DeFi bluechip momentum strategy for some backtesting cycles and see any analysis code does not break.."""
import datetime
import logging
import os

from pathlib import Path
from typing import Tuple

import pandas as pd
import pytest
import plotly.graph_objects as go
from IPython.core.display_functions import display

from tradeexecutor.analysis.alpha_model_analyser import create_alpha_model_timeline_all_assets, render_alpha_model_plotly_table, analyse_alpha_model_weights, \
    create_pair_weight_analysis_summary_table
from tradeexecutor.analysis.fee_analyser import analyse_trading_fees, create_pair_trading_fee_summary_table
from tradeexecutor.backtest.backtest_runner import run_backtest, setup_backtest
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.state.state import State
from tradeexecutor.statistics.summary import calculate_summary_statistics
from tradeexecutor.strategy.execution_context import ExecutionMode
from tradeexecutor.analysis.trade_analyser import build_trade_analysis
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.visual.single_pair import visualise_single_pair, visualise_single_pair_positions_with_duration_and_slippage
from tradingstrategy.chain import ChainId



pytestmark = pytest.mark.skipif(
    (os.environ.get("TRADING_STRATEGY_API_KEY") is None or os.environ.get("SKIP_SLOW_TEST") is not None),
    reason="Set TRADING_STRATEGY_API_KEY environment variable to run this test, and SKIP_SLOW_TEST must not be set"
)


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


@pytest.fixture(scope="module")
def strategy_path() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "test_only", "defi-bluechip-momentum-trailing-sl.py"))



@pytest.fixture(scope="module")
def end_at() -> datetime.datetime:
    """What's the end of the backtesting date."""
    return  datetime.datetime(2022, 6, 1)


@pytest.fixture(scope="module")
def backtest_result(
        logger: logging.Logger,
        strategy_path: Path,
        persistent_test_client,
        end_at,
) -> Tuple[State, TradingStrategyUniverse, dict]:
    """Runs DeFi blugchip strategy for one day and checks that various analytics functions work on the resulting state.

    - Check the strategy does not crash

    - Run for one year
    """

    client = persistent_test_client

    # Run backtest over 6 months, daily
    setup = setup_backtest(
        strategy_path,
        start_at=datetime.datetime(2021, 6, 1),
        end_at=end_at,
        initial_deposit=10_000,
    )

    state, universe, debug_dump = run_backtest(setup, client)

    return state, universe, debug_dump


@pytest.fixture()
def state(backtest_result) -> State:
    return backtest_result[0]


@pytest.fixture()
def universe(backtest_result) -> TradingStrategyUniverse:
    return backtest_result[1]


@pytest.fixture()
def debug_dump(backtest_result):
    return backtest_result[2]


def test_trailing_stop_loss_check(
    state,
    debug_dump,
    end_at,
    ):
    """See we got some trailing stop losses."""

    # Check we got trailing stop losses triggered
    stop_loss_positions = [p for p in state.portfolio.get_all_positions() if p.is_stop_loss()]
    trailing_stop_loss_positions = [p for p in state.portfolio.get_all_positions() if p.is_trailing_stop_loss()]
    assert len(stop_loss_positions) == 93
    assert len(trailing_stop_loss_positions) == 93


