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
from tradeexecutor.strategy.execution_context import ExecutionMode, unit_test_execution_context
from tradeexecutor.analysis.trade_analyser import build_trade_analysis
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.visual.equity_curve import calculate_compounding_realised_trading_profitability
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
    return Path(os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "test_only", "defi-bluechip-momentum.py"))



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
def strategy_universe(universe) -> TradingStrategyUniverse:
    return universe


@pytest.fixture()
def debug_dump(backtest_result):
    return backtest_result[2]


def test_summary_statistics(
    state,
    debug_dump,
    end_at,
    ):
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


def test_alpha_model_timeline(
    state,
        strategy_universe,
    ):
    """Create a table where we have per asset column of taken positions, see that these functions do not crash"""
    df = create_alpha_model_timeline_all_assets(state, strategy_universe)
    figure, table = render_alpha_model_plotly_table(df)
    assert figure is not None
    assert table is not None


def test_trade_analysis(
    state,
    strategy_universe,
    ):
    """See trade analysis calculations do not crash."""
    analysis = build_trade_analysis(state.portfolio)
    summary = analysis.calculate_summary_statistics()
    df = summary.to_dataframe(format_headings=False)
    assert isinstance(df, pd.DataFrame)

    crv_usd = strategy_universe.get_pair_by_human_description((ChainId.ethereum, "uniswap-v2", "CRV", "WETH"))

    fig = visualise_single_pair(
        state,
        unit_test_execution_context,
        strategy_universe.data_universe.candles,
        pair_id=crv_usd.internal_id,
    )
    assert isinstance(fig, go.Figure)

    candles = strategy_universe.data_universe.candles.get_candles_by_pair(crv_usd.internal_id)
    fig = visualise_single_pair_positions_with_duration_and_slippage(
        state,
        unit_test_execution_context,
        candles=candles,
        pair_id=crv_usd.internal_id,
    )
    assert isinstance(fig, go.Figure)

    profit1 = calculate_compounding_realised_trading_profitability(state).iloc[-1]
    profit2 = summary.return_percent
    # TODO: This is don't match in alpha model strategies, need to investigate
    # assert profit1 == pytest.approx(profit2)


def test_alpha_model_weight_analysis(
    state,
        strategy_universe,
    ):
    """See alpha model weight analysis does not crash."""

    weight_analysis = analyse_alpha_model_weights(state, strategy_universe)
    assert len(weight_analysis) == 189

    pair_weight_summary = create_pair_weight_analysis_summary_table(weight_analysis)
    assert pair_weight_summary.iloc[0].name == "AAVE-WETH"
    # normalised_weight  max     1.000000
    #                    mean    0.097497
    #                    min     0.000000
    # signal             max     0.469029
    #                    mean    0.135494
    #                    min     0.004682
    assert pair_weight_summary.iloc[0]["normalised_weight"]["max"] == 1.0
    #display(pair_weight_summary)


def test_trading_fee_analysis(
    state,
    ):
    """See trading fee analysis does not crash."""
    fee_analysis = analyse_trading_fees(state)
    pair_fee_summary = create_pair_trading_fee_summary_table(fee_analysis)
    assert pair_fee_summary.iloc[0].name == ("AAVE-WETH", "buy")

