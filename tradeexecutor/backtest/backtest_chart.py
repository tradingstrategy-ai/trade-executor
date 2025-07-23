"""Test chart subsystem in backtesting.
"""

import os
from pathlib import Path

import pandas as pd
import pytest
from plotly.graph_objs import Figure

from tradeexecutor.analysis.credit import calculate_yield_metrics, YieldType, display_vault_position_table
from tradeexecutor.analysis.multipair import analyse_multipair
from tradeexecutor.backtest.backtest_module import run_backtest_for_module
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.strategy.chart.renderer import ChartBacktestRenderingSetup
from tradeexecutor.strategy.execution_context import unit_test_execution_context
from tradeexecutor.strategy.pnl import calculate_pnl
from tradeexecutor.visual.position import calculate_position_timeline, visualise_position


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


@pytest.fixture()
def strategy_file() -> Path:
    return Path(os.path.join(os.path.dirname(__file__), "../..", "strategies", "test_only", "bnb-local-high.py"))



@pytest.mark.slow_test_group
def test_backtest_charts(
    persistent_test_client,
    strategy_file,
    logger,
):
    """Run a backtest and render all charts."""
    client = persistent_test_client

    # Run backtest over 6 months, daily
    result = run_backtest_for_module(
        strategy_file=strategy_file,
        cache_path=client.transport.cache_path,
        execution_context=unit_test_execution_context,
    )

    mod = result.strategy_module
    assert mod.create_charts

    charts = mod.create_charts(
        timestamp=None,
        parameters=mod.parameters,
        strategy_universe=result.universe,
        execution_context=unit_test_execution_context,
    )

    # Attempt to render all charts defined in the strategy module.
    # There are a lot of them.
    chart_renderer = ChartBacktestRenderingSetup(
        registry=charts,
        strategy_input_indicators=result.indicators,
        backtest_end_at=mod.parameters.backtest_end,
    )

    for func in charts.by_function.keys():
        result = chart_renderer.render(func)
