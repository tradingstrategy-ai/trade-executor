"""Test chart subsystem in backtesting.
"""

import os
from pathlib import Path

import pytest

from tradeexecutor.backtest.backtest_module import run_backtest_for_module
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.strategy.chart.definition import ChartParameters
from tradeexecutor.strategy.chart.renderer import ChartBacktestRenderingSetup, render_for_web
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.execution_context import unit_test_execution_context


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
        mod_overrides={
            # Set in 1s to live trading test, but breaks backtest valuation
            "trading_strategy_cycle": CycleDuration.cycle_1h,
        }
    )

    mod = result.strategy_module
    assert mod.create_charts

    charts = mod.create_charts(
        timestamp=None,
        parameters=mod.parameters,
        strategy_universe=result.strategy_universe,
        execution_context=unit_test_execution_context,
    )

    # Attempt to render all charts defined in the strategy module.
    # There are a lot of them.
    chart_renderer = ChartBacktestRenderingSetup(
        registry=charts,
        strategy_input_indicators=result.indicators,
        backtest_end_at=mod.parameters.backtest_end,
        state=result.state,
    )

    # Speed up rendering
    parameters = ChartParameters(width=256, height=256)

    for func in charts.by_function.keys():
        result = chart_renderer.render(func)
        assert result is not None
        rendered = render_for_web(parameters, result, func)
        assert not rendered.error, f"Error rendering chart {func}: {rendered.error}"
