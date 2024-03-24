"""Visualise grid search results.

- Different visualisation tools to compare grid search results
"""

import plotly.graph_objects as go

from tradeexecutor.analysis.multi_asset_benchmark import get_benchmark_data
from tradeexecutor.backtest.grid_search import GridSearchResult
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.visual.benchmark import visualise_equity_curves
from tradingstrategy.types import USDollarAmount


def visualise_grid_search_result_benchmark(
    result: GridSearchResult,
    strategy_universe: TradingStrategyUniverse,
    initial_cash: USDollarAmount | None = None,
) -> go.Figure:
    """Draw one equity curve from grid search results.

    - Compare the equity curve againt buy and hold assets from the trading universe

    - Use :func:`find_best_grid_search_results` to find some equity curves.

    See also

    - :py:func:`tradeexecutor.visual.benchmark.visualise_equity_curves`

    - :py:func:`tradeexecutor.analysis.multi_asset_benchmark.get_benchmark_data`

    :param result:
        Picked grid search result
    :param strategy_universe:
        Used to get benechmark indexes
    :param initial_cash:
        Not needed. Automatically filled in by grid search.

        Legacy param.
    :return:
        Plotly figure
    """

    assert isinstance(result, GridSearchResult)
    assert isinstance(strategy_universe, TradingStrategyUniverse)

    # Get daily returns
    equity = result.equity_curve

    benchmarks = get_benchmark_data(
        strategy_universe,
        cumulative_with_initial_cash=result.initial_cash or initial_cash,
    )

    benchmark_series = [v for k, v in benchmarks.items()]

    fig = visualise_equity_curves(
        [equity] + benchmark_series
    )

    return fig
