"""Visualise grid search results.

- Different visualisation tools to compare grid search results
"""
from typing import List

import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objs import Figure, Scatter

from tradeexecutor.analysis.grid_search import _get_hover_template
from tradeexecutor.analysis.multi_asset_benchmark import get_benchmark_data
from tradeexecutor.backtest.grid_search import GridSearchResult
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.visual.benchmark import visualise_equity_curves
from tradingstrategy.types import USDollarAmount


def visualise_single_grid_search_result_benchmark(
    result: GridSearchResult,
    strategy_universe: TradingStrategyUniverse,
    initial_cash: USDollarAmount | None = None,
) -> go.Figure:
    """Draw one equity curve from grid search results.

    - Compare the equity curve againt buy and hold assets from the trading universe

    - Used to plot "the best" equity curve

    - Use :func:`find_best_grid_search_results` to find some equity curves.

    See also

    - :py:func:`visualise_grid_search_equity_curves`

    - :py:func:`tradeexecutor.visual.benchmark.visualise_equity_curves`

    - :py:func:`tradeexecutor.analysis.multi_asset_benchmark.get_benchmark_data`

    - :py:func:`tradeexecutor.analysis.grid_search.find_best_grid_search_results`

    Example:

    .. code-block:: python

        from tradeexecutor.analysis.grid_search import find_best_grid_search_results
        from tradeexecutor.visual.grid_search import visualise_single_grid_search_result_benchmark

        # Show the equity curve of the best grid search performer
        best_results = find_best_grid_search_results(grid_search_results)
        fig = visualise_single_grid_search_result_benchmark(best_results.cagr[0], strategy_universe)
        fig.show()

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
    equity.attrs["name"] = result.get_label()

    benchmarks = get_benchmark_data(
        strategy_universe,
        cumulative_with_initial_cash=result.initial_cash or initial_cash,
    )

    benchmark_series = [v for k, v in benchmarks.items()]

    fig = visualise_equity_curves(
        [equity] + benchmark_series
    )

    return fig


def visualise_grid_search_equity_curves(
    results: List[GridSearchResult],
    name: str | None = None,
    benchmark_indexes: pd.DataFrame | None = None,
    height=1200,
    colour="rgba(160, 160, 160, 0.5)",
    log_y=False,
) -> Figure:
    """Draw multiple equity curves in the same chart.

    - See how all grid searched strategies work

    - Benchmark against buy and hold of various assets

    - Benchmark against hold all cash

    .. note ::

        Only good up to ~hundreds results. If more than thousand result, rendering takes too long time.

    Example that draws equity curves of a grid search results.

    .. code-block:: python

        from tradeexecutor.visual.grid_search import visualise_grid_search_equity_curves
        from tradeexecutor.analysis.multi_asset_benchmark import get_benchmark_data

        # Automatically create BTC and ETH buy and hold benchmark if present
        # in the trading universe
        benchmark_indexes = get_benchmark_data(
            strategy_universe,
            cumulative_with_initial_cash=ShiftedStrategyParameters.initial_cash,
        )

        fig = visualise_grid_search_equity_curves(
            grid_search_results,
            name="8h clock shift, stop loss added and adjusted momentum",
            benchmark_indexes=benchmark_indexes,
            log_y=False,
        )
        fig.show()

    :param results:
        Results from the grid search.

    :param benchmark_indexes:
        List of other asset price series displayed on the timeline besides equity curve.

        DataFrame containing multiple series.

        - Asset name is the series name.
        - Setting `colour` for `pd.Series.attrs` allows you to override the colour of the index

    :param height:
        Chart height in pixels

    :param start_at:
        When the backtest started

    :param end_at:
        When the backtest ended

    :param additional_indicators:
        Additional technical indicators drawn on this chart.

        List of indicator names.

        The indicators must be plotted earlier using `state.visualisation.plot_indicator()`.

        **Note**: Currently not very useful due to Y axis scale

    :param log_y:
        Use logarithmic Y-axis.

        Because we accumulate larger treasury over time,
        the swings in the value will be higher later.
        We need to use a logarithmic Y axis so that we can compare the performance
        early in the strateg and late in the strategy.

    """

    if name is None:
        name = "Grid search equity curve comparison"

    fig = Figure()

    for result in results:
        curve = result.equity_curve
        label = result.get_label()
        template =_get_hover_template(result)
        scatter = Scatter(
            x=curve.index,
            y=curve,
            mode="lines",
            name="",  # Hides hover legend, use hovertext only
            line=dict(color=colour),
            showlegend=False,
            hovertemplate=template,
            hovertext=None,
        )
        fig.add_trace(scatter)

    if benchmark_indexes is not None:
        for benchmark_name, curve in benchmark_indexes.items():
            benchmark_colour = curve.attrs.get("colour", "black")
            scatter = Scatter(
                x=curve.index,
                y=curve,
                mode="lines",
                name=benchmark_name,
                line=dict(color=benchmark_colour),
                showlegend=True,
            )
            fig.add_trace(scatter)

    fig.update_layout(title=f"{name}", height=height)
    if log_y:
        fig.update_yaxes(title="Value $ (logarithmic)", showgrid=False, type="log")
    else:
        fig.update_yaxes(title="Value $", showgrid=False)
    fig.update_xaxes(rangeslider={"visible": False})

    # Move legend to the bottom so we have more space for
    # time axis in narrow notebook views
    # https://plotly.com/python/legend/
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    return fig
