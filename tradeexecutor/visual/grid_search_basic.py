"""Visualise grid search results.

- Different visualisation tools to compare grid search results
"""
from typing import List, Callable

import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objs import Figure, Scatter

from tradeexecutor.analysis.curve import CurveType, DEFAULT_BENCHMARK_COLOURS
from tradeexecutor.analysis.grid_search import _get_hover_template, order_grid_search_results_by_metric
from tradeexecutor.analysis.multi_asset_benchmark import get_benchmark_data
from tradeexecutor.backtest.grid_search import GridSearchResult
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.visual.benchmark import visualise_equity_curves
from tradingstrategy.types import USDollarAmount


def visualise_single_grid_search_result_benchmark(
    result: GridSearchResult,
    strategy_universe: TradingStrategyUniverse,
    initial_cash: USDollarAmount | None = None,
    name="Picked search result",
    log_y=False,
    asset_count=3,
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

    :param name:
        Chart title

    :param initial_cash:
        Not needed. Automatically filled in by grid search.

        Legacy param.

    :param asset_count:
        Draw this many comparison buy-and-hold curves from well-known assets.

    :return:
        Plotly figure
    """

    assert isinstance(result, GridSearchResult)
    assert isinstance(strategy_universe, TradingStrategyUniverse)

    # Get daily returns
    equity = result.equity_curve
    equity.attrs["name"] = result.get_truncated_label()
    equity.attrs["curve"] = CurveType.equity
    equity.attrs["colour"] = DEFAULT_BENCHMARK_COLOURS["Strategy"]

    if result.state is not None:
        start_at = result.state.get_trading_time_range()[0]
    else:
        start_at = equity.index[0]

    benchmarks = get_benchmark_data(
        strategy_universe,
        cumulative_with_initial_cash=initial_cash or getattr(result, "initial_cash", None),  # Legacy support hack
        start_at=start_at,
        max_count=asset_count,
    )

    benchmark_series = [v for k, v in benchmarks.items()]

    fig = visualise_equity_curves(
        [equity] + benchmark_series,
        name=name,
        log_y=log_y,
        start_at=start_at,
    )

    return fig


def visualise_grid_search_equity_curves(
    results: List[GridSearchResult],
    name: str | None = None,
    benchmark_indexes: pd.DataFrame | None = None,
    height=1200,
    colour=None,
    log_y=False,
    alpha=0.7,
    label_func: Callable = None,
    annotation_xshift=200,
) -> Figure:
    """Draw multiple equity curves in the same chart.

    - See how all grid searched strategies work

    - Benchmark against buy and hold of various assets

    - Benchmark against hold all cash

    TODO: A lot of parameter descriptions are not up-to-date.

    .. note ::

        Only good up to ~hundreds results. If more than thousand result, rendering takes too long time.

    Example that draws equity curve comparison with custom labels:

    .. code-block:: python

        from tradeexecutor.visual.grid_search_basic import visualise_grid_search_equity_curves
        from tradeexecutor.analysis.multi_asset_benchmark import get_benchmark_data

        # Automatically create BTC and ETH buy and hold benchmark if present
        # in the trading universe
        benchmark_indexes = get_benchmark_data(
            strategy_universe,
            cumulative_with_initial_cash=Parameters.initial_cash,
        )

        fig = visualise_grid_search_equity_curves(
            grid_search_results,
            benchmark_indexes=benchmark_indexes,
            log_y=False,
            label_func=lambda x: f"Decision cycle {x.get_parameter('cycle_duration').value}, CARG {x.get_cagr():.0%}, Sharpe {x.get_sharpe():.1f}",
        )
        fig.show()

    Example that draws equity curves of a grid search results.

    .. code-block:: python

        from tradeexecutor.visual.grid_search_basic import visualise_grid_search_equity_curves
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

    :param colour:
        Colour of the equity curve e.g. "rgba(160, 160, 160, 0.5)". If provided, all equity curves will be drawn with this colour.

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

    :param label_func:
        Create custom legend to compare

    """

    if name is None:
        name = "Grid search equity curve comparison"

    fig = Figure()

    colors = _generate_grey_alpha(len(results))

    results = order_grid_search_results_by_metric(results)

    for result in results:
        curve = result.equity_curve
        label = result.get_truncated_label()
        template =_get_hover_template(result)

        if label_func is not None:
            text = label_func(result)
        else:
            text = None

        scatter = Scatter(
            x=curve.index,
            y=curve,
            mode="lines",
            name="",  # Hides hover legend, use hovertext only
            line=dict(color=colors.pop(0)),
            showlegend=False,
        )
        fig.add_trace(scatter)

        if text:
            text_color = "red"
            fig.add_annotation(
                x=curve.index[-1],
                y=curve.iloc[-1],
                text=text,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=text_color,
                ax=20 + annotation_xshift,
                ay=-20,
                font=dict(size=12, color=text_color, weight='bold'),
                xshift=0,  # Small shift to avoid overlap with line end
                yshift=0
            )

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


def _generate_broad_bluered_colors(num_colors, alpha):
    """
    Generate a list of RGBA colors along a broad blue-purple-red color scale.

    Parameters:
    num_colors (int): Number of colors to generate.
    alpha (float): Alpha value for the colors (0 to 1).

    Returns:
    list: List of RGBA color tuples.
    """
    colors = []
    for i in range(num_colors):
        ratio = i / (num_colors - 1)
        if ratio < 0.7:
            red = int(255 * (1.5 * ratio))
            blue = 255
        else:
            red = 255
            blue = int(255 * (2 * (1 - ratio)))
        green = max(0, int(255 * (1 - abs(ratio - 0.7) * 2)))
        color = (red, green, blue, alpha)
        colors.append(f"rgba{color}")
    return colors


def _generate_grey_alpha(num_colors):
    colors = []
    for i in range(num_colors):
        red = 33 + 128 * i / num_colors
        green = red
        blue = red
        alpha = red
        color = (red, green, blue, alpha)
        colors.append(f"rgba{color}")
    return colors