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
    group_by: str | None = None,
    group_by_secondary: str | None = None,
) -> Figure:
    """Draw multiple equity curves in the same chart.

    - See how all grid searched strategies work

    - Benchmark against buy and hold of various assets

    - Benchmark against hold all cash

    - Optionally group and colour curves by parameter values

    .. note ::

        Only good up to ~hundreds results. If more than thousand result, rendering takes too long time.

    Example with grouped equity curves coloured by parameter:

    .. code-block:: python

        fig = visualise_grid_search_equity_curves(
            grid_search_results,
            benchmark_indexes=benchmark_indexes,
            log_y=False,
            group_by="weighting_method",
            group_by_secondary="weight_function",
        )
        fig.show()

    :param results:
        Results from the grid search.

    :param benchmark_indexes:
        List of other asset price series displayed on the timeline besides equity curve.

        DataFrame containing multiple series.

        - Asset name is the series name.
        - Setting ``colour`` for ``pd.Series.attrs`` allows you to override the colour of the index

    :param height:
        Chart height in pixels

    :param colour:
        Colour of the equity curve e.g. ``"rgba(160, 160, 160, 0.5)"``.
        If provided, all equity curves will be drawn with this colour.

    :param log_y:
        Use logarithmic Y-axis.

    :param label_func:
        Create custom annotation labels for individual curves.

    :param group_by:
        Primary parameter name to group curves by (e.g. ``"weighting_method"``).
        Each unique value gets a distinct colour family.
        When not set, all curves are drawn in greyscale.

    :param group_by_secondary:
        Secondary parameter name to group curves by (e.g. ``"weight_function"``).
        Each unique value gets a distinct line dash style.
        Only used when ``group_by`` is also set.

    """

    if name is None:
        name = "Grid search equity curve comparison"

    fig = Figure()

    results = order_grid_search_results_by_metric(results)

    use_grouping = group_by is not None

    if use_grouping:
        # Build colour and dash mappings from unique parameter values
        primary_values = list(dict.fromkeys(
            str(r.get_parameter(group_by)) for r in results
        ))
        group_hues = _get_group_colour_palette(len(primary_values))
        primary_colour_map = {val: group_hues[i] for i, val in enumerate(primary_values)}

        dash_styles = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
        if group_by_secondary is not None:
            secondary_values = list(dict.fromkeys(
                str(r.get_parameter(group_by_secondary)) for r in results
            ))
            secondary_dash_map = {val: dash_styles[i % len(dash_styles)] for i, val in enumerate(secondary_values)}
        else:
            secondary_dash_map = None

        # Pre-compute per-group rank for opacity
        from collections import defaultdict
        group_counts = defaultdict(int)
        group_indices = defaultdict(int)
        for r in results:
            key = str(r.get_parameter(group_by))
            group_counts[key] += 1

        shown_legend_groups = set()
    else:
        colors = _generate_grey_alpha(len(results))

    for result in results:
        curve = result.equity_curve
        template = _get_hover_template(result)

        if label_func is not None:
            text = label_func(result)
        else:
            text = None

        if use_grouping:
            primary_val = str(result.get_parameter(group_by))
            r, g, b = primary_colour_map[primary_val]

            # Opacity: best in group = most opaque, worst = least
            count = group_counts[primary_val]
            rank = group_indices[primary_val]
            group_indices[primary_val] += 1
            opacity = 0.9 - (0.6 * rank / max(count - 1, 1))

            line_colour = f"rgba({r},{g},{b},{opacity:.2f})"

            if secondary_dash_map is not None:
                secondary_val = str(result.get_parameter(group_by_secondary))
                dash = secondary_dash_map[secondary_val]
                legend_group = f"{primary_val} / {secondary_val}"
                legend_name = f"{primary_val} / {secondary_val}"
            else:
                dash = "solid"
                legend_group = primary_val
                legend_name = primary_val

            show_legend = legend_group not in shown_legend_groups
            if show_legend:
                shown_legend_groups.add(legend_group)

            scatter = Scatter(
                x=curve.index,
                y=curve,
                mode="lines",
                name=legend_name,
                line=dict(color=line_colour, dash=dash),
                legendgroup=legend_group,
                showlegend=show_legend,
            )
        else:
            scatter = Scatter(
                x=curve.index,
                y=curve,
                mode="lines",
                name="",
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
                xshift=0,
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

    # Place legend below the chart so it does not overlap curves
    # https://plotly.com/python/legend/
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.15,
        xanchor="left",
        x=0
    ))

    return fig


def _get_group_colour_palette(num_groups):
    """Return a list of distinct RGB tuples for grouping equity curves.

    Uses well-separated hues that remain distinguishable
    when rendered with varying opacity.
    """
    base_colours = [
        (31, 119, 180),   # blue
        (214, 39, 40),    # red
        (44, 160, 44),    # green
        (255, 127, 14),   # orange
        (148, 103, 189),  # purple
        (23, 190, 207),   # teal
        (188, 189, 34),   # olive
        (227, 119, 194),  # pink
    ]
    # Cycle if more groups than base colours
    return [base_colours[i % len(base_colours)] for i in range(num_groups)]


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