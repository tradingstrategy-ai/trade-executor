"""Grid search result analysis.

- Breaddown of performance of different grid search combinations

- Heatmap and other comparison methods

"""
import textwrap
import warnings
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

import plotly.express as px
from pandas.io.formats.style import Styler
from plotly.graph_objs import Figure

from tradeexecutor.analysis.grid_search_format import (
    build_format_dict,
    normalise_enum_values,
)
from tradeexecutor.backtest.grid_search import GridSearchResult
from tradeexecutor.utils.sort import unique_sort

VALUE_COLS = ["Optim", "CAGR", "Max DD", "Sharpe", "Sortino", "Avg pos", "Med pos", "Win rate", "Time in market"]

CALMAR_VALUE_COLS = ["Optim", "CAGR", "Max DD", "Sharpe", "Sortino", "Calmar", "Time in market", "Avg capital util", "Max cash %", "Avg pos", "Med pos", "Win rate"]

PERCENT_COLS = ["CAGR", "Max DD", "Avg pos", "Med pos", "Time in market", "Win rate", "Avg capital util", "Max cash %"]

DATA_COLS = ["Positions", "Trades"]


def analyse_combination(
    r: GridSearchResult,
    min_positions_threshold: int,
) -> dict:
    """Create a grid search result table row.

    - Create columns we can use to compare different grid search combinations

    :param min_positions_threshold:
        If we did less positions than this amount, do not consider this a proper strategy.

        Filter out one position outliers.

    """

    row = {}
    param_names = []
    for param in r.combination.parameters:

        # Skip parameters that are single fixed value
        # and do not affect the grid search results
        if param.single and not param.optimise:
            continue

        row[param.name] = param.value
        param_names.append(param.name)

    def clean(x):
        if x == "-":
            return np.nan
        elif x == "":
            return np.nan

        if type(x) == int:
            return float(x)

        return x

    row.update({
        "Positions": r.summary.total_positions,
        "Trades": r.summary.total_trades,
    })

    if r.optimiser_search_value is not None:
        # Display raw optimiser search values
        # See analyse_optimiser_result()
        row.update({
            "Optim": clean(r.optimiser_search_value),
        })

    row.update({
        # "Return": r.summary.return_percent,
        # "Return2": r.summary.annualised_return_percent,
        #"Annualised profit": clean(r.metrics.loc["Expected Yearly"][0]),
        "CAGR": clean(r.metrics.loc["Annualised return (raw)"].iloc[0]),
        "Max DD": clean(r.metrics.loc["Max Drawdown"].iloc[0]),
        "Sharpe": clean(r.metrics.loc["Sharpe"].iloc[0]),
        "Sortino": clean(r.metrics.loc["Sortino"].iloc[0]),
        # "Combination": r.combination.get_label(),
        "Time in market": clean(r.metrics.loc["Time in Market"].iloc[0]),
        "Win rate": clean(r.get_win_rate()),
        "Avg pos": r.summary.average_trade,  # Average position
        "Med pos": r.summary.median_trade,  # Median position
    })

    # Calmar ratio: CAGR / |max drawdown|
    cagr_val = row.get("CAGR")
    max_dd_val = row.get("Max DD")
    if cagr_val is not None and max_dd_val is not None and not np.isnan(cagr_val) and not np.isnan(max_dd_val) and abs(max_dd_val) > 0:
        row["Calmar"] = cagr_val / abs(max_dd_val)
    else:
        row["Calmar"] = np.nan

    # Capital utilisation metrics from portfolio stats time-series
    try:
        state = r.hydrate_state()
        if state and state.stats and state.stats.portfolio:
            cash_ratios = []
            for ps in state.stats.portfolio:
                equity = ps.total_equity or 0
                cash = ps.free_cash
                if equity > 0 and cash is not None:
                    cash_ratios.append(cash / equity)
            if cash_ratios:
                row["Avg capital util"] = clean(1.0 - np.mean(cash_ratios))
                row["Max cash %"] = clean(np.max(cash_ratios))
            else:
                row["Avg capital util"] = np.nan
                row["Max cash %"] = np.nan
        else:
            row["Avg capital util"] = np.nan
            row["Max cash %"] = np.nan
    except Exception:
        row["Avg capital util"] = np.nan
        row["Max cash %"] = np.nan

    # Clear all values except position count if this is not a good trade series
    if r.summary.total_positions < min_positions_threshold:
        for k in row.keys():
            if k != "Positions" and k not in param_names:
                row[k] = np.nan

    return row


def analyse_grid_search_result(
    results: List[GridSearchResult],
    min_positions_threshold: int = 5,
    drop_duplicates=True,
) -> pd.DataFrame:
    """Create aa table showing grid search result of each combination.

    - Each row have labeled parameters of its combination

    - Each row has some metrics extracted from the results by :py:func:`analyse_combination`

    The output has the following row for each parameter combination:

    - Combination parameters
    - Positions and trade counts
    - CAGR (Communicative annualized growth return, compounding)
    - Max drawdown
    - Sharpe
    - Sortino

    See also :py:func:`analyse_combination`.

    :param results:
        Output from :py:meth:`tradeexecutor.backtest.grid_search.perform_grid_search`.

    :param min_positions_threshold:
        If we has less closed positions than this amount, do not consider this a proper trading strategy.

        It is just random noise. Do not write a result line for such parameter combinations.

    :return:
        Table of grid search combinations
    """
    assert len(results) > 0, "analyse_grid_search_result(): the result set is empty - likely none of the backtested strategies made any trades"
    rows = [analyse_combination(r, min_positions_threshold) for r in results]
    df = pd.DataFrame(rows)

    duplicate_cols_count = df.columns.duplicated(keep='first').sum()
    if duplicate_cols_count > 0:
        raise RuntimeError(f"Duplicate columns: {df.columns}")

    r = results[0]
    param_names = [p.name for p in r.combination.searchable_parameters]
    # display(df)
    df = df.set_index(param_names)

    # Optimiser may result to the duplicate grid combinations
    # as the optimiser searches are down in parallel
    # - however this would break styles of this df.
    # We remove duplicates here.
    if drop_duplicates:
        df = df.drop_duplicates(keep="first")

    # duplicates = df[df.index.duplicated(keep='first')]
    # if len(duplicates)> 0:
    #    for r in duplicates.iterrows():
    #        print(f"Row: {r}")
    #    raise RuntimeError(f"Duplicate indexes found: {duplicates}")

    df = df.sort_index()
    return df


def visualise_table(*args, **kwargs):
    warnings.warn('This function is deprecated. Use render_grid_search_result_table() instead', DeprecationWarning, stacklevel=2)
    return render_grid_search_result_table(*args, **kwargs)


def render_grid_search_result_table(
    results: pd.DataFrame | list[GridSearchResult],
    calmar: bool = False,
    sharpe: bool = True,
    sortino: bool = True,
) -> Styler:
    """Render a grid search combination table to notebook output.

    - Highlight winners and losers

    - Gradient based on the performance of a metric

    - Stripes for the input

    Example:

    .. code-block:: python

            grid_search_results = perform_grid_search(
                decide_trades,
                strategy_universe,
                combinations,
                max_workers=get_safe_max_workers_count(),
                trading_strategy_engine_version="0.5",
                multiprocess=True,
            )
            render_grid_search_result_table(grid_search_results)

    :param results:
        Output from :py:func:`perform_grid_search`.

    :param calmar:
        Include the Calmar ratio (CAGR / |max drawdown|) column.

    :param sharpe:
        Include the Sharpe ratio column.

    :param sortino:
        Include the Sortino ratio column.

    :return:
        Styled DataFrame for the notebook output
    """

    if isinstance(results, pd.DataFrame):
        df = results
    else:
        df = analyse_grid_search_result(results)

    return _style_grid_search_table(df, calmar=calmar, sharpe=sharpe, sortino=sortino)


def render_grid_search_result_table_avg(
    results: pd.DataFrame | list[GridSearchResult],
    avg_by: str,
    calmar: bool = False,
    sharpe: bool = True,
    sortino: bool = True,
) -> Styler:
    """Render a grid search table with rows averaged by a single parameter.

    - Groups all combinations by ``avg_by`` parameter and averages their metrics
    - Shows one row per unique value of ``avg_by``
    - Includes an ``n`` column with the number of combinations in each group

    Example:

    .. code-block:: python

        render_grid_search_result_table_avg(df, avg_by="calmar_signal_transform", calmar=True)

    :param results:
        Output from :py:func:`perform_grid_search` or :py:func:`analyse_grid_search_result`.

    :param avg_by:
        Parameter name to group by.

    :param calmar:
        Include the Calmar ratio column.

    :param sharpe:
        Include the Sharpe ratio column.

    :param sortino:
        Include the Sortino ratio column.

    :return:
        Styled DataFrame for the notebook output
    """
    if isinstance(results, pd.DataFrame):
        df = results
    else:
        df = analyse_grid_search_result(results)

    # Parameters are in the MultiIndex — reset to make them regular columns
    df = df.reset_index()

    if avg_by not in df.columns:
        raise ValueError(f"Parameter '{avg_by}' not found in results. Available columns: {list(df.columns)}")

    # Identify numeric columns to average
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    grouped = df.groupby(avg_by, sort=True)
    avg_df = grouped[numeric_cols].mean()
    avg_df.insert(0, "n", grouped.size())

    return _style_grid_search_table(avg_df, calmar=calmar, sharpe=sharpe, sortino=sortino)


def _style_grid_search_table(
    df: pd.DataFrame,
    calmar: bool = False,
    sharpe: bool = True,
    sortino: bool = True,
) -> Styler:
    """Apply standard grid search table styling.

    Shared helper for :py:func:`render_grid_search_result_table`
    and :py:func:`render_grid_search_result_table_avg`.
    """

    # https://stackoverflow.com/a/57152529/315168
    # TODO: Diverge color gradient around zero
    # https://stackoverflow.com/a/60654669/315168

    cols = list(CALMAR_VALUE_COLS if calmar else VALUE_COLS)
    if not sharpe:
        cols = [c for c in cols if c != "Sharpe"]
    if not sortino:
        cols = [c for c in cols if c != "Sortino"]
    value_cols = [v for v in cols if v in df.columns]

    # Drop hidden columns from the DataFrame
    drop_cols = []
    if not sharpe and "Sharpe" in df.columns:
        drop_cols.append("Sharpe")
    if not sortino and "Sortino" in df.columns:
        drop_cols.append("Sortino")
    if not calmar and "Calmar" in df.columns:
        drop_cols.append("Calmar")
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Reorder columns to match the desired display order:
    # parameter columns first (in their original order), then value columns
    param_cols = [c for c in df.columns if c not in value_cols and c not in DATA_COLS]
    data_cols_present = [c for c in DATA_COLS if c in df.columns]
    ordered_cols = param_cols + data_cols_present + value_cols
    df = df[[c for c in ordered_cols if c in df.columns]]

    format_dict = build_format_dict(value_cols, PERCENT_COLS, DATA_COLS)
    df = normalise_enum_values(df)

    # Not sure what triggers duplicate result in the result set
    # KeyError: 'Styler.apply and .map are not compatible with non-unique index or columns'
    df = df[~df.index.duplicated(keep='first')]

    # Set NaN cells to white background, overriding any gradient
    def highlight_nan_white(s):
        return ['background-color: white' if pd.isna(v) else '' for v in s]

    formatted = df.style.background_gradient(
        axis = 0,
        subset = value_cols,
    ).highlight_min(
        color = 'pink',
        axis = 0,
        subset = value_cols,
    ).highlight_max(
        color = 'darkgreen',
        axis = 0,
        subset = value_cols,
    ).apply(
        highlight_nan_white,
        subset = value_cols,
    ).format(
        format_dict,
        na_rep="",
    )
    return formatted


def visualise_heatmap_2d(
    result: pd.DataFrame,
    parameter_1: str,
    parameter_2: str,
    metric: str,
    color_continuous_scale='Bluered_r',
    continuous_scale: bool | None = None,
) -> Figure:
    """Draw a heatmap square comparing two different parameters.

    Directly shows the resulting matplotlib figure.

    :param parameter_1:
        Y axis

    :param parameter_2:
        X axis

    :param metric:
        Value to examine

    :param result:
        Grid search results as a DataFrame.

        Created by :py:func:`analyse_grid_search_result`.

    :param color_continuous_scale:
        The name of Plotly gradient used for the colour scale.

    :param continuous_scale:
        Are the X and Y scales continuous.

        X and Y scales cannot be continuous if they contain values like None or NaN.
        This will stretch the scale to infinity or zero.

        Set `True` to force continuous, `False` to force discreet steps, `None` to autodetect.

    :return:
        Plotly Figure object
    """

    # Reset multi-index so we can work with parameter 1 and 2 as series
    df = result.reset_index()

    # Backwards compatibiltiy
    if metric == "Annualised return" and ("Annualised return" not in df.columns) and "CAGR" in df.columns:
        metric = "CAGR"

    # Detect any non-number values on axes
    if continuous_scale is None:
        continuous_scale = not(df[parameter_1].isna().any() or df[parameter_2].isna().any())

    # setting all column values to string will hint
    # Plotly to make all boxes same size regardless of value
    if not continuous_scale:
        df[parameter_1] = df[parameter_1].astype(str)
        df[parameter_2] = df[parameter_2].astype(str)

    df = df.pivot(index=parameter_1, columns=parameter_2, values=metric)

    # Format percents inside the cells and mouse hovers
    if metric in PERCENT_COLS:
        text = df.map(lambda x: f"{x * 100:,.2f}%")
    else:
        text = df.map(lambda x: f"{x:,.2f}")

    fig = px.imshow(
        df,
        labels=dict(x=parameter_2, y=parameter_1, color=metric),
        aspect="auto",
        title=metric,
        color_continuous_scale=color_continuous_scale,
    )

    fig.update_traces(text=text, texttemplate="%{text}")

    fig.update_layout(
        title={"text": metric},
        height=600,
    )
    return fig


def visualise_3d_scatter(
    flattened_result: pd.DataFrame,
    parameter_x: str,
    parameter_y: str,
    parameter_z: str,
    measured_metric: str,
    color_continuous_scale="Bluered_r",  # Reversed, blue = best
    height=600,
) -> Figure:
    """Draw a 3D scatter plot for grid search results.

    Create an interactive 3d chart to explore three different parameters and one performance measurement
    of the grid search results.

    Example:

    .. code-block:: python

        from tradeexecutor.analysis.grid_search import analyse_grid_search_result
        table = analyse_grid_search_result(grid_search_results)
        flattened_results = table.reset_index()
        flattened_results["Annualised return %"] = flattened_results["Annualised return"] * 100
        fig = visualise_3d_scatter(
            flattened_results,
            parameter_x="rsi_days",
            parameter_y="rsi_high",
            parameter_z="rsi_low",
            measured_metric="Annualised return %"
        )
        fig.show()

    :param flattened_result:
        Grid search results as a DataFrame.

        Created by :py:func:`analyse_grid_search_result`.

    :param parameter_x:
        X axis

    :param parameter_y:
        Y axis

    :param parameter_z:
        Z axis

    :param parameter_colour:
        Output we compare.

        E.g. `Annualised return`

    :param color_continuous_scale:
        The name of Plotly gradient used for the colour scale.

        `See the Plotly continuos scale color gradient options <https://plotly.com/python/builtin-colorscales/>`__.

    :return:
        Plotly figure to display
    """

    assert isinstance(flattened_result, pd.DataFrame)
    assert type(parameter_x) == str
    assert type(parameter_y) == str
    assert type(parameter_z) == str
    assert type(measured_metric) == str

    fig = px.scatter_3d(
        flattened_result,
        x=parameter_x,
        y=parameter_y,
        z=parameter_z,
        color=measured_metric,
        color_continuous_scale=color_continuous_scale,
        height=height,
    )

    return fig


def _get_hover_template(
    result: GridSearchResult,
    key_metrics = ("CAGR﹪", "Max Drawdown", "Time in Market", "Sharpe", "Sortino"),  # See quantstats
    percent_metrics = ("CAGR﹪", "Max Drawdown", "Time in Market"),
):

    # Get metrics calculated with QuantStats
    data = result.metrics["Strategy"]
    metrics = {}
    for name in key_metrics:
        metrics[name] = data[name]

    template = textwrap.dedent(f"""<b>{result.get_label()}</b><br><br>""")

    for k, v in metrics.items():
        if type(v) == int:
            v = float(v)

        if v in ("", None, "-"):  # Messy third party code does not know how to mark no value
            template += f"{k}: -<br>"
        elif k in percent_metrics:
            assert type(v) == float, f"Got unknown type: {k}: {v} ({type(v)}"
            v *= 100
            template += f"{k}: {v:.2f}%<br>"
        else:
            assert type(v) == float, f"Got unknown type: {k}: {v} ({type(v)}"
            template += f"{k}: {v:.2f}<br>"

    # Get trade metrics
    for k, v in result.summary.get_trading_core_metrics().items():
        template += f"{k}: {v}<br>"

    return template


@dataclass(slots=True)
class TopGridSearchResult:
    """Sorted best grid search results."""

    #: Top returns
    cagr: list[GridSearchResult]

    #: Top Sharpe
    sharpe: list[GridSearchResult]


def find_best_grid_search_results(grid_search_results: list[GridSearchResult], count=20, unique_only=True) -> TopGridSearchResult:
    """From all grid search results, filter out the best one to be displayed.

    :param unique_only:
        Return unique value matches only.

        If multiple grid search results share the same metric (CAGR),
        filter out duplicates. Otherwise the table will be littered with duplicates.

    :return:
        Top lists
    """

    if unique_only:
        sorter = unique_sort
    else:
        sorter = sorted

    result = TopGridSearchResult(
        cagr=sorter(grid_search_results, key=lambda r: r.get_cagr(), reverse=True)[0: count],
        sharpe=sorter(grid_search_results, key=lambda r: r.get_sharpe(), reverse=True)[0: count],
    )
    return result


def visualise_grid_search_equity_curves(*args, **kwags):
    """Deprecated."""
    warnings.warn("use tradeexecutor.visual.grid_search.visualise_grid_search_equity_curves instead", DeprecationWarning, stacklevel=2)
    from tradeexecutor.visual.grid_search import visualise_grid_search_equity_curves
    return visualise_grid_search_equity_curves(*args, **kwags)


def order_grid_search_results_by_metric(results: List[GridSearchResult], metric: str = 'Cumulative Return') -> List[GridSearchResult]:
    """Order grid search results by a metric. Default is Cumulative Return.
    
    :param results: List of GridSearchResult
    :param metric: Metric to order by. Default is 'Cumulative Return'
    :return: List of GridSearchResult ordered by the metric
    """
    return sorted(results, key=lambda x: x.get_metric(metric), reverse=True)
