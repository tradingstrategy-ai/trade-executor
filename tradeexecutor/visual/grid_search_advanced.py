"""Plot evolving sharpe ratio in grid search results.

- Calculate rolling metrics using :py:func:`calculate_rolling_metrics`,
  either for 1 parameter or 2 parameters visualisation

- Visualise with :py:func:`visualise_grid_rolling_metric_heatmap` or :py:func:`visualise_grid_rolling_metric_line_chart`
"""
import enum
from typing import Any
import logging

import numpy as np
import pandas as pd
from jedi.inference.gradual.typing import TypeAlias

from plotly.graph_objs import Figure
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tradeexecutor.backtest.grid_search import GridSearchResult, GridCombination


logger = logging.getLogger(__name__)


#: A period like MS for month start
HumanPeriod: TypeAlias = pd.DateOffset | str


class BenchmarkMetric(enum.Enum):
    sharpe = "sharpe"


class BenchmarkVisualisationType(enum.Enum):
    line_chart = "line_chart"
    heatmap = "heatmap"


def _calc_sharpe(
    backtest_start_at: pd.Timestamp,
    backtest_end_at: pd.Timestamp,
    step: pd.Timedelta,
    sharpe_period: pd.Timedelta,
):
    pass


def check_inputs(
    visualised_parameters: str | tuple[str, str],
    fixed_parameters: dict,
    combination: GridCombination,
):
    """Raise if we have a human error."""

    if type(visualised_parameters) == str:
        visualised_parameters = [visualised_parameters]
    else:
        visualised_parameters = list(visualised_parameters)

    parameter_name_list = visualised_parameters + list(fixed_parameters.keys())

    for p in combination.searchable_parameters:
        if p.name not in parameter_name_list:
            raise AssertionError(f"Visualisation logic missing coverage for parameter {p.name} - we have {parameter_name_list}")



def prepare_comparisons(
    visualised_parameters: str | tuple[str, str],
    fixed_parameters: dict,
    grid_search_results: list[GridSearchResult],
) -> tuple[list[GridSearchResult], list[Any]]:
    """Construct X axis.

    - Get running values for the visualised paramter

    - Discard grid search results that do not match otherwise fixed parameters

    :return:
        (Grid search results we need to visualise, unique values we are going to have)
    """

    # Get all fixed_parameter values that go to the x axis
    x_axis = []
    uniq = set()
    for r in grid_search_results:
        params = r.combination.parameters
        param_map = {p.name: p.value for p in params}

        if not all([key for key, value in fixed_parameters.items() if param_map.get(key) == value]):
            # This grid search result is not in the scope of visualisation,
            # as we are looking different fixed parameters
            continue

        x_axis.append(r)
        if type(visualised_parameters) == tuple:
            assert len(visualised_parameters) == 2
            uniq.add(
                (param_map.get(visualised_parameters[0]), param_map.get(visualised_parameters[1]))
            )
        else:
            uniq.add(param_map.get(visualised_parameters))

    uniq = sorted(list(uniq))

    return x_axis, uniq


def calculate_sharpe_at_timestamps(
    index: pd.DatetimeIndex,
    lookback: pd.Timedelta,
    returns: pd.Series,
    days_in_year=365,
    risk_free_rate=0,
) -> pd.Series:
    """Calculate rolling sharpe at certain points of time."""

    assert isinstance(returns, pd.Series)
    assert isinstance(returns.index, pd.DatetimeIndex)

    annualisation_factor = pd.Timedelta(days=days_in_year) // (returns.index[1] - returns.index[0])
    period_rf_rate = (1 + risk_free_rate) ** (1/annualisation_factor) - 1
    excess_returns = returns - period_rf_rate

    data = []

    for timestamp in index:
        period_returns = excess_returns.loc[timestamp - lookback:timestamp]
        mean = period_returns.mean()
        std = period_returns.std()
        annualized_mean = mean * annualisation_factor
        annualized_std = std * np.sqrt(annualisation_factor)
        sharpe_ratio = annualized_mean / annualized_std
        data.append(sharpe_ratio)

    return pd.Series(data, index=index)


def crunch_1d(
    visualised_parameter: str,
    unique_visualise_parameters: list[Any],
    benchmarked_results: list[GridSearchResult],
    index: pd.DatetimeIndex,
    lookback: pd.Timedelta,
    visualised_metric: BenchmarkMetric,
) -> pd.DataFrame:
    """Calculate raw results.


    TODO: Use rolling functions or not?
    """

    assert type(visualised_parameter) == str

    data = {}

    for uniq in unique_visualise_parameters:
        logger.info("Calculating %s = %s", visualised_parameter, uniq)
        found = False
        for res in benchmarked_results:
            if res.combination.get_parameter(visualised_parameter) == uniq:
                returns = res.returns
                sharpes = calculate_sharpe_at_timestamps(
                    index=index,
                    lookback=lookback,
                    returns=returns,
                )
                data[uniq] = sharpes
                found = True
        assert found, f"Zero result match for {visualised_parameter} = {uniq}, we have {len(benchmarked_results)} results"

    return pd.DataFrame(data)


def crunch_2d(
    visualised_parameters: tuple[str, str],
    unique_visualise_parameters: list[Any],
    benchmarked_results: list[GridSearchResult],
    index: pd.DatetimeIndex,
    lookback: pd.Timedelta,
    visualised_metric: BenchmarkMetric,
) -> pd.DataFrame:
    """Calculate raw results.


    TODO: Use rolling functions or not?
    """

    assert type(visualised_parameters) == tuple
    assert len(visualised_parameters) == 2

    data = {}

    param_name_1 = visualised_parameters[0]
    param_name_2 = visualised_parameters[1]

    for uniq_1, uniq_2 in unique_visualise_parameters:
        logger.info(
            "Calculating %s = %s, %s = %s",
            param_name_1,
            uniq_1,
            param_name_2,
            uniq_2
        )
        found = False
        for res in benchmarked_results:
            if res.combination.get_parameter(param_name_1) == uniq_1 and res.combination.get_parameter(param_name_2) == uniq_2:
                returns = res.returns
                sharpes = calculate_sharpe_at_timestamps(
                    index=index,
                    lookback=lookback,
                    returns=returns,
                )
                # Pandas DataFrame allows tuples as column keys
                data[(uniq_1, uniq_2)] = sharpes
                found = True
        assert found, f"Zero result match for {param_name_1} = {uniq_1} and {param_name_2} = {uniq_2}, we have {len(benchmarked_results)} results"

    return pd.DataFrame(data)


def calculate_rolling_metrics(
    grid_search_result: list[GridSearchResult],
    visualised_parameters: str | tuple[str, str],
    fixed_parameters: dict,
    sample_freq: HumanPeriod="MS",
    lookback=pd.Timedelta(days=3*30),
    benchmarked_metric=BenchmarkMetric.sharpe,
) -> pd.DataFrame:
    """Calculate rolling metrics for grid search.

    We can have two parameters e.g.
    - N: size of traded basket
    - M: number of different pick sizes

    For each N:
    - Calc rolling sharpe using last 3 months of returns (do in pandas)
    - This will give you an M-sized array or returns

    For each quarter:
    - Look back 3 months and plot
    - yaxis: sharpe ratios
    - x-axis: array of Ns

    Example output if using a single visualised parameter:

    .. code-block:: text

                        0.50      0.75      0.99
        2021-06-01       NaN       NaN       NaN
        2021-07-01 -7.565988 -5.788797 -7.554848
        2021-08-01 -3.924643 -1.919256 -3.914840
        2021-09-01 -1.807489 -1.050918 -1.798897
        2021-10-01 -1.849303 -1.604062 -1.841385
        2021-11-01 -3.792905 -3.924210 -3.784793
        2021-12-01 -4.156751 -4.186192 -4.148683

    Example of 2d heatmap output:

    .. code-block:: text

                        0.50                0.75                0.99
                           a         b         a         b         a         b
        2021-06-01       NaN       NaN       NaN       NaN       NaN       NaN
        2021-07-01 -7.565988 -7.565988 -5.788797 -5.788797 -7.554848 -7.554848
        2021-08-01 -3.924643 -3.924643 -1.919256 -1.919256 -3.914840 -3.914840
        2021-09-01 -1.807489 -1.807489 -1.050918 -1.050918 -1.798897 -1.798897
        2021-10-01 -1.849303 -1.849303 -1.604062 -1.604062 -1.841385 -1.841385
        2021-11-01 -3.792905 -3.792905 -3.924210 -3.924210 -3.784793 -3.784793
        2021-12-01 -4.156751 -4.156751 -4.186192 -4.186192 -4.148683 -4.148683

    :parma visualised_parameters:
        Single parameter name for a line chart, two parameter name tuple for a heatmap.

    :param sample_freq:
        What is the frequency of calculating rolling value

    :param lookback:
        For trailing sharpe, how far look back

    :return:

        DataFrame where

        - Index is timestamp, by `step`
        - Each column is value of visualisation parameter
        - Each row value is the visualised metric for that parameter and that timestamp

        The first row contains NaNs as it cannot be calculated due to lack of data.
    """

    logger.info(
        "calculate_rolling_metrics(), %d results",
        len(grid_search_result),
    )

    assert benchmarked_metric == BenchmarkMetric.sharpe, "Only sharpe supported at the moment"
    assert len(grid_search_result) > 0
    first_result = grid_search_result[0]

    # Different parmaeters may start trading at different times,
    # so we copy the defined backtesting period from the first grid search
    # backtest_start, backtest_end = first_result.universe_options.start_at, first_result.universe_options.end_at

    check_inputs(
        visualised_parameters,
        fixed_parameters,
        first_result.combination,
    )

    benchmarked_results, unique_visualise_parameters = prepare_comparisons(
        visualised_parameters,
        fixed_parameters,
        grid_search_result,
    )

    for res in benchmarked_results[0:3]:
        logger.info("Example result: %s", res.combination)

    logger.info(
        "We have %d unique combinations to analyse over %d results",
        len(unique_visualise_parameters),
        len(benchmarked_results),
    )

    range_start = first_result.backtest_start
    range_end = first_result.backtest_end

    logger.info(
        "Range is %s - %s",
        range_start,
        range_end,
    )

    assert range_end - range_start > pd.Timedelta("1d"), f"Range looks too short: {range_start} - {range_end}"

    # Prepare X axis
    index = pd.date_range(
        start=range_start,
        end=range_end,
        freq=sample_freq,
    )

    assert len(index) > 0, f"Could not generate index: {range_start} - {range_end}, freq {sample_freq}"

    if type(visualised_parameters) == tuple:
        df = crunch_2d(
            visualised_parameters=visualised_parameters,
            unique_visualise_parameters=unique_visualise_parameters,
            benchmarked_results=benchmarked_results,
            index=index,
            lookback=lookback,
            visualised_metric=benchmarked_metric
        )
    else:
        df = crunch_1d(
            visualised_parameter=visualised_parameters,
            unique_visualise_parameters=unique_visualise_parameters,
            benchmarked_results=benchmarked_results,
            index=index,
            lookback=lookback,
            visualised_metric=benchmarked_metric
        )

    df.attrs["metric_name"] = benchmarked_metric.name
    df.attrs["param_name"] = visualised_parameters
    df.attrs["lookback"] = lookback
    df.attrs["type"] = BenchmarkVisualisationType.heatmap if type(visualised_parameters) == tuple else BenchmarkVisualisationType.line_chart

    return df


def visualise_grid_single_rolling_metric(
    df: pd.DataFrame,
    width=None,
    height=800,
) -> Figure:
    """Create a single figure for a grid search parameter how results evolve over time.

    :param df:
        Created by :py:func:`calculate_rolling_metrics`
    """

    assert isinstance(df, pd.DataFrame)

    assert df.attrs["type"] == BenchmarkVisualisationType.line_chart

    metric_name = df.attrs["metric_name"]
    param_name = df.attrs["param_name"].replace("_", " ").capitalize()
    lookback = df.attrs["lookback"]

    # Rename columns for human readable labels
    for col in list(df.columns):
        df.rename(columns={col: f"{param_name} = {col}"}, inplace=True)

    # Create figure
    fig = go.Figure()

    # Add traces for each column
    for column in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[column],
                name=column,
                mode='lines',
            )
        )

    # Update layout
    fig.update_layout(
        title=f"Rolling {metric_name} for {param_name} parameter, with lookback of {lookback}",
        yaxis_title=metric_name,
        xaxis_title='Date',
        hovermode='x unified',
        showlegend=True,
        template='plotly_white',  # Clean white background
        height=height,
        width=width,
    )

    # Add range slider
    # fig.update_xaxes(rangeslider_visible=True)

    return fig


def visualise_grid_rolling_metric_line_chart(
    df: pd.DataFrame,
    width=1200,
    height_per_row=500,
    extra_height_margin=100,
    charts_per_row=3,
    range_start=None,
    range_end=None,
) -> Figure:
    """Create an "animation" for a single grid search parameter how results evolve over time as a line chart.

    :param df:
        Created by :py:func:`calculate_rolling_metrics`

    :param charts_per_row:
        How many mini charts display per Plotly row

    :param range_start:
        Visualise slice of full backtest period.

        Inclusive.

    :param range_end:
        Visualise slice of full backtest period.

        Inclusive.

    :return:
        List of figure  s, one for each index timestamp.
    """

    assert isinstance(df, pd.DataFrame)

    if range_start is not None:
        df = df.loc[range_start:range_end]

    metric_name = df.attrs["metric_name"]
    lookback = df.attrs["lookback"]
    param_name = df.attrs["param_name"].replace("_", " ").capitalize()

    total_rows = len(df.index) // charts_per_row + 1

    if total_rows == 1:
        charts_per_row = min(len(df.index), charts_per_row)

    logger.info(
        "visualise_grid_rolling_metric_multi_chart(): entries %d, rows %d, cols %d",
        len(df.index),
        total_rows,
        charts_per_row,
    )

    titles = []
    for timestamp in df.index:
        titles.append(
            f"{timestamp}"
        )

    fig = make_subplots(
        rows=total_rows, cols=charts_per_row,
        subplot_titles=titles,
        horizontal_spacing=0.05,
        vertical_spacing=0.05
    )

    for idx, timestamp in enumerate(df.index):

        # Get one row as one chart
        row_series = df.loc[timestamp]

        sub_fig = px.line(
            row_series,
            title=f"{timestamp}",
        )

        col = (idx % charts_per_row) + 1
        row = (idx // charts_per_row) + 1
        fig.add_trace(sub_fig.data[0], row=row, col=col)

    height = height_per_row * total_rows + extra_height_margin

    fig.update_layout(
        height=height,
        width=width,
        title_text=f"{metric_name.capitalize()} for parameter {param_name} with lookback of {lookback}",
        title_x=0.5,
        showlegend=False
    )

    # You can also adjust the overall margins of the figure
    fig.update_layout(
        margin=dict(
            l=0,  # left margin
            r=0,  # right margin
            t=extra_height_margin,  # top margin
            b=0  # bottom margin
        )
    )
    return fig


def visualise_grid_rolling_metric_heatmap(
    df: pd.DataFrame,
    width=1200,
    height_per_row=500,
    extra_height_margin=100,
    charts_per_row=3,
    range_start=None,
    range_end=None,
    discrete_paramaters=True,
) -> Figure:
    """Create an "animation" for two grid search parameters how results evolve over time as a heatmap.

    :param df:
        Created by :py:func:`calculate_rolling_metrics`

    :param charts_per_row:
        How many mini charts display per Plotly row

    :param range_start:
        Visualise slice of full backtest period.

        Inclusive.

    :param range_end:
        Visualise slice of full backtest period.

        Inclusive.

    :param discrete_paramaters:
        Measured parameters are category like, not linear.

    :return:
        List of figure  s, one for each index timestamp.
    """

    assert isinstance(df, pd.DataFrame)

    assert df.attrs["type"] == BenchmarkVisualisationType.heatmap

    if range_start is not None:
        df = df.loc[range_start:range_end]

    metric_name = df.attrs["metric_name"]
    param_1 = df.attrs["param_name"][0]
    param_2 = df.attrs["param_name"][1]
    param_name = f"""{param_1.replace("_", " ").capitalize()} (Y) and {param_2.replace("_", " ").capitalize()} (X)"""

    lookback = df.attrs["lookback"]

    total_rows = len(df.index) // charts_per_row + 1

    if total_rows == 1:
        charts_per_row = min(len(df.index), charts_per_row)

    logger.info(
        "visualise_grid_rolling_metric_multi_chart(): entries %d, rows %d, cols %d",
        len(df.index),
        total_rows,
        charts_per_row,
    )

    titles = []
    for timestamp in df.index:
        titles.append(
            f"{timestamp}"
        )

    fig = make_subplots(
        rows=total_rows, cols=charts_per_row,
        subplot_titles=titles,
        horizontal_spacing=0.05,
        vertical_spacing=0.05
    )

    for idx, timestamp in enumerate(df.index):

        # Get one row as one chart
        row_series = df.loc[timestamp]

        index_levels = [sorted(row_series.index.get_level_values(i).unique())
                        for i in range(row_series.index.nlevels)]

        # Create a 2D array for the heatmap
        z = np.zeros((len(index_levels[0]), len(index_levels[1])))

        # Fill the array with values from the series
        for row_idx, value in row_series.items():
            i = index_levels[0].index(row_idx[0])
            j = index_levels[1].index(row_idx[1])

            if discrete_paramaters:
                z[i][j] = value
            else:
                z[i][j] = value

        # Create the heatmap
        if idx == 0:
            trace = go.Heatmap(
                z=z,
                x=index_levels[1],  # Second index level for x-axis
                y=index_levels[0],  # First index level for y-axis
                colorscale='Blues',
                # text=[[f'{val:.1f}%' for val in row] for row in z],
                text=[],
                #texttemplate='%{text}',
                textfont={"size": 12},
                showscale=True,
                colorbar=None,
            )
        else:
            trace = go.Heatmap(
                z=z,
                x=index_levels[1],  # Second index level for x-axis
                y=index_levels[0],  # First index level for y-axis
                colorscale='Blues',
                # text=[[f'{val:.1f}%' for val in row] for row in z],
                text=[],
                #texttemplate='%{text}',
                textfont={"size": 12},
                showscale=False,
                colorbar=None,
            )
        col = (idx % charts_per_row) + 1
        row = (idx // charts_per_row) + 1
        fig.add_trace(trace, row=row, col=col)

    height = height_per_row * total_rows + extra_height_margin

    fig.update_layout(
        height=height,
        width=width,
        title_text=f"{metric_name.capitalize()} for parameters {param_name} with lookback of {lookback}",
        title_x=0.5,
        showlegend=False
    )

    # You can also adjust the overall margins of the figure
    fig.update_layout(
        margin=dict(
            l=0,  # left margin
            r=0,  # right margin
            t=extra_height_margin,  # top margin
            b=0  # bottom margin
        )
    )
    return fig

