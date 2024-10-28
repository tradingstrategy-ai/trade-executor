"""Plot evolving sharpe ratio in grid search results.

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

from tradeexecutor.backtest.grid_search import GridSearchResult, GridCombination


logger = logging.getLogger(__name__)


#: A period like MS for month start
HumanPeriod: TypeAlias = pd.DateOffset | str


class BenchmarkFunction(enum.Enum):
    sharpe = "sharpe"



def _calc_sharpe(
    backtest_start_at: pd.Timestamp,
    backtest_end_at: pd.Timestamp,
    step: pd.Timedelta,
    sharpe_period: pd.Timedelta,
):
    pass


def check_inputs(
    visualised_parameter: str,
    fixed_parameters: dict,
    combination: GridCombination,
):
    """Raise if we have a human error."""
    parameter_name_list = [visualised_parameter] + list(fixed_parameters.keys())

    for p in combination.searchable_parameters:
        if p.name not in parameter_name_list:
            raise AssertionError(f"Visualisation logic missing coverage for parameter {p.name} - we have {parameter_name_list}")



def prepare_comparisons(
    visualised_parameter: str,
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
        uniq.add(param_map.get(visualised_parameter))

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


def crunch(
    visualised_parameter: str,
    unique_visualise_parameters: list[Any],
    benchmarked_results: list[GridSearchResult],
    index: pd.DatetimeIndex,
    lookback: pd.Timedelta,
    visualised_metric: BenchmarkFunction,
) -> pd.DataFrame:
    """Calculate raw results.


    TODO: Use rolling functions or not?
    """

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



def calculate_rolling_metrics(
    grid_search_result: list[GridSearchResult],
    visualised_parameter: str,
    fixed_parameters: dict,
    sample_freq: HumanPeriod="MS",
    lookback=pd.Timedelta(days=3*30),
    visualised_metric=BenchmarkFunction.sharpe,
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

    Example output:

    .. code-block:: text

                        0.50      0.75      0.99
        2021-06-01       NaN       NaN       NaN
        2021-07-01 -7.565988 -5.788797 -7.554848
        2021-08-01 -3.924643 -1.919256 -3.914840
        2021-09-01 -1.807489 -1.050918 -1.798897
        2021-10-01 -1.849303 -1.604062 -1.841385
        2021-11-01 -3.792905 -3.924210 -3.784793
        2021-12-01 -4.156751 -4.186192 -4.148683

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

    assert visualised_metric == BenchmarkFunction.sharpe, "Only sharpe supported at the moment"
    assert len(grid_search_result) > 0
    first_result = grid_search_result[0]

    # Different parmaeters may start trading at different times,
    # so we copy the defined backtesting period from the first grid search
    # backtest_start, backtest_end = first_result.universe_options.start_at, first_result.universe_options.end_at

    check_inputs(
        visualised_parameter,
        fixed_parameters,
        first_result.combination,
    )

    benchmarked_results, unique_visualise_parameters = prepare_comparisons(
        visualised_parameter,
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

    df = crunch(
        visualised_parameter=visualised_parameter,
        unique_visualise_parameters=unique_visualise_parameters,
        benchmarked_results=benchmarked_results,
        index=index,
        lookback=lookback,
        visualised_metric=visualised_metric
    )

    df.attrs["metric_name"] = visualised_metric.name
    df.attrs["param_name"] = visualised_parameter

    return df


def visualise_grid_sharpe_for_parameter(
    df: pd.DataFrame,
    width=None,
    height=800,
) -> Figure:
    """Create an animation for a grid search parameter how results evolve over time.

    :param df:
        Created by :py:func:`calculate_rolling_metrics`
    """

    metric_name = df.attrs["metric_name"]
    param_name = df.attrs["param_name"].capitalize()

    # Rename columns for human readable labels
    for col in df.colums:
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
        title=f"Rolling {metric_name} for {param_name} parameter",
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

