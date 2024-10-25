"""Plot evolving sharpe ratio in grid search results.

"""
import pandas as pd
from plotly.graph_objs import Figure

from tradeexecutor.backtest.grid_search import GridSearchResult, GridCombination


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


def prepare_axes(
    fixed_parameters: dict,
    grid_search_results: list[GridSearchResult],
) -> list[GridSearchResult]:
    """Construct X axis.

    - Get running values for the visualised paramter

    - Discard grid search results that do not match otherwise fixed parameters
    """

    # Get all fixed_parameter values that go to the x axis
    x_axis = []
    for r in grid_search_results:
        params = r.combination.parameters
        param_map = {p.name: p.value for p in params}

        if not all([key for key, value in fixed_parameters.items() if param_map.get(key) == value])
            # This grid search result is not in the scope of visualisation,
            # as we are looking different fixed parameters
            continue

        x_axis.append(r)

    return x_axis


def visualise_grid_sharpe_for_parameter(
    grid_search_result: list[GridSearchResult],
    visualised_parameter: str,
    fixed_parameters: dict,
    step=pd.Timedelta(months=3),
    sharpe_period=pd.Timedelta(months=3),
    visualised_metric="sharpe",
) -> list[Figure]:
    """Create an animation for a grid search parameter how results evolve over time.

    We can have two parameters e.g.
    - N: size of traded basket
    - M: number of different pick sizes

    For each N:
    - Calc rolling sharpe using last 3 months of returns (do in pandas)
    - This will give you an M-sized array or returns

    For each quarter:
        Look back 3 months and plot
        yaxis: sharpe ratios
        x-axis: array of Ns
    """

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

    # Make X-axis of individual grid search result
    x_axis = prepare_axes(
        fixed_parameters,
        grid_search_result,
    )

    x_axis.sort(key=lambda r: r.combination.parameters.get_parameter(visualised_parameter))

