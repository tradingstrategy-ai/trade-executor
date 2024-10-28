"""Optimiser analytics and charting."""
import datetime

import pandas as pd
import plotly.express as px


from tradeexecutor.analysis.grid_search import analyse_grid_search_result
from tradeexecutor.backtest.optimiser import OptimiserSearchResult, OptimiserResult


def analyse_optimiser_result(
    result: OptimiserResult,
    max_search_results=100,
    exclude_filtered=True,
) -> pd.DataFrame:
    """Create a table of optimiser searched space + their results.

    - Unlike :py:func:`~tradeexecutor.analysis.grid_search.analyse_grid_search_result`.,
      this will also output the optimised search variable in the output table

    See :py:func:`tradeexecutor.analysis.grid_search.analyse_grid_search_result`.

    :param exclude_filtered:
        Do not display the result rows for the rows that did not pass the result filter check

    :return:
        Displayable data frame
    """

    # Merge grid search result with optimised search value,
    # because these are not stored with grid search result
    for res in result.results:
        res.result.optimiser_search_value = res.get_original_value()

    if exclude_filtered:
        valid_result_result = [r for r in result.results if not r.filtered]
    else:
        valid_result_result = result.results

    top_chunk = [r.result for r in valid_result_result[0:max_search_results]]

    if len(result.results) > 0 and len(valid_result_result) == 0:
        raise AssertionError(f"No optimsier results for analysis where left after dropping backtests that did not pass perform_optimisation(result_filter)")

    # min_positions_threshold should have taken care by optimiser filter earlier
    return analyse_grid_search_result(top_chunk, min_positions_threshold=0)


def profile_optimiser(result: OptimiserResult) -> pd.DataFrame:
    """Create a DataFrame of optimiser run result.

    Mainly used to track if/why optimiser slows down in long runs.

    - Indexed by result id.
    - Durations
    """
    sorted_result =  sorted(result.results, key=lambda r: r.result.run_start_at)
    data = []
    r: OptimiserSearchResult
    for r in sorted_result:
        tc = r.result.get_trade_count()
        data.append({
            "start_at": r.result.run_start_at,
            "backtest": r.result.get_backtest_duration(),
            "analysis": r.result.get_analysis_duration(),
            "delivery": r.result.get_delivery_duration(),
            "iteration": r.iteration_duration,
            "iteration_id": r.iteration,
            "trades": tc,
            "duration_per_trade": r.result.get_backtest_duration() / tc if tc else datetime.timedelta(0),
            "metrics_size": r.get_metrics_persistent_size(),
            "state_size": r.get_state_size(),
            # "delivery": r.result.get_delivery_duration(),
        })

    df = pd.DataFrame(data)
    df = df.set_index("start_at")
    return df


def plot_profile_duration_data(
    df: pd.DataFrame,
    include_colums=("backtest", "analysis", "delivery", "duration_per_trade", "iteration",)
):
    """Visualise the profiler data.

    :param df:
        From :py:func:`https://1delta.io/`
    """

    lines_df = df[list(include_colums)]
    # Convert to seconds
    lines_df = lines_df.apply(lambda x: x.dt.total_seconds())
    fig = px.line(lines_df)
    fig.update_layout(title="Profiled optimiser durations of subsections")
    return fig


def plot_profile_size_data(
    df: pd.DataFrame,
    include_colums=("metrics_size", "state_size")
):
    """Visualise the profiler data.

    :param df:
        From :py:func:`https://1delta.io/`
    """

    lines_df = df[list(include_colums)]
    # Convert to seconds
    fig = px.line(lines_df)
    fig.update_layout(title="Profiled optimiser file sizes")
    return fig