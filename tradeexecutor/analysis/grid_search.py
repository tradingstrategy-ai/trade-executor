"""Grid search result analysis."""

from typing import Iterable, List

import numpy as np
import pandas as pd
from IPython.core.display_functions import display

from tradeexecutor.backtest.grid_search import GridSearchResult


VALUE_COLS = ["Annualised profit", "Max drawdown", "Sharpe", "Average position", "Median position"]

PERCENT_COLS = ["Annualised profit", "Max drawdown", "Average position", "Median position"]


def analyse_combination(r: GridSearchResult) -> dict:
    """Create a grid search result table row."""

    row = {}

    for param in r.combination.parameters:
        row[param.name] = param.value

    def clean(x):
        if x == "-":
            return np.NaN
        elif x == "":
            return np.NaN
        return x

    row.update({
        # "Combination": r.combination.get_label(),
        "Positions": r.summary.total_positions,
        "Annualised profit": r.summary.annualised_return_percent,
        "Max drawdown": clean(r.metrics.loc["Max Drawdown"][0]),
        "Sharpe": clean(r.metrics.loc["Sharpe"][0]),
        "Average position": r.summary.average_trade,
        "Median position": r.summary.median_trade,
    })
    return row


def analyse_grid_search_result(results: List[GridSearchResult], drop_index=True) -> pd.DataFrame:
    """Create aa table showing grid search result of each combination.

    - Each row have labeled parameters of its combination

    - Each row has some metrics extracted from the results by :py:func:`analyse_combination`
    """
    assert len(results) > 0, "No results"
    rows = [analyse_combination(r) for r in results]
    df = pd.DataFrame(rows)
    r = results[0]
    param_names = [p.name for p in r.combination.parameters]
    df = df.set_index(param_names, drop=drop_index)
    df = df.sort_index()
    return df


def visualise_table(df: pd.DataFrame):
    """Render a grid search combination table to notebook output.

    - Highlight winners and losers
    """

    # https://stackoverflow.com/a/57152529/315168

    formatted = df.style.background_gradient(
        axis = 0,
        subset = VALUE_COLS,
    ).highlight_min(
        color = 'pink',
        axis = 0,
        subset = VALUE_COLS,
    ).format(
        formatter="{:.2%}",
        subset = PERCENT_COLS,
    )

    # formatted = df.style.highlight_max(
    #     color = 'lightgreen',
    #     axis = 0,
    #     subset = VALUE_COLS,
    # ).highlight_min(
    #     color = 'pink',
    #     axis = 0,
    #     subset = VALUE_COLS,
    # ).format(
    #     formatter="{:.2%}",
    #     subset = PERCENT_COLS,
    # )

    display(formatted)


def visualise_heatmap_2d(
        result: pd.DataFrame,
        parameter_1: str,
        parameter_2: str,
        metric: str,
):
    """Draw a heatmap square comparing two different parameters.

    :param parameter_1:
        X axis

    :param parameter_2:
        Y axis

    :param metric:
        Value to examine

    :param result:
        Grid search results as a DataFrame.

        Created by :py:func:`analyse_grid_search_result`.
    """

    import seaborn as sns
    df = result.reset_index().pivot(parameter_1, parameter_2, metric)
    sns.heatmap(df, annot=True)

