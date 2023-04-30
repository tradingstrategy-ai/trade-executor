"""Grid search result analysis."""

from typing import Iterable

import pandas as pd

from tradeexecutor.backtest.grid_search import GridSearchResult


def analyse_combination(r: GridSearchResult) -> dict:
    """Create a grid search result table row."""

    row = {}

    for param in r.combination.parameters:
        row[param.name] = param.value

    row.update({
        "Combination": r.combination.get_label(),
        "Annualised profit": r.summary.annualised_return_percent,
        "Max drawdown": r.metrics.loc["Max Drawdown"][0]
    })
    return row


def analyse_grid_search_result(results: Iterable[GridSearchResult]) -> pd.DataFrame:
    """Create aa table showing grid search result of each combination."""
    rows = [analyse_combination(r) for r in results]
    df = pd.DataFrame(rows)
    df = df.set_index("Combination")
    return df


def visualise_heatmap_2d(
        result: pd.DataFrame,
        metric: str,
        parameter_1: str,
        parameter_2: str,
):
    """Draw a heatmap square comparing two different parameters.

    :param result:
        Grid search results as a DataFrame.

        Created by :py:func:`analyse_grid_search_result`.
    """

