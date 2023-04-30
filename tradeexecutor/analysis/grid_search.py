"""Grid search result analysis."""

from typing import Iterable, List

import pandas as pd

from tradeexecutor.backtest.grid_search import GridSearchResult


def analyse_combination(r: GridSearchResult) -> dict:
    """Create a grid search result table row."""

    row = {}

    for param in r.combination.parameters:
        row[param.name] = param.value

    row.update({
        # "Combination": r.combination.get_label(),
        "Annualised profit": r.summary.annualised_return_percent,
        "Max drawdown": r.metrics.loc["Max Drawdown"][0]
    })
    return row


def analyse_grid_search_result(results: List[GridSearchResult]) -> pd.DataFrame:
    """Create aa table showing grid search result of each combination.

    - Each row have labeled parameters of its combination

    - Each row has some metrics extracted from the results by :py:func:`analyse_combination`
    """
    assert len(results) > 0, "No results"
    rows = [analyse_combination(r) for r in results]
    df = pd.DataFrame(rows)
    r = results[0]
    param_names = [p.name for p in r.combination.parameters]
    df = df.set_index(param_names, drop=False)
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

