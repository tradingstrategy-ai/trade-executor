from typing import Iterable

import pandas as pd

from tradeexecutor.backtest.grid_search import GridSearchResult


def analyse_combination(r: GridSearchResult) -> dict:
    """Create a grid search result table row."""

    return {
        "Combination": r.combination.get_label(),
        "Annualised profit": r.summary.annualised_return_percent,
        "Maximum drawdown": r.metrics["Maximum drawdown"][0]
    }


def analyse_grid_search_result(results: Iterable[GridSearchResult]) -> pd.DataFrame:
    """Create aa table showing grid search result of each combination."""
    rows = [analyse_combination(r) for r in results]
    df = pd.DataFrame(rows)
    df = df.set_index("Combination")
    return df

