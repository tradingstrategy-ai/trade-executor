import pandas as pd

from tradeexecutor.analysis.grid_search import analyse_grid_search_result
from tradeexecutor.backtest.optimiser import OptimiserSearchResult, OptimiserResult


def analyse_optimiser_result(
    result: OptimiserResult,
    max_search_results=100,
) -> pd.DataFrame:
    """Create a table of optimiser searched space + their results.

    - Unlike :py:func:`~tradeexecutor.analysis.grid_search.analyse_grid_search_result`.,
      this will also output the optimised search variable in the output table

    See :py:func:`tradeexecutor.analysis.grid_search.analyse_grid_search_result`.
    """

    # Merge grid search result with optimised search value,
    # because these are not stored with grid search result
    for res in result.results:
        res.result.optimiser_search_value = res.get_original_value()

    top_chunk = [r.result for r in result.results[0:max_search_results]]

    # min_positions_threshold should have taken care by optimiser filter earlier
    return analyse_grid_search_result(top_chunk, min_positions_threshold=0)



def profile_optimiser(result: OptimiserResult) -> pd.DataFrame:
    """Create a DataFrame of optimiser run result.

    - Indexed by result id.
    - Durations
    """


