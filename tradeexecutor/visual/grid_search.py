from tradeexecutor.backtest.grid_search import GridSearchResult
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


def visualise_grid_search_result_benchmark(
    result: GridSearchResult,
    strategy_universe: TradingStrategyUniverse,
) -> go.Figure:
    """Draw one equity curve from grid search results.

    - Use :func:`find_best_grid_search_results` to find some equity curves.

    :param result:
        Picked grid search result
    :param strategy_universe:
        Used to get benechmark indexes
    :return:
        Plotly figure
    """
