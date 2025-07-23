"""Performance metrics tables."""

from plotly.graph_objects import Figure

from tradeexecutor.strategy.chart.definition import ChartInput
from tradeexecutor.analysis.multi_asset_benchmark import get_benchmark_data, DEFAULT_BENCHMARK_ASSETS, compare_strategy_backtest_to_multiple_assets
from tradeexecutor.visual.benchmark import visualise_equity_curve_benchmark


def performance_metrics(
    input: ChartInput,
    max_benchmark_count=4,
    benchmark_token_symbols: list[str]=None,
) -> Figure:
    """Render performance metrics table.

    - Render the backtesting or live trade equity curve based on the state

    :param max_benchmark_count:
        Max number of benchmark assets

    :param benchmark_token_symbols:
        What tokens we wish to show in the equity curve as a benchmark.

        Must have a corresponding price data loaded in the strategy universe.

    :return:
        Equity curve figure
    """

    state = input.state
    strategy_universe = input.strategy_universe

    if benchmark_token_symbols is None:
        benchmark_token_symbols = DEFAULT_BENCHMARK_ASSETS

    df = compare_strategy_backtest_to_multiple_assets(
        state,
        strategy_universe,
        display=True,
        interesting_assets=benchmark_token_symbols,
        asset_count=max_benchmark_count,
    )
    return df