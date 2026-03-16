"""Performance metrics tables."""

import pandas as pd

from plotly.graph_objects import Figure
from tradeexecutor.analysis.multi_asset_benchmark import (
    DEFAULT_BENCHMARK_ASSETS, compare_strategy_backtest_to_multiple_assets,
    get_benchmark_data)
from tradeexecutor.strategy.chart.definition import ChartInput
from tradeexecutor.visual.benchmark import visualise_equity_curve_benchmark


EXTENDED_PERFORMANCE_METRICS = [
    "Prob. Sharpe Ratio",
    "Ulcer Index",
    "Expected Shortfall (cVaR)",
    "Recovery Factor",
    "Longest DD Days",
    "Time in Market",
]


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

    assert state is not None, "State must be provided to render performance metrics."

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


def extended_performance_metrics(
    input: ChartInput,
    max_benchmark_count=4,
    benchmark_token_symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Render extended risk metrics table for the strategy and benchmarks.

    Includes PSR, Ulcer Index, cVaR, recovery factor, longest drawdown
    duration, and time in market in a web-renderable table layout.

    For glossary definitions of these metrics, see
    https://tradingstrategy.ai/glossary.
    """
    state = input.state
    strategy_universe = input.strategy_universe

    assert state is not None, "State must be provided to render performance metrics."

    if benchmark_token_symbols is None:
        benchmark_token_symbols = DEFAULT_BENCHMARK_ASSETS

    df = compare_strategy_backtest_to_multiple_assets(
        state,
        strategy_universe,
        display=True,
        interesting_assets=benchmark_token_symbols,
        asset_count=max_benchmark_count,
    )

    available_metrics = [metric for metric in EXTENDED_PERFORMANCE_METRICS if metric in df.index]
    filtered = df.loc[available_metrics].copy()
    filtered.index.name = "Metric"
    return filtered.reset_index()
