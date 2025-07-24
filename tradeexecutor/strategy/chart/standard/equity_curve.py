"""Equity curve charts."""

from plotly.graph_objects import Figure

from tradeexecutor.strategy.chart.definition import ChartInput
from tradeexecutor.analysis.multi_asset_benchmark import get_benchmark_data, DEFAULT_BENCHMARK_ASSETS
from tradeexecutor.visual.benchmark import visualise_equity_curve_benchmark
from tradeexecutor.visual.equity_curve import calculate_equity_curve, calculate_returns, visualise_equity_curve
from matplotlib.figure import Figure as MatplotlibFigure

def equity_curve(
    input: ChartInput,
    max_benchmark_count=4,
    benchmark_token_symbols: list[str]=None,
) -> Figure:
    """Render equity curve for the strategy.

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

    benchmark_indexes = get_benchmark_data(
        strategy_universe,
        cumulative_with_initial_cash=state.portfolio.get_initial_cash(),
        max_count=max_benchmark_count,
        start_at=state.get_trading_time_range()[0],
        interesting_assets=benchmark_token_symbols,
    )

    fig = visualise_equity_curve_benchmark(
        state=state,
        benchmark_indexes=benchmark_indexes,
        height=800,
        log_y=True,
    )
    return fig


def equity_curve_with_drawdown(
    input: ChartInput,
) -> MatplotlibFigure:
    """Equity curve with drawdown.

    - Render the backtesting or live trade equity curve based on the state

    :return:
        Matplotlib figure
    """
    state = input.state
    curve = calculate_equity_curve(state)
    returns = calculate_returns(curve)
    fig = visualise_equity_curve(returns)
    return fig
