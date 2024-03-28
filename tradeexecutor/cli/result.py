"""Simple command line based backtesting result display."""
from IPython.core.display_functions import display

from tradeexecutor.analysis.trade_analyser import build_trade_analysis
from tradeexecutor.state.state import State
from tradeexecutor.statistics.key_metric import calculate_key_metrics
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.analysis.multi_asset_benchmark import compare_strategy_backtest_to_multiple_assets


def display_backtesting_results(
        state: State,
        strategy_universe: TradingStrategyUniverse = None
):
    """Print backtest result summary to terminal.

    - Used when running individual backtests from the terminal

    :param state:
        Backtest result state

    :param universe:
        The trading universe.

        If given, also output a benchmark.
    """
    analysis = build_trade_analysis(state.portfolio)
    summary = analysis.calculate_summary_statistics()
    display(summary.to_dataframe(format_headings=False))

    if strategy_universe is not None:
        # Nothing further to display
        portfolio_comparison = compare_strategy_backtest_to_multiple_assets(
            state,
            strategy_universe
        )
        display(portfolio_comparison)

    key_metrics = calculate_key_metrics(
        State(),
        state,
    )

    print("Individual stats")
    for metric in key_metrics:
        print(f"{metric.kind.name} = {metric.value}, from {metric.source}")

