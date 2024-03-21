"""Simple command line based backtesting result display."""
from IPython.core.display_functions import display

from tradeexecutor.analysis.trade_analyser import build_trade_analysis
from tradeexecutor.state.state import State
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


def display_backtesting_results(state: State, universe: TradingStrategyUniverse):
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

    if universe is None:
        # Nothing further to display
        return



