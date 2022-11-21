"""Simple command line based backtesting result display."""
from tradeexecutor.analysis.trade_analyser import build_trade_analysis
from tradeexecutor.state.state import State


def display_backtesting_results(state: State):
    """Print backtest result summary to terminal."""

    analysis = build_trade_analysis(state.portfolio)
    summary = analysis.calculate_summary_statistics()
    print(summary.to_dataframe())
