"""Summary statistics are displayed on the summary tiles of the strategies."""
import datetime

from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.state import State
from tradeexecutor.strategy.execution_context import ExecutionMode
from tradeexecutor.strategy.summary import StrategySummaryStatistics


def calculate_summary_statistics(
        clock: datetime.datetime,
        state: State,
        execution_mode: ExecutionMode) -> StrategySummaryStatistics:
    """Preprocess the strategy statistics for the summary card."""

    portfolio = state.portfolio

    first_trade_at, last_trade_at = portfolio.get_first_and_last_executed_trade()

    if first_trade_at:
        enough_data = datetime.datetime.utcnow() - first_trade_at
    else:
        enough_data = False

    current_value = portfolio.get_total_equity()

    