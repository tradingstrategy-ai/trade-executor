"""Summary statistics are displayed on the summary tiles of the strategies."""
import datetime
from typing import Optional

import pandas as pd
from numpy import isnan

from tradeexecutor.state.state import State
from tradeexecutor.state.statistics import calculate_naive_profitability
from tradeexecutor.strategy.execution_context import ExecutionMode
from tradeexecutor.strategy.summary import StrategySummaryStatistics
from tradeexecutor.visual.equity_curve import calculate_compounding_realised_trading_profitability


def calculate_summary_statistics(
        state: State,
        execution_mode: ExecutionMode,
        time_window = pd.Timedelta(days=90),
        now_: Optional[pd.Timestamp | datetime.datetime] = None,
        legacy_workarounds=False,
) -> StrategySummaryStatistics:
    """Preprocess the strategy statistics for the summary card in the web frontend.

    To test out in the :ref:`console`:

    .. code-block:: python

        from tradeexecutor.statistics.summary import calculate_summary_statistics
        from tradeexecutor.strategy.execution_context import ExecutionMode

        calculate_summary_statistics(state, ExecutionMode.preflight_check)

    :param state:
        Strategy state from which we calculate the summary

    :param execution_mode:
        If we need to skip calculations during backtesting

    :param time_window:
        How long we look back for the summary statistics

    :param now_:
        Override current time for unit testing.

        Set this to the date of the last trade.

    :param legacy_workarounds:
        Skip some calculations on old data, because data is missing.

    :return:
        Summary calculations for the summary tile,
        or empty `StrategySummaryStatistics` if cannot be calculated.
    """

    portfolio = state.portfolio

    # We can alway get the current value even if there are no trades
    current_value = portfolio.get_total_equity()

    first_trade, last_trade = portfolio.get_first_and_last_executed_trade()
    if first_trade is None:
        # No trades
        # Cannot calculate anything
        return StrategySummaryStatistics(current_value=current_value)

    first_trade_at = first_trade.executed_at
    last_trade_at = last_trade.executed_at

    if not now_:
        now_ = pd.Timestamp.utcnow().tz_localize(None)

    start_at = now_ - time_window

    stats = state.stats

    profitability_90_days = None
    enough_data = False
    performance_chart_90_days = None

    if len(stats.portfolio) > 0 and not legacy_workarounds:
        profitability = calculate_compounding_realised_trading_profitability(state)
        enough_data = len(profitability.index) > 1 and profitability.index[0] <= start_at
        profitability_time_windowed = profitability[start_at:]
        if len(profitability_time_windowed) > 0:
            profitability_daily = profitability_time_windowed.resample(pd.offsets.Day()).max()
            # We do not generate entry for dates without trades so forward fill from the previous day
            profitability_daily = profitability_daily.ffill()
            profitability_90_days = profitability_daily[-1]
            performance_chart_90_days = [(index.to_pydatetime(), value) for index, value in profitability_daily.iteritems()]
        else:
            profitability_90_days = None
            performance_chart_90_days = None

    return StrategySummaryStatistics(
        first_trade_at=first_trade_at,
        last_trade_at=last_trade_at,
        enough_data=enough_data,
        current_value=current_value,
        profitability_90_days=profitability_90_days,
        performance_chart_90_days=performance_chart_90_days,
    )
