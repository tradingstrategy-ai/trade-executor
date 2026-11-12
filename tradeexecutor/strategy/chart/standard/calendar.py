"""Quarterly and annual whole-wallet profit for fixed-capital backtests."""

import pandas as pd
from plotly.graph_objects import Figure

from tradeexecutor.strategy.chart.definition import ChartInput
from tradeexecutor.visual.calendar import calculate_calendar_profit, visualise_quarterly_profit
from tradeexecutor.visual.equity_curve import calculate_equity_curve


def quarterly_profit(
    input: ChartInput,
    title: str = "Quarterly whole-wallet returns",
    valuation_interval: pd.Timedelta = pd.Timedelta(days=1),
) -> tuple[Figure, pd.DataFrame]:
    """Return the percentage-only quarterly heatmap and its full-precision evidence table.

    Register as ``ChartKind.state_all_pairs``. Daily valuation grid by default;
    intraday backtests must pass their actual interval. Live fund flows require
    flow-adjusted returns and are deliberately unsupported here.
    """
    if input.live:
        raise ValueError("quarterly_profit requires a fixed-capital backtest")
    table = calculate_calendar_profit(
        calculate_equity_curve(input.state), input.state.portfolio.get_initial_cash(),
        input.start_at, input.end_at, valuation_interval=valuation_interval,
    )
    return visualise_quarterly_profit(table, title=title), table
