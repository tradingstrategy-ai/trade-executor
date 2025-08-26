"""Common day trading metrics."""

import pandas as pd

from tradeexecutor.strategy.chart.definition import ChartInput

from tradeexecutor.analysis.trade_analyser import build_trade_analysis

def trading_metrics(
    input: ChartInput,
) -> pd.DataFrame:
    """Common day trading metrics.

    :return: DataFrame with profit/loss breakdown per trading pair.
    """
    state = input.state


    analysis = build_trade_analysis(state.portfolio)
    summary = analysis.calculate_summary_statistics()
    df = summary.to_dataframe(format_headings=False)
    return df
