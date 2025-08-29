"""Profit/loss breakdown calculations."""
import pandas as pd

from tradeexecutor.strategy.chart.definition import ChartInput

from tradeexecutor.analysis.multipair import analyse_multipair
from tradeexecutor.analysis.multipair import format_multipair_summary


def trading_pair_breakdown(
    input: ChartInput,
    limit=None,
) -> pd.DataFrame:
    """Calculate profit/loss breakdown for each trading pair.

    Example:

    .. code-block:: python

        html = chart_renderer.render(trading_pair_breakdown, limit=5)
        display(html)

    :param limit:
        Only display top N entries, not to clutter output.

    :return: DataFrame with profit/loss breakdown per trading pair.
    """
    state = input.state
    multipair_summary = analyse_multipair(state)
    df = format_multipair_summary(multipair_summary, sort_column="Total return %", limit=limit)
    return df