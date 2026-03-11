"""Profit/loss breakdown calculations."""
import pandas as pd

from tradeexecutor.strategy.chart.definition import ChartInput

from tradeexecutor.analysis.multipair import analyse_multipair
from tradeexecutor.analysis.multipair import format_multipair_summary


def trading_pair_breakdown(
    input: ChartInput,
    limit=None,
    show_chain: bool = False,
    show_address: bool = False,
) -> pd.DataFrame:
    """Calculate profit/loss breakdown for each trading pair.

    Example:

    .. code-block:: python

        html = chart_renderer.render(trading_pair_breakdown, limit=5)
        display(html)

    :param limit:
        Only display top N entries, not to clutter output.

    :param show_chain:
        Add a "Chain" column with the chain name for each pair.

    :param show_address:
        Display the trading pair smart contract address.

    :return: DataFrame with profit/loss breakdown per trading pair.
    """
    state = input.state
    multipair_summary = analyse_multipair(state, show_chain=show_chain, show_address=show_address)
    df = format_multipair_summary(multipair_summary, sort_column="Total return %", limit=limit)
    return df