"""Strategy thinking and decide_trades() logging output."""

import pandas as pd

from pandas.io.formats.style import Styler

from tradeexecutor.strategy.chart.definition import ChartInput


def last_messages(
    input: ChartInput,
    count=2,
) -> Styler:
    """Open positions at the end of the backtest/currently.
    """
    state = input.state
    messages = state.visualisation.get_messages_tail(2)

    table = pd.Series(
        data=list(messages.values()),
        index=list(messages.keys()),
    )

    df = table.to_frame()

    styled = df.style.set_properties(**{
        'text-align': 'left',
        'white-space': 'pre-wrap',
    })
    return styled
