"""Alpha model charts and diagnostics."""

import pandas as pd

from tradeexecutor.strategy.alpha_model import format_signals
from tradeexecutor.strategy.chart.definition import ChartInput


def alpha_model_diagnostics(
    input: ChartInput,
    count=2,
) -> pd.DataFrame:
    """Alpha model output for the last cycle.
    """
    state = input.state
    alpha_model = state.visualisation.discardable_data["alpha_model"]
    df = format_signals(alpha_model, signal_type="all")
    return df
