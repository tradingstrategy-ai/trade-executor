"""Alpha model charts and diagnostics."""

import pandas as pd
from tradeexecutor.analysis.vault_missed_events import analyse_missed_vault_deposit_redemption_events
from tradeexecutor.strategy.alpha_model import format_signals
from tradeexecutor.strategy.chart.definition import ChartInput


def alpha_model_diagnostics(
    input: ChartInput,
    count=2,
) -> pd.DataFrame:
    """Alpha model output for the last cycle.
    """
    state = input.state
    alpha_model = state.visualisation.discardable_data.get("alpha_model")
    if alpha_model is None:
        return pd.DataFrame([])
    df = format_signals(alpha_model, signal_type="all")
    return df


def missed_vault_deposit_redemption_events(input: ChartInput) -> pd.DataFrame:
    """Missed vault deposit and redemption event summary."""
    return analyse_missed_vault_deposit_redemption_events(input.state)
