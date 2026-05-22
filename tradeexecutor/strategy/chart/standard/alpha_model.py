"""Alpha model charts and diagnostics."""

import pandas as pd
import plotly.express as px
from pandas.io.formats.style import Styler
from plotly.graph_objects import Figure

from tradeexecutor.analysis.vault_missed_events import (
    analyse_missed_vault_deposit_redemption_event_timeline,
    analyse_missed_vault_deposit_redemption_events,
    format_missed_vault_deposit_redemption_events,
)
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


def missed_vault_deposit_redemption_events(input: ChartInput) -> Styler:
    """Missed vault deposit and redemption event summary."""
    df = analyse_missed_vault_deposit_redemption_events(input.state)
    return format_missed_vault_deposit_redemption_events(df)


def missed_vault_deposit_redemption_timeline(input: ChartInput) -> Figure:
    """Missed vault deposit and redemption events over time."""
    df = analyse_missed_vault_deposit_redemption_event_timeline(input.state)

    if df.empty:
        fig = Figure()
        fig.update_layout(
            title="Missed vault deposit and redemption events over time",
            xaxis_title="Time",
            yaxis_title="Missed US dollar",
            template="plotly_dark",
        )
        return fig

    fig = px.bar(
        df,
        x="Timestamp",
        y="Missed US dollar",
        color="Vault name",
        facet_row="Event type",
        barmode="stack",
        title="Missed vault deposit and redemption events over time",
        labels={
            "Timestamp": "Time",
            "Missed US dollar": "Missed US dollar",
            "Vault name": "Vault",
            "Event type": "Event type",
            "Missed event count": "Missed events",
        },
        hover_data={
            "Missed event count": True,
            "Missed US dollar": ":$,.2f",
        },
        template="plotly_dark",
    )
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    fig.update_layout(legend_title_text="Vault", bargap=0.05)
    return fig
