"""Technical indicator visualisation.

- Technical indicators are :py:class:`tradeexecutor.state.visualisation.Plot` objects populated during decide_trades().

- We use Plotly Scatter to draw the indicators
"""

import logging
from typing import Optional

import pandas as pd
from plotly import graph_objects as go


from tradeexecutor.state.visualisation import Visualisation, Plot, PlotKind


logger = logging.getLogger(__name__)


def overlay_all_technical_indicators(
        fig: go.Figure,
        visualisation: Visualisation,
        start_at: Optional[pd.Timestamp] = None,
        end_at: Optional[pd.Timestamp] = None
):
    """Draw all technical indicators from the visualisation over candle chart.

    :param start_at:
        Crop range

    :param end_at:
        Crop range
    """

    current_row = 2
    
    # https://plotly.com/python/graphing-multiple-chart-types/
    for plot_id, plot in visualisation.plots.items():
        trace = visualise_technical_indicator(
            plot,
            start_at,
            end_at,
        )
        if trace is not None:
            if plot.kind == PlotKind.technical_indicator_on_price:
                fig.add_trace(trace, row=1, col=1)
            elif plot.kind == PlotKind.technical_indicator_detached:
                fig.print_grid()
                fig.add_trace(trace, row=current_row, col=1)
                current_row += 1
            else:
                raise ValueError("Unknown plot kind: %s" % plot.plot_kind)


def visualise_technical_indicator(
        plot: Plot,
        start_at: Optional[pd.Timestamp] = None,
        end_at: Optional[pd.Timestamp] = None
) -> Optional[go.Scatter]:
    """Convert a single technical indicator to Plotly trace object.

    :param plot:
        A stateful plot object created in decide_trades()

    :return:
        Plotly Trace object.

        If there are no data points to draw return `None`.
    """

    logger.debug("Visualising %s, %s - %s", plot.name, start_at, end_at)

    # Convert data columnar format
    df = export_plot_as_dataframe(plot, start_at, end_at)

    if len(df) > 0:
        return go.Scatter(
            x=df["timestamp"],
            y=df["value"],
            mode="lines",
            name=plot.name,
            line=dict(color=plot.colour),
            line_shape=plot.plot_shape.value
        )
    else:
        return None


def export_plot_as_dataframe(
        plot: Plot,
        start_at: Optional[pd.Timestamp] = None,
        end_at: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Convert visualisation state to Plotly friendly df.

    :param plot:
        Internal Plot object from the strategy state.

    :param start_at:
        Crop range

    :param end_at:
        Crop range
    """
    data = []
    for time, value in plot.points.items():
        time = pd.to_datetime(time, unit='s')

        if start_at or end_at:
            if time < start_at or time > end_at:
                continue

        data.append({
            "timestamp": time,
            "value": value,
        })

    # set_index fails if the frame is empty
    if len(data) == 0:
        return pd.DataFrame()

    # Convert timestamp to pd.Timestamp column
    df = pd.DataFrame(data)
    df = df.set_index(pd.DatetimeIndex(df["timestamp"]))
    return df
