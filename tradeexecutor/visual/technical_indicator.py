"""Technical indicator visualisation.

- Technical indicators are :py:class:`tradeexecutor.state.visualisation.Plot` objects populated during decide_trades().

- We use Plotly Scatter to draw the indicators
"""

import logging
from typing import Optional

import pandas as pd
from plotly import graph_objects as go


from tradeexecutor.state.visualisation import (
    Visualisation, Plot, PlotKind, PlotShape
)

from tradingstrategy.charting.candle_chart import VolumeBarMode


logger = logging.getLogger(__name__)


def overlay_all_technical_indicators(
        fig: go.Figure,
        visualisation: Visualisation,
        start_at: Optional[pd.Timestamp] = None,
        end_at: Optional[pd.Timestamp] = None,
        volume_bar_mode: VolumeBarMode = None,
):
    """Draw all technical indicators from the visualisation over candle chart.

    :param start_at:
        Crop range

    :param end_at:
        Crop range
    
    :param volume_bar_mode:
        How to draw volume bars e.g. overlay, seperate, hidden
    """

    # get starting row for indicators
    cur_row = _get_initial_row(volume_bar_mode)

    # https://plotly.com/python/graphing-multiple-chart-types/
    for plot in visualisation.plots.values():
        
        # get trace which is unattached to plot
        trace = visualise_technical_indicator(
            plot,
            start_at,
            end_at,
        )
        
        # must have something to draw for plot
        if trace is None:
            raise ValueError(f"Unknown plot kind: {plot.plot_kind}")
        
        # add trace to plot
        if plot.kind == PlotKind.technical_indicator_detached:
            cur_row += 1
            fig.add_trace(trace, row=cur_row, col=1)
        elif plot.kind == PlotKind.technical_indicator_on_price:
            # don't increment current row
            fig.add_trace(trace, row=1, col=1)
        elif plot.kind == PlotKind.technical_indicator_overlay_on_detached:
            # don't increment current row
            fig.add_trace(trace, row=cur_row, col=1)
        else:
            raise ValueError(f"Unknown plot kind: {plot.plot_kind}")

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

    if len(df) <= 0:
        return None
    
    return _get_plot(df, plot)

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


def _get_initial_row(volume_bar_mode: VolumeBarMode):
    """Get first row for plot based on volume bar mode."""
    if volume_bar_mode in {VolumeBarMode.hidden, VolumeBarMode.overlay}:
        cur_row = 1
    elif volume_bar_mode == VolumeBarMode.separate:
        cur_row = 2
    else:
        raise ValueError("Unknown volume bar mode: %s" % volume_bar_mode)
    return cur_row

def _get_plot(df: pd.DataFrame, plot: Plot):
    """Get plot based on plot shape and plot size."""
    if plot.plot_shape == PlotShape.markers:
        return _get_marker_plot(df, plot)
    elif plot.plot_shape in {PlotShape.linear, PlotShape.horizontal_vertical}:
        return _get_linear_plot(df, plot)
    else:
        raise ValueError(f"Unknown plot shape: {plot.plot_shape}")

def _get_marker_plot(df: pd.DataFrame, plot: Plot) -> go.Scatter:
    """Get marker plot for event indicators i.e. individual points."""
    if plot.indicator_size:
        return go.Scatter(
            x=df["timestamp"],
            y=df["value"],
            mode="markers",
            name=plot.name,
            marker=dict(color=plot.colour, size=plot.indicator_size),
        )
    else:
        return go.Scatter(
            x=df["timestamp"],
            y=df["value"],
            mode="markers",
            name=plot.name,
            marker=dict(color=plot.colour),
        )

def _get_linear_plot(df: pd.DataFrame, plot: Plot) -> go.Scatter:
    """Get linear plot for standard indicators i.e. lines."""    
    if plot.indicator_size:
        return go.Scatter(
            x=df["timestamp"],
            y=df["value"],
            mode="lines",
            name=plot.name,
            line=dict(color=plot.colour, width=plot.indicator_size),
            line_shape=plot.plot_shape.value
        )
    else:
        return go.Scatter(
            x=df["timestamp"],
            y=df["value"],
            mode="lines",
            name=plot.name,
            line=dict(color=plot.colour),
            line_shape=plot.plot_shape.value
        )