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

        # add horizontal line if needs be
        if line_val := plot.horizontal_line:
            _add_hline(fig, line_val, start_at, end_at, cur_row, plot)

        

def _get_initial_row(volume_bar_mode: VolumeBarMode):
    """Get first row for plot based on volume bar mode."""
    if volume_bar_mode in {VolumeBarMode.hidden, VolumeBarMode.overlay}:
        cur_row = 1
    elif volume_bar_mode == VolumeBarMode.separate:
        cur_row = 2
    else:
        raise ValueError("Unknown volume bar mode: %s" % volume_bar_mode)
    return cur_row

def _add_hline(
    fig: go.Figure, 
    line_val: float,
    start_at: pd.Timestamp | None, 
    end_at: pd.Timestamp | None, 
    cur_row: int, 
    plot: Plot,
):
    """Add horizontal line to plot"""
    
    minimum = min(plot.points.values())
    maximum = max(plot.points.values())
    
    assert minimum < line_val < maximum, f"Horizontal line value must be within range of plot. Plot range: {minimum} - {maximum}"
    
    start_at, end_at = _get_start_and_end(start_at, end_at, plot)

    horizontal_line_colour = plot.horizontal_line_colour or "grey"

    # Add a horizontal line to the first subplot
    fig.add_shape(
            type="line",
            x0=start_at,
            y0=line_val,
            x1=end_at,
            y1=line_val,
            line=dict(color=horizontal_line_colour, width=1),
            row=cur_row, col=1
        )

def _get_start_and_end(
    start_at: pd.Timestamp | None, 
    end_at: pd.Timestamp | None, 
    plot: Plot,
):
    """Get first and last entry from plot if start_at and end_at are not set."""
    if not start_at and end_at:
        start_at = plot.get_first_entry()[0]
        end_at = plot.get_last_entry()[0]
    return start_at,end_at


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
        if plot.plot_shape == PlotShape.marker:
            return go.Scatter(
                x=df["timestamp"],
                y=df["value"],
                mode="markers",
                name=plot.name,
                line=dict(color=plot.colour),
            )
        elif plot.plot_shape in {PlotShape.linear, PlotShape.horizontal_vertical}:
            return go.Scatter(
                x=df["timestamp"],
                y=df["value"],
                mode="lines",
                name=plot.name,
                line=dict(color=plot.colour),
                line_shape=plot.plot_shape.value
            )
        else:
            raise ValueError(f"Unknown plot shape: {plot.plot_shape}")
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
