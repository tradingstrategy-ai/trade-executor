"""Technical indicator visualisation.

- Technical indicators are :py:class:`tradeexecutor.state.visualisation.Plot` objects populated during decide_trades().

- We use Plotly Scatter to draw the indicators
"""

import logging
from typing import Optional

import pandas as pd
from plotly import graph_objects as go


from tradeexecutor.state.visualisation import Visualisation, Plot, PlotKind

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
    """

    # get starting row for indicators
    cur_row = _get_initial_row(volume_bar_mode)

    # https://plotly.com/python/graphing-multiple-chart-types/
    for plot in visualisation.plots.values():
        trace = visualise_technical_indicator(
            plot,
            start_at,
            end_at,
        )
        
        # must have something to draw for plot
        if trace is None:
            raise ValueError(f"Unknown plot kind: {plot.plot_kind}")
        
        # add trace to plot
        fig.add_trace(trace, row=cur_row, col=1)

        # add horizontal line if needs be
        _add_hline(fig, start_at, end_at, cur_row, plot)

        # on to next row
        cur_row += 1

def _get_initial_row(volume_bar_mode):
    """Get first row for plot based on volume bar mode."""
    if volume_bar_mode in {VolumeBarMode.hidden, VolumeBarMode.overlay}:
        cur_row = 1
    elif volume_bar_mode == VolumeBarMode.separate:
        cur_row = 3
    else:
        raise ValueError("Unknown volume bar mode: %s" % volume_bar_mode)
    return cur_row

def _add_hline(fig, start_at, end_at, cur_row, plot):
    """Add horizontal line to plot if needed."""
    if line_val := plot.horizontal_line:
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

def _get_start_and_end(start_at, end_at, plot):
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
