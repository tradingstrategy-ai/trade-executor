"""Technical indicator visualisation.

- Technical indicators are :py:class:`tradeexecutor.state.visualisation.Plot` objects populated during decide_trades().

- We use Plotly Scatter to draw the indicators
"""

import logging
from typing import Optional

import pandas as pd
from plotly import graph_objects as go


from tradeexecutor.state.visualisation import (
    Visualisation, Plot, PlotKind, PlotShape, RecordingTime, PlotLabel
)

from tradingstrategy.charting.candle_chart import VolumeBarMode


logger = logging.getLogger(__name__)


def overlay_all_technical_indicators(
    fig: go.Figure,
    visualisation: Visualisation,
    start_at: Optional[pd.Timestamp] = None,
    end_at: Optional[pd.Timestamp] = None,
    volume_bar_mode: VolumeBarMode = None,
    pair_id: Optional[int] = None,
    start_row: int = None,
    detached_indicators: bool = True,
):
    """Draw all technical indicators from the visualisation over candle chart.

    :param start_at:
        Crop range

    :param end_at:
        Crop range
    
    :param volume_bar_mode:
        How to draw volume bars e.g. overlay, seperate, hidden

    :param pair_id:
        If set, only draw indicators for this pair

    :param start_row:
        If set, start drawing indicators from this row. Used for multipair visualisation.

    :param detached_indicators:
        If set, draw detached indicators.
    """

    # get starting row for indicators
    # should start on candlestick row if volume is overlayed or hidden
    cur_row = _get_initial_row(volume_bar_mode)

    if start_row:
        cur_row = cur_row + start_row - 1

        # currently, this line breaks single pair visualisation without breaking changes, so needs to be here
        plots = [plot for plot in visualisation.plots.values() if getattr(plot.pair, "internal_id", None) == pair_id]

    else:
        start_row = 1
        plots = visualisation.plots.values()
    
    # only for PlotKind.technical_indicator_detached
    detached_plots_to_row = {}

    # https://plotly.com/python/graphing-multiple-chart-types/
    for plot in plots:

        if plot.kind in {PlotKind.technical_indicator_detached, PlotKind.technical_indicator_overlay_on_detached} and not detached_indicators:
            continue

        # get trace which is unattached to plot
        trace = visualise_technical_indicator(
            plot,
            start_at,
            end_at,
        )

        if trace is None:
            # Likely zero data points
            logger.info("Did not receive trace for plot %s. Maybe this plot had zero data points?", plot)
            return
        
        # add trace to plot
        # only increment row if detached plot
        if plot.kind == PlotKind.technical_indicator_detached:
            cur_row += 1
            detached_plots_to_row[plot.name] = cur_row
            fig.add_trace(trace, row=cur_row, col=1)
        elif plot.kind == PlotKind.technical_indicator_on_price:
            fig.add_trace(trace, row=start_row, col=1)
        elif plot.kind == PlotKind.technical_indicator_overlay_on_detached:
            fig.add_trace(trace, row=detached_plots_to_row[plot.detached_overlay_name], col=1)
        else:
            raise NotImplementedError(f"Unknown plot kind: {plot.kind}")


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
    correct_look_ahead_bias_negation=True,
) -> pd.DataFrame:
    """Convert visualisation state to Plotly friendly df.

    :param plot:
        Internal Plot object from the strategy state.

    :param start_at:
        Crop range

    :param end_at:
        Crop range

    :param correct_look_ahead_bias_negation:
        How many candles to shift the data if the source plot was adjusted for the look ahead bias.

        See :py:class:`tradeexecutor.state.visualisation.RecordingTime` for more information.
    """
    if start_at or end_at:
        start_at_unix = start_at.timestamp()
        end_at_unix = end_at.timestamp()

        data = [{"timestamp": time, "value": value} for time, value in plot.points.items() if start_at_unix <= time <= end_at_unix]
    else:
        data = [{"timestamp": time, "value": value} for time, value in plot.points.items()]

    # set_index fails if the frame is empty
    if len(data) == 0:
        return pd.DataFrame()

    # Convert timestamp to pd.Timestamp column
    df = pd.DataFrame(data)
    # .values claims to offer extra speedup trick
    df["timestamp"] = pd.to_datetime(df["timestamp"].values, unit='s', utc=True).tz_localize(None)
    df = df.set_index(pd.DatetimeIndex(df["timestamp"]))

    # TODO: Not a perfect implementation, will
    # fix this later, as currently we are not recording the decision making cycle on plots
    if correct_look_ahead_bias_negation:
        if plot.recording_time == RecordingTime.decision_making_time:
            df = df.shift(periods=-1)
            # We cannot use timestamp as a column anymore, because it is now out of sync,
            # use index only
            df = df.drop(labels=["timestamp"], axis="columns")
            # TODO: Some functions consume timestamp column,
            # when they should use index
            df["timestamp"] = df.index

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


def _get_plot(df: pd.DataFrame, plot: Plot) -> go.Scatter:
    """Get plot based on plot shape and plot size."""
    if plot.plot_shape == PlotShape.markers:
        scatter = _get_marker_plot(df, plot)
    elif plot.plot_shape in {PlotShape.linear, PlotShape.horizontal_vertical}:
        scatter = _get_linear_plot(df, plot)
    else:
        raise ValueError(f"Unknown plot shape: {plot.plot_shape}")

    return scatter


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