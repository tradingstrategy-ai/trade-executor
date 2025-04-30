"""Visualisation of a strategy.

The visualisation part of the state shows "how strategy is thinking."
All information stored is dianogtics information and is not consumed in the actual
decision making - the data is just derivates for decision making process and raw data.

Visualisation data is filled by the backtest, or by timepoint-by-timepoint
by a live strategy. Visualisation includes

- Debug messages

- Technical indicators on a graph
"""

import datetime
import enum
from dataclasses import dataclass, field
from types import NoneType
from typing import List, Dict, Optional, Any, Union, Tuple, Callable

import numpy as np
import pandas as pd
import logging

from dataclasses_json import dataclass_json
from tradingstrategy.pair import DEXPair
from tradingstrategy.types import PrimaryKey

from tradeexecutor.utils.timestamp import convert_and_validate_timestamp, convert_and_validate_timestamp_as_int
from tradeexecutor.state.identifier import TradingPairIdentifier

logger = logging.getLogger(__name__)


class PlotKind(enum.Enum):
    """What different plots a strategy can output.

    See :py:meth:`Visualisation.plot_indicator`.
    """

    #: This plot is drawn on the top of the price graph
    technical_indicator_on_price = "technical_indicator_on_price"

    #: This plot is drawn below the price graph as a separate chart.
    #:
    #:
    technical_indicator_detached = "technical_indicator_detached"

    #: This plot is overlaid on a detached indicator plot.
    #:
    #:
    technical_indicator_overlay_on_detached = "technical_indicator_overlay"


class PlotLabel(enum.Enum):
    """How do we render the plot label.

    See :py:meth:`Visualisation.plot_indicator`.

    Example:

    .. code-block:: python

        # Draw BTC + ETH RSI between its trigger zones for this pair of we got a valid value for RSI for this pair

        # BTC RSI daily
        visualisation.plot_indicator(
            timestamp,
            f"RSI",
            PlotKind.technical_indicator_detached,
            current_rsi_values[btc_pair],
            colour="orange",
        )

        # ETH RSI daily
        visualisation.plot_indicator(
            timestamp,
            f"RSI ETH",
            PlotKind.technical_indicator_overlay_on_detached,
            current_rsi_values[eth_pair],
            colour="blue",
            label=PlotLabel.hidden,
            detached_overlay_name=f"RSI",
        )

        # Low (vertical line)
        visualisation.plot_indicator(
            timestamp,
            f"RSI low trigger",
            PlotKind.technical_indicator_overlay_on_detached,
            rsi_low,
            detached_overlay_name=f"RSI",
            plot_shape=PlotShape.horizontal_vertical,
            colour="red",
            label=PlotLabel.hidden,
        )

        # High (vertical line)
        visualisation.plot_indicator(
            timestamp,
            f"RSI high trigger",
            PlotKind.technical_indicator_overlay_on_detached,
            rsi_high,
            detached_overlay_name=f"RSI",
            plot_shape=PlotShape.horizontal_vertical,
            colour="red",
            label=PlotLabel.hidden,
        )
    """

    #: Render the plot label as the axis description left or right.
    axis = "axis"

    #: This plot label is hidden on the axis.
    #:
    #: Plot names etc. will still appear in mouse hovers.
    #:
    hidden = "hidden"


class PlotShape(enum.Enum):
    """What kind of shape is this plot.

    Describes the various shapes that a line can take in a plot. See discussion: https://github.com/tradingstrategy-ai/trade-executor/pull/156#discussion_r1058819823

    See :py:meth:`Visualisation.plot_indicator`.
    """

    #: Standard linear line. Used in most cases. 
    linear = "linear"

    #: Is the line horizontal-vertical. Used for stop loss line.
    #:
    #: See https://plotly.com/python/line-charts/?_ga=2.83222870.1162358725.1672302619-1029023258.1667666588#interpolation-with-line-plots
    horizontal_vertical = "hv"

    #: Individually specified points.
    #: 
    #: Typically used for event indicators e.g. cross over of two lines.
    markers = "markers"


class PlotKind(enum.Enum):
    """What different plots a strategy can output.

    See :py:meth:`Visualisation.plot_indicator`.
    """

    #: This plot is drawn on the top of the price graph
    technical_indicator_on_price = "technical_indicator_on_price"

    #: This plot is drawn below the price graph as a separate chart.
    #:
    #:
    technical_indicator_detached = "technical_indicator_detached"

    #: This plot is overlaid on a detached indicator plot.
    #:
    #:
    technical_indicator_overlay_on_detached = "technical_indicator_overlay"


class RecordingTime(enum.Enum):
    """At what timestamp this data is being recorded.

    Are we avoiding look ahead bias or not.

    All decisions are made on a candle open. During the decision,
    we can access previous candle close that is more or less equivalent
    of the current candle open. But we cannot access the current candle high,
    low, or close values because those have not happened yet.

    When the data is recorded by `decide_trades` it is recording
    the previous cycle data used in the decision making. For daily candles, a decision today
    is based on the data of yesterday. Thus this data does not represent
    technical indicator calculations over time series, but decision making inputs.
    This is to avoid look ahead bias.

    This causes one decision making cycle lag charts when comparing charts
    with market analysis charts, because those charts operate on the current
    candle open, high, low, close values.

    Because decision making visualisation is uncommon and causes confusion,
    we will later correct this in :py:mod:`tradeexecutor.visual.techical_indicator`,
    so that plots are shifted to match their market occurence timestamps.

    See :py:meth:`Visualisation.plot_indicator`.
    """

    #: The plot value was recorded for decision making.
    #:
    #: Shift data to match market timestamps.
    decision_making_time = "decision_making_time"

    #: The plot value represents technical indicator at the time of the market.
    #:
    #: Accessing the latest item in this plot cannot be used
    #: for decision making, but charts are natively sync
    #: with other market analysis charts.
    market_time = "market_time"


@dataclass_json
@dataclass(slots=True)
class Plot:
    """Descibe singe plot on a strategy.

    Plot is usually displayed as an overlay line over the price chart.
    E.g. simple moving average over price candles.
    """

    #: Name of this plot
    name: str

    #: What kind of a plot we are drawing
    kind: PlotKind

    #: One of Plotly colour names
    #: https://community.plotly.com/t/plotly-colours-list/11730/2
    colour: Optional[str] = None

    #: Points of this plot.
    #:
    #: TODO:
    #: Because we cannot use datetime.datetime directly as a key in JSON,
    #: we use UNIX timestamp here to keep our state easily serialisable.
    #:
    #: Also note that entries may not be in order - you might need
    #: to sort the output yourself.
    points: Dict[int, float] = field(default_factory=dict)

    #: Standard is linear. 
    #: Alternative is horizontal-vertical which can be used for stop loss line. 
    #: See https://plotly.com/python/line-charts/?_ga=2.83222870.1162358725.1672302619-1029023258.1667666588#interpolation-with-line-plots
    plot_shape: Optional[PlotShape] = PlotShape.linear

    #: If this plot is overlayed on top of a detached technical indicator, this is the name of the overlay it should be attached to.
    detached_overlay_name: Optional[str] = None

    #: Optional indicator to determine the size of the indicator. 
    #: 
    #: For a line, this is the width of the line. 
    #:
    #: For a marker, this is the size of the marker.
    indicator_size: Optional[float] = None

    #: What is the recording time for this plot.
    #:
    #: Are we adjusted for look ahead bias or not.
    #:
    recording_time: RecordingTime = RecordingTime.decision_making_time

    #: The trading pair this plot is for.
    #:
    #: Plots are not necessarily restricted to a single trading pair, so this is optional.
    #:
    pair: Optional[TradingPairIdentifier] = None

    #: How do we render label for this plot
    #:
    label: Optional[PlotLabel] = None

    #: Height hint for the rendering.
    #:
    #: Currently not supported. See :py:meth:`Visualisation.plot_indicator` for comments.
    #:
    height: Optional[int] = None

    def __repr__(self):
        return f"<Plot name:{self.name} kind:{self.kind.name} with {len(self.points)} points>"

    def add_point(self,
                  timestamp: datetime.datetime,
                  value: float,
                  ):
        assert isinstance(timestamp, datetime.datetime)
        assert isinstance(value, (float, NoneType)), f"Got {value} ({value.__class__})"
        timestamp = convert_and_validate_timestamp_as_int(timestamp)
        logger.info("Plotting %s at %s: %s", self.name, timestamp, value)
        # This condition is untrue if we run --run-single-cycle twice to debug the strategy
        # assert timestamp not in self.points, f"Plot {self.name} aleady has point for timestamp {timestamp}"
        self.points[timestamp] = value

    def get_last_value(self) -> float:
        """Assume points is an ordered dict."""
        # https://stackoverflow.com/a/63059166/315168
        key = next(reversed(self.points.keys()))
        return self.points[key]

    def get_last_entry(self) -> Tuple[datetime.datetime, float]:
        """Get the last entry in this plot.

        :return:
            timestamp, value tuple
        """
        last_entry = max(self.points.keys())
        last_value = self.points[last_entry]

        return datetime.datetime.utcfromtimestamp(last_entry), last_value

    def get_first_entry(self) -> Tuple[datetime.datetime, float]:
        """Get the first entry in this plot.

        :return:
            timestamp, value tuple
        """
        first_entry = min(self.points.keys())
        first_value = self.points[first_entry]

        return datetime.datetime.utcfromtimestamp(first_entry), first_value

    def get_entries(self) -> List[Tuple[datetime.datetime, float]]:
        """Get entries as a sorted list."

        :return:
            List[timestamp, value], oldest timestamp first
        """
        sorted_entries = []
        keys = sorted(self.points.keys())
        for k in keys:
            timestamp = datetime.datetime.utcfromtimestamp(k)
            v = self.points[k]
            sorted_entries.append((timestamp, v))
        return sorted_entries


@dataclass_json
@dataclass
class Visualisation:
    """Strategy visualisation helper.

    This object is returned from the strategy execution cycle.
    It allows you to plot values, add debug messages, etc.
    It is not used in any trading, but can help and visualize
    trade backtesting and execution.

    - See :py:meth:`plot_indicator` for how to draw plots inside `decide_trades()`

    - See :py:func:`get_series` how to extract data for charting as :py:class:`pd.Series`.
    """

    #: Messages for each strategy cycle.
    #:
    #: Because we cannot use datetime.datetime directly as a key in JSON,
    #: we use UNIX timestamp here to keep our state easily serialisable.
    messages: Dict[int, List[str]] = field(default_factory=dict)

    #: Extra calculation diagnostics for each strategy cycle.
    #:
    #: Cycle -> dict of values mappings.
    #:
    #: Currently used to record the alpha model state when doing
    #: doing portfolio construction modelling.
    #:
    #: Because we cannot use datetime.datetime directly as a key in JSON,
    #: we use UNIX timestamp here to keep our state easily serialisable.
    calculations: Dict[int, dict] = field(default_factory=dict)

    # Extra technical indicator outputs.
    #:
    #: Name -> Plot value mappings
    #:
    plots: Dict[str, Plot] = field(default_factory=dict)

    #: Trading pair IDs to visualise instead of the default trading pairs from 
    #: strategy universe.
    pair_ids: list[PrimaryKey] = field(default_factory=list)

    #: Data for which we only keep the last value.
    #:
    #: Most useful in unit test/backtest debugging.
    #:
    #: See :py:meth:`set_discardable_data`.
    #:
    discardable_data: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        plot_count = len(self.plots)
        point_count = sum([len(p.points) for p in self.plots.values()])
        return f"<Visualisation with {plot_count} plots and {point_count} data points>"

    def set_visualised_pairs(self, pairs: list[TradingPairIdentifier]) -> None:
        """Set the trading pair for this plot.

        :param pairs:
            Use these trading pairs to plot candle charts instead of the default
            trading pairs from strategy universe.
        """
        assert isinstance(pairs, list)
        for pair in pairs:
            assert isinstance(pair, TradingPairIdentifier), f"Unexpected pair type, got {pair} of type {type(pair)}"
            self.pair_ids.append(pair.internal_id)

    def add_message(
            self,
            timestamp: datetime.datetime,
            content: str
    ):
        """Write a debug message.

        - Each message is associated to a different timepoint

        - Multiple messages per timepoint are possible

        - The first message is in strategy thinking Discord/Telegram logging output for each cycle

        To display in backtesting notebook:

        For example see :py:meth:`get_messages_tail`.

        :param timestamp:
            The current strategy cycle timestamp

        :param content:
            The contents of the message

        """
        assert type(content) == str, f"Got {type(content)}"
        timestamp = convert_and_validate_timestamp_as_int(timestamp)
        timepoint_messages = self.messages.get(timestamp, list())
        timepoint_messages.append(content)
        self.messages[timestamp] = timepoint_messages

    def get_messages_tail(self, count: int, filter_func: Callable = None) -> dict[datetime.datetime, str]:
        """Get N latest messages.

        - If there are multiple messages per timepoint get only one.

        Example:

        .. code-block:: python

            # Find rebalance event
            def find_rebalance_log_messages(msg: str):
                return "Rebalanced: 👍" in msg

            messages = state.visualisation.get_messages_tail(10, filter_func=find_rebalance_log_messages)

            table = pd.Series(
                data=list(messages.values()),
                index=list(messages.keys()),
            )

            df = table.to_frame()

            display(df.style.set_properties(**{
                'text-align': 'left',
                'white-space': 'pre-wrap',
            }))

        :param count:
            How many last messages to return.

        :param filter_func:
            Match messages against this filter.

            Matches the first message for a timepoint.
        """
        message_tuples = sorted(self.messages.items(), key=lambda x: x[0], reverse=True)
        result = {}

        if filter_func:
            for msg_data in message_tuples:
                unix_time = msg_data[0]
                messages = msg_data[1]

                if messages:

                    if filter_func is not None:
                        if not filter_func(messages[0]):
                            continue

                    # We only want the first message for this timepoint
                    timestamp = datetime.datetime.utcfromtimestamp(unix_time)
                    result[timestamp] = "\n".join(messages)

            last_n = dict(list(result.items())[-count:])
            return last_n

        else:
            for msg_data in message_tuples[0:count]:
                unix_time = msg_data[0]
                messages = msg_data[1]

                if messages:
                    # We only want the first message for this timepoint
                    timestamp = datetime.datetime.utcfromtimestamp(unix_time)
                    result[timestamp] = "\n".join(messages)

        return result

    def add_calculations(
            self,
            timestamp: datetime.datetime,
            cycle_calculations: dict
    ):
        """Update strategy cycle calculations diagnostics.

        - Each strategy cycle can dump whatever intermediate
          calculations state on the visualisation record keeping,
          so that it can be later pulled up in the analysis.

        - Currently this is used to store the alpha model calculations
          for portfolio construction model.


        .. note ::

            Using this method may slow down your backtests because serialising ``cycle_calculations``
            might be slow. Avoid if not needed.

        :param timestamp:
            The current strategy cycle timestamp

        :param cycle_calculations:
            The contents of the calculations.

            Must be JSON serialisable dict.

        """

        assert isinstance(cycle_calculations, dict)
        timestamp = convert_and_validate_timestamp_as_int(timestamp)
        self.calculations[timestamp] = cycle_calculations

    def plot_indicator(
            self,
            timestamp: Union[datetime.datetime, pd.Timestamp],
            name: str,
            kind: PlotKind,
            value: float,
            colour: Optional[str] = None,
            plot_shape: Optional[PlotShape] = PlotShape.linear,
            detached_overlay_name: Optional[str] = None,
            indicator_size: Optional[float] = None,
            recording_time: Optional[RecordingTime] = RecordingTime.decision_making_time,
            pair: Optional[TradingPairIdentifier] = None,
            label: PlotLabel = PlotLabel.axis,
            height: Optional[int] = None,
    ):
        """Add a value to the output data and diagram.
        
        Plots are stored by their name.

        Example how to draw a detached RSI indicator and top/bottom indicator line for it:

        .. code-block:: python

            # Current daily
            visualisation.plot_indicator(
                timestamp,
                f"RSI {token}",
                PlotKind.technical_indicator_detached,
                current_rsi_values[pair],
            )

            # Low (vertical line)
            visualisation.plot_indicator(
                timestamp,
                f"RSI {token} low trigger",
                PlotKind.technical_indicator_overlay_on_detached,
                rsi_low,
                detached_overlay_name=f"RSI {token}",
                plot_shape=PlotShape.horizontal_vertical,
            )

            # High (vertical line)
            visualisation.plot_indicator(
                timestamp,
                f"RSI {token} high trigger",
                PlotKind.technical_indicator_overlay_on_detached,
                rsi_high,
                detached_overlay_name=f"RSI {token}",
                plot_shape=PlotShape.horizontal_vertical,
            )

        :param timestamp:
            The current strategy cycle timestamp

        :param name:
            The plot label

        :param kind:
            The plot type

        :param value:
            Current value e.g. price as USD

        :param colour:
            Optional colour 

        :param plot_shape:
            PlotShape enum value e.g. Plotshape.linear or Plotshape.horizontal_vertical
        
        :param detached_overlay_name:
            If this plot is overlayed on top of a detached technical indicator, this is the name of the overlay it should be attached to.
            
        :param indicator_size:
            Optional indicator to determine the size of the indicator. For a line, this is the width of the line. For a marker, this is the size of the marker.

        :param recording_time:
            Optional recording time to determine when the plot should be recorded. For example, if you want to record the plot at the decision making time, you can set this to RecordingTime.decision_making_time. Default is RecordingTime.decision_making_time.

        :param label:
            How to render the label for this plot.

            The last set value is effective.

        :param height:
            Currently not supported.

            Plotly does not support setting heights of individual subplots.
            Instead, you can adjust the overall :py:class:`plotly.Figure` size in pixels
            and then % of subplot height in them.
        """

        def _get_helper_message(variable_name: str = None):
            """Get a helper message to help the user fix the error. 
            
            Theses errors show up while running the backtest, so we can point the user directly to decide_trades to fix the error."""

            if not variable_name:
                return ". You can adjust plot_indicator parameters in decide_trades to fix this error."
            else:
                return f". You can adjust {variable_name} in plot_indicator parameters in decide_trades to fix this error."

        assert (
                type(name) == str
        ), f"Got name{name} of type {str(type(name))}" + _get_helper_message("name")

        # Convert numpy.float32 and numpy.float64 to serializable float instances,
        # e.g. prices
        if value is not None:
            try:
                value = float(value)
            except TypeError as e:
                raise RuntimeError(f"Could not convert value {value} {value.__class__} to float" + _get_helper_message("value") + ". Make sure you provide a float or int, not a series, to plot_indicator.") from e

            if pd.isna(value):
                value = None

            # assert not pd.isna(value), f"Cannot plot NaN (not a number) values. {name} received {value} at timestamp {timestamp}. Please convert to None or do not call plot_indicator() for NaN values."

        if detached_overlay_name:
            assert type(detached_overlay_name) is str, "Detached overlay must be a string" + _get_helper_message("detached_overlay_name")
            assert kind == PlotKind.technical_indicator_overlay_on_detached, "Detached overlay must be a PlotKind.technical_indicator_overlay_on_detached" + _get_helper_message("kind")

        if kind == PlotKind.technical_indicator_overlay_on_detached:
            assert detached_overlay_name and type(detached_overlay_name) == str, "Detached overlay must be a string for PlotKind.technical_indicator_overlay_on_detached" + _get_helper_message("detached_overlay_name")

        plot = self.plots.get(name, Plot(name=name, kind=kind))

        plot.add_point(timestamp, value)
        plot.kind = kind
        plot.plot_shape = plot_shape
        plot.detached_overlay_name = detached_overlay_name
        plot.indicator_size = indicator_size
        plot.label = label
        if colour:
            plot.colour = colour
        plot.recording_time = recording_time
        plot.pair = pair
        plot.height = height
        self.plots[name] = plot

    def get_timestamp_range(self, plot_name: Optional[str] = None) -> Tuple[Optional[datetime.datetime], Optional[datetime.datetime]]:
        """Get the time range for which we have data.

        :param plot_name:
            Use range from a specific plot.

            If not given use the first plot.

        :return:
            UTC started at, ended at.

            Return None, None if no data.
        """

        if len(self.plots) == 0:
            return None, None

        if plot_name is None:
            plot_name = next(iter(self.plots.keys()))

        plot = self.plots[plot_name]

        if len(plot.points) == 0:
            return None, None

        first_timestamp, _ = plot.get_first_entry()
        last_timestamp, _ = plot.get_last_entry()

        return first_timestamp, last_timestamp

    def get_total_points(self) -> int:
        """Get number of data points stored in all plots."""
        return sum([len(p.points) for p in self.plots.values()])

    def get_series(
            self,
            name: str,
    ) -> pd.Series:
        """Get plot data for charts and tables.

        Example from a backtesting notebook:

        .. code-block:: python

            import plotly.express as px

            visualisation = state.visualisation

            df = pd.DataFrame({
                "assets_available": visualisation.get_series("assets_available"),
                "assets_chosen": visualisation.get_series("assets_chosen")
            })

            fig = px.line(df, title='Assets tradeable vs. chosen to the basket')
            fig.update_yaxes(title="Number of assets")
            fig.update_xaxes(title="Time")
            fig.show()

        :param name:
            Same as in :py:func:`plot_indicator`

        :return:
            Pandas series with DateTimeIndex and float values.
        """
        plot = self.plots.get(name)
        assert plot is not None, f"No plot named {name}, we have {list(self.plots.keys())}"
        timestamps = list(plot.points.keys())
        values = plot.points.values()
        datetime_index = pd.to_datetime(timestamps, unit='s')
        series = pd.Series(values, index=datetime_index)
        return series

    def set_discardable_data(self, key, value):
        """Set data for which we only keep the last set."""
        self.discardable_data[key] = value
