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
from typing import List, Dict, Optional, Any, Union, Tuple
import pandas as pd
import logging

from dataclasses_json import dataclass_json

from tradeexecutor.utils.timestamp import convert_and_validate_timestamp, convert_and_validate_timestamp_as_int


logger = logging.getLogger(__name__)


class PlotKind(enum.Enum):
    """What different plots a strategy can output."""

    #: This plot is drawn on the top of the price graph
    technical_indicator_on_price = "technical_indicator_on_price"
    
    #: This plot is drawn below the price graph
    technical_indicator_detached = "technical_indicator_detached"
    
    #: This plot is overlaid on a detached indicator plot
    technical_indicator_overlay_on_detached = "technical_indicator_overlay"


class PlotShape(enum.Enum):
    """
    Describes the various shapes that a line can take in a plot. See discussion: https://github.com/tradingstrategy-ai/trade-executor/pull/156#discussion_r1058819823
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


@dataclass_json
@dataclass
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
    detached_overlay_name: Optional[str]= None
    
    #: Optional indicator to determine the size of the indicator. 
    #: 
    #: For a line, this is the width of the line. 
    #:
    #: For a marker, this is the size of the marker.
    indicator_size: Optional[float] = None

    def add_point(self,
                  timestamp: datetime.datetime,
                  value: float,
                  ):
        assert isinstance(timestamp, datetime.datetime)
        assert isinstance(value, (float, NoneType)), f"Got {value} ({value.__class__})"
        timestamp = convert_and_validate_timestamp_as_int(timestamp)
        logger.info("Plotting %s at %s: %s", self.name, timestamp, value)
        assert timestamp not in self.points, f"Plot {self.name} aleady has point for timestamp {timestamp}"
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
    """
    This object is returned from the strategy execution cycle.
    It allows you to plot values, add debug messages, etc.
    It is not used in any trading, but can help and visualize
    trade backtesting and execution.
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

    def add_message(self,
                    timestamp: datetime.datetime,
                    content: str):
        """Write a debug message.

        - Each message is associated to a different timepoint.

        :param timestamp:
            The current strategy cycle timestamp

        :param content:
            The contents of the message

        """
        timestamp = convert_and_validate_timestamp_as_int(timestamp)
        timepoint_messages = self.messages.get(timestamp, list())
        timepoint_messages.append(content)
        self.messages[timestamp] = timepoint_messages

    def add_calculations(self,
                         timestamp: datetime.datetime,
                         cycle_calculations: dict):
        """Update strategy cycle calculations diagnostics.

        - Each strategy cycle can dump whatever intermediate
          calculations state on the visualisation record keeping,
          so that it can be later pulled up in the analysis.

        - Currently this is used to store the alpha model calculations
          for portfolio construction model.

        :param timestamp:
            The current strategy cycle timestamp

        :param cycle_calculations:
            The contents of the calculations.

            Must be JSON serialisable dict.

        """

        assert isinstance(cycle_calculations, dict)
        timestamp = convert_and_validate_timestamp_as_int(timestamp)
        self.calculations[timestamp] = cycle_calculations

    def plot_indicator(self,
             timestamp: Union[datetime.datetime, pd.Timestamp],
             name: str,
             kind: PlotKind,
             value: float,
             colour: Optional[str] = None,
             plot_shape: Optional[PlotShape] = PlotShape.linear,
             detached_overlay_name: str | None = None,
             indicator_size: Optional[float] = None,
        ):
        """Add a value to the output data and diagram.
        
        Plots are stored by their name.

        :param timestamp:
            The current strategy cycle timestamp

        :param name:
            The plot label

        :param kind:
            The plot typre

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
                raise RuntimeError(f"Could not convert value {value} {value.__class__} to float" + _get_helper_message("value")) from e

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

        if colour:
            plot.colour = colour

        self.plots[name] = plot

    def get_timestamp_range(self, plot_name: Optional[str]=None) -> Tuple[Optional[datetime.datetime], Optional[datetime.datetime]]:
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
        

