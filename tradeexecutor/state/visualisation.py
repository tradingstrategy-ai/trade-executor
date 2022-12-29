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
from typing import List, Dict, Optional, Any, Union
import pandas as pd

from dataclasses_json import dataclass_json

from tradeexecutor.utils.timestamp import convert_and_validate_timestamp, convert_and_validate_timestamp_as_int


class PlotKind(enum.Enum):
    """What different plots a strategy can output."""

    #: This plot is drawn on the top of the price graph
    technical_indicator_on_price = "technical_indicator_on_price"

class PlotShape(enum.Enum):
    """
    Describes the various shapes that a line can take in a plot. See discussion: https://github.com/tradingstrategy-ai/trade-executor/pull/156#discussion_r1058819823
    """

    #: Standard linear line. Used in most cases. 
    linear = "linear"

    #: Is the line horizontal-vertical. Used for stop loss line.
    #: See https://plotly.com/python/line-charts/?_ga=2.83222870.1162358725.1672302619-1029023258.1667666588#interpolation-with-line-plots
    horizontal_vertical = "hv"

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
    points: Dict[int, float] = field(default_factory=dict)

    #: Is the line horizontal-vertical. Used for stop loss line. See https://plotly.com/python/line-charts/?_ga=2.83222870.1162358725.1672302619-1029023258.1667666588#interpolation-with-line-plots
    plot_shape: Optional[PlotShape] = PlotShape.linear

    def add_point(self,
                  timestamp: datetime.datetime,
                  value: float,
                  ):
        assert isinstance(timestamp, datetime.datetime)
        assert isinstance(value, (float, NoneType)), f"Got {value} ({value.__class__})"
        timestamp = convert_and_validate_timestamp_as_int(timestamp)
        self.points[timestamp] = value

    def get_last_value(self) -> float:
        """Assume points is an ordered dict."""
        # https://stackoverflow.com/a/63059166/315168
        key = next(reversed(self.points.keys()))
        return self.points[key]


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
    #: TODO:
    #: Because we cannot use datetime.datetime directly as a key in JSON,
    #: we use UNIX timestamp here to keep our state easily serialisable.
    messages: Dict[int, List[str]] = field(default_factory=dict)

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

    def plot_indicator(self,
             timestamp: Union[datetime.datetime, pd.Timestamp],
             name: str,
             kind: PlotKind,
             value: float,
             colour: Optional[str] = None,
             plot_shape: Optional[PlotShape] = PlotShape.linear):
        # sourcery skip: remove-unnecessary-cast
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
        """

        assert type(name) == str, "Got name"

        # Convert numpy.float32 and numpy.float64 to serializable float instances,
        # e.g. prices
        if value is not None:
            try:
                value = float(value)
            except TypeError as e:
                raise RuntimeError(f"Could not convert value {value} {value.__class__} to float") from e

        plot = self.plots.get(name, Plot(name=name, kind=kind))

        plot.add_point(timestamp, value)

        plot.kind = kind

        plot.plot_shape = plot_shape

        if colour:
            plot.colour = colour

        self.plots[name] = plot