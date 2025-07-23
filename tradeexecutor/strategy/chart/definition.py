"""Chart definition for the trade executor strategy."""
import enum
import typing
from dataclasses import dataclass
from functools import wraps
from typing import Literal

import pandas as pd
import plotly.graph_objects as go

from tradeexecutor.state.state import State
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInputIndicators


class ChartKind(enum.Enum):
    """What kind of charts we can define"""
    #: A Chart for a single pair based on indicator data
    indicator_single_pair = "indicator_single_pair"
    #: A Chart for a multiple pairs based on indicator data
    indicator_multi_pair = "indicator_multi_pair"
    #: A Chart for a all trading pairs once based on indicator data
    indicator_all_pairs = "indicator_universe"
    #: A Chart for a all trading pairs once based on state data
    state_all_pairs= "universe_state"
    #: A Chart for a single pair based on indicator data
    state_single_pair = "state_single_pair"


@dataclass(slots=True, frozen=False)
class ChartInput:
    """Input state needed to render a chart.

    - Any of the input fields may be filled
    - What parameters the chart function needs to
    """
    state: State | None = None
    strategy_input_indicators: StrategyInputIndicators = None
    pair_id: int | None = None

    def __post_init__(self):
        if self.state is not None:
            assert isinstance(self.state, State), "State must be an instance of State."

        if self.strategy_input_indicators is not None:
            assert isinstance(self.strategy_input_indicators, StrategyInputIndicators), \
                "strategy_input_indicators must be an instance of StrategyInputIndicators."

        if self.pair_id is not None:
            assert isinstance(self.pair_id, int), "pair_id must be an integer."

    @property
    def strategy_universe(self):
        return self.strategy_input_indicators.strategy_universe


@dataclass(slots=True, frozen=False)
class ChartParameters:
    width = 1200
    height = 800
    format: Literal["png", "svg"] = "png"


@dataclass(slots=True, frozen=True)
class ChartRenderingResult:
    """Server-side rendered result, ready to send over a wire."""
    data: bytes
    content_type: Literal["image/png", "image/svg", "text/html"] = "image/png"
    error: str | None = None

    @staticmethod
    def error_out(msg: str) -> "ChartRenderingResult":
        """Create an error output."""
        return ChartRenderingResult(
            data=b"",
            content_type="text/plain",
            error=msg,
        )


ChartOutput = go.Figure | pd.DataFrame


class ChartFunction(typing.Protocol):
    """Chart rendering protocol definition.

    - Define function arguments for calling chart functions
    """

    def __call__(self, input: ChartInput) -> ChartOutput:
        """Render a chart based on the provided input.
        """



@dataclass(slots=True, frozen=True)
class ChartCallback:
    """One function serving chats.
    """
    name: str
    func: ChartFunction
    kind: ChartKind



class ChartRegistry:
    """Registry for charts.

    - Makes charts discoverable by name in the frontend
    """

    def __init__(self):
        #: Name -> registered functions mappings
        self.registry: dict[str, ChartCallback] = {}

    def get_chart_function(self, name: str) -> ChartCallback | None:
        """Get a chart function by name."""
        return self.registry.get(name)

    def define(
        self,
        kind: ChartKind,
        name: str | None = None,
    ):
        def decorator(func):
            nonlocal name

            self.register(func, kind, name)

            @wraps(func)
            def wrapper(*args, **kwargs):
                # Call the original function with the original arguments
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def register(
        self,
        func: ChartCallback,
        kind: ChartKind,
        name: str | None = None,
    ):
        """Manually register a chart function."""


        name = name or func.__name__

        assert not " " in name, f"Chart name '{name}' cannot contain spaces."

        self.registry[name] = ChartCallback(
            name=name,
            func=func,
            kind=kind,
        )





