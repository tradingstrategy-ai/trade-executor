"""Chart definition for the trade executor strategy."""
import datetime
import enum
import typing
from dataclasses import dataclass
from functools import wraps
from typing import Literal

import pandas as pd
import plotly.graph_objects as go
from matplotlib.figure import Figure as MatplotlibFigure
from pandas.io.formats.style import Styler

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInputIndicators
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


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
    """Input state and choises needed to render a chart.

    - Any of the input fields may be filled
    - What parameters the chart function needs to
    """

    #: Are we running live or backtest
    execution_context: ExecutionContext
    state: State | None = None
    strategy_input_indicators: StrategyInputIndicators = None
    pairs: typing.Collection[TradingPairIdentifier] | None = None

    #: Passed when setting up `ChartBacktestRenderingSetup`.
    #:
    #: Use :py:meth:`end_at` for access.
    backtest_end_at: datetime.datetime | None = None

    #: Cached calculations in backtesting notebook
    cache = {}

    def __post_init__(self):
        if self.state is not None:
            assert isinstance(self.state, State), "State must be an instance of State."

        if self.strategy_input_indicators is not None:
            assert isinstance(self.strategy_input_indicators, StrategyInputIndicators), \
                "strategy_input_indicators must be an instance of StrategyInputIndicators."

        if self.pairs is not None:
            assert isinstance(self.pairs, (list, set, tuple)), f"pairs must be a collection, got {type(self.pairs)}."
            for p in self.pairs:
                assert isinstance(p, TradingPairIdentifier), f"Each pair must be a TradingPairIdentifier, got {type(p)}."

    @property
    def strategy_universe(self) -> TradingStrategyUniverse:
        return self.strategy_input_indicators.strategy_universe

    @property
    def live(self) -> bool:
        return self.execution_context.live_trading

    @property
    def backtest(self) -> bool:
        return not self.execution_context.live_trading

    @property
    def end_at(self) -> datetime.datetime:
        """The end timestamp of the charting.

        - Backtesting: backtest end timestamp
        - Live trading: The latest completed cycle timestamp
        """
        if self.execution_context.live_trading:
            raise NotImplementedError()
        else:
            return self.backtest_end_at


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


#: Chart functions can return
#: - Plotly Figure
#: - DataFrame for rendering a table
#: - Both
#: - List of figures (for each pair, vault, etc.)
#: - Matplotlib Figure
#: - Pandas Styler styled dataframe for rendering a HTML table
ChartOutput = go.Figure | pd.DataFrame | tuple[go.Figure, pd.DataFrame] | list[go.Figure] | MatplotlibFigure | Styler

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
    description: str



class ChartRegistry:
    """Registry for charts.

    - Makes charts discoverable by name in the frontend
    """

    def __init__(self):

        #: Name -> registered functions mappings
        self.registry: dict[str, ChartCallback] = {}

        #: Function -> registered functions mappings.
        #: Only useful for backtesting notebooks.
        self.by_function: dict[ChartFunction, ChartCallback] = {}

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
        func: ChartFunction,
        kind: ChartKind,
        name: str | None = None,
    ):
        """Manually register a chart function."""
        name = name or func.__name__

        docstring = func.__doc__
        assert docstring, f"Chart function '{func}' must have a docstring as a description."

        description = docstring.strip().split("\n")[0]

        assert not " " in name, f"Chart name '{name}' cannot contain spaces."
        callback = ChartCallback(
            name=name,
            func=func,
            kind=kind,
            description=description
        )
        self.registry[name] = callback
        self.by_function[func] = callback







