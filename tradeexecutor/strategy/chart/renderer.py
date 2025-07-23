import traceback
from dataclasses import dataclass
from typing import Collection, Callable

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.strategy.chart.definition import ChartRegistry, ChartParameters, ChartRenderingResult, ChartInput, ChartOutput
from tradeexecutor.strategy.execution_context import notebook_execution_context
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInputIndicators


def render_chart(
    registry: ChartRegistry,
    chart_name: str,
    parameters: ChartParameters,
    input: ChartInput,
) -> ChartRenderingResult:
    """Render a chart using the provided registry and parameters.

    - Call from the web API endpoint
    - In backtesting use :py:class:`ChartBackt  estRenderingSetup`

    :param registry: The chart registry containing available charts.
    :param chart_name: The name of the chart to render.
    :param parameters: Parameters for rendering the chart.
    :param input: Input data required for rendering the chart.
    :return: ChartOutput containing the rendered chart data or an error message.
    """
    try:
        assert isinstance(registry, ChartRegistry), "Invalid chart registry provided."
        assert isinstance(chart_name, str), "Chart name must be a string."
        assert isinstance(parameters, ChartParameters), "Parameters must be of type ChartParameters."
        assert isinstance(input, ChartInput), "Input must be of type ChartInput."

        chart_function = registry.get_chart_function(chart_name)
        if not chart_function:
            return ChartRenderingResult.error_out(f"Chart '{chart_name}' not found in registry.")

        # Call the chart function with the provided parameters and input
        data = chart_function(input, parameters)
        return data

    except Exception as e:
        tb_str = traceback.format_exc()
        return ChartRenderingResult.error_out(f"Error rendering chart '{chart_name}': {str(e)}\n{tb_str}")


@dataclass(slots=True, frozen=True)
class ChartBacktestRenderingSetup:
    """Define a setup that tells which pairs we are about to render"""

    registry: ChartRegistry

    strategy_input_indicators: StrategyInputIndicators

    #: Selected pairs.
    #:
    #: Examine these assets in backtesting rendering functions.
    pairs: Collection[TradingPairIdentifier] | None

    # :Where do we run the renderer
    execution_context = notebook_execution_context

    #: Backtesting or live trading state
    state: State | None = None

    def __post_init__(self):
        assert self.strategy_input_indicators is not None, "strategy_input_indicators must be provided."
        for pair in self.pairs:
            assert isinstance(pair, TradingPairIdentifier), f"pairs must contain TradingPairIdentifier instances, got {type(pair)}: {pair}"

    def render(self, func: Callable, **kwargs) -> ChartOutput:
        """Render the chart using the provided function.

        :param func:
            Python function object registered with ``ChartRegistry.register()``.

        :param kwargs:
            Extra arguments passed to the function
        """

        assert func in self.registry.by_function, f"Function {func} is not registered in the chart registry."

        input = ChartInput(
            strategy_input_indicators=self.strategy_input_indicators,
            pairs=self.pairs,
        )
        result = func(input, **kwargs)
        assert result is not None, f"Chart rendering function {func} returned None."
        return result
