import base64
import datetime
import io
import traceback
from dataclasses import dataclass
from typing import Collection, Callable, Any

import pandas as pd
from pandas.io.formats.style import Styler
from plotly.graph_objects import Figure
import IPython.core.display

import matplotlib.figure

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.strategy.chart.definition import ChartRegistry, ChartParameters, ChartRenderingResult, ChartInput, ChartOutput
from tradeexecutor.strategy.execution_context import notebook_execution_context
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInputIndicators
from tradeexecutor.visual.image_output import render_plotly_figure_as_image_file


@dataclass(slots=True, frozen=False)
class MultiImage:
    """Helper to render multi-image HTTP responses"""
    png: bytes
    title: str | None = None


def render_matplotlib_figure_as_image_file(
    figure: matplotlib.figure.Figure,
    format: str = "png",
    width: int = 512,
    height: int = 512,
    dpi = 300,
) -> bytes:
    """Some legacy charts use Matplotlib"""
    assert isinstance(figure, matplotlib.figure.Figure), f"Expected a Matplotlib Figure, got {type(figure)}"
    figure.set_size_inches(width / dpi, height / dpi)
    buf = io.BytesIO()
    figure.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()


def render_multi_image_result(
    images: list[MultiImage],
) -> bytes:
    """Convert multiple images to a format suitable for the frontend.

    - This is our internal format, there is no web standard
    """
    assert isinstance(images, list), f"Expected a list of PNG images, got {type(pngs)}"
    assert len(images) > 0, "No images to render"

    buf = io.BytesIO()
    buf.write(b'<div class="analysis-charts">')

    for img in images:
        assert isinstance(img, MultiImage), f"Expected MultiImage, got {type(img)}"
        buf.write(b'<img alt="')
        buf.write(img.title.encode("utf-8") if img.title else b"")
        buf.write(b'" class="analysis-chart-image" src="data:image/png;base64,')
        buf.write(base64.b64encode(img.png))
        buf.write(b'"/>')
    buf.write(b'</div>')

    return buf.getvalue()


def render_for_web(
    parameters: ChartParameters,
    func_result: Any,
    func: Callable,
) -> ChartRenderingResult:

    # Fix Time Axis
    # https://github.com/plotly/plotly.py/issues/5210
    from tradeexecutor.monkeypatch import plotly

    # We do not support multi-content output yet,
    # so discard other parts of the result
    if type(func_result) == tuple:
        func_result = func_result[0]

    if isinstance(func_result, Figure):
        # Render Plotly Figure as PNG image file
        data = render_plotly_figure_as_image_file(
            func_result,
            format=parameters.format,
            width=parameters.width,
            height=parameters.height
        )
        return ChartRenderingResult(
            data=data,
            content_type="image/png",
        )
    elif isinstance(func_result, matplotlib.figure.Figure):
        # Render Matplotlib Figure as PNG image file
        data = render_matplotlib_figure_as_image_file(
            func_result,
            format=parameters.format,
            width=parameters.width,
            height=parameters.height
        )
        return ChartRenderingResult(
            data=data,
            content_type="image/png",
        )
    elif isinstance(func_result, pd.DataFrame):
        # Render DataFrame as HTML table
        html_table = func_result.to_html(classes='table table-striped', index=False)
        return ChartRenderingResult(
            data=html_table.encode("utf-8"),
            content_type="text/html",
        )
    elif isinstance(func_result, Styler):
        # Render DataFrame as HTML table
        html_table = func_result.to_html(classes='table table-striped', index=False)
        return ChartRenderingResult(
            data=str(html_table).encode("utf-8"),
            content_type="text/html",
        )
    elif isinstance(func_result, list):
        # List of images.
        # For now, only render the first one.

        assert isinstance(func_result[0], Figure), f"Expected list of Plotly Figures, got {type(func_result[0])}."

        images = []
        for result in func_result:
            images.append(
                MultiImage(
                    png=render_plotly_figure_as_image_file(
                        result,
                        format=parameters.format,
                        width=parameters.width,
                        height=parameters.height
                    ),
                    title=result.layout.title.text if result.layout.title else None
                )
            )

        data = render_multi_image_result(images)
        return ChartRenderingResult(
            data=data,
            content_type="text/html",
        )
    elif isinstance(func_result, IPython.core.display.HTML):
        data = str(func_result).encode("utf-8")
        return ChartRenderingResult(
            data=data,
            content_type="text/html",
        )
    else:
        raise NotImplementedError(f"Chart function {func}: Unsupported chart output type: {type(func_result)}. Expected Figure or ChartOutput.")


def render_chart(
    registry: ChartRegistry,
    chart_id: str,
    parameters: ChartParameters,
    input: ChartInput,
) -> ChartRenderingResult:
    """Render a chart using the provided registry and parameters.

    - Call from the web API endpoint
    - In backtesting use :py:class:`ChartBackt  estRenderingSetup`

    :param registry: The chart registry containing available charts.
    :param chart_id: The name of the chart to render.
    :param parameters: Parameters for rendering the chart.
    :param input: Input data required for rendering the chart.
    :return: ChartOutput containing the rendered chart data or an error message.
    """
    try:
        assert isinstance(registry, ChartRegistry), f"Invalid chart registry provided: {type(registry)}"
        assert isinstance(chart_id, str), "Chart name must be a string."
        assert isinstance(parameters, ChartParameters), "Parameters must be of type ChartParameters."
        assert isinstance(input, ChartInput), "Input must be of type ChartInput."

        chart_entry = registry.get_chart_function(chart_id)
        if not chart_entry:
            return ChartRenderingResult.error_out(f"Chart '{chart_id}' not found in registry.")

        chart_function = chart_entry.func

        assert callable(chart_function), f"Chart function must be callable: {type(chart_function)}"

        # Call the chart function with the provided parameters and input
        func_result = chart_function(input)
        return render_for_web(parameters, func_result, chart_function)

    except Exception as e:
        tb_str = traceback.format_exc()
        return ChartRenderingResult.error_out(f"Error rendering chart '{chart_id}': {str(e)}\n{tb_str}")


@dataclass(slots=True, frozen=False)
class ChartBacktestRenderingSetup:
    """Define a setup that tells which pairs we are about to render"""

    registry: ChartRegistry

    strategy_input_indicators: StrategyInputIndicators

    # :Where do we run the renderer
    execution_context = notebook_execution_context

    #: Backtesting or live trading state
    state: State | None = None

    #: Backtest end time hint, if backtest is not run yet.
    #: Backtest end time hint, if backtest is not run yet.
    backtest_end_at: datetime.datetime | None = None

    #: Selected pairs.
    #:
    #: Examine these assets in backtesting rendering functions.
    #: If not given pick ``ChartRegistry.default_pairs``.
    pairs: Collection[TradingPairIdentifier] | None = None


    def __post_init__(self):
        assert self.strategy_input_indicators is not None, "strategy_input_indicators must be provided."

        if self.pairs is None:
            strategy_universe = self.strategy_input_indicators.strategy_universe
            self.pairs = [strategy_universe.get_pair_by_human_description(desc) for desc in self.registry.default_benchmark_pairs]

        assert self.pairs, "pairs must not be empty."

        for pair in self.pairs:
            assert isinstance(pair, TradingPairIdentifier), f"pairs must contain TradingPairIdentifier instances, got {type(pair)}: {pair}"

        if self.backtest_end_at:
            assert isinstance(self.backtest_end_at, datetime.datetime), f"end_at must be a datetime, got {type(self.end_at)}: {self.end_at}"

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
            execution_context=self.execution_context,
            backtest_end_at= self.backtest_end_at,
            state=self.state,
        )
        result = func(input, **kwargs)
        assert result is not None, f"Chart rendering function {func} returned None."
        return result
