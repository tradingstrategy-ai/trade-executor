"""Volatility across pairs in the universe."""
from typing import Collection

from plotly.graph_objects import Figure

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.chart.chart_definition import ChartInput


def volatility_benchmark(
    input: ChartInput,
    pairs: Collection[TradingPairIdentifier],
    avg_volatility_indicator_name: str = "avg_volatility",
) -> Figure:
    """Render historical volatility for some pairs based on the provided volatility indicator name."""

