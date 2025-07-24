"""Portfolio weights visualisation."""


import pandas as pd
from plotly.graph_objects import Figure

from tradeexecutor.analysis.weights import calculate_asset_weights, visualise_weights, calculate_weights_statistics
from tradeexecutor.strategy.chart.definition import ChartInput


def _calculate_and_cache_weights(input: ChartInput) -> pd.Series:
    """Calculate and cache asset weights for the input."""
    state = input.state
    weights_series = input.cache.get_indicator_series("weights")
    if weights_series is None:
        weights_series = calculate_asset_weights(state)
        input.cache["weights"] = weights_series
    return weights_series


def volatile_weights_by_percent(
    input: ChartInput,
) -> Figure:
    """Return volatile asset weights, 100% stacked.
    """
    weights_series = calculate_asset_weights(input.state)
    fig = visualise_weights(
        weights_series,
        normalised=True,
        include_reserves=False,
    )
    return fig


def volatile_and_non_volatile_percent(
    input: ChartInput,
) -> Figure:
    """Return volatile asset weights, 100% stacked.
    """
    weights_series = calculate_asset_weights(input.state)
    fig = visualise_weights(
        weights_series,
        normalised=True,
        include_reserves=True,
    )
    return fig


def equity_curve_by_asset(
    input: ChartInput,
) -> Figure:
    """Equity curve with assets colored.
    """
    weights_series = calculate_asset_weights(input.state)
    fig = visualise_weights(
        weights_series,
        normalised=False,
    )
    return fig


def weight_allocation_statistics(
    input: ChartInput,
) -> pd.DataFrame:
    """Statistics about portfolio mixture.
    """
    weights_series = calculate_asset_weights(input.state)
    stats = calculate_weights_statistics(weights_series)
    return stats
