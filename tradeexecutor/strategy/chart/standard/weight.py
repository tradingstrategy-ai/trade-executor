"""Portfolio weights visualisation."""


import pandas as pd
import plotly.colors as colors
import plotly.express as px
from plotly.graph_objects import Figure

from tradingstrategy.chain import _chain_data

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


def equity_curve_by_chain(input: ChartInput) -> tuple[Figure, pd.DataFrame]:
    """Equity curve with positions grouped by chain.

    Shows the USD value allocated to each blockchain over time,
    so concentration risk across chains is visible at a glance.

    Returns (fig, df) tuple so diagnostics can inspect the underlying data.
    """
    state = input.state

    # Map position_id -> chain name
    position_chain_map = {}
    for p in state.portfolio.get_all_positions():
        chain_id = p.pair.chain_id
        chain_entry = _chain_data.get(chain_id, {})
        chain_name = chain_entry.get("name", f"Chain {chain_id}")
        position_chain_map[p.position_id] = chain_name

    # Build reserve rows using derived cash to avoid double-counting.
    # free_cash can lag behind position openings at the same timestamp,
    # so we derive it as total_equity - open_position_equity.
    reserve_asset, _price = state.portfolio.get_default_reserve_asset()
    reserve_chain_entry = _chain_data.get(reserve_asset.chain_id, {})
    reserve_chain_name = reserve_chain_entry.get("name", f"Chain {reserve_asset.chain_id}")
    reserve_rows = [{
        "timestamp": ps.calculated_at,
        "chain": reserve_chain_name,
        "value": ps.total_equity - (ps.open_position_equity or 0),
    } for ps in state.stats.portfolio]

    # Build position rows grouped by chain
    position_rows = [{
        "timestamp": ps.calculated_at,
        "chain": position_chain_map[position_id],
        "value": ps.value,
    } for position_id, position_stats in state.stats.positions.items()
      for ps in position_stats]

    df = pd.DataFrame(reserve_rows + position_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.groupby(["timestamp", "chain"])["value"].sum().reset_index()
    df = df.sort_values("timestamp")
    df = df.pivot(index="timestamp", columns="chain", values="value").fillna(0)

    # Sort columns alphabetically
    df = df[sorted(df.columns)]

    fig = px.area(
        df,
        title="Asset weights (USD) by chain",
        labels={"index": "Time", "value": "US dollar size"},
        color_discrete_sequence=colors.qualitative.Light24,
        template="plotly_dark",
    )
    fig.update_traces(line_width=0)
    return fig, df
