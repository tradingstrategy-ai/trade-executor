"""Portfolio weights visualisation."""


from collections.abc import Callable

import pandas as pd
import plotly.colors as colors
import plotly.express as px
from plotly.graph_objects import Figure

from tradingstrategy.chain import _chain_data
from tradingstrategy.vault import VaultMetadata

from tradeexecutor.analysis.multipair import calculate_pair_annualised_average_yield
from tradeexecutor.analysis.weights import (
    PENDING_SETTLEMENT_LABEL_SUFFIX,
    calculate_asset_weights,
    calculate_weights_statistics,
    get_pending_settlement_asset_label,
    visualise_weights,
)
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.strategy.chart.asset_weight_legend import (
    AssetWeightLegendEntry,
    add_asset_weight_legend,
    merge_asset_weight_legend_entries,
)
from tradeexecutor.strategy.chart.definition import ChartInput
from tradeexecutor.strategy.phase_aware import (
    EVENT_CLOSE,
    EVENT_PARK,
    EVENT_PROMOTE,
    is_queue_vault_position,
    iter_all_events,
)

#: Band label for the yield-bearing queue venue, split out from directional chain bands and idle cash.
QUEUE_VENUE_BAND = "Queue venue"

#: Reserve-like colour for YieldManager-managed queue venue positions.
YIELD_MANAGER_RESERVE_COLOUR = "#666"

#: Neutral swatch colour used for asset-weight legend rows.
ASSET_WEIGHT_LEGEND_SWATCH_COLOUR = "#a8b1c1"


#: Full allocation already represented by ordinary vault/spot position value.
ALLOCATED_CAPITAL_BAND = "Allocated capital"

#: Raw reserve capital that is neither invested nor earmarked for a settlement.
CASH_BAND = "Cash"

#: Yield-bearing queue-venue capital with no pending target vault deposit.
QUEUE_BAND = "Queue"

#: Queue-venue capital earmarked for a vault whose deposit window is closed.
PENDING_DEPOSITS_BAND = "Pending deposits"

#: Capital sent to an async vault but not yet represented by vault shares.
PENDING_SETTLEMENTS_BAND = "Pending settlements"


LIQUIDITY_STATE_COLOURS = {
    ALLOCATED_CAPITAL_BAND: "#4c78a8",
    CASH_BAND: "#a8b1c1",
    QUEUE_BAND: YIELD_MANAGER_RESERVE_COLOUR,
    PENDING_DEPOSITS_BAND: "#f2a900",
    PENDING_SETTLEMENTS_BAND: "#9c6ade",
}


def _get_queue_asset_label(pair) -> str:
    """Legend label for a YieldManager venue in asset allocation charts."""
    venue_name = pair.get_chart_label() or pair.get_ticker()
    return f"{venue_name} [queue]"


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
    reserve_asset_symbol = input.state.portfolio.get_default_reserve_asset()[0].token_symbol
    queue_venue_labels = {}
    for position in input.state.portfolio.get_all_positions():
        if is_queue_vault_position(position):
            queue_venue_labels[position.position_id] = _get_queue_asset_label(position.pair)

    weights_series = calculate_asset_weights(input.state, position_asset_label_overrides=queue_venue_labels)
    queue_asset_labels = list(queue_venue_labels.values())
    fig = visualise_weights(
        weights_series,
        normalised=False,
        extra_colours={label: YIELD_MANAGER_RESERVE_COLOUR for label in queue_asset_labels},
        extra_sort_order={
            reserve_asset_symbol: -1000,
            **{label: -999 for label in queue_asset_labels},
        },
    )
    return fig


def weight_allocation_statistics(
    input: ChartInput,
) -> pd.DataFrame:
    """Statistics about portfolio mixture.
    """
    weights_series = calculate_asset_weights(input.state)
    stats = calculate_weights_statistics(weights_series, state=input.state)
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

    # Queue-venue (YieldManager-managed) positions render as their own band, split out from both the
    # directional chain bands and the idle-cash reserve rows, so idle USDC vs yield-bearing venue is
    # visible at a glance. Identified from state alone via the durable yield-decision trade marker.
    # Note: this classifies ANY YieldManager-managed vault venue, not only phase-aware queue vaults,
    # so a non-phase-aware YieldManager strategy also gets its yield venue rendered as a distinct band
    # (an intended generalisation - a yield venue is reserve-like for every strategy that uses one).
    venue_position_ids = {
        p.position_id for p in state.portfolio.get_all_positions() if is_queue_vault_position(p)
    }

    # Build position rows grouped by chain (queue-venue positions get their own band).
    position_rows = [{
        "timestamp": ps.calculated_at,
        "chain": QUEUE_VENUE_BAND if position_id in venue_position_ids else position_chain_map[position_id],
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


def _calculate_pending_settlements_by_timestamp(input: ChartInput) -> dict[object, float]:
    """Reuse the asset-weight accounting for unsettled asynchronous deposits."""
    weights = calculate_asset_weights(input.state)
    pending_symbols = weights.attrs.get("pending_settlement_symbols", [])
    if not pending_symbols:
        return {}
    pending_weights = weights[weights.index.get_level_values(1).isin(pending_symbols)]
    return pending_weights.groupby(level=0).sum().to_dict()


def _calculate_waiting_deposits_by_timestamp(state: State) -> dict[object, float]:
    """Reconstruct phase-aware queue deposits which wait for a deposit window."""
    timestamps = sorted(state.stats.portfolio, key=lambda stats: stats.calculated_at)
    values: dict[object, float] = {stats.calculated_at: 0.0 for stats in timestamps}
    event_timestamps = [
        (pd.Timestamp(event.timestamp), event)
        for event in iter_all_events(state.other_data)
        if event.timestamp is not None
    ]
    event_timestamps.sort(key=lambda item: item[0])

    event_index = 0
    open_deposits: dict[int, float] = {}
    for portfolio_stats in timestamps:
        timestamp = portfolio_stats.calculated_at
        while event_index < len(event_timestamps) and event_timestamps[event_index][0] <= timestamp:
            event = event_timestamps[event_index][1]
            if event.kind == EVENT_PARK:
                open_deposits[event.vault_internal_id] = event.usd
            elif event.kind in (EVENT_PROMOTE, EVENT_CLOSE):
                open_deposits.pop(event.vault_internal_id, None)
            event_index += 1
        values[timestamp] = sum(open_deposits.values())

    return values


def equity_curve_by_liquidity_state(input: ChartInput) -> tuple[Figure, pd.DataFrame]:
    """Show the complete portfolio split by liquid-capital state.

    This complements the per-vault asset-weight map. It separates free cash,
    unassigned queue-venue capital, deposits parked for a closed target window,
    and deposits already in flight to asynchronous vaults. The remaining band is
    capital represented by ordinary open positions, so every timestamp sums to
    total portfolio equity.
    """
    state = input.state
    queue_position_ids = {
        position.position_id
        for position in state.portfolio.get_all_positions()
        if is_queue_vault_position(position)
    }
    queue_values_by_timestamp: dict[object, float] = {}
    for position_id in queue_position_ids:
        for position_stats in state.stats.positions.get(position_id, []):
            queue_values_by_timestamp[position_stats.calculated_at] = (
                queue_values_by_timestamp.get(position_stats.calculated_at, 0.0)
                + position_stats.value
            )

    pending_settlements = _calculate_pending_settlements_by_timestamp(input)
    waiting_deposits = _calculate_waiting_deposits_by_timestamp(state)
    rows = []
    for portfolio_stats in state.stats.portfolio:
        timestamp = portfolio_stats.calculated_at
        total_equity = float(portfolio_stats.total_equity)
        open_position_equity = float(portfolio_stats.open_position_equity or 0.0)
        queue_value = queue_values_by_timestamp.get(timestamp, 0.0)
        pending_deposits = min(waiting_deposits.get(timestamp, 0.0), queue_value)
        pending_settlement_value = pending_settlements.get(timestamp, 0.0)
        rows.append({
            "timestamp": timestamp,
            ALLOCATED_CAPITAL_BAND: open_position_equity - queue_value,
            CASH_BAND: total_equity - open_position_equity - pending_settlement_value,
            QUEUE_BAND: queue_value - pending_deposits,
            PENDING_DEPOSITS_BAND: pending_deposits,
            PENDING_SETTLEMENTS_BAND: pending_settlement_value,
        })

    columns = [
        CASH_BAND,
        QUEUE_BAND,
        PENDING_DEPOSITS_BAND,
        PENDING_SETTLEMENTS_BAND,
        ALLOCATED_CAPITAL_BAND,
    ]
    df = pd.DataFrame(rows, columns=["timestamp", *columns])
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").fillna(0.0)
    else:
        df = pd.DataFrame(columns=columns)

    fig = px.area(
        df,
        title="Asset weights by liquidity state (USD)",
        labels={"index": "Time", "value": "US dollar size"},
        color_discrete_map=LIQUIDITY_STATE_COLOURS,
        template="plotly_dark",
    )
    fig.update_traces(line_width=0)
    for trace in fig.data:
        if trace.name == PENDING_DEPOSITS_BAND:
            trace.update(fillpattern={"shape": "/"})
        elif trace.name == PENDING_SETTLEMENTS_BAND:
            trace.update(fillpattern={"shape": "x"})
    return fig, df


def get_asset_weight_label(position: TradingPosition) -> str:
    """Get the trace label used for a position in the asset weight chart.

    Queue venue positions get the ``[queue]`` suffixed label,
    other positions use the pair chart label.
    """
    if is_queue_vault_position(position):
        return _get_queue_asset_label(position.pair)
    return position.pair.get_chart_label() or position.pair.get_ticker()


def build_asset_weight_legend_entries(state: State) -> list[AssetWeightLegendEntry]:
    """Map each asset-weight trace label to its chain and vault metadata.

    Builds one entry for the reserve asset and one per position,
    carrying the capital- and time-weighted annualised yield of each pair,
    for :py:func:`tradeexecutor.strategy.chart.asset_weight_legend.add_asset_weight_legend`.
    """
    reserve_asset, _ = state.portfolio.get_default_reserve_asset()
    _, strategy_end_at = state.get_strategy_time_range()
    annualised_yield_percent_by_pair_id = {
        pair.internal_id: calculate_pair_annualised_average_yield(pair, state.portfolio, end_at=strategy_end_at) * 100
        for pair in state.portfolio.get_all_traded_pairs()
    }
    entries = [
        AssetWeightLegendEntry(
            label=reserve_asset.token_symbol,
            colour=ASSET_WEIGHT_LEGEND_SWATCH_COLOUR,
            chain_id=reserve_asset.chain_id,
            metadata=None,
            annualised_yield_percent=0.0,
        ),
    ]

    for position in state.portfolio.get_all_positions(pending=True):
        metadata = position.pair.get_token_metadata()
        if not isinstance(metadata, VaultMetadata):
            metadata = None
        entries.append(AssetWeightLegendEntry(
            label=get_asset_weight_label(position),
            colour=ASSET_WEIGHT_LEGEND_SWATCH_COLOUR,
            chain_id=position.pair.chain_id,
            chain_ids=(position.pair.chain_id,),
            metadata=metadata,
            annualised_yield_percent=annualised_yield_percent_by_pair_id.get(position.pair.internal_id, 0.0),
        ))
        if any(trade.is_buy() and trade.other_data.get("vault_async_flow") for trade in position.trades.values()):
            entries.append(AssetWeightLegendEntry(
                label=get_pending_settlement_asset_label(get_asset_weight_label(position)),
                colour=ASSET_WEIGHT_LEGEND_SWATCH_COLOUR,
                chain_id=position.pair.chain_id,
                chain_ids=(position.pair.chain_id,),
                metadata=metadata,
                annualised_yield_percent=0.0,
            ))

    return merge_asset_weight_legend_entries(entries)


def make_asset_weight_legend_sort_key(state: State) -> Callable[[str, float], tuple]:
    """Build the default asset-weight legend row order.

    Keeps the reserve asset first and queue venues second. Directional vaults
    are grouped by label, with each pending-settlement claim immediately after
    its settled vault allocation.
    """
    reserve_symbol = state.portfolio.get_default_reserve_asset()[0].token_symbol
    queue_labels = {
        get_asset_weight_label(position)
        for position in state.portfolio.get_all_positions()
        if is_queue_vault_position(position)
    }

    def sort_key(label: str, allocation_pct: float) -> tuple:
        if label == reserve_symbol:
            priority = 0
        elif label in queue_labels:
            priority = 1
        else:
            priority = 2

        is_pending_settlement = label.endswith(PENDING_SETTLEMENT_LABEL_SUFFIX)
        vault_label = label.removesuffix(PENDING_SETTLEMENT_LABEL_SUFFIX)

        # Group the claim immediately after its settled allocation. We use the
        # base vault label for group ordering because the pending claim's own
        # allocation can otherwise separate it from the vault it belongs to.
        return priority, vault_label, is_pending_settlement

    return sort_key


def equity_curve_by_asset_with_legend(input: ChartInput) -> Figure:
    """Equity curve by asset with the vault-logo legend below the chart.

    Renders :py:func:`equity_curve_by_asset` and replaces the native Plotly
    legend with the aligned allocation, chain, protocol and curator logo rows.
    """
    state = input.state
    fig = equity_curve_by_asset(input)
    fig.update_layout(title="Portfolio asset weights by token")
    add_asset_weight_legend(
        fig,
        build_asset_weight_legend_entries(state),
        legend_layout="vertical",
        legend_sort_key=make_asset_weight_legend_sort_key(state),
    )
    return fig
