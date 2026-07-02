"""Vault charts."""
import datetime
import warnings

import pandas as pd
import plotly.colors as colors
import plotly.express as px
from plotly.graph_objects import Figure
from tradeexecutor.analysis.credit import display_vault_position_table, display_vault_daily_pnl_table
from tradeexecutor.analysis.vault_position_helpers import find_latest_position_for_pair
from tradeexecutor.analysis.vault import visualise_vaults
from tradeexecutor.strategy.chart.definition import ChartInput
from tradeexecutor.strategy.phase_aware import EVENT_CLOSE, EVENT_PARK, EVENT_PROMOTE, iter_all_events
from tradeexecutor.visual.position import (calculate_position_timeline,
                                           visualise_position)


def all_vaults_share_price_and_tvl(
    input: ChartInput,
    printer=print,
    max_count=3,
) -> list[Figure]:
    """Render share price and TVL for all vaults.

    - Get vault pairs from the strategy universe

    :return:
        List of figures
    """
    figures = visualise_vaults(input.strategy_universe, printer=printer, max_count=max_count)

    if not figures:
        raise ValueError("No chart data available for vault pairs - missing candle or liquidity data")

    return figures

def vault_position_timeline(
    input: ChartInput,
    cut_off_date: pd.Timestamp = None,
    height=2000,
    width=1200,
) -> tuple[Figure | None, pd.DataFrame | None]:
    """How a single vault position evolved over time.

    - Takes vault pair as an input

    :return:
        Figure visualising the position timeline and a DataFrame with individual trades
    """
    assert input.state

    state = input.state
    strategy_universe = input.strategy_universe

    assert input.pairs and len(input.pairs) == 1, "This chart only supports a single vault pair."
    pair = input.pairs[0]

    position = find_latest_position_for_pair(state, pair)
    if position is None:
        return None, None

    if input.execution_context.mode.is_backtesting():
        assert state.backtest_data
        end_at = state.backtest_data.end_at
    else:
        start_at, end_at = state.get_strategy_start_and_end()

    position_df = calculate_position_timeline(
        strategy_universe,
        position,
        end_at,
    )

    if cut_off_date:
        position_df = position_df[position_df.index < cut_off_date]

    fig = visualise_position(
        position,
        position_df,
        extended=True,
        autosize=False,
        height=height,
        width=width,
    )

    with pd.option_context('display.min_rows', 500):  # Show up to 50 rows
        # Assuming df is your DataFrame and condition is your boolean mask
        mask = position_df.delta != 0  # Your original condition
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            extended_mask = mask | mask.shift(1).fillna(False) | mask.shift(-1).fillna(False)

        df = position_df[extended_mask]
        return fig, df


def all_vault_positions(
    chart_input: ChartInput,
    sort_by: str = "Opened",
    sort_ascending: bool = True,
    show_address: bool = False,
    top_n: int | None = 10,
    bottom_n: int | None = 10,
) -> pd.DataFrame:
    """Display vault positions in a table.

    By default shows the top 10 winners and bottom 10 losers.

    :param sort_by:
        Column name to sort by. E.g. "Profit USD", "Profit % annualised", "Opened".

    :param sort_ascending:
        Sort order.

    :param show_address:
        Display the vault smart contract address.

    :param top_n:
        Show the N most profitable positions. Set to ``None`` to show all.

    :param bottom_n:
        Show the N least profitable positions. Set to ``None`` to show all.
    """
    state = chart_input.state
    vault_df = display_vault_position_table(
        state,
        sort_by=sort_by,
        sort_ascending=sort_ascending,
        show_address=show_address,
        top_n=top_n,
        bottom_n=bottom_n,
    )
    return vault_df


def all_vault_daily_gains_losses(
    chart_input: ChartInput,
    top_n: int = 10,
    bottom_n: int = 10,
) -> pd.DataFrame:
    """Display the biggest daily gains and losses for vault positions.

    :param top_n:
        Show the N biggest daily gains.

    :param bottom_n:
        Show the N biggest daily losses.
    """
    state = chart_input.state
    return display_vault_daily_pnl_table(
        state,
        top_n=top_n,
        bottom_n=bottom_n,
    )


def _get_trade_pending_window(trade) -> tuple[datetime.datetime, datetime.datetime | None] | None:
    """Reconstruct the pending settlement window of an async vault trade.

    Settlement clears ``trade.vault_settlement_pending_at``, so the request
    timestamp must come from the durable ``other_data`` copy. Older state
    files predate ``vault_settlement_requested_at``; fall back to the live
    pending marker and finally the trade start.

    :return:
        ``(requested_at, settled_at)`` tuple, ``settled_at`` is None when the
        request is still pending. None if the trade is not an async vault flow.
    """
    if not trade.other_data.get("vault_async_flow"):
        return None

    requested_at_raw = trade.other_data.get("vault_settlement_requested_at")
    if requested_at_raw:
        requested_at = datetime.datetime.fromisoformat(requested_at_raw)
    elif trade.vault_settlement_pending_at is not None:
        requested_at = trade.vault_settlement_pending_at
    else:
        requested_at = trade.started_at or trade.opened_at

    if requested_at is None:
        return None

    if trade.is_success():
        settled_at = trade.executed_at
    elif trade.failed_at is not None:
        # Reclaimed/failed request (Ostium reclaim path) - the buffer ends at failure
        settled_at = trade.failed_at
    else:
        settled_at = None
    return requested_at, settled_at


def pending_vault_settlements(input: ChartInput) -> tuple[Figure, pd.DataFrame]:
    """In-flight async vault settlement buffers over time.

    Like the asset weight maps, but shows the capital queued in two-stage
    (ERC-7540 / Ostium) vault deposit and redemption requests on each strategy
    cycle: deposits as positive areas, redemptions as negative areas. The
    pending window of a request is half-open ``[requested_at, settled_at)`` —
    on the settlement cycle the capital is live again, so it no longer counts
    as a buffer.

    Returns (fig, df) tuple so diagnostics can inspect the underlying data.
    """
    state = input.state

    timestamps = [ps.calculated_at for ps in state.stats.portfolio]
    if not timestamps:
        return Figure(), pd.DataFrame()
    last_timestamp = max(timestamps)

    rows = []
    for position in state.portfolio.get_all_positions(pending=True):
        ticker = position.pair.get_vault_name() or position.pair.get_ticker()
        for trade in position.trades.values():
            window = _get_trade_pending_window(trade)
            if window is None:
                continue
            requested_at, settled_at = window
            # Half-open window: still-pending requests extend to the last statistics timestamp
            end = settled_at if settled_at is not None else last_timestamp + datetime.timedelta(seconds=1)
            if trade.is_buy():
                series = f"{ticker} deposit"
                value = float(trade.planned_reserve)
            else:
                series = f"{ticker} redeem"
                value = -abs(float(trade.planned_quantity)) * float(trade.planned_price or 0)
            for ts in timestamps:
                if requested_at <= ts < end:
                    rows.append({"timestamp": ts, "series": series, "value": value})

    if not rows:
        fig = Figure()
        fig.update_layout(
            title="Pending vault settlement buffers",
            xaxis_title="Time",
            yaxis_title="US dollar size",
            template="plotly_dark",
        )
        return fig, pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.groupby(["timestamp", "series"])["value"].sum().reset_index()
    df = df.sort_values("timestamp")
    df = df.pivot(index="timestamp", columns="series", values="value").fillna(0)
    df = df[sorted(df.columns)]

    fig = px.area(
        df,
        title="Pending vault settlement buffers (deposits positive, redemptions negative)",
        labels={"index": "Time", "value": "US dollar size"},
        color_discrete_sequence=colors.qualitative.Light24,
        template="plotly_dark",
    )
    fig.update_traces(line_width=0)
    return fig, df


def pending_trigger_queue(input: ChartInput) -> tuple[Figure, pd.DataFrame]:
    """Waiting-deposit buffers parked in the queue venue (not yet in flight) over time.

    The phase-aware sibling of :py:func:`pending_vault_settlements`: rather than in-flight async
    settlements, it shows cash *parked* in the queue venue for a target vault whose deposit window is
    closed, reconstructed from the durable park / promote / close event log in ``state.other_data``
    (never per-position ``other_data``, which is lost on resize/close). A park opens a waiting deposit;
    a later promote (deposited once the window opened) or close (no longer targeted) ends it. The event
    log records deposits only, so this shows a waiting-deposit band (there is no redemption band).

    Returns (fig, df) tuple so diagnostics can inspect the underlying data.
    """
    state = input.state

    def _empty() -> tuple[Figure, pd.DataFrame]:
        fig = Figure()
        fig.update_layout(
            title="Waiting deposit queue (parked in the queue venue)",
            xaxis_title="Time",
            yaxis_title="US dollar size",
            template="plotly_dark",
        )
        return fig, pd.DataFrame()

    timestamps = [ps.calculated_at for ps in state.stats.portfolio]
    # Events carry an ISO timestamp (older logs may not); those without one cannot be placed on the
    # time axis and are skipped. iter_all_events yields in cycle (chronological) order.
    event_ts = [(pd.Timestamp(e.timestamp), e) for e in iter_all_events(state.other_data) if e.timestamp is not None]
    if not timestamps or not event_ts:
        return _empty()

    # For each statistics timestamp, fold every event at or before it to get the open-park set, so
    # the waiting-deposit buffer persists between the (sparse) event cycles - a proper step series.
    rows = []
    for ts in timestamps:
        cutoff = pd.Timestamp(ts)
        open_usd: dict[int, float] = {}
        for e_ts, event in event_ts:
            if e_ts > cutoff:
                break
            if event.kind == EVENT_PARK:
                open_usd[event.vault_internal_id] = event.usd
            elif event.kind in (EVENT_PROMOTE, EVENT_CLOSE):
                open_usd.pop(event.vault_internal_id, None)
        total = sum(open_usd.values())
        if total:
            rows.append({"timestamp": cutoff, "series": "Waiting deposits", "value": total})

    if not rows:
        return _empty()

    df = pd.DataFrame(rows)
    df = df.groupby(["timestamp", "series"])["value"].sum().reset_index()
    df = df.sort_values("timestamp")
    df = df.pivot(index="timestamp", columns="series", values="value").fillna(0)

    fig = px.area(
        df,
        title="Waiting deposit queue (parked in the queue venue)",
        labels={"index": "Time", "value": "US dollar size"},
        color_discrete_sequence=colors.qualitative.Light24,
        template="plotly_dark",
    )
    fig.update_traces(line_width=0)
    return fig, df


def vault_data_freshness(input: ChartInput) -> pd.DataFrame:
    """Data freshness timestamps for all vault pairs showing candle and TVL staleness.

    Reports the last real (non-forward-filled) timestamp and the latest
    timestamp for both candle and TVL data per vault. When available, also
    includes universe-level vault history startup diagnostics such as cache
    age, remote metadata, and pre-filter / post-filter / post-resample
    freshness markers. Vaults with no real data at all appear at the top,
    followed by stalest-first ordering.
    """
    from tradeexecutor.ethereum.vault.checks import get_vault_data_freshness
    return get_vault_data_freshness(input.strategy_universe)
