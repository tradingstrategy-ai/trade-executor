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
from tradeexecutor.strategy.phase_aware import (
    EVENT_CLOSE,
    EVENT_PARK,
    EVENT_PROMOTE,
    EVENT_REDEEM_BLOCK,
    EVENT_REDEEM_CLEAR,
    iter_all_events,
)
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
    """Waiting-to-trigger buffers (not yet in flight): waiting deposits positive, redemption-locked negative.

    The phase-aware sibling of :py:func:`pending_vault_settlements`: rather than in-flight async
    settlements, it shows the buffers still waiting for a window to open, reconstructed from the
    durable event log in ``state.other_data`` (never per-position ``other_data``, which is lost on
    resize/close):

    - **Waiting deposits** (positive): cash parked in the queue venue for a target vault whose
      deposit window is closed. A park opens the buffer; a later promote (deposited once the window
      opened) or close (no longer targeted) ends it.
    - **Redemption locked** (negative): value held in positions whose redemption is currently
      blocked (closed redemption window / lock-up) - the capital that could not be exited this
      cycle even if wanted. A redeem-block opens the buffer; a redeem-clear (the window opened,
      the lock-up expired, or the position is gone) ends it.

    Returns (fig, df) tuple so diagnostics can inspect the underlying data.
    """
    state = input.state

    def _empty() -> tuple[Figure, pd.DataFrame]:
        fig = Figure()
        fig.update_layout(
            title="Waiting trigger queue (waiting deposits positive, redemption-locked negative)",
            xaxis_title="Time",
            yaxis_title="US dollar size",
            template="plotly_dark",
        )
        return fig, pd.DataFrame()

    timestamps = [ps.calculated_at for ps in state.stats.portfolio]
    # Events carry an ISO timestamp (older logs may not); those without one cannot be placed on the
    # time axis and are skipped.
    event_ts = [(pd.Timestamp(e.timestamp), e) for e in iter_all_events(state.other_data) if e.timestamp is not None]
    if not timestamps or not event_ts:
        return _empty()
    event_ts.sort(key=lambda pair: pair[0])

    # Single chronological sweep: apply each event once as the statistics timestamps advance,
    # keeping the open buffer sets as rolling state - the waiting buffers persist between the
    # (sparse) event cycles, a proper step series.
    rows = []
    event_idx = 0
    open_deposit_usd: dict[int, float] = {}
    open_redeem_usd: dict[int, float] = {}
    for ts in sorted(pd.Timestamp(t) for t in timestamps):
        while event_idx < len(event_ts) and event_ts[event_idx][0] <= ts:
            event = event_ts[event_idx][1]
            if event.kind == EVENT_PARK:
                open_deposit_usd[event.vault_internal_id] = event.usd
            elif event.kind in (EVENT_PROMOTE, EVENT_CLOSE):
                open_deposit_usd.pop(event.vault_internal_id, None)
            elif event.kind == EVENT_REDEEM_BLOCK:
                open_redeem_usd[event.vault_internal_id] = event.usd
            elif event.kind == EVENT_REDEEM_CLEAR:
                open_redeem_usd.pop(event.vault_internal_id, None)
            event_idx += 1
        deposit_total = sum(open_deposit_usd.values())
        if deposit_total:
            rows.append({"timestamp": ts, "series": "Waiting deposits", "value": deposit_total})
        redeem_total = sum(open_redeem_usd.values())
        if redeem_total:
            rows.append({"timestamp": ts, "series": "Redemption locked", "value": -redeem_total})

    if not rows:
        return _empty()

    df = pd.DataFrame(rows)
    df = df.groupby(["timestamp", "series"])["value"].sum().reset_index()
    df = df.sort_values("timestamp")
    df = df.pivot(index="timestamp", columns="series", values="value").fillna(0)

    fig = px.area(
        df,
        title="Waiting trigger queue (waiting deposits positive, redemption-locked negative)",
        labels={"index": "Time", "value": "US dollar size"},
        color_discrete_sequence=colors.qualitative.Light24,
        template="plotly_dark",
    )
    fig.update_traces(line_width=0)
    return fig, df


def _resolve_vault_label(input: ChartInput, vault_internal_id: int) -> dict:
    """Resolve a queue-event vault id to display metadata."""
    pair = None
    strategy_universe = None
    if input.strategy_input_indicators is not None:
        strategy_universe = input.strategy_universe

    if strategy_universe is not None:
        try:
            pair = strategy_universe.get_pair_by_id(vault_internal_id)
        except KeyError:
            pair = None

    if pair is None:
        return {
            "Vault id": vault_internal_id,
            "Vault": f"Vault {vault_internal_id}",
            "Ticker": str(vault_internal_id),
            "Protocol": "",
            "Chain": "",
        }

    return {
        "Vault id": vault_internal_id,
        "Vault": pair.get_vault_name() or pair.get_ticker(),
        "Ticker": pair.get_ticker(),
        "Protocol": pair.get_vault_protocol() or "",
        "Chain": pair.chain_id,
    }


def _build_queue_wait_intervals(input: ChartInput) -> pd.DataFrame:
    """Build deposit-wait intervals from phase-aware park/promote/close events."""
    state = input.state
    if state is None:
        return pd.DataFrame()

    event_ts = [
        (pd.Timestamp(e.timestamp), e)
        for e in iter_all_events(state.other_data)
        if e.timestamp is not None and e.kind in (EVENT_PARK, EVENT_PROMOTE, EVENT_CLOSE)
    ]
    if not event_ts:
        return pd.DataFrame()
    event_ts.sort(key=lambda pair: pair[0])

    stats_timestamps = [pd.Timestamp(ps.calculated_at) for ps in state.stats.portfolio if ps.calculated_at is not None]
    if stats_timestamps:
        final_ts = max(stats_timestamps)
    elif input.end_at is not None:
        final_ts = pd.Timestamp(input.end_at)
    else:
        final_ts = max(ts for ts, _event in event_ts)

    open_waits: dict[int, dict] = {}
    rows = []

    def _accrue(wait: dict, ts: pd.Timestamp) -> None:
        elapsed_days = max((ts - wait["last_update"]).total_seconds() / 86400, 0.0)
        wait["usd_days"] += wait["usd"] * elapsed_days
        wait["last_update"] = ts

    def _close_wait(vault_id: int, end_ts: pd.Timestamp, event_kind: str) -> None:
        wait = open_waits.pop(vault_id, None)
        if wait is None:
            return
        _accrue(wait, end_ts)
        duration_days = max((end_ts - wait["started_at"]).total_seconds() / 86400, 0.0)
        metadata = _resolve_vault_label(input, vault_id)
        rows.append({
            **metadata,
            "Started at": wait["started_at"],
            "Ended at": end_ts,
            "Status": "promoted" if event_kind == EVENT_PROMOTE else "closed",
            "Duration days": duration_days,
            "Start USD": wait["start_usd"],
            "End USD": wait["usd"],
            "Max USD": wait["max_usd"],
            "USD days": wait["usd_days"],
            "Updates": wait["updates"],
        })

    for ts, event in event_ts:
        vault_id = event.vault_internal_id
        if event.kind == EVENT_PARK:
            wait = open_waits.get(vault_id)
            if wait is None:
                open_waits[vault_id] = {
                    "started_at": ts,
                    "last_update": ts,
                    "start_usd": event.usd,
                    "usd": event.usd,
                    "max_usd": event.usd,
                    "usd_days": 0.0,
                    "updates": 1,
                }
            else:
                _accrue(wait, ts)
                wait["usd"] = event.usd
                wait["max_usd"] = max(wait["max_usd"], event.usd)
                wait["updates"] += 1
        elif event.kind in (EVENT_PROMOTE, EVENT_CLOSE):
            _close_wait(vault_id, ts, event.kind)

    for vault_id in list(open_waits):
        wait = open_waits.pop(vault_id)
        _accrue(wait, final_ts)
        duration_days = max((final_ts - wait["started_at"]).total_seconds() / 86400, 0.0)
        metadata = _resolve_vault_label(input, vault_id)
        rows.append({
            **metadata,
            "Started at": wait["started_at"],
            "Ended at": pd.NaT,
            "Status": "open",
            "Duration days": duration_days,
            "Start USD": wait["start_usd"],
            "End USD": wait["usd"],
            "Max USD": wait["max_usd"],
            "USD days": wait["usd_days"],
            "Updates": wait["updates"],
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.sort_values(["Started at", "Vault id"]).reset_index(drop=True)


def phase_aware_queue_duration_summary(input: ChartInput) -> pd.DataFrame:
    """Average phase-aware queue wait duration by target vault."""
    intervals = _build_queue_wait_intervals(input)
    if intervals.empty:
        return pd.DataFrame()

    group_cols = ["Vault id", "Vault", "Ticker", "Protocol", "Chain"]
    rows = []
    for key, group in intervals.groupby(group_cols, dropna=False, sort=False):
        duration = group["Duration days"]
        usd_days = group["USD days"].sum()
        max_usd = group["Max USD"].max()
        start_usd = group["Start USD"].mean()
        rows.append({
            "Vault id": key[0],
            "Vault": key[1],
            "Ticker": key[2],
            "Protocol": key[3],
            "Chain": key[4],
            "Episodes": len(group),
            "Promoted": int((group["Status"] == "promoted").sum()),
            "Closed": int((group["Status"] == "closed").sum()),
            "Open": int((group["Status"] == "open").sum()),
            "Average wait days": duration.mean(),
            "Median wait days": duration.median(),
            "Max wait days": duration.max(),
            "Average start USD": start_usd,
            "Max queued USD": max_usd,
            "USD days": usd_days,
        })

    df = pd.DataFrame(rows)
    return df.sort_values(["USD days", "Average wait days"], ascending=[False, False]).reset_index(drop=True)


def phase_aware_queue_duration(input: ChartInput) -> tuple[Figure, pd.DataFrame]:
    """Average phase-aware queue wait duration by target vault."""
    df = phase_aware_queue_duration_summary(input)
    if df.empty:
        fig = Figure()
        fig.update_layout(
            title="Phase-aware queue wait duration by target vault",
            xaxis_title="Average wait days",
            yaxis_title="Target vault",
            template="plotly_dark",
        )
        return fig, df

    fig = px.bar(
        df.sort_values("Average wait days", ascending=True),
        x="Average wait days",
        y="Vault",
        orientation="h",
        color="Status" if "Status" in df.columns else "Protocol",
        title="Average phase-aware queue wait duration by target vault",
        labels={
            "Average wait days": "Average wait days",
            "Vault": "Target vault",
            "Protocol": "Protocol",
        },
        hover_data={
            "Episodes": True,
            "Promoted": True,
            "Closed": True,
            "Open": True,
            "Max wait days": ":.2f",
            "USD days": ":,.2f",
            "Max queued USD": ":$,.2f",
        },
        template="plotly_dark",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
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
