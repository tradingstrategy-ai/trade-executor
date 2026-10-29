"""Exchange-account charts: internal share price, shares, value and profit over time.

An exchange-account position, such as a Lighter master account or
sub-account, measures its performance with an internal share price that
capital flows mint and burn shares against without moving. These charts show
that series from the per-cycle :py:class:`~tradeexecutor.state.statistics.PositionStatistics`,
with a marker at each flow and valuation, so a person can check that flows
leave the price flat and only valuations move it.

See ``docs/plans/lighter-subaccount-flow-neutral-profit.md``.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

from tradeexecutor.analysis.vault_position_helpers import find_latest_position_for_pair
from tradeexecutor.state.balance_update import BalanceUpdateCause
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeFlag
from tradeexecutor.strategy.chart.definition import ChartInput

#: Marker style per event label: plotly symbol and colour.
EVENT_MARKER_STYLES: dict[str, tuple[str, str]] = {
    "Deposit": ("triangle-up", "green"),
    "Allocation": ("square", "green"),
    "Withdrawal": ("triangle-down", "red"),
    "Transfer in": ("diamond", "royalblue"),
    "Transfer out": ("diamond", "darkorange"),
    "Valuation": ("circle", "grey"),
    "Write-off": ("x", "black"),
}


def get_position_statistics_frame(state: State, position: TradingPosition) -> pd.DataFrame:
    """Read a position's recorded statistics as a time series.

    :return:
        DataFrame indexed by ``timestamp`` with columns ``share_price``,
        ``units``, ``value`` and ``profit_usd``, sorted by time; empty if no
        statistics were recorded. Points before the share-price state existed
        have ``NaN`` share price and units.
    """
    rows = [
        {
            "timestamp": stat.calculated_at,
            "share_price": stat.internal_share_price,
            "units": stat.internal_total_supply,
            "value": stat.value,
            "profit_usd": stat.internal_profit_usd,
        }
        for stat in state.stats.positions.get(position.position_id, [])
    ]
    frame = pd.DataFrame(rows, columns=["timestamp", "share_price", "units", "value", "profit_usd"])
    return frame.set_index("timestamp").sort_index().astype(float)


def get_exchange_account_events(state: State, position: TradingPosition) -> pd.DataFrame:
    """List the flows and valuations of an exchange-account position.

    Deposits, withdrawals, allocations and incoming transfers are trades on
    the position. An outgoing transfer is a funding trade on another
    position that names this one as ``funding_source_position_id``; its fee
    and principal balance updates on this position are not listed again as
    valuations. Every other balance update is a valuation, or a write-off
    when its cause is a correction.

    :return:
        DataFrame with columns ``timestamp`` and ``event``, one of the keys of
        :py:data:`EVENT_MARKER_STYLES`, sorted by time.
    """
    events = []
    for trade in position.trades.values():
        if not trade.is_success() or trade.executed_at is None:
            continue
        flags = trade.flags or set()
        if TradeFlag.external_account_transfer in flags:
            events.append((trade.executed_at, "Deposit" if trade.is_buy() else "Withdrawal"))
        elif TradeFlag.internal_account_transfer in flags:
            events.append((trade.executed_at, "Transfer in"))
        elif trade.is_buy() and (trade.executed_reserve or 0) > 1:
            events.append((trade.executed_at, "Allocation"))

    funding_balance_update_ids = set()
    for trade in state.portfolio.get_all_trades():
        other_data = trade.other_data or {}
        if trade.is_internal_account_transfer() and trade.is_success() and int(other_data.get("funding_source_position_id", -1)) == position.position_id:
            events.append((trade.executed_at, "Transfer out"))
            funding_balance_update_ids.update(other_data.get("funding_source_balance_update_ids", []))

    for balance_update in position.balance_updates.values():
        if balance_update.balance_update_id in funding_balance_update_ids:
            continue
        label = "Write-off" if balance_update.cause == BalanceUpdateCause.correction else "Valuation"
        events.append((balance_update.block_mined_at, label))

    return pd.DataFrame(events, columns=["timestamp", "event"]).sort_values("timestamp", kind="stable").reset_index(drop=True)


def exchange_account_share_price(
    input: ChartInput,
    height: int = 900,
    width: int = 1200,
) -> Figure | None:
    """Internal share price, shares, value and USD profit of one exchange-account position.

    Takes one exchange-account pair in ``input.pairs`` and charts its latest
    position from the recorded per-cycle statistics, in three panels: the
    internal share price with a marker at each flow and valuation, the
    outstanding internal shares, and the value with the USD profit. Flows
    should leave the share price flat and move only the shares; valuations
    should move only the share price.

    :return:
        The figure, or ``None`` if the pair has no position.
    """
    assert input.state, "This chart needs the state"
    assert input.pairs and len(input.pairs) == 1, "This chart takes a single exchange-account pair"
    pair = input.pairs[0]
    assert pair.is_exchange_account(), f"Not an exchange-account pair: {pair}"

    position = find_latest_position_for_pair(input.state, pair)
    if position is None:
        return None

    series = get_position_statistics_frame(input.state, position)
    # Start where the share-price series starts: before the first flow or
    # valuation the account is a placeholder at zero.
    tracked = series["share_price"].notna()
    if tracked.any():
        series = series.loc[tracked.idxmax():]
    events = get_exchange_account_events(input.state, position)

    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Internal share price", "Internal shares", "Value and USD profit"),
    )
    # Step lines: every figure changes only at an event, never in between.
    figure.add_trace(go.Scatter(x=series.index, y=series["share_price"], mode="lines", name="Share price", line={"shape": "hv", "color": "black"}), row=1, col=1)

    priced = series["share_price"].dropna()
    if len(priced) and len(events):
        # Place each marker on the share price recorded at or before its event.
        located = pd.merge_asof(events, priced.rename("y").reset_index(), on="timestamp", direction="backward")
        for label, group in located.dropna(subset=["y"]).groupby("event", sort=False):
            symbol, colour = EVENT_MARKER_STYLES[label]
            figure.add_trace(
                go.Scatter(x=group["timestamp"], y=group["y"], mode="markers", name=label, marker={"symbol": symbol, "color": colour, "size": 11, "line": {"width": 1, "color": "white"}}),
                row=1,
                col=1,
            )

    figure.add_trace(go.Scatter(x=series.index, y=series["units"], mode="lines", name="Shares", line={"shape": "hv", "color": "royalblue"}), row=2, col=1)
    figure.add_trace(go.Scatter(x=series.index, y=series["value"], mode="lines", name="Value (USD)", line={"shape": "hv", "color": "teal"}), row=3, col=1)
    figure.add_trace(go.Scatter(x=series.index, y=series["profit_usd"], mode="lines", name="Profit (USD)", line={"shape": "hv", "color": "purple", "dash": "dot"}), row=3, col=1)

    status = "closed" if position.is_closed() else "open"
    figure.update_layout(
        title=f"{pair.get_ticker()} position #{position.position_id}, {status}: return {position.get_total_profit_percent():.2%}, profit {position.get_total_profit_usd():,.4f} USD",
        height=height,
        width=width,
        legend={"orientation": "h", "y": -0.08},
    )
    return figure


def all_exchange_account_share_prices(
    input: ChartInput,
    height: int = 700,
    width: int = 1200,
) -> Figure:
    """Internal share price of every exchange-account position on one chart.

    Charts the positions of ``input.pairs`` if given, otherwise every
    exchange-account position in the state, one line each, from the recorded
    per-cycle statistics, for comparing accounts side by side.

    :return:
        The figure.
    """
    assert input.state, "This chart needs the state"
    positions = [
        position
        for position in input.state.portfolio.get_all_positions()
        if position.is_exchange_account() and (not input.pairs or position.pair in input.pairs)
    ]
    figure = go.Figure()
    for position in sorted(positions, key=lambda position: position.position_id):
        series = get_position_statistics_frame(input.state, position)["share_price"].dropna()
        if len(series):
            figure.add_trace(go.Scatter(x=series.index, y=series, mode="lines", line={"shape": "hv"}, name=f"{position.pair.get_ticker()} #{position.position_id}"))
    figure.update_layout(title="Internal share price of exchange accounts", height=height, width=width, yaxis_title="Share price")
    return figure
