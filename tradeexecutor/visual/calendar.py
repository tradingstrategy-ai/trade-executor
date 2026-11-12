"""Calendar profit comparisons for fixed-capital backtests.

These figures use whole-wallet equity, including unrealised gains and recorded
costs. They are not closed-trade PnL, and must not be used on an equity curve
containing external deposits or withdrawals without adjusting those flows first.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def calculate_calendar_profit(
    equity: pd.Series,
    initial_cash: float,
    start_at,
    end_at,
    valuation_interval: pd.Timedelta = pd.Timedelta(days=1),
) -> pd.DataFrame:
    """Calculate Q1–Q4 and calendar-year profits on a UTC evaluation calendar.

    ``[start_at, end_at)`` contains valuation timestamps. Each timestamp books
    the change since the preceding valuation; the first uses ``initial_cash``.
    A quarter's opening equity is the preceding quarter's final valuation, not
    its own first valuation. Thus first-day fees are retained and USD profits
    telescope to final equity less initial cash. Returns compound across periods.

    Input must be a complete regular valuation grid: missing/duplicate,
    unordered, non-finite or non-positive values fail rather than becoming zero
    profit. No forward filling or extrapolation is performed here. Upstream
    carried/stale NAVs cannot be detected from an account curve alone.

    ``partial`` labels calendar periods clipped by the evaluation boundaries.
    Unobserved quarters are absent from the table and displayed as N/A by the
    renderer. Year totals cover observed support and are not annualised.
    """
    columns = ["profit_usd", "return", "opening_equity", "closing_equity",
               "observations", "partial", "start_at", "end_at"]
    if not isinstance(equity, pd.Series) or not isinstance(equity.index, pd.DatetimeIndex):
        raise ValueError("equity must be a Series with a DatetimeIndex")
    start, end = pd.Timestamp(start_at), pd.Timestamp(end_at)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    interval = pd.Timedelta(valuation_interval)
    if pd.isna(start) or pd.isna(end) or start >= end or interval <= pd.Timedelta(0):
        raise ValueError("Require finite start_at < end_at and a positive valuation interval")
    if not np.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("initial_cash must be finite and positive")
    if equity.empty:
        return pd.DataFrame(columns=columns, index=pd.MultiIndex.from_tuples([], names=["year", "period"]))
    curve = equity.astype(float).copy()
    curve.index = pd.to_datetime(curve.index, utc=True)
    if curve.index.hasnans or not curve.index.is_unique or not curve.index.is_monotonic_increasing:
        raise ValueError("Valuation timestamps must be finite, unique and increasing")
    if not np.isfinite(curve.to_numpy()).all() or (curve <= 0).any():
        raise ValueError("Equity must be finite and positive")
    expected = pd.date_range(start, end, freq=interval, inclusive="left")
    if not curve.index.equals(expected):
        raise ValueError("Equity must cover the complete evaluation valuation grid; missing observations are not zero profit")
    local_index = curve.index.tz_localize(None)
    records = []
    for frequency in ("Q", "Y"):
        opening = float(initial_cash)
        for period, values in curve.groupby(local_index.to_period(frequency)):
            closing = float(values.iloc[-1])
            period_start = period.start_time.tz_localize("UTC")
            period_end = (period + 1).start_time.tz_localize("UTC")
            records.append({
                "year": period.year,
                "period": f"Q{period.quarter}" if frequency == "Q" else "Year",
                "profit_usd": closing - opening,
                "return": closing / opening - 1,
                "opening_equity": opening,
                "closing_equity": closing,
                "observations": len(values),
                "partial": start > period_start or end < period_end,
                "start_at": max(start, period_start),
                "end_at": min(end, period_end),
            })
            opening = closing
    return pd.DataFrame(records).set_index(["year", "period"])[columns].sort_index()


def visualise_quarterly_profit(
    table: pd.DataFrame,
    title: str = "Quarterly whole-wallet profit",
    colour_limit: float | None = None,
) -> go.Figure:
    """Render quarterly USD profit, percentage returns and observed year totals.

    Colours encode USD profit on a symmetric zero-centred scale. Pass the same
    ``colour_limit`` to comparable wallets for a shared colour scale. N/A cells
    are grey; an asterisk identifies partial quarters/years. The full-precision
    table from :func:`calculate_calendar_profit` is the accompanying evidence.
    """
    periods = ["Q1", "Q2", "Q3", "Q4", "Year"]
    years = sorted(table.index.get_level_values("year").unique())
    if not years:
        fig = go.Figure()
        fig.add_annotation(text="No observed valuations", showarrow=False)
        fig.update_layout(title=title)
        return fig
    grid = pd.MultiIndex.from_product([years, periods], names=["year", "period"])
    ordered = table.reindex(grid)
    z = ordered.profit_usd.to_numpy(dtype=float).reshape(len(years), 5)
    text = []
    for year in years:
        row = []
        for period in periods:
            entry = ordered.loc[(year, period)]
            if pd.isna(entry.profit_usd):
                row.append("N/A")
            else:
                marker = "*" if entry.partial else ""
                row.append(f"${entry.profit_usd:+,.0f}<br>{entry['return']:+.2%}{marker}")
        text.append(row)
    limit = float(np.nanmax(np.abs(z))) if colour_limit is None else float(colour_limit)
    if not np.isfinite(limit) or limit < 0 or limit < float(np.nanmax(np.abs(z))):
        raise ValueError("colour_limit must be finite and cover every plotted profit")
    limit = max(limit, 1.0)
    # A separate text trace keeps N/A visible even where heatmap z is missing.
    fig = go.Figure(go.Heatmap(
        z=z, x=periods, y=[str(y) for y in years], colorscale="RdYlGn",
        zmin=-limit, zmax=limit, zmid=0, xgap=3, ygap=3,
        hovertemplate="%{y} %{x}: $%{z:,.2f}<extra></extra>",
        colorbar=dict(title="Profit USD"), hoverongaps=False,
    ))
    fig.add_trace(go.Scatter(
        x=periods * len(years), y=[str(y) for y in years for _ in periods],
        mode="text", text=[cell for row in text for cell in row],
        textfont=dict(size=16, color="black"), hoverinfo="skip", showlegend=False,
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=21)), template="plotly_white",
        plot_bgcolor="#e8e8e8", width=1100, height=max(350, 180 + 90 * len(years)),
        margin=dict(t=90, b=90, l=80, r=130),
        xaxis=dict(title="Calendar quarter (UTC); Year = observed total", side="top"),
        yaxis=dict(title="Year", autorange="reversed", type="category"),
        annotations=[dict(
            text="USD profit includes unrealised PnL and recorded fees · % uses opening equity · * partial period · N/A unobserved",
            x=0, y=-0.24, xref="paper", yref="paper", showarrow=False,
            xanchor="left", font=dict(size=12),
        )],
    )
    return fig
