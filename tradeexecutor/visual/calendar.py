"""Calendar profit comparisons for fixed-capital backtests.

These figures use whole-wallet equity, including unrealised gains and recorded
costs. They are not closed-trade PnL, and must not be used on an equity curve
containing external deposits or withdrawals without adjusting those flows first.
"""

from html import escape
from textwrap import wrap

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
    title: str = "Quarterly whole-wallet returns",
    colour_limit: float | None = None,
) -> go.Figure:
    """Render quarterly percentage returns and compounded observed year totals.

    Cells, hover text and colours use returns only. ``colour_limit`` is a
    fractional return (0.10 means 10%); share it across comparable wallets.
    N/A cells are grey, and an asterisk identifies partial quarters/years.
    USD accounting remains in the accompanying calculation table.

    All chart text uses the same 16 px Arial font. Title, calendar subtitle,
    quarter labels and footnotes occupy separate bands; long titles wrap and
    increase the top margin without reducing the cell area.
    """
    periods = ["Q1", "Q2", "Q3", "Q4", "Year"]
    font = dict(family="Arial", size=16, color="#222222")
    title_lines = wrap(title, width=82) or [""]
    title_text = "<br>".join(escape(line) for line in title_lines)
    years = sorted(table.index.get_level_values("year").unique())
    top = 120 + 24 * (len(title_lines) - 1)
    height = top + max(140, 64 * len(years)) + 100
    fig = go.Figure()
    fig.update_layout(
        font=font, template="plotly_white", width=1100, height=height,
        title=dict(text=f"<b>{title_text}</b>", font=font,
                   x=80 / 1100, xanchor="left", y=1 - 20 / height, yanchor="top"),
        margin=dict(t=top, b=100, l=80, r=150), plot_bgcolor="#e8e8e8",
    )
    if not years:
        fig.add_annotation(text="No observed valuations", showarrow=False, font=font)
        return fig
    grid = pd.MultiIndex.from_product([years, periods], names=["year", "period"])
    ordered = table.reindex(grid)
    z = ordered["return"].to_numpy(dtype=float).reshape(len(years), 5)
    observed_max = float(np.nanmax(np.abs(z)))
    limit = observed_max if colour_limit is None else float(colour_limit)
    if not np.isfinite(limit) or limit < 0 or limit < observed_max:
        raise ValueError("colour_limit must be finite and cover every plotted return")
    limit = max(limit, 0.01)
    text = []
    text_colours = []
    for _, entry in ordered.iterrows():
        value = entry["return"]
        if pd.isna(value):
            text.append("N/A")
            text_colours.append(font["color"])
        else:
            marker = "*" if entry.partial else ""
            text.append(f"{value:+.2%}{marker}")
            text_colours.append("white" if abs(value) >= 0.65 * limit else font["color"])
    fig.add_trace(go.Heatmap(
        z=z, x=periods, y=[str(y) for y in years], colorscale="RdYlGn",
        zmin=-limit, zmax=limit, zmid=0, xgap=3, ygap=3,
        text=np.asarray(text).reshape(len(years), 5),
        hovertemplate="%{y} %{x}: %{text}<extra></extra>",
        colorbar=dict(title=dict(text="Return", font=font), tickfont=font,
                      tickformat="+.0%", thickness=20, len=0.95),
        hoverongaps=False,
    ))
    # A separate text trace keeps N/A visible where the heatmap has no z value.
    fig.add_trace(go.Scatter(
        x=periods * len(years), y=[str(y) for y in years for _ in periods],
        mode="text", text=text,
        textfont=dict(family=font["family"], size=font["size"], color=text_colours),
        hoverinfo="skip", showlegend=False,
    ))
    start, end = pd.Timestamp(table.start_at.min()), pd.Timestamp(table.end_at.max())
    date_format = "%Y-%m-%d" if start == start.normalize() and end == end.normalize() else "%Y-%m-%d %H:%M:%S"
    calendar = f"UTC {start.strftime(date_format)} to {end.strftime(date_format)} (end exclusive)"
    fig.update_layout(
        xaxis=dict(side="top", tickfont=font, fixedrange=True),
        yaxis=dict(autorange="reversed", type="category", tickfont=font, fixedrange=True),
        hoverlabel=dict(font=font),
        annotations=[
            dict(text=calendar, x=0, y=1, yshift=42, xref="paper", yref="paper",
                 xanchor="left", yanchor="bottom", showarrow=False, font=font),
            dict(text="* partial period · N/A unobserved · Year = compounded observed return",
                 x=0, y=0, yshift=-24, xref="paper", yref="paper",
                 xanchor="left", yanchor="top", showarrow=False, font=font),
            dict(text="Includes unrealised gains and recorded fees; returns use period-opening equity.",
                 x=0, y=0, yshift=-50, xref="paper", yref="paper",
                 xanchor="left", yanchor="top", showarrow=False, font=font),
        ],
    )
    return fig
