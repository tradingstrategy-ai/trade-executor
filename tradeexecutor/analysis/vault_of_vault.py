"""Vault-of-vault analysis helpers."""

import datetime

import pandas as pd
import plotly.graph_objects as go
from eth_defi.chain import get_chain_name
from plotly.graph_objs import Figure

from tradeexecutor.statistics.key_metric import (
    calculate_cagr,
    calculate_max_drawdown,
    calculate_sharpe,
)
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


def _to_datetime_indexed_series(series: pd.Series) -> pd.Series:
    """Convert a series to a sorted datetime-indexed series."""
    series = series.copy()

    if isinstance(series.index, pd.MultiIndex):
        series.index = pd.to_datetime(series.index.get_level_values("timestamp"))
    else:
        series.index = pd.to_datetime(series.index)

    return series.sort_index()


def _window_series(
    series: pd.Series,
    start_at: datetime.datetime,
    end_at: datetime.datetime,
) -> pd.Series:
    """Slice a series to the requested analysis window."""
    series = _to_datetime_indexed_series(series.dropna().astype("float64"))
    return series.loc[
        (series.index >= pd.Timestamp(start_at))
        & (series.index <= pd.Timestamp(end_at))
    ]


def _normalise_equity_curve(
    series: pd.Series,
    start_at: datetime.datetime,
    end_at: datetime.datetime,
) -> pd.Series:
    """Normalise an equity curve to start at ``1.0``."""
    series = _window_series(series, start_at, end_at)

    if len(series) < 2:
        return pd.Series(dtype="float64")

    return series / series.iloc[0]


def _build_vault_label(pair) -> str:
    """Build a display label for a vault pair."""
    return f"{pair.get_vault_name()} ({get_chain_name(pair.chain_id)})"


def _build_metrics_row(
    label: str,
    equity_curve: pd.Series,
    tvl_series: pd.Series | None,
    periods: int,
) -> dict:
    """Build a metrics row for the comparison table."""
    returns = equity_curve.pct_change().dropna()
    cagr = calculate_cagr(returns)
    sharpe = calculate_sharpe(returns, periods=periods)
    max_dd = calculate_max_drawdown(returns)
    calmar = cagr / abs(max_dd) if pd.notna(cagr) and pd.notna(max_dd) and abs(max_dd) > 0 else float("nan")
    last_tvl = tvl_series.iloc[-1] if tvl_series is not None and len(tvl_series) else float("nan")
    avg_tvl = tvl_series.mean() if tvl_series is not None and len(tvl_series) else float("nan")

    return {
        "Series": label,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Calmar": calmar,
        "Max DD": max_dd,
        "Last TVL": last_tvl,
        "Avg TVL": avg_tvl,
    }


def visualise_strategy_vs_selected_vaults(
    strategy_universe: TradingStrategyUniverse,
    strategy_equity_curve: pd.Series,
    start_at: datetime.datetime,
    end_at: datetime.datetime,
    strategy_label: str = "Strategy",
    periods: int = 365,
) -> tuple[Figure, pd.DataFrame]:
    """Create a strategy-vs-selected-vaults chart and metrics table.

    :return:
        Plotly figure and a numeric metrics DataFrame.
    """
    strategy_metric_curve = _normalise_equity_curve(
        strategy_equity_curve,
        start_at,
        end_at,
    ).rename(strategy_label)

    if len(strategy_metric_curve) < 2:
        raise ValueError("Strategy equity curve does not have enough data in the selected window")

    vault_metric_curves: dict[str, pd.Series] = {}
    vault_tvl_series: dict[str, pd.Series] = {}

    for pair in strategy_universe.iterate_pairs():
        if not pair.is_vault():
            continue

        candles = strategy_universe.data_universe.candles.get_candles_by_pair(pair.internal_id)
        if candles is None or len(candles) == 0:
            continue

        liquidity_candles = strategy_universe.data_universe.liquidity.get_liquidity_samples_by_pair(pair.internal_id)
        if liquidity_candles is None or len(liquidity_candles) == 0:
            continue

        metric_curve = _normalise_equity_curve(
            candles["close"],
            start_at,
            end_at,
        )
        if len(metric_curve) < 2:
            continue

        tvl_series = _window_series(liquidity_candles["close"], start_at, end_at)
        if len(tvl_series) == 0:
            continue

        tvl_series = tvl_series.reindex(metric_curve.index).ffill()
        label = _build_vault_label(pair)
        vault_metric_curves[label] = metric_curve
        vault_tvl_series[label] = tvl_series

    comparison_index = strategy_metric_curve.index
    for curve in vault_metric_curves.values():
        comparison_index = comparison_index.union(curve.index)
    comparison_index = comparison_index.sort_values()

    strategy_comparison_curve = strategy_metric_curve.reindex(comparison_index).ffill()
    vault_comparison_curves = pd.DataFrame(
        {
            label: curve.reindex(comparison_index).ffill()
            for label, curve in sorted(vault_metric_curves.items())
        }
    )

    fig = go.Figure()

    for vault_name in vault_comparison_curves.columns:
        fig.add_trace(
            go.Scatter(
                x=vault_comparison_curves.index,
                y=vault_comparison_curves[vault_name],
                mode="lines",
                name=vault_name,
                line=dict(color="rgba(128, 128, 128, 0.5)", width=1),
                showlegend=False,
                hovertemplate="%{fullData.name}<br>%{x|%Y-%m-%d}<br>Normalised equity: %{y:.2f}x<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=strategy_comparison_curve.index,
            y=strategy_comparison_curve,
            mode="lines",
            name=strategy_label,
            line=dict(color="#0b7285", width=3),
            hovertemplate=f"{strategy_label}<br>%{{x|%Y-%m-%d}}<br>Normalised equity: %{{y:.2f}}x<extra></extra>",
        )
    )

    fig.update_layout(
        title="Strategy equity curve versus selected vaults",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title="Normalised equity", tickformat=".2f")
    fig.update_xaxes(title="Time")

    metrics_rows = [
        _build_metrics_row(
            strategy_label,
            strategy_metric_curve,
            None,
            periods=periods,
        )
    ]
    metrics_rows.extend(
        _build_metrics_row(
            vault_name,
            vault_metric_curves[vault_name],
            vault_tvl_series[vault_name],
            periods=periods,
        )
        for vault_name in sorted(vault_metric_curves)
    )

    metrics_df = pd.DataFrame(metrics_rows).set_index("Series")
    return fig, metrics_df


def format_strategy_vs_selected_vaults_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Format strategy-vs-selected-vaults metrics for notebook display."""
    display_df = metrics_df.copy()
    display_df["CAGR"] = display_df["CAGR"].map(lambda value: f"{value:.2%}" if pd.notna(value) else "—")
    display_df["Sharpe"] = display_df["Sharpe"].map(lambda value: f"{value:.2f}" if pd.notna(value) else "—")
    display_df["Calmar"] = display_df["Calmar"].map(lambda value: f"{value:.2f}" if pd.notna(value) else "—")
    display_df["Max DD"] = display_df["Max DD"].map(lambda value: f"{value:.2%}" if pd.notna(value) else "—")
    display_df["Last TVL"] = display_df["Last TVL"].map(lambda value: f"{value:,.0f} USD" if pd.notna(value) else "—")
    display_df["Avg TVL"] = display_df["Avg TVL"].map(lambda value: f"{value:,.0f} USD" if pd.notna(value) else "—")
    return display_df
