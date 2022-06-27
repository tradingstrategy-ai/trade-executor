"""Compare portfolio performance against other strategies."""
from typing import Optional, List

import plotly.graph_objects as go
import pandas as pd


from tradeexecutor.state.statistics import PortfolioStatistics


def visualise_portfolio_statistics(
        name: str,
        portfolio_statistics: List[PortfolioStatistics],
        colour="#008800") -> go.Scatter:
    """Draw portfolio performance."""

    plot = []
    for s in portfolio_statistics:
        plot.append({
            "timestamp": pd.Timestamp(s.calculated_at),
            "value": s.total_equity,
        })

    df = pd.DataFrame(plot)
    df.set_index("timestamp", inplace=True)

    return go.Scatter(
        x=df.index,
        y=df["value"],
        mode="lines",
        name=name,
        line=dict(color=colour),
    )

def visualise_all_cash(
        start_at: pd.Timestamp,
        end_at: pd.Timestamp,
        all_cash: float,
        colour="#000088") -> go.Scatter:
    """Draw portfolio performance."""

    plot = []
    plot.append({
        "timestamp": start_at,
        "value": all_cash,
    })

    plot.append({
        "timestamp": end_at,
        "value": all_cash,
    })

    df = pd.DataFrame(plot)
    df.set_index("timestamp", inplace=True)

    return go.Scatter(
        x=df.index,
        y=df["value"],
        mode="lines",
        name="All cash",
        line=dict(color=colour),
    )


def visualise_buy_and_hold(
        buy_and_hold_asset_name: str,
        price_series: pd.Series,
        all_cash: float,
        colour="#880000") -> go.Scatter:
    """Draw portfolio performance."""

    # Whatever we bought at the start
    initial_inventory = all_cash / float(price_series.iloc[0])

    series = price_series * initial_inventory

    return go.Scatter(
        x=series.index,
        y=series,
        mode="lines",
        name=f"Buy and hold {buy_and_hold_asset_name}",
        line=dict(color=colour),
    )


def visualise_benchmark(
    name: Optional[str]=None,
    portfolio_statistics: Optional[List[PortfolioStatistics]]=None,
    all_cash: Optional[float]=None,
    buy_and_hold_asset_name: Optional[str]=None,
    buy_and_hold_price_series: Optional[pd.Series]=None,
    height=800,
) -> go.Figure:
    """Visualise strategy performance against benchmarks.

    - Live or backtested strategies

    - Right axis is portfolio value

    :param portfolio_statistics:
        Portfolio performance record.

    :param all_cash:
        Set a linear line of just holding X amount

    :param buy_and_hold:
        Visualise holding all_cash amount in the asset,
        bought at the start.
        This is basically price * all_cash.

    :param height:
        Chart height in pixels
    """

    fig = go.Figure()

    assert portfolio_statistics

    start_at = pd.Timestamp(portfolio_statistics[0].calculated_at)
    end_at = pd.Timestamp(portfolio_statistics[-1].calculated_at)

    scatter = visualise_portfolio_statistics(name, portfolio_statistics)
    fig.add_trace(scatter)

    if all_cash:
        scatter = visualise_all_cash(start_at, end_at, all_cash)
        fig.add_trace(scatter)

    if buy_and_hold_price_series is not None:
        scatter = visualise_buy_and_hold(buy_and_hold_asset_name, buy_and_hold_price_series, all_cash)
        fig.add_trace(scatter)

    if name:
        fig.update_layout(title=f"{name} portfolio value", height=height)
    else:
        fig.update_layout(title=f"Portfolio value", height=height)

    fig.update_xaxes(rangeslider={"visible": False})

    return fig
