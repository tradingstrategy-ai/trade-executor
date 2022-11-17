"""Compare portfolio performance against other strategies."""
import datetime
from typing import Optional, List, Union

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
    height=1200,
    start_at: Optional[Union[pd.Timestamp, datetime.datetime]]=None,
    end_at: Optional[Union[pd.Timestamp, datetime.datetime]]=None,

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

    :param start_at:
        When the backtest started

    :param end_at:
        When the backtest ended
    """

    fig = go.Figure()

    assert portfolio_statistics

    if isinstance(start_at, datetime.datetime):
        start_at = pd.Timestamp(start_at)

    if isinstance(end_at, datetime.datetime):
        end_at = pd.Timestamp(end_at)

    if start_at is not None:
        assert isinstance(start_at, pd.Timestamp)

    if end_at is not None:
        assert isinstance(end_at, pd.Timestamp)

    if not start_at:
        start_at = pd.Timestamp(portfolio_statistics[0].calculated_at)

    if not end_at:
        end_at = pd.Timestamp(portfolio_statistics[-1].calculated_at)

    scatter = visualise_portfolio_statistics(name, portfolio_statistics)
    fig.add_trace(scatter)

    if all_cash:
        scatter = visualise_all_cash(start_at, end_at, all_cash)
        fig.add_trace(scatter)

    if buy_and_hold_price_series is not None:

        # Clamp backtest to the strategy range even if we have more candles
        buy_and_hold_price_series = buy_and_hold_price_series[start_at:end_at]

        scatter = visualise_buy_and_hold(buy_and_hold_asset_name, buy_and_hold_price_series, all_cash)
        fig.add_trace(scatter)

    if name:
        fig.update_layout(title=f"{name} portfolio value", height=height)
    else:
        fig.update_layout(title=f"Portfolio value", height=height)

    fig.update_yaxes(title="Value $", showgrid=False)

    fig.update_xaxes(rangeslider={"visible": False})

    # Move legend to the bottom so we have more space for
    # time axis in narrow notebook views
    # https://plotly.com/python/legend/
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    return fig
