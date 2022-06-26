"""Tools to visualise live trading/backtest outcome."""

import logging
from typing import Iterable

import plotly.graph_objects as go
import pandas as pd

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.visualisation import Visualisation, Plot
from tradingstrategy.candle import GroupedCandleUniverse


logger = logging.getLogger(__name__)


def export_trade_for_dataframe(t: TradeExecution, marker_size=12) -> dict:
    """Export data for a Pandas dataframe presentation"""

    if t.is_failed():
        marker = dict(symbol='triangle-down-open')
        label = f"Faild trade"
    else:
        if t.is_sell():
            marker = dict(symbol='triangle-down-open')
            label = f"Sell {t.pair.base.token_symbol} @ {t.executed_price}"
        else:
            marker = dict(symbol='triangle-up-open')
            label = f"Buy {t.pair.base.token_symbol} @ {t.executed_price}"

    # See Plotly Scatter usage https://stackoverflow.com/a/61349739/315168
    return {
        "timestamp": t.executed_at,
        "success": t.is_success(),
        "type": t.is_buy() and "buy" or "sell",
        "marker": marker,
        "label": label,
    }


def export_trades_as_dataframe(trades: Iterable[TradeExecution]) -> pd.DataFrame:
    """Convert executed trades to a dataframe, so it is easier to work with them in Plotly."""
    data = [export_trade_for_dataframe(t) for t in trades]
    return pd.DataFrame(data)


def export_plot_as_dataframe(plot: Plot) -> pd.DataFrame:
    """Convert visualisation state to Plotly friendly df."""
    data = []
    for time, value in plot.points.items():
        data.append({
            "timestamp": time,
            "value": value,
        })
    return pd.DataFrame(data)


def add_technical_indicators(fig: go.Figure, visualisation: Visualisation):
    """Draw technical indicators over candle chart."""

    # https://plotly.com/python/graphing-multiple-chart-types/
    # https://plotly.com/python/graphing-multiple-chart-types/
    for plot_id, plot in visualisation.plots.items():
        df = export_plot_as_dataframe(plot)
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["value"],
            mode="lines",
            name=plot.name,
            line=dict(color=plot.color),
        ))


def visualise_trades(
        fig: go.Figure,
        candles: pd.DataFrame,
        trades_df: pd.DataFrame,):

    buys_df = trades_df.loc[trades_df["type"] == "buy"]
    sells_df = trades_df.loc[trades_df["type"] == "sell"]

    # Buys
    fig.add_trace(
        go.Scatter(
            name="Buys",
            mode="markers+text",
            x=buys_df["timestamp"],
            y=candles["high"],
            text=buys_df["label"],
            marker={"symbol": "triangle-up-open"}
        )
    )

    # Sells
    fig.add_trace(
        go.Scatter(
            name="Sells",
            mode="markers+text",
            x=sells_df["timestamp"],
            y=candles["high"],
            text=sells_df["label"],
            marker={"symbol": "triangle-up-open"}
        )
    )

    return fig



def visualise_single_pair(
        state: State,
        candle_universe: GroupedCandleUniverse,
        start_ts=None,
        end_ts=None) -> go.Figure:
    """Visualise single-pair trade execution."""

    assert candle_universe.get_pair_count() == 1, "visualise_single_pair() can be only used for a trading universe with a single pair"
    candles = candle_universe.get_single_pair_data()

    first_trade, last_trade = state.portfolio.get_first_and_last_executed_trade()

    if not start_ts:
        if first_trade:
            start_ts = first_trade.executed_at
        else:
            # No trades made, use the first candle timestamp
            start_ts = candle_universe.get_timestamp_range()[0]

    if not end_ts:
        if last_trade:
            end_ts = last_trade.executed_at
        else:
            end_ts = candle_universe.get_timestamp_range()[1]

    logger.info(f"Visualising single pair strategy for range {start_ts} = {end_ts}")

    # Candles define our diagram X axis
    # Crop it to the trading range
    candles = candles.loc[candles["timestamp"].between(start_ts, end_ts)]

    trades_df = export_trades_as_dataframe(state.portfolio.get_all_trades())

    # set up figure with values not high and not low
    # include candlestick with rangeselector
    fig = go.Figure()

    # https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html#plotly.graph_objects.Figure.add_candlestick
    fig.add_candlestick(
        x=candles['timestamp'],
        open=candles['open'],
        high=candles['high'],
        low=candles['low'],
        close=candles['close'])

    add_technical_indicators(fig, state.visualisation)

    # Add trade markers if any trades have been made
    if len(trades_df) > 0:
        visualise_trades(fig, candles, trades_df)

    return fig


