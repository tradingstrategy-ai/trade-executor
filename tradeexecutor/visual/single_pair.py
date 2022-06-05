"""Tools to visualise live trading/backtest outcome."""

from typing import Iterable

import plotly.graph_objects as go
import pandas as pd

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.visualisation import Visualisation
from tradingstrategy.universe import Universe


def export_trade_for_dataframe(t: TradeExecution, marker_size=12) -> dict:
    """Export data for a Pandas dataframe presentation"""

    if t.is_failed():
        marker = dict(symbol='triangle-down-open', size=marker_size)
        label = f"Faild trade"
    else:
        if t.is_sell():
            marker = dict(symbol='triangle-down-open', size=marker_size)
            label = f"Sell {t.pair.base.token_symbol} @ {t.executed_price}"
        else:
            marker = dict(symbol='triangle-up-open', size=marker_size)
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
    return pd.DataFrame.from_dict(data)


def visualise_single_pair(state: State, universe: Universe, output: Visualisation) -> go.Figure:
    """Visualise single-pair trade execution."""

    assert universe.pairs.get_count() == 1, "visualise_single_pair() can be only used for a trading universe with a single pair"
    pair = universe.pairs.get_single()
    candles = universe.candles.get_candles_by_pair(pair.pair_id)

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

    # Add trade markers
    fig.add_trace(
        go.Scatter(
            name="Trades",
            mode="markers+text",
            x=trades_df["timestamp"],
            y=candles["high"],
            text=trades_df["label"],
            marker=trades_df["marker"],
        )
    )

    return fig
