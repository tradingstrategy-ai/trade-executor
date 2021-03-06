"""Tools to visualise live trading/backtest outcome for strategies trading only one pair."""
import datetime
import logging
from typing import Iterable, Optional, Union

import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.visualisation import Visualisation, Plot
from tradingstrategy.candle import GroupedCandleUniverse


logger = logging.getLogger(__name__)


def export_trade_for_dataframe(t: TradeExecution) -> dict:
    """Export data for a Pandas dataframe presentation"""

    if t.is_failed():
        label = f"Faild trade"
    else:
        if t.is_sell():
            label = f"Sell {t.pair.base.token_symbol} @ {t.executed_price}"
        else:
            label = f"Buy {t.pair.base.token_symbol} @ {t.executed_price}"

    # See Plotly Scatter usage https://stackoverflow.com/a/61349739/315168
    return {
        "timestamp": t.executed_at,
        "success": t.is_success(),
        "type": t.is_buy() and "buy" or "sell",
        "label": label,
        "price": t.executed_price,
    }


def export_trades_as_dataframe(trades: Iterable[TradeExecution]) -> pd.DataFrame:
    """Convert executed trades to a dataframe, so it is easier to work with them in Plotly."""
    data = [export_trade_for_dataframe(t) for t in trades]
    return pd.DataFrame(data)


def export_plot_as_dataframe(plot: Plot) -> pd.DataFrame:
    """Convert visualisation state to Plotly friendly df."""
    data = []
    for time, value in plot.points.items():
        time = pd.to_datetime(time, unit='s')
        data.append({
            "timestamp": time,
            "value": value,
        })

    # Convert timestamp to pd.Timestamp column
    df = pd.DataFrame(data)
    df = df.set_index(pd.DatetimeIndex(df["timestamp"]))
    return df


def visualise_technical_indicators(fig: go.Figure, visualisation: Visualisation):
    """Draw technical indicators over candle chart."""

    # https://plotly.com/python/graphing-multiple-chart-types/
    for plot_id, plot in visualisation.plots.items():
        df = export_plot_as_dataframe(plot)
        start_ts = df["timestamp"].min()
        end_ts = df["timestamp"].max()
        logger.info(f"Visualisation {plot_id} has data for range {start_ts} - {end_ts}")
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["value"],
            mode="lines",
            name=plot.name,
            line=dict(color=plot.colour),
        ))


def visualise_trades(
        fig: go.Figure,
        candles: pd.DataFrame,
        trades_df: pd.DataFrame,):
    """Plot individual trades over the candlestick chart."""

    buys_df = trades_df.loc[trades_df["type"] == "buy"]
    sells_df = trades_df.loc[trades_df["type"] == "sell"]

    # Buys
    fig.add_trace(
        go.Scatter(
            name="Buys",
            mode="markers",
            x=buys_df["timestamp"],
            y=buys_df["price"],
            text=buys_df["label"],
            marker={"symbol": 'triangle-right', "size": 12, "line": {"width": 2, "color": "black"}},
        ),
        secondary_y=False,
    )

    # Sells
    fig.add_trace(
        go.Scatter(
            name="Sells",
            mode="markers",
            x=sells_df["timestamp"],
            y=sells_df["price"],
            text=sells_df["label"],
            marker={"symbol": 'triangle-left', "size": 12, "line": {"width": 2, "color": "black"}}
        ),
        secondary_y=False,
    )

    return fig


def visualise_single_pair(
        state: State,
        candle_universe: GroupedCandleUniverse,
        start_at: Optional[Union[pd.Timestamp, datetime.datetime]]=None,
        end_at: Optional[Union[pd.Timestamp, datetime.datetime]]=None,
        height=800) -> go.Figure:
    """Visualise single-pair trade execution.

    :param state:
        The recorded state of the strategy execution.
        Either live or backtest.

    :param candle_universe:
        Price candles we used for the strategy

    :param height:
        Chart height in pixels

    :param start_at:
        When the backtest started

    :param end_at:
        When the backtest ended
    """

    logger.info("Visualising %s", state)

    if isinstance(start_at, datetime.datetime):
        start_at = pd.Timestamp(start_at)

    if isinstance(end_at, datetime.datetime):
        end_at = pd.Timestamp(end_at)

    if start_at is not None:
        assert isinstance(start_at, pd.Timestamp)

    if end_at is not None:
        assert isinstance(end_at, pd.Timestamp)

    assert candle_universe.get_pair_count() == 1, "visualise_single_pair() can be only used for a trading universe with a single pair"
    candles = candle_universe.get_single_pair_data()

    try:
        first_trade = next(iter(state.portfolio.get_all_trades()))
    except StopIteration:
        first_trade = None

    if first_trade:
        pair_name = f"{first_trade.pair.base.token_symbol} - {first_trade.pair.quote.token_symbol}"
    else:
        pair_name = None

    if not start_at:
        # No trades made, use the first candle timestamp
        start_at = candle_universe.get_timestamp_range()[0]

    if not end_at:
        end_at = candle_universe.get_timestamp_range()[1]

    logger.info(f"Visualising single pair strategy for range {start_at} = {end_at}")

    # Candles define our diagram X axis
    # Crop it to the trading range
    candles = candles.loc[candles["timestamp"].between(start_at, end_at)]

    candle_start_ts = candles["timestamp"].min()
    candle_end_ts = candles["timestamp"].max()
    logger.info(f"Candles are {candle_start_ts} = {candle_end_ts}")

    trades_df = export_trades_as_dataframe(state.portfolio.get_all_trades())

    # set up figure with values not high and not low
    # include candlestick with rangeselector

    candlesticks = go.Candlestick(
        x=candles.index,
        open=candles['open'],
        high=candles['high'],
        low=candles['low'],
        close=candles['close'],
        showlegend=False
    )

    fig = make_subplots(specs=[[{"secondary_y": False}]])

    if state.name:
        fig.update_layout(title=f"{state.name} trades", height=height)
    else:
        fig.update_layout(title=f"Trades", height=height)

    if pair_name:
        fig.update_yaxes(title=f"{pair_name} price", secondary_y=False, showgrid=True)
    else:
        fig.update_yaxes(title="Price $", secondary_y=False, showgrid=True)

    fig.update_xaxes(rangeslider={"visible": False})

    # Synthetic data may not have volume available
    if "volume" in candles.columns:
        volume_bars = go.Bar(
            x=candles.index,
            y=candles['volume'],
            showlegend=False,
            marker={
                "color": "rgba(128,128,128,0.5)",
            }
        )
        fig.add_trace(volume_bars, secondary_y=True)
        fig.update_yaxes(title="Volume $", secondary_y=True, showgrid=False)

    fig.add_trace(candlesticks, secondary_y=False)

    visualise_technical_indicators(fig, state.visualisation)

    # Add trade markers if any trades have been made
    if len(trades_df) > 0:
        visualise_trades(fig, candles, trades_df)

    # Move legend to the bottom so we have more space for
    # time axis in narrow notebook views
    # https://plotly.com/python/legend/f
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    return fig


