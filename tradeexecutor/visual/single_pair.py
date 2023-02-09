"""Tools to visualise live trading/backtest outcome for strategies trading only one pair."""
import datetime
import logging
import textwrap
from typing import Optional, Union, List, Tuple

import plotly.graph_objects as go
import pandas as pd
from plotly.graph_objs.layout import Annotation

from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.visualisation import Visualisation, Plot
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.charting.candle_chart import visualise_ohlcv, make_candle_labels

logger = logging.getLogger(__name__)


def export_trade_for_dataframe(p: Portfolio, t: TradeExecution) -> dict:
    """Export data for a Pandas dataframe presentation.

    - Decimal roundings are based on rule of thumb and may need to be tuned
    """

    position = p.get_position_by_id(t.position_id)

    price_prefix = f"{t.pair.base.token_symbol} / USD"

    label = []

    if t.is_failed():
        label += [f"Failed trade"]
        type = "failed"
    else:
        if t.is_sell():
            if t.is_stop_loss():
                label += [f"Stop loss {t.pair.base.token_symbol}", "",
                          f"Trigger was at {position.stop_loss:.4f} {price_prefix}"]
                type = "stop-loss"
            else:
                label += [f"Sell {t.pair.base.token_symbol}"]
                type = "sell"
        else:
            if t.is_take_profit():
                type = "take-profit"
                label += [f"Take profit {t.pair.base.token_symbol}", "",
                          "Trigger was at {position.take_profit:.4f} {price_prefix}"]
            else:
                type = "buy"
                label += [f"Buy {t.pair.base.token_symbol}"]

        label += [
            "",
            f"Executed at: {t.executed_at}",
            f"Value: {t.get_value():.4f} USD",
            f"Quantity: {abs(t.get_position_quantity()):.6f} {t.pair.base.token_symbol}",
            "",
        ]

        label += [
            f"Mid-price: {t.planned_mid_price:.4f} {price_prefix}" if t.planned_mid_price else "",
            f"Executed at price: {t.executed_price:.4f} {price_prefix}",
            f"Estimated execution price: {t.planned_price:.4f} {price_prefix}",
            "",
        ]

        if t.lp_fees_estimated is not None:
            if t.executed_price and t.planned_mid_price:
                realised_fees = abs(1 - t.planned_mid_price / t.executed_price)
                label += [
                    f"Fees paid: {t.get_fees_paid():.4f} USD",
                    f"Fees planned: {t.lp_fees_estimated:.4f} USD",
                    f"Fees: {realised_fees:.4f} %"
                ]
            else:
                label += [
                    f"Fees paid: {t.get_fees_paid():.4f} USD",
                    f"Fees planned: {t.lp_fees_estimated:.4f} USD",
                ]

    # See Plotly Scatter usage https://stackoverflow.com/a/61349739/315168
    return {
        "timestamp": t.executed_at,
        "success": t.is_success(),
        "type": type,
        "label": "<br>".join(label),
        "price": t.planned_mid_price if t.planned_mid_price else t.planned_price,
    }


def export_trades_as_dataframe(
        portfolio: Portfolio,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Convert executed trades to a dataframe, so it is easier to work with them in Plotly.

    :param start_at:
        Crop range

    :param end_at:
        Crop range
    """

    if start:
        assert isinstance(start, pd.Timestamp)

    if end:
        assert isinstance(end, pd.Timestamp)
        assert start

    data = []

    for t in portfolio.get_all_trades():
        # Crop
        if start or end:

            if not t.started_at:
                # Hotfix to some invalid data?
                logger.warning("Trade lacks start date: %s", t)
                continue

            if t.started_at < start or t.started_at > end:
                continue

        data.append(export_trade_for_dataframe(portfolio, t))
    return pd.DataFrame(data)


def export_plot_as_dataframe(
        plot: Plot,
        start_at: Optional[pd.Timestamp] = None,
        end_at: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Convert visualisation state to Plotly friendly df.

    :param start_at:
        Crop range

    :param end_at:
        Crop range
    """
    data = []
    for time, value in plot.points.items():
        time = pd.to_datetime(time, unit='s')

        if start_at or end_at:
            if time < start_at or time > end_at:
                continue

        data.append({
            "timestamp": time,
            "value": value,
        })

    # set_index fails if the frame is empty
    if len(data) == 0:
        return pd.DataFrame()

    # Convert timestamp to pd.Timestamp column
    df = pd.DataFrame(data)
    df = df.set_index(pd.DatetimeIndex(df["timestamp"]))
    return df


def visualise_technical_indicators(
        fig: go.Figure,
        visualisation: Visualisation,
        start_at: Optional[pd.Timestamp] = None,
        end_at: Optional[pd.Timestamp] = None
):
    """Draw technical indicators over candle chart.

    :param start_at:
        Crop range

    :param end_at:
        Crop range
    """

    # https://plotly.com/python/graphing-multiple-chart-types/
    for plot_id, plot in visualisation.plots.items():
        df = export_plot_as_dataframe(plot, start_at, end_at)
        if len(df) > 0:
            start_ts = df["timestamp"].min()
            end_ts = df["timestamp"].max()
            logger.info(f"Visualisation {plot_id} has data for range {start_ts} - {end_ts}")
            fig.add_trace(go.Scatter(
                x=df["timestamp"],
                y=df["value"],
                mode="lines",
                name=plot.name,
                line=dict(color=plot.colour),
                line_shape=plot.plot_shape.value
            ))


def visualise_trades(
        fig: go.Figure,
        candles: pd.DataFrame,
        trades_df: pd.DataFrame, ):
    """Plot individual trades over the candlestick chart."""

    # If we have used stop loss, do different categories
    advanced_trade_types = ("stop-loss", "take-profit")
    advanced_trades = len(trades_df.loc[trades_df["type"].isin(advanced_trade_types)]) > 0

    if advanced_trades:
        buys_df = trades_df.loc[trades_df["type"] == "buy"]
        sells_df = trades_df.loc[trades_df["type"] == "sell"]
        stop_loss_df = trades_df.loc[trades_df["type"] == "stop-loss"]
        take_profit_df = trades_df.loc[trades_df["type"] == "take-profit"]
    else:
        buys_df = trades_df.loc[trades_df["type"] == "buy"]
        sells_df = trades_df.loc[trades_df["type"] == "sell"]
        stop_loss_df = None
        take_profit_df = None

    # Buys
    fig.add_trace(
        go.Scatter(
            name="Buy",
            mode="markers",
            x=buys_df["timestamp"],
            y=buys_df["price"],
            text=buys_df["label"],
            marker={"color": "#aaaaff", "symbol": 'triangle-right', "size": 12,
                    "line": {"width": 1, "color": "#3333aa"}},
            hoverinfo="text",
        ),
        secondary_y=False,
    )

    # Sells
    fig.add_trace(
        go.Scatter(
            name="Sell",
            mode="markers",
            x=sells_df["timestamp"],
            y=sells_df["price"],
            text=sells_df["label"],
            marker={"color": "#aaaaff", "symbol": 'triangle-left', "size": 12,
                    "line": {"width": 1, "color": "#3333aa"}},
            hoverinfo="text",
        ),
        secondary_y=False,
    )

    if stop_loss_df is not None:
        fig.add_trace(
            go.Scatter(
                name="Stop loss",
                mode="markers",
                x=stop_loss_df["timestamp"],
                y=stop_loss_df["price"],
                text=stop_loss_df["label"],
                marker={"symbol": 'triangle-left', "size": 12, "line": {"width": 1, "color": "black"}},
                hoverinfo="text",
            ),
            secondary_y=False,
        )

    if take_profit_df is not None:
        fig.add_trace(
            go.Scatter(
                name="Take profit",
                mode="markers",
                x=take_profit_df["timestamp"],
                y=take_profit_df["price"],
                text=take_profit_df["label"],
                marker={"symbol": 'triangle-left', "size": 12, "line": {"width": 1, "color": "black"}},
                hoverinfo="text",
            ),
            secondary_y=False,
        )

    return fig


def get_position_hover_text(p: TradingPosition) -> str:
    """Get position hover text for Plotly."""

    # First draw a position as a re
    first_trade = p.get_first_trade()
    last_trade = p.get_last_trade()

    duration = last_trade.executed_at - first_trade.executed_at

    started_at = first_trade.started_at.strftime("%Y-%m-%d, %H:%M:%S UTC")
    ended_at = last_trade.executed_at.strftime("%Y-%m-%d, %H:%M:%S UTC")

    entry_diff = (first_trade.executed_price - first_trade.planned_price) / first_trade.planned_price
    entry_dur = (first_trade.executed_at - first_trade.started_at)
    exit_diff = (last_trade.executed_price - last_trade.planned_price) / last_trade.planned_price
    exit_dur = (last_trade.executed_at - last_trade.started_at)

    text = []

    text += [
        f"Position #{p.position_id}",
        ""
    ]

    # Add remarks
    if p.is_open():
        text += [
            "Position currently open",
            ""
        ]
    elif p.is_stop_loss():
        text += [
            f"Stop loss triggered at: {p.stop_loss:.2f} USD",
            ""
        ]
    else:
        pass

    if p.is_closed():
        text += [
            f"Profit: {p.get_realised_profit_usd():.2f} USD",
            f"Profit: {p.get_total_profit_percent() * 100:.4f} %",
            ""
        ]

    text += [
        f"Entry price: {first_trade.planned_mid_price:.2f} USD (mid price)",
        f"Entry price: {first_trade.planned_price:.2f} USD (expected)",
        f"Entry price: {first_trade.executed_price:.2f} USD (executed)",
        f"Entry slippage: {entry_diff * 100:.4f} %",
        f"Entry duration: {entry_dur}",
        ""
    ]

    if p.is_closed():
        text += [
            f"Exit price: {last_trade.planned_price:.2f} USD (expected)",
            f"Exit price: {last_trade.executed_price:.2f} USD (executed)",
            f"Exit slippage: {exit_diff * 100:.4f} %",
            f"Exit duration: {exit_dur}",
        ]

    if p.has_buys() or p.has_sells():
        if p.has_buys():
            text += [
                f"Avg buy price: {p.get_average_buy():.2f} USD",
            ]
        if p.has_sells():
            text += [
                f"Avg sell price: {p.get_average_sell():.2f} USD",
            ]
        text += [""]

    if p.is_closed():
        text += [
            f"Duration: {duration}",
            f"Started: {started_at} (first trade started)",
            f"Ended: {ended_at} (last trade executed at)",
            ""
        ]
    else:
        text += [
            f"Started: {started_at} (first trade started)",
            ""
        ]
    return "<br>".join(text)


def visualise_positions_with_duration_and_slippage(
        fig: go.Figure,
        candles: pd.DataFrame,
        positions: List[TradingPosition]):
    """Visualise trades as coloured area over time.

    Add arrow indicators to point start and end duration,
    and slippage.
    """

    # TODO: Figure out how to add a Y coordinate
    # for Scatter in Plotly paper space
    max_price = max(candles["high"])

    # https://stackoverflow.com/a/58128982/315168
    annotations: List[Annotation] = []

    buys = {
        "x": [],
        "y": [],
        "text": [],
    }

    sells = {
        "x": [],
        "y": [],
        "text": [],
    }

    for position in positions:

        # First draw a position as a re
        first_trade = position.get_first_trade()
        last_trade = position.get_last_trade()

        left_x = pd.Timestamp(first_trade.started_at)
        right_x = pd.Timestamp(last_trade.executed_at)

        if position.is_profitable():
            colour = "LightGreen"
        else:
            colour = "LightPink"

        # https://plotly.com/python/shapes/
        fig.add_vrect(
            x0=left_x,
            x1=right_x,
            xref="x",
            fillcolor=colour,
            opacity=0.5,
            layer="below",
            line_width=0,
        )

        position_text = get_position_hover_text(position)

        # Add tooltips as the dot market at the top left corner
        # of the position
        fig.add_trace(
            go.Scatter(
                x=[left_x + (right_x - left_x) / 2],
                y=[max_price],
                hovertext=position_text,
                hoverinfo="text",
                showlegend=False,
                mode='markers',
                marker={"color": colour, "size": 12}
            ))

        # Visualise trades as lines
        # TODO: Plotly arrow drawing broken for small arrows
        t: TradeExecution
        for t in position.trades.values():

            colour = "black"

            fig.add_shape(
                type="line",
                x0=t.started_at,
                x1=t.executed_at,
                xref="x",
                y0=t.planned_price,
                y1=t.executed_price,
                yref="y",
                line={
                    "color": colour,
                    "width": 1,
                }
            )

            if t.is_buy():
                trade_markers = buys
            else:
                trade_markers = sells

            trade_markers["x"].append(t.executed_at)
            trade_markers["y"].append(t.executed_price)
            trade_markers["text"].append(str(t))

            # Plotly does not render arrows if they are
            # too small.
            #
            # ann = {
            #     "showarrow": True,
            #     "ax": t.started_at,
            #     "axref": "x",
            #     "x": t.executed_at,
            #     "xref": "x",
            #     "ay":t.planned_price,
            #     "ayref": "y",
            #     "y" :t.executed_price,
            #     "yref": "y",
            #     "arrowwidth": 2,
            #     "arrowhead": 5,
            #     "arrowcolor": colour,
            # }
            #
            # annotations.append(ann)

            # dict(
            #     x= x_end,
            #     y= y_end,
            #     xref="x", yref="y",
            #     text="",
            #     showarrow=True,
            #     axref = "x", ayref='y',
            #     ax= x_start,
            #     ay= y_start,
            #     arrowhead = 3,
            #     arrowwidth=1.5,
            #     arrowcolor='rgb(255,51,0)',)
            # )

    # Add "arrowheads" to trade lines

    fig.add_trace(
        go.Scatter(
            x=buys["x"],
            y=buys["y"],
            text=buys["text"],
            showlegend=False,
            mode='markers',
            marker={"symbol": "arrow-right", "color": "black", "size": 12, "line": {"width": 0}},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=sells["x"],
            y=sells["y"],
            text=sells["text"],
            showlegend=False,
            mode='markers',
            marker={"symbol": "arrow-left", "color": "black", "size": 12, "line": {"width": 0}},
        )
    )

    # TODO: Currently does not work
    # https://stackoverflow.com/questions/58095322/draw-multiple-arrows-using-plotly-python
    if annotations:
        print(annotations)
        fig.update_layout(annotations=annotations)

    return fig


def visualise_single_pair(
        state: State,
        candle_universe: GroupedCandleUniverse | pd.DataFrame,
        start_at: Optional[Union[pd.Timestamp, datetime.datetime]] = None,
        end_at: Optional[Union[pd.Timestamp, datetime.datetime]] = None,
        height=800,
        axes=True,
        title: Union[str, bool] = True,
        theme="plotly_white",
) -> go.Figure:
    """Visualise single-pair trade execution.

    :param state:
        The recorded state of the strategy execution.
        Either live or backtest.

    :param candle_universe:
        Price candles we used for the strategy

    :param height:
        Chart height in pixels

    :param start_at:
        When the backtest started or when we crop the content

    :param end_at:
        When the backtest ended or when we crop the content

    :param axes:
        Draw axes labels

    :param title:
        Draw the chart title.

        Set to string to give your own name.

        Set `True` to use the state name as a title.
        TODO: True is a legacy option and will be removed.

    :param theme:
        Plotly colour scheme to use
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

    if isinstance(candle_universe, GroupedCandleUniverse):
        assert candle_universe.get_pair_count() == 1, "visualise_single_pair() can be only used for a trading universe with a single pair"
        candles = candle_universe.get_single_pair_data()
    else:
        # Raw dataframe
        candles = candle_universe

    try:
        first_trade = next(iter(state.portfolio.get_all_trades()))
    except StopIteration:
        first_trade = None

    if first_trade:
        pair_name = f"{first_trade.pair.base.token_symbol} - {first_trade.pair.quote.token_symbol}"
        pair = first_trade.pair
        base_token = pair.base.token_symbol
        quote_token = pair.quote.token_symbol
    else:
        pair_name = None
        base_token = None
        quote_token = None

    if not start_at:
        # No trades made, use the first candle timestamp
        start_at = candle_universe.get_timestamp_range()[0]

    if not end_at:
        end_at = candle_universe.get_timestamp_range()[1]

    logger.info(f"Visualising single pair strategy for range {start_at} - {end_at}")

    # Candles define our diagram X axis
    # Crop it to the trading range
    candles = candles.loc[candles["timestamp"].between(start_at, end_at)]

    candle_start_ts = candles["timestamp"].min()
    candle_end_ts = candles["timestamp"].max()
    logger.info(f"Candles are {candle_start_ts} = {candle_end_ts}")

    trades_df = export_trades_as_dataframe(
        state.portfolio,
        start_at,
        end_at,
    )

    if title is True:
        title_text = state.name
    elif type(title) == str:
        title_text = title
    else:
        title_text = None

    if axes:
        axes_text = pair_name
        volume_text = "Volume USD"
    else:
        axes_text = None
        volume_text = None

    labels = make_candle_labels(
        candles,
        base_token_name=base_token,
        quote_token_name=quote_token,
    )

    fig = visualise_ohlcv(
        candles,
        height=height,
        theme=theme,
        chart_name=title_text,
        y_axis_name=axes_text,
        volume_axis_name=volume_text,
        labels=labels,
    )

    # Draw EMAs etc.
    visualise_technical_indicators(
        fig,
        state.visualisation,
        start_at,
        end_at,
    )

    # Add trade markers if any trades have been made
    if len(trades_df) > 0:
        visualise_trades(fig, candles, trades_df)

    return fig


def visualise_single_pair_positions_with_duration_and_slippage(
        state: State,
        candles: pd.DataFrame,
        start_at: Optional[Union[pd.Timestamp, datetime.datetime]] = None,
        end_at: Optional[Union[pd.Timestamp, datetime.datetime]] = None,
        height=800,
        axes=True,
        title: Union[bool, str] = True,
        theme="plotly_white",
) -> go.Figure:
    """Visualise performance of a live trading strategy.

    Unlike :py:func:`visualise_single_pair`
    attempt to visualise

    - position duration, as a colored area

    - more position tooltip text

    - trade duration (started at - executed)

    - slippage

    :param state:
        The recorded state of the strategy execution.
        Either live or backtest.

    :param candle_universe:
        Price candles we used for the strategy

    :param height:
        Chart height in pixels

    :param start_at:
        When the backtest started or when we crop the content

    :param end_at:
        When the backtest ended or when we crop the content

    :param axes:
        Draw axes labels

    :param title:
        Draw the chart title.

        Set to string to give your own name.

        Set `True` to use the state name as a title.
        TODO: True is a legacy option and will be removed.


    :param theme:
        Plotly colour scheme to use
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

    try:
        first_trade = next(iter(state.portfolio.get_all_trades()))
    except StopIteration:
        first_trade = None

    if first_trade:
        pair_name = f"{first_trade.pair.base.token_symbol} - {first_trade.pair.quote.token_symbol}"
    else:
        pair_name = None

    candle_start_ts = candles.iloc[0]["timestamp"]
    if not start_at:
        # No trades made, use the first candle timestamp
        start_at = candle_start_ts

    candle_end_ts = candles.iloc[-1]["timestamp"]

    if not end_at:
        end_at = candle_end_ts

    logger.info(f"Visualising single pair strategy for range {start_at} - {end_at}")

    # Candles define our diagram X axis
    # Crop it to the trading range
    candles = candles.loc[candles["timestamp"].between(start_at, end_at)]

    logger.info(f"Candles are {candle_start_ts} - {candle_end_ts}")

    positions = [p for p in state.portfolio.get_all_positions()]

    if title is True:
        title_text = state.name
    elif type(title) == str:
        title_text = title
    else:
        title_text = None

    if axes:
        axes_text = pair_name
        volume_text = "Volume USD"
    else:
        axes_text = None
        volume_text = None

    fig = visualise_ohlcv(
        candles,
        height=height,
        theme=theme,
        chart_name=title_text,
        y_axis_name=axes_text,
        volume_axis_name=volume_text,
    )

    visualise_technical_indicators(
        fig,
        state.visualisation,
        start_at,
        end_at,
    )

    # Add trade markers if any trades have been made
    visualise_positions_with_duration_and_slippage(fig, candles, positions)

    return fig
