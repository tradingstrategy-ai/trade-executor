"""Tools to visualise live trading/backtest outcome for strategies trading only one pair."""
import datetime
import logging

from typing import Optional, Union, List, Collection

import plotly.graph_objects as go
import pandas as pd
from plotly.graph_objs.layout import Annotation

from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.trade_pricing import format_fees_dollars
from tradeexecutor.state.visualisation import PlotKind

from tradeexecutor.state.types import PairInternalId
from tradeexecutor.visual.technical_indicator import overlay_all_technical_indicators

from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.charting.candle_chart import visualise_ohlcv, make_candle_labels, VolumeBarMode

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
                    f"Fees paid: {format_fees_dollars(t.get_fees_paid())}",
                    f"Fees planned: {format_fees_dollars(t.lp_fees_estimated)}",
                    f"Fees: {realised_fees:.4f} %"
                ]
            else:
                label += [
                    f"Fees paid: {format_fees_dollars(t.get_fees_paid())}",
                    f"Fees planned: {format_fees_dollars(t.lp_fees_estimated)}",
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
        pair_id: PairInternalId,
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

        if t.pair.internal_id != pair_id:
            continue

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
        positions: Collection[TradingPosition]):
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
        state: Optional[State],
        candle_universe: GroupedCandleUniverse | pd.DataFrame,
        start_at: Optional[Union[pd.Timestamp, datetime.datetime]] = None,
        end_at: Optional[Union[pd.Timestamp, datetime.datetime]] = None,
        pair_id: Optional[PairInternalId] = None,
        height=800,
        axes=True,
        technical_indicators=True,
        title: Union[str, bool] = True,
        theme="plotly_white",
        volume_bar_mode=VolumeBarMode.overlay,
        vertical_spacing = 0.05,
) -> go.Figure:
    """Visualise single-pair trade execution.

    :param state:
        The recorded state of the strategy execution.

        You must give either `state` or `positions`.

    :param pair_id:
        The visualised pair in the case the strategy contains trades for multiple pairs.

        If the strategy contains trades only for one pair this is not needed.

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

    :param technical_indicators:
        Extract technical indicators from the state and overlay them on the price action.

        Only makes sense if the indicators were drawn against the price action of this pair.

    :param title:
        Draw the chart title.

        Set to string to give your own name.

        Set `True` to use the state name as a title.
        TODO: True is a legacy option and will be removed.

    :param theme:
        Plotly colour scheme to use
        
    :param volume_bar_mode:
        How to draw the volume bars
    
    :param vertical_spacing:
        Vertical spacing between the subplots. Default is 0.05.
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
        if not pair_id:
            assert candle_universe.get_pair_count() == 1, "visualise_single_pair() can be only used for a trading universe with a single pair, please pass pair_id"
            pair_id = next(iter(candle_universe.get_pair_ids()))
        candles = candle_universe.get_candles_by_pair(pair_id)
    else:
        # Raw dataframe
        candles = candle_universe

    # Get all positions for the trading pair we want to visualise
    positions = [p for p in state.portfolio.get_all_positions() if p.pair.internal_id == pair_id]

    if len(positions) > 0:
        first_trade = positions[0].get_first_trade()
    else:
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
        pair_id,
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

    num_detached_indicators = sum(
        plot.kind == PlotKind.technical_indicator_detached
        for plot in state.visualisation.plots.values()
    )
    
    # since python dicts are now ordered by insertion
    relative_sizing = [plot.relative_size for plot in state.visualisation.plots.values() if plot.kind == PlotKind.technical_indicator_detached]
    
    subplot_names = [plot.name for plot in state.visualisation.plots.values() if plot.kind == PlotKind.technical_indicator_detached]
    
    # visualise candles and volume and create empty grid space for technical indicators
    fig = visualise_ohlcv(
        candles,
        height=height,
        theme=theme,
        chart_name=title_text,
        y_axis_name=axes_text,
        volume_axis_name=volume_text,
        labels=labels,
        volume_bar_mode=volume_bar_mode,
        num_detached_indicators=num_detached_indicators,
        vertical_spacing=vertical_spacing,
        relative_sizing=relative_sizing,
        subplot_names=subplot_names,
    )

    # Draw EMAs etc.
    if technical_indicators:
        overlay_all_technical_indicators(
            fig,
            state.visualisation,
            start_at,
            end_at,
            volume_bar_mode,
        )

    # Add trade markers if any trades have been made
    if len(trades_df) > 0:
        visualise_trades(fig, candles, trades_df)

    return fig


def visualise_single_pair_positions_with_duration_and_slippage(
        state: State,
        candles: pd.DataFrame,
        pair_id: Optional[PairInternalId] = None,
        start_at: Optional[Union[pd.Timestamp, datetime.datetime]] = None,
        end_at: Optional[Union[pd.Timestamp, datetime.datetime]] = None,
        height=800,
        axes=True,
        title: Union[bool, str] = True,
        theme="plotly_white",
        technical_indicators=True,
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

    :param pair_id:
        The visualised pair in the case the strategy contains trades for multiple pairs.

        If the strategy contains trades only for one pair this is not needed.

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

    :param technical_indicators:
        Extract technical indicators from the state and overlay them on the price action.

        Only makes sense if the indicators were drawn against the price action of this pair.

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

    if not pair_id:
        pair_id = candles.iloc[0]["pair_id"]

    logger.info(f"Candles are {candle_start_ts} - {candle_end_ts}")

    positions = [p for p in state.portfolio.get_all_positions() if p.pair.internal_id == pair_id]

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

    num_detached_indicators = sum(
        plot.kind == PlotKind.technical_indicator_detached
        for plot in state.visualisation.plots.values()
    )

    fig = visualise_ohlcv(
        candles,
        height=height,
        theme=theme,
        chart_name=title_text,
        y_axis_name=axes_text,
        volume_axis_name=None,
        volume_bar_mode=VolumeBarMode.hidden,
        num_detached_indicators=num_detached_indicators
    )

    if technical_indicators:
        overlay_all_technical_indicators(
            fig,
            state.visualisation,
            start_at,
            end_at,
        )

    # Add trade markers if any trades have been made
    visualise_positions_with_duration_and_slippage(fig, candles, positions)

    return fig
