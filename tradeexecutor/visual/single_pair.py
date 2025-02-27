"""Tools to visualise live trading/backtest outcome for strategies trading only one pair."""
import datetime
import logging

from typing import Optional, Union, List, Collection

import plotly.graph_objects as go
import pandas as pd
from plotly.graph_objs.layout import Annotation

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution

from tradeexecutor.state.types import PairInternalId
from tradeexecutor.strategy.execution_context import ExecutionContext, notebook_execution_context
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.visual.technical_indicator import overlay_all_technical_indicators

from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.charting.candle_chart import visualise_ohlcv, make_candle_labels, VolumeBarMode

from tradeexecutor.visual.utils import get_all_positions, get_pair_name_from_first_trade, get_all_text, get_num_detached_and_names, get_pair_base_quote_names, get_start_and_end, export_trades_as_dataframe, visualise_trades


logger = logging.getLogger(__name__)


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
    execution_context: ExecutionContext,
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
    subplot_font_size = 11,
    relative_sizing: list[float] = None,
    volume_axis_name: str = "Volume USD",
    candle_decimals: int = 4,
    detached_indicators: bool = True,
    hover_text: bool = True,
    include_credit_supply_positions: bool = False,
    legend: bool = True,
) -> go.Figure:
    """Visualise single-pair trade execution.

    Example:

    .. code-block:: python

        from tradeexecutor.visual.single_pair import visualise_single_pair
        from tradingstrategy.charting.candle_chart import VolumeBarMode

        pair = token_map["BANANA"]

        all_trades = [t for t in state.portfolio.get_all_trades() if t.pair == pair]
        print(f"We have total {len(all_trades)} trades on {pair}")

        figure = visualise_single_pair(
            state,
            candle_universe=strategy_universe.data_universe.candles,
            pair_id=pair.internal_id,
            volume_bar_mode=VolumeBarMode.hidden,
            execution_context=notebook_execution_context,
            title=f"Trades on {pair}",
        )

        figure.show()

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
    
    :param subplot_font_size:
        Font size of the subplot titles. Default is 11.
    
    :param relative_sizing:
        Optional relative sizes of each plot. Starts with first main candle plot, then the volume plot if it is detached, then the other detached technical indicators. 
        
        e.g. [1, 0.2, 0.3, 0.3] would mean the second plot is 20% the size of the first, and the third and fourth plots are 30% the size of the first.
        
        Remember to account for whether the volume subplot is detached or not. If it is detached, it should take up the second element in the list. 
    
    :param volume_axis_name:
        Name of the volume axis. Default is "Volume USD".
    
    :param candle_decimals:
        Number of decimal places to round the candlesticks to. Default is 4.

    :param detached_indicators:
        If set, draw detached indicators. Has no effect if `technical_indicators` is False.

    :param hover_text:
        If True, show all standard hover text. If False, show no hover text at all.
    """

    assert isinstance(execution_context, ExecutionContext)
    
    logger.info("Visualising %s", state)

    if not (start_at and end_at):
        start_at, end_at = state.get_strategy_start_and_end()

    start_at, end_at = get_start_and_end(start_at, end_at)

    if isinstance(candle_universe, GroupedCandleUniverse):
        if not pair_id:
            assert candle_universe.get_pair_count() == 1, "visualise_single_pair() can be only used for a trading universe with a single pair, please pass pair_id or use visualise_multiple_pairs()"
            pair_id = next(iter(candle_universe.get_pair_ids()))
        elif candle_universe.get_pair_count() > 1:
            assert all(p.pair and p.pair.internal_id for p in state.visualisation.plots.values()), "Please make sure you provide the `pair` argument to `plot_indicator` inside `decide_trades`, since you are using `visualise_single_pair` in a multipair universe."
        candles = candle_universe.get_candles_by_pair(pair_id)
    else:
        # Raw dataframe
        candles = candle_universe

    pair_name, base_token, quote_token = get_pair_base_quote_names(state, pair_id)

    if not start_at:
        # No trades made, use the first candle timestamp
        start_at = candle_universe.get_timestamp_range()[0]

    if not end_at:
        end_at = candle_universe.get_timestamp_range()[1]

    logger.info(f"Visualising single pair for pair ({pair_name}) strategy for range {start_at} - {end_at}")

    # Candles define our diagram X axis
    # Crop it to the trading range
    candles = candles.loc[candles["timestamp"].between(start_at, end_at)]

    candle_start_ts = candles["timestamp"].min()
    candle_end_ts = candles["timestamp"].max()
    logger.info(f"Candles are {candle_start_ts} = {candle_end_ts}, having {len(candles)} candles")

    trades_df = export_trades_as_dataframe(
        state.portfolio,
        pair_id,
        start_at,
        end_at,
        include_credit_supply_positions=include_credit_supply_positions,
    )

    labels = make_candle_labels(
        candles,
        base_token_name=base_token,
        quote_token_name=quote_token,
        candle_decimals=candle_decimals
    )

    fig = _get_grid_with_candles_volume_indicators(
        state=state,
        execution_context=execution_context,
        start_at=start_at, 
        end_at=end_at, 
        height=height, 
        axes=axes, 
        technical_indicators=technical_indicators, 
        title=title, 
        theme=theme, 
        volume_bar_mode=volume_bar_mode, 
        vertical_spacing=vertical_spacing, 
        subplot_font_size=subplot_font_size, 
        relative_sizing=relative_sizing, 
        candles=candles,
        pair_name=pair_name,
        labels=labels,
        volume_axis_name=volume_axis_name,
        pair_id=pair_id,
        detached_indicators=detached_indicators,
        hover_text=hover_text,
    )


    # Add trade markers if any trades have been made
    if len(trades_df) > 0:
        visualise_trades(fig, candles, trades_df, include_credit_supply_positions=include_credit_supply_positions)

    if not legend:
        fig.update_layout(showlegend=False)

    return fig


def visualise_single_pair_positions_with_duration_and_slippage(
    state: State,
    execution_context: ExecutionContext,
    candles: pd.DataFrame,
    pair_id: Optional[PairInternalId] = None,
    start_at: Optional[Union[pd.Timestamp, datetime.datetime]] = None,
    end_at: Optional[Union[pd.Timestamp, datetime.datetime]] = None,
    height=800,
    axes=True,
    title: Union[bool, str] = True,
    theme="plotly_white",
    technical_indicators=True,
    vertical_spacing = 0.05,
    relative_sizing: list[float] = None,
    subplot_font_size: int = 11,
) -> go.Figure:
    """Visualise performance of a live trading strategy.

    Unlike :py:func:`visualise_single_pair`
    attempt to visualise

    - position duration, as a colored area

    - more position tooltip text

    - trade duration (started at - executed)

    - slippage

    Example:

    .. code-block:: python

        from tradeexecutor.visual.single_pair import visualise_single_pair_positions_with_duration_and_slippage
        from tradingstrategy.charting.candle_chart import VolumeBarMode

        pair = token_map["BANANA"]

        all_positions = [t for t in state.portfolio.get_all_positions() if t.pair == pair]
        print(f"We have total {len(all_positions)} positions on {pair}")

        figure = visualise_single_pair_positions_with_duration_and_slippage(
            state=state,
            candles=strategy_universe.data_universe.candles.get_candles_by_pair(pair.internal_id),
            pair_id=pair.internal_id,
            execution_context=notebook_execution_context,
            title=f"Positions on {pair}",
        )

        figure.show()

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
    
    :param vertical_spacing:
        Vertical spacing between subplots
    
    :param relative_sizing:
        Optional relative sizes of each plot. Starts with first main candle plot. In this function, there is no volume plot (neither overlayed, hidden, or detached), so the first plot is the candle plot, and the rest are the technical indicator plots.
        
        e.g. [1, 0.2, 0.3, 0.3] would mean the second plot is 20% the size of the first, and the third and fourth plots are 30% the size of the first.

    :param subplot_font_size:
        Font size of the subplot titles

    :return:
        Plotly figure
    """

    assert isinstance(execution_context, ExecutionContext)

    logger.info("Visualising %s", state)

    if not (start_at and end_at):
        start_at, end_at = state.get_strategy_start_and_end()

    start_at, end_at = get_start_and_end(start_at, end_at)

    try:
        first_trade = next(iter(state.portfolio.get_all_trades()))
    except StopIteration:
        first_trade = None

    if first_trade:
        pair_name = get_pair_name_from_first_trade(first_trade)
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
        pair_id = int(candles.iloc[0]["pair_id"])

    logger.info(f"Candles are {candle_start_ts} - {candle_end_ts}")

    positions = get_all_positions(state, pair_id)

    logging.info("State has %d positions for pair id %d", len(positions), pair_id)

    if start_at or end_at:
        positions = [p for p in positions if start_at <= p.opened_at <= end_at]
        logging.info(
            "After limiting range %s %s, we have %d positions for pair id %d",
            start_at,
            end_at,
            len(positions),
            pair_id,
        )
    
    # hide volume bar
    volume_bar_mode = VolumeBarMode.hidden

    fig = _get_grid_with_candles_volume_indicators(
        state=state,
        execution_context=execution_context,
        start_at=start_at, 
        end_at=end_at, 
        height=height, 
        axes=axes, 
        technical_indicators=technical_indicators, 
        title=title, 
        theme=theme, 
        volume_bar_mode=volume_bar_mode, 
        vertical_spacing=vertical_spacing, 
        subplot_font_size=subplot_font_size, 
        relative_sizing=relative_sizing, 
        candles=candles,
        pair_name=pair_name,
        labels=None,
    )

    # Add trade markers if any trades have been made
    visualise_positions_with_duration_and_slippage(fig, candles, positions)

    return fig


def _get_grid_with_candles_volume_indicators(
    *,
    state: State,
    execution_context: ExecutionContext,
    start_at: pd.Timestamp | None, 
    end_at: pd.Timestamp | None, 
    height: int, 
    axes: bool, 
    technical_indicators: bool, 
    title: str | bool, 
    theme: str, 
    volume_bar_mode: VolumeBarMode, 
    vertical_spacing: float, 
    subplot_font_size: int, 
    relative_sizing: list[float], 
    candles: pd.DataFrame, 
    pair_name: str | None, 
    labels: pd.Series,
    volume_axis_name: str = "Volume USD",
    pair_id: int | None = None,
    detached_indicators: bool = True,
    hover_text: bool = True,
):
    """Gets figure grid with candles, volume, and indicators overlayed.
    
    .. warning:: Currently only `compatible with visualise_single_pair` and `visualise_single_pair_positions_with_duration_and_slippage`.
    """

    assert isinstance(execution_context, ExecutionContext)
    
    title_text, axes_text, volume_text = get_all_text(state.name, axes, title, pair_name, volume_axis_name)

    if all(p.pair and p.pair.internal_id for p in state.visualisation.plots.values()) and pair_id:
        plots = [plot for plot in state.visualisation.plots.values() if plot.pair.internal_id == pair_id]
        start_row=1
    else:
        plots = state.visualisation.plots.values()
        start_row=None
    
    num_detached_indicators, subplot_names = get_num_detached_and_names(
        technical_indicators,
        plots, 
        execution_context, 
        volume_bar_mode, 
        volume_text, 
        pair_name=None, 
        detached_indicators=detached_indicators
    )

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
        subplot_font_size=subplot_font_size,
    )


    # Draw EMAs etc.
    if technical_indicators:
        overlay_all_technical_indicators(
            fig,
            state.visualisation,
            start_at,
            end_at,
            volume_bar_mode,
            pair_id,
            detached_indicators=detached_indicators,
            start_row=start_row,
        )

    fig.update_yaxes(showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dot', spikethickness=1)

    if hover_text:
        fig.update_layout(hovermode='x unified')

    fig.update_traces(xaxis='x')
        
    return fig



def visualise_single_position(
    state: State,
    strategy_universe: TradingStrategyUniverse,
    position: TradingPosition,
    area_around: int | pd.Timedelta = 100,
    execution_context: ExecutionContext = notebook_execution_context,
) -> go.Figure:
    """Inspect individual won or lost trade.

    - Give you an overview what failed in the trade

    - We draw the technical indicators if the source data has them (`input.visualisation == True`).

    Example:

    .. code-block:: python

        from tradeexecutor.utils.list import get_linearly_sampled_items
        from tradeexecutor.visual.single_pair import visualise_single_position
        from tradeexecutor.utils.profile import profiled

        # We are always using interactive charting mode for single position visualisation
        setup_charting_and_output(OutputMode.interactive)

        state = best_pick.hydrate_state()
        portfolio = state.portfolio

        pair = strategy_universe.get_pair_by_human_description(trading_pairs[0])  # We examine positions for this trading pair only
        examinaned_position_count = 3  # We are linearly sampling this many failing trades for visualisation
        area_around_candels = 40  # How many candles before and after entry and exit

        all_positions = list(portfolio.get_all_positions())
        positions_lost = [p for p in all_positions if p.is_loss()]

        print(f"Total lost positions for {pair.get_ticker()} is {len(positions_lost)} / {len(all_positions)}. We are visualising {examinaned_position_count} of these.")
        # Take 3 positions equally weighted from the time line

        examined_positions = get_linearly_sampled_items(positions_lost, count=examinaned_position_count)

        for position in examined_positions:
            position_summary = pd.Series(position.get_human_summary())
            display(position_summary)
            fig = visualise_single_position(
                state=state,
                strategy_universe=strategy_universe,
                position=position,
                area_around=area_around_candels,
            )
            display(fig)

    :param position:
        A single trading position to visualise

    :param area_around:
        How many candles display before and start of the position
    """

    assert position.is_closed(), "We can only visualise closed positions for now"

    start_at = position.get_first_trade().executed_at
    end_at = position.get_last_trade().executed_at
    candle_width = strategy_universe.data_universe.time_bucket.to_pandas_timedelta()

    if type(area_around) == int:
        buffer = area_around * candle_width
    else:
        assert isinstance(area_around, pd.Timedelta)
        buffer = area_around

    start_at_around = start_at - buffer
    end_at_around = end_at + buffer

    fig = visualise_single_pair(
        state=state,
        execution_context=execution_context,
        candle_universe=strategy_universe.data_universe.candles,
        start_at=start_at_around,
        end_at=end_at_around,
        pair_id=position.pair.internal_id,
    )

    # title = f"Position #{position.position_id}: {position.get_unrealised_and_realised_profit_percent()*100}%, {position.get_realised_profit_usd()} USD"
    # No title needed to display
    fig.update_layout(
        title="",
    )

    # https://community.plotly.com/t/excessive-margins-in-graphs-how-to-remove/49094/2
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
    )

    return fig


def display_positions_table(
    state: State,
    pair: TradingPairIdentifier,
    sort_by="PnL %",
    ascending=True,
) -> pd.DataFrame:
    """Prepare a table of positions taken for a trading pair.

    Example:

    .. code-block:: python

        from tradeexecutor.visual.single_pair import display_positions_table
        pair = token_map["JOE"]  # Get previously resolved trading pair
        display_positions_table(state, pair, sort_by="PnL USD", ascending=False)

    :return:
        DataFrame that includes basic diagnostics info about positions taken in a backtest.
    """
    data = []
    positions = [p for p in state.portfolio.get_all_positions() if p.pair == pair]

    if len(positions) == 0:
        return pd.DataFrame([])

    for p in positions:
        data.append({
            "Open": p.opened_at,
            "Close": p.closed_at,
            "Duration": p.get_duration(),
            "PnL %": p.get_realised_profit_percent() * 100,
            "PnL USD": p.get_realised_profit_usd(),
            "Trades": p.get_trade_count(),
        })

    df = pd.DataFrame(data)
    df = df.set_index("Open")
    df = df.sort_values(by=[sort_by], ascending=ascending)
    return df
