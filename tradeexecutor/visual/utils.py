from typing import Optional
import pandas as pd
import logging

import datetime

import plotly.graph_objects as go

from tradeexecutor.strategy.execution_context import ExecutionContext
from tradingstrategy.charting.candle_chart import VolumeBarMode

from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.types import PairInternalId
from tradeexecutor.state.visualisation import Plot, PlotKind, PlotLabel
from tradeexecutor.strategy.trade_pricing import format_fees_dollars


logger = logging.getLogger(__name__)


def get_start_and_end(
    start_at: pd.Timestamp | datetime.datetime | None,
    end_at: pd.Timestamp | datetime.datetime | None,
):
    """Get and validate start and end timestamps"""
    if isinstance(start_at, datetime.datetime):
        start_at = pd.Timestamp(start_at)

    if isinstance(end_at, datetime.datetime):
        end_at = pd.Timestamp(end_at)

    if start_at is not None:
        assert isinstance(start_at, pd.Timestamp)

    if end_at is not None:
        assert isinstance(end_at, pd.Timestamp)
    return start_at, end_at


def export_trade_for_dataframe(p: Portfolio, t: TradeExecution) -> dict:
    """Export data for a Pandas dataframe presentation.
    - Decimal roundings are based on rule of thumb and may need to be tuned
    """

    position = p.get_position_by_id(t.position_id)
    reserve_token_symbol = position.reserve_currency.token_symbol
    base_token_symbol = position.pair.get_pricing_pair().base.token_symbol
    price_prefix = f"{base_token_symbol} / USD"

    label = ["-" * 60]

    is_profitable = None

    if t.is_failed():
        label += ["Failed trade"]
        type = "failed"
    elif t.is_repaired():
        label += ["Repaired trade"]
        type = "failed"
    else:
        
        if t.is_stop_loss():
            position = p.find_position_for_trade(t)
            profitable = position.is_profitable()

            type = "stop-loss"
            label += [
                f"Stop loss {base_token_symbol}",
                "",
                f"Triggered at: {position.stop_loss:.4f} {price_prefix}",
            ]
            is_profitable = True if profitable else False
        elif t.is_take_profit():
            type = "take-profit"
            label += [
                f"Take profit {base_token_symbol}",
                "",
                f"Triggered at: {position.take_profit:.4f} {price_prefix}",
            ]
        elif t.is_credit_supply():
            if t.is_buy():
                type = "open-credit-supply"
                label += [
                    f"Open credit supply {reserve_token_symbol}",
                    "",
                ]
            else:
                type = "close-credit-supply"
                label += [
                    f"Close credit supply {reserve_token_symbol}",
                    "",
                ]
        elif t.is_sell():
            type = "sell"
            label += [
                f"Sell {base_token_symbol}",
                "",
            ]
        elif t.is_buy():
            type = "buy"
            label += [
                f"Buy {base_token_symbol}",
                "",
            ]

        label += [
            # "",
            f"Executed at: {t.executed_at}",
            f"Value: {t.get_value():.4f} USD",
            f"Quantity: {abs(t.get_position_quantity()):.6f} {base_token_symbol}",
            # "",
        ]

        label += [
            # f"Mid-price: {t.planned_mid_price:.4f} {price_prefix}"
            # if t.planned_mid_price
            # else "",
            f"Executed at price: {t.executed_price:.4f} {price_prefix}"
            if t.executed_price
            else "",
            # f"Estimated execution price: {t.planned_price:.4f} {price_prefix}"
            # if t.planned_price
            # else "",
            # "",
        ]

        if t.is_success() and t.lp_fees_paid is not None:
            label += [
                f"Fees paid: {format_fees_dollars(t.get_fees_paid())}",
            ]
        
        if t.cost_of_gas:
            label += [f"Gas fee: {t.cost_of_gas:.4f}"]

        if t.notes:
            label += t.notes.split("\n")

    # See Plotly Scatter usage https://stackoverflow.com/a/61349739/315168
    return {
        "timestamp": t.executed_at,
        "success": t.is_success(),
        "type": type,
        "label": "<br>".join(label),
        "price": t.planned_mid_price if t.planned_mid_price else t.planned_price,
        "is_profitable": is_profitable,
    }


def export_trades_as_dataframe(
    portfolio: Portfolio,
    pair_id: PairInternalId,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    include_credit_supply_positions: bool = False,
) -> pd.DataFrame:
    """Convert executed trades to a dataframe, so it is easier to work with them in Plotly.

    :param start_at:
        Crop range

    :param end_at:
        Crop range
    """

    if start:
        if isinstance(start, datetime.datetime):
            start = pd.Timestamp(start)
        assert isinstance(start, pd.Timestamp), f"Got {start} {start.__class__}"

    if end:
        if isinstance(end, datetime.datetime):
            end = pd.Timestamp(end)

        assert isinstance(end, pd.Timestamp), f"Got {end} {end.__class__}"
        assert end

    data = []

    for t in portfolio.get_all_trades():
        if t.is_credit_supply():
            if not include_credit_supply_positions:
                continue
        elif (
            pair_id is not None 
            and t.pair.get_pricing_pair().internal_id != pair_id
        ):
            continue

        # Crop
        if start or end:
            s = t.opened_at or t.started_at
            
            if not s:
                # Hotfix to some invalid data?
                logger.info("Trade lacks start date: %s", t)
                continue

            if s < start or s > end:
                continue

        data.append(export_trade_for_dataframe(portfolio, t))
    return pd.DataFrame(data)


def visualise_trades(
    fig: go.Figure,
    candles: pd.DataFrame,
    trades_df: pd.DataFrame,
    candlestick_row: int | None = None,
    column: int | None = None,
    include_credit_supply_positions: bool = False,
):
    """Plot individual trades over the candlestick chart."""

    # If we have used stop loss, do different categories
    if include_credit_supply_positions:
        advanced_trade_types = ("stop-loss", "take-profit", "open-credit-supply", "close-credit-supply")
    else: 
        advanced_trade_types = ("stop-loss", "take-profit")
    advanced_trades = (
        len(trades_df.loc[trades_df["type"].isin(advanced_trade_types)]) > 0
    )

    stop_loss_df = None
    take_profit_df = None
    open_credit_supply_df = None
    close_credit_supply_df = None
    if advanced_trades:
        buys_df = trades_df.loc[trades_df["type"] == "buy"]
        sells_df = trades_df.loc[trades_df["type"] == "sell"]
        stop_loss_df = trades_df.loc[trades_df["type"] == "stop-loss"].copy()
        take_profit_df = trades_df.loc[trades_df["type"] == "take-profit"]
        if include_credit_supply_positions:
            open_credit_supply_df = trades_df.loc[trades_df["type"] == "open-credit-supply"]
            close_credit_supply_df = trades_df.loc[trades_df["type"] == "close-credit-supply"]
    else:
        buys_df = trades_df.loc[trades_df["type"] == "buy"]
        sells_df = trades_df.loc[trades_df["type"] == "sell"]

    # Buys
    fig.add_trace(
        go.Scatter(
            name="Buy",
            mode="markers",
            x=buys_df["timestamp"],
            y=buys_df["price"],
            text=buys_df["label"],
            marker={
                "color": "#aaaaff",
                "symbol": "triangle-right",
                "size": 12,
                "line": {"width": 1, "color": "#3333aa"},
            },
            hoverinfo="text",
        ),
        secondary_y=False,
        row=candlestick_row,
        col=column,
    )

    # Sells
    fig.add_trace(
        go.Scatter(
            name="Sell",
            mode="markers",
            x=sells_df["timestamp"],
            y=sells_df["price"],
            text=sells_df["label"],
            marker={
                "color": "#aaaaff",
                "symbol": "triangle-left",
                "size": 12,
                "line": {"width": 1, "color": "#3333aa"},
            },
            hoverinfo="text",
        ),
        secondary_y=False,
        row=candlestick_row,
        col=column,
    )

    if stop_loss_df is not None:
        stop_loss_df['colour'] = stop_loss_df['is_profitable'].apply(lambda x: 'green' if x == True else 'orangered')
        fig.add_trace(
            go.Scatter(
                name="Stop loss",
                mode="markers",
                x=stop_loss_df["timestamp"],
                y=stop_loss_df["price"],
                text=stop_loss_df["label"],
                marker={
                    "symbol": "arrow-down",
                    "size": 12,
                    "line": {"width": 1, "color": "black"},
                    "color": stop_loss_df['colour'],
                },
                hoverinfo="text",
            ),
            secondary_y=False,
            row=candlestick_row,
            col=column,
        )

    if take_profit_df is not None:
        fig.add_trace(
            go.Scatter(
                name="Take profit",
                mode="markers",
                x=take_profit_df["timestamp"],
                y=take_profit_df["price"],
                text=take_profit_df["label"],
                marker={
                    "symbol": "arrow-up",
                    "size": 12,
                    "line": {"width": 1, "color": "black"},
                    "color": "lightgreen",
                },
                hoverinfo="text",
            ),
            secondary_y=False,
            row=candlestick_row,
            col=column,
        )

    if open_credit_supply_df is not None:
        fig.add_trace(
            go.Scatter(
                name="Open credit supply",
                mode="markers",
                x=open_credit_supply_df["timestamp"],
                y=open_credit_supply_df["price"],
                text=open_credit_supply_df["label"],
                marker={
                    "symbol": "triangle-ne",
                    "size": 12,
                    "line": {"width": 1, "color": "black"},
                    "color": "green",
                },
                hoverinfo="text",
            ),
            secondary_y=False,
            row=candlestick_row,
            col=column,
        )

    if close_credit_supply_df is not None:
        fig.add_trace(
            go.Scatter(
                name="Close credit supply",
                mode="markers",
                x=close_credit_supply_df["timestamp"],
                y=close_credit_supply_df["price"],
                text=close_credit_supply_df["label"],
                marker={
                    "symbol": "triangle-sw",
                    "size": 12,
                    "line": {"width": 1, "color": "black"},
                    "color": "orange",
                },
                hoverinfo="text",
            ),
            secondary_y=False,
            row=candlestick_row,
            col=column,
        )

    return fig


def get_all_positions(state: State, pair_id):
    """Get all positions for a given pair"""
    assert type(pair_id) == int

    positions = [
        p 
        for p in state.portfolio.get_all_positions() 
        if p.pair.get_pricing_pair().internal_id == pair_id
    ]

    return positions


def get_pair_name_from_first_trade(first_trade: TradeExecution):
    return (
        f"{first_trade.pair.base.token_symbol} - {first_trade.pair.quote.token_symbol}"
    )


def get_pair_base_quote_names(state: State, pair_id: int | None):
    """Get all positions for the trading pair we want to visualise"""

    if pair_id:
        positions = get_all_positions(state, pair_id)
    else:
        positions = []

    if len(positions) > 0:
        first_trade = positions[0].get_first_trade()
    else:
        first_trade = None

    if first_trade:
        pair_name = get_pair_name_from_first_trade(first_trade)
        pair = first_trade.pair
        base_token = pair.base.token_symbol
        quote_token = pair.quote.token_symbol
    else:
        pair_name = None
        base_token = None
        quote_token = None

    return pair_name, base_token, quote_token


def _get_title(name: str, title: str):
    if title is True:
        return name
    elif type(title) == str:
        return title
    else:
        return None


def _get_axes_and_volume_text(
    axes: bool, pair_name: str | None, volume_axis_name: str = "Volume USD"
):
    """Get axes and volume text"""
    if axes:
        axes_text = pair_name
        volume_text = volume_axis_name
    else:
        axes_text = None
        volume_text = None
    return axes_text, volume_text


def get_all_text(
    state_name: str,
    axes: bool,
    title: str | None,
    pair_name: str | None,
    volume_axis_name: str,
):
    title_text = _get_title(state_name, title)
    axes_text, volume_text = _get_axes_and_volume_text(
        axes, pair_name, volume_axis_name
    )

    return title_text, axes_text, volume_text


def _get_num_detached_indicators(
        plots: list[Plot],
        execution_context: ExecutionContext,
        volume_bar_mode: VolumeBarMode,
        detached_indicators: bool):
    """Get the number of detached technical indicators"""

    assert isinstance(execution_context, ExecutionContext)

    if detached_indicators:
        num_detached_indicators = sum(
            plot.kind == PlotKind.technical_indicator_detached for plot in plots
        )
    else:
        num_detached_indicators = 0

    if volume_bar_mode in {VolumeBarMode.hidden, VolumeBarMode.overlay}:
        pass
    elif volume_bar_mode == VolumeBarMode.separate:
        num_detached_indicators += 1
    else:
        raise ValueError(f"Unknown volume bar mode {VolumeBarMode}")

    return num_detached_indicators


def _get_plot_name_and_separator(plot: Plot) -> str:
    if plot.label == PlotLabel.hidden:
        return ""
    return f"<br> + {plot.name}"


def _get_subplot_names(
    plots: list[Plot],
    execution_context: ExecutionContext,
    volume_bar_mode: VolumeBarMode,
    volume_axis_name: str = "Volume USD",
    pair_name: str = None,
):
    """Get subplot names for detached technical indicators.
    Overlaid names are appended to the detached plot name."""

    assert isinstance(execution_context, ExecutionContext)

    if volume_bar_mode in {VolumeBarMode.hidden, VolumeBarMode.overlay}:
        subplot_names = []
        detached_without_overlay_count = 0
    else:
        subplot_names = [volume_axis_name]
        detached_without_overlay_count = 1

    # for allowing multiple overlays on detached plots
    # list of detached plot names that already have overlays
    already_overlaid_names = []

    for plot in plots:

        # get subplot names for detached technical indicators without any overlay
        if (plot.kind == PlotKind.technical_indicator_detached) and (
            plot.name
            not in [
                plot.detached_overlay_name
                for plot in plots
                if plot.kind == PlotKind.technical_indicator_overlay_on_detached
            ]
        ):
            subplot_names.append(plot.name)
            detached_without_overlay_count += 1

        # get subplot names for detached technical indicators with overlay
        if plot.kind == PlotKind.technical_indicator_overlay_on_detached:
            # check that detached plot exists
            detached_plots = [
                plot.name
                for plot in plots
                if plot.kind == PlotKind.technical_indicator_detached
            ]

            # Don't crash live trading due to visualisation bugs.
            # This can e.g. happen if the strategy code changes
            # and the state file contains data for old indicators
            if not execution_context.mode.is_live_trading():
                assert (
                    plot.detached_overlay_name in detached_plots
                ), f"Overlay name {plot.detached_overlay_name} not in available detached plots {detached_plots}. Make sure there is a matching plot with the same name and kind PlotKind.technical_indicator_detached in the visualisation data."

            # check if another overlay exists
            if plot.detached_overlay_name in already_overlaid_names:
                # add to existing overlay
                subplot_names[
                    detached_without_overlay_count
                    + already_overlaid_names.index(plot.detached_overlay_name)
                ] += _get_plot_name_and_separator(plot)
            else:
                # add to list
                subplot_names.append(plot.detached_overlay_name + _get_plot_name_and_separator(plot))
                already_overlaid_names.append(plot.detached_overlay_name)

    # Insert blank name for main candle chart
    subplot_names.insert(0, pair_name)

    return subplot_names


def get_num_detached_and_names(
    plots: list[Plot],
    execution_context: ExecutionContext,
    volume_bar_mode: VolumeBarMode,
    volume_axis_name: str,
    pair_name: str | None = None,
    detached_indicators: bool = True,
):
    """Get num_detached_indicators and subplot_names"""

    assert isinstance(execution_context, ExecutionContext), f"Expected ExecutionContext, got {type(execution_context)}"

    num_detached_indicators = _get_num_detached_indicators(plots, execution_context, volume_bar_mode, detached_indicators)
    
    if detached_indicators:
        subplot_names = _get_subplot_names(
            plots, execution_context, volume_bar_mode, volume_axis_name, pair_name
        )
    elif volume_bar_mode == VolumeBarMode.separate:
        subplot_names = [pair_name, volume_axis_name]
    else:
        subplot_names = [pair_name]

    return num_detached_indicators, subplot_names


def get_num_detached_and_names_no_indicators(
    execution_context: ExecutionContext,
    volume_bar_mode: VolumeBarMode,
    volume_axis_name: str,
    pair_name: str | None = None,
):
    """Get num_detached_indicators and subplot_names. Used when technical_indicators == False"""

    assert isinstance(execution_context, ExecutionContext)

    if volume_bar_mode == VolumeBarMode.separate:
        num_detached_indicators = 1
    else:
        num_detached_indicators = 0
    
    if volume_bar_mode == VolumeBarMode.separate:
        subplot_names = [pair_name, volume_axis_name]
    else:
        subplot_names = [pair_name]

    return num_detached_indicators, subplot_names
