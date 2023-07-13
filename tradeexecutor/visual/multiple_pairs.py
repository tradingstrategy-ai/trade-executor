"""Tools to visualise live trading/backtest outcome for strategies trading only one pair."""
import datetime
import logging

from typing import Optional, Union, List, Collection
from collections import namedtuple
from dataclasses import dataclass, field

import plotly.graph_objects as go
import pandas as pd
from plotly.graph_objs.layout import Annotation
from plotly.subplots import make_subplots

from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.visualisation import Plot
from tradeexecutor.strategy.trade_pricing import format_fees_dollars
from tradeexecutor.state.visualisation import PlotKind

from tradeexecutor.state.types import PairInternalId
from tradeexecutor.visual.technical_indicator import overlay_all_technical_indicators
from tradeexecutor.visual.single_pair import export_trades_as_dataframe, export_trades_as_dataframe, visualise_trades, _get_pair_base_quote_names

from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.charting.candle_chart import visualise_ohlcv, make_candle_labels, VolumeBarMode, _get_secondary_y, _get_specs

logger = logging.getLogger(__name__)


@dataclass
class PairSubplotInfo:
    """Information required to fill in the subplot grid for a single pair."""

    pair_id: PairInternalId

    candlestick_row: int

    volume_bar_mode: VolumeBarMode

    num_detached_indicators: int

    #: includes main candlestick name
    subplot_names: List[str]

    candles: pd.DataFrame

    candle_labels: pd.DataFrame

    trades: pd.DataFrame

    relative_sizing: list[float]

    specs: list[list[dict]] = field(init=False)

    detached_indicator_start_row: int = field(init=False)

    def __post_init__(self):
        is_secondary_y = _get_secondary_y(self.volume_bar_mode)
        self.specs = _get_specs(self.num_detached_indicators, is_secondary_y)

        if self.volume_bar_mode in {VolumeBarMode.overlay, VolumeBarMode.hidden}:
            self.detached_indicator_start_row = self.candlestick_row + 1
        elif VolumeBarMode == VolumeBarMode.separate:
            self.detached_indicator_start_row = self.candlestick_row + 2
        else:
            raise ValueError(f"Unknown volume bar mode {VolumeBarMode}")

    def get_volume_row(self):
        if self.volume_bar_mode == VolumeBarMode.overlay:
            return self.candlestick_row
        elif self.volume_bar_mode == VolumeBarMode.separate:
            return self.candlestick_row + 1
        else:
            return None
        

def visualise_multiple_pairs(
        state: Optional[State],
        candle_universe: GroupedCandleUniverse | pd.DataFrame,
        start_at: Optional[Union[pd.Timestamp, datetime.datetime]] = None,
        end_at: Optional[Union[pd.Timestamp, datetime.datetime]] = None,
        pair_ids: Optional[list[PairInternalId]] = None,
        height=2000,
        width=1000,
        axes=True,
        technical_indicators=True,
        title: Union[str, bool] = True,
        theme="plotly_white",
        volume_bar_modes: Optional[list[VolumeBarMode]] = None,
        vertical_spacing = 0.03,
        subplot_font_size = 11,
        relative_sizing: list[float] = None,
        volume_axis_name: str = "Volume USD",
        candle_decimals: int = 4,
) -> go.Figure:
    """Visualise single-pair trade execution.

    :param state:
        The recorded state of the strategy execution.

        You must give either `state` or `positions`.

    :param pair_ids:
        The visualised pairs.

        If the user would like to visualise all pairs this is not needed.

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

    """
    
    logger.info("Visualising %s", state)

    start_at, end_at = _get_start_and_end(start_at, end_at)

    # convert candles to raw dataframe
    if isinstance(candle_universe, GroupedCandleUniverse):
        if pair_ids is None:
            pair_ids = [int(pair_id) for pair_id in list(candle_universe.get_pair_ids())]

        df = candle_universe.df
        candles = df.loc[df["pair_id"].isin(pair_ids)]
    else:
        candles = candle_universe

    pair_ids = [int(pair_id) for pair_id in pair_ids]

    if not volume_bar_modes:
        volume_bar_modes = [VolumeBarMode.overlay] * len(pair_ids)
    else:
        assert len(volume_bar_modes) == len(pair_ids), "volume_bar_modes must be the same length as pair_ids"

    pair_subplot_infos: list[PairSubplotInfo] = []
    current_candlestick_row = 1

    for i, pair_id in enumerate(pair_ids):

        assert "pair_id" in candles.columns, "candles must have a pair_id column"
        
        c = candles.loc[candles["pair_id"] == pair_id]
        
        pair_name, base_token, quote_token = _get_pair_base_quote_names(state, pair_id)

        if not start_at:
            # No trades made, use the first candle timestamp
            start_at = candle_universe.get_timestamp_range()[0]

        if not end_at:
            end_at = candle_universe.get_timestamp_range()[1]

        logger.info(f"Visualising single pair strategy for range {start_at} - {end_at}")

        # Candles define our diagram X axis
        # Crop it to the trading range
        c = c.loc[c["timestamp"].between(start_at, end_at)]

        candle_start_ts = c["timestamp"].min()
        candle_end_ts = c["timestamp"].max()
        logger.info(f"Candles are {candle_start_ts} = {candle_end_ts}")

        trades_df = export_trades_as_dataframe(
            state.portfolio,
            pair_id,
            start_at,
            end_at,
        )

        labels = make_candle_labels(
            c,
            base_token_name=base_token,
            quote_token_name=quote_token,
            candle_decimals=candle_decimals
        )

        plots = [plot for plot in state.visualisation.plots.values() if plot.pair.internal_id == pair_id]

        title_text, axes_text, volume_text = _get_all_text(state.name, axes, title, pair_name, volume_axis_name)

        num_detached_indicators, subplot_names = _get_num_detached_and_names(plots, volume_bar_modes[i], volume_text, pair_name)

        if not relative_sizing:
            _relative_sizing = [1] + [0.3] * num_detached_indicators

        pair_subplot_infos.append(
            PairSubplotInfo(
                pair_id=pair_id,
                candlestick_row=current_candlestick_row,
                volume_bar_mode=volume_bar_modes[i],
                num_detached_indicators=num_detached_indicators,
                subplot_names=subplot_names,
                candles=c,
                candle_labels = labels,
                trades=trades_df,
                relative_sizing = _relative_sizing
            )
        )

        current_candlestick_row += num_detached_indicators + 1
    
    # Create empty grid space

    relative_sizing = [item for pair_subplot_info in pair_subplot_infos for item in pair_subplot_info.relative_sizing]

    subplot_names = [item for pair_subplot_info in pair_subplot_infos for item in pair_subplot_info.subplot_names]

    specs = [item for pair_subplot_info in pair_subplot_infos for item in pair_subplot_info.specs]
    
    num_rows = current_candlestick_row - 1

    assert len(relative_sizing) == len(subplot_names) == len(specs) == num_rows, f"Sanity check. Should not happen. Expected {num_rows}, got {len(relative_sizing), len(subplot_names), len(specs)}"

    subplot = make_subplots(
        rows=num_rows,
        cols=1,
        row_heights=relative_sizing,
        row_titles=subplot_names,
        shared_xaxes=True,
        specs = specs,
        vertical_spacing=vertical_spacing,
    )

    """
    1. create overall grid of subplots based on pair_subplot_infos
    2. iteratively get candle and volume traces for each pair and plot them using pair_subplot_infos
    3. create function to get current row for technical indicators (based on candlestick row and volume_bar_mode)
    4. plot technical indicators for each pair in the grid
    5. trades (same row as candlestick)
    """
    
    for pair_subplot_info in pair_subplot_infos:

        _candles = pair_subplot_info.candles
        text = pair_subplot_info.candle_labels

        candlesticks = go.Candlestick(
            x=_candles.index,
            open=_candles['open'],
            high=_candles['high'],
            low=_candles['low'],
            close=_candles['close'],
            showlegend=False,
            text=text,
            hoverinfo="text",
        )

        subplot.add_trace(
            candlesticks,
            row=pair_subplot_info.candlestick_row,
            col=1,
        )

        
        if "volume" in candles.columns and pair_subplot_info.volume_bar_mode != VolumeBarMode.hidden:
            volume_bars = go.Bar(
                    x=candles.index,
                    y=candles['volume'],
                    showlegend=False,
                    marker={
                        "color": "rgba(128,128,128,0.5)",
                    }
                )
            
            subplot.add_trace(
                volume_bars,
                row=pair_subplot_info.get_volume_row(),
                col=1,
            )
        else:
            volume_bars = None

        if technical_indicators:
            overlay_all_technical_indicators(
                subplot,
                state.visualisation,
                start_at,
                end_at,
                pair_subplot_info.volume_bar_mode,
                pair_subplot_info.pair_id,
                start_row=pair_subplot_info.candlestick_row,
            )

        # Add trade markers if any trades have been made
        trades_df = pair_subplot_info.trades
        if len(trades_df) > 0:
            visualise_trades(
                subplot,
                candles, 
                trades_df, 
                pair_subplot_info.candlestick_row,
                1,
            )

    subplot.update_xaxes(rangeslider={"visible": False})

    subplot.update_annotations(font_size=subplot_font_size)

    subplot.update_layout(dict(
        showlegend = False,
        height=height,
        width=width,
        template=theme,
    ))

    return subplot

# changed
def _get_grid_with_candles_volume_indicators(
    *,
    state: State, 
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
):
    """Gets figure grid with candles, volume, and indicators overlayed."""
    
    title_text, axes_text, volume_text = _get_all_text(state.name, axes, title, pair_name, volume_axis_name)

    num_detached_indicators, subplot_names = _get_num_detached_and_names(plots, volume_bar_mode, volume_axis_name)
    
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
        )
        
    return fig


def _get_all_text(
    state_name: str, 
    axes: bool, 
    title: str | None, 
    pair_name: str | None,
    volume_axis_name: str,
):
    title_text = _get_title(state_name, title)
    axes_text, volume_text = _get_axes_and_volume_text(axes, pair_name, volume_axis_name)
    
    return title_text,axes_text,volume_text


# changed
def _get_num_detached_and_names(
    plots: list[Plot], 
    volume_bar_mode: VolumeBarMode, 
    volume_axis_name: str,
    title_text: str,
):
    """Get num_detached_indicators and subplot_names"""

    num_detached_indicators = _get_num_detached_indicators(plots, volume_bar_mode)
    subplot_names = _get_subplot_names(plots, volume_bar_mode, volume_axis_name, title_text)

    return num_detached_indicators,subplot_names


def _get_title(name: str, title: str):
    if title is True:
        return name
    elif type(title) == str:
        return title
    else:
        return None

def _get_num_detached_indicators(plots: list[Plot], volume_bar_mode: VolumeBarMode):
    """Get the number of detached technical indicators"""
    
    num_detached_indicators = sum(
        plot.kind == PlotKind.technical_indicator_detached
        for plot in plots
    )
    
    if volume_bar_mode in {VolumeBarMode.hidden, VolumeBarMode.overlay}:
        pass
    elif volume_bar_mode == VolumeBarMode.separate:
        num_detached_indicators += 1
    else:
        raise ValueError(f"Unknown volume bar mode {VolumeBarMode}")
    
    return num_detached_indicators


# changed
def _get_subplot_names(
    plots: list[Plot], 
    volume_bar_mode: VolumeBarMode, 
    volume_axis_name: str = "Volume USD",
    title_text: str = None,
):
    """Get subplot names for detached technical indicators. 
    
    Overlaid names are appended to the detached plot name."""
    
    
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
        if (plot.kind == PlotKind.technical_indicator_detached) and (plot.name not in [plot.detached_overlay_name for plot in plots if plot.kind == PlotKind.technical_indicator_overlay_on_detached]):
            subplot_names.append(plot.name)
            detached_without_overlay_count += 1
            
        # get subplot names for detached technical indicators with overlay
        if plot.kind == PlotKind.technical_indicator_overlay_on_detached:
            # check that detached plot exists
            detached_plots = [plot.name for plot in plots if plot.kind == PlotKind.technical_indicator_detached]
            assert plot.detached_overlay_name in detached_plots, f"Overlay name {plot.detached_overlay_name} not in available detached plots {detached_plots}"
            
            # check if another overlay exists
            if plot.detached_overlay_name in already_overlaid_names:
                # add to existing overlay
                subplot_names[detached_without_overlay_count + already_overlaid_names.index(plot.detached_overlay_name)] += f"<br> + {plot.name}"
            else:
                # add to list
                subplot_names.append(plot.detached_overlay_name + f"<br> + {plot.name}")
                already_overlaid_names.append(plot.detached_overlay_name)
    
    # Insert blank name for main candle chart    
    subplot_names.insert(0, title_text)
    
    return subplot_names

def _get_start_and_end(
    start_at: pd.Timestamp | datetime.datetime | None, 
    end_at: pd.Timestamp | datetime.datetime | None
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
    return start_at,end_at


def _get_all_positions(state: State, pair_id):
    """Get all positions for a given pair"""
    assert type(pair_id) == int
    positions = [p for p in state.portfolio.get_all_positions() if p.pair.internal_id == pair_id]
    return positions


def _get_axes_and_volume_text(axes: bool, pair_name: str | None, volume_axis_name: str = "Volume USD"):
    """Get axes and volume text"""
    if axes:
        axes_text = pair_name
        volume_text = volume_axis_name
    else:
        axes_text = None
        volume_text = None
    return axes_text,volume_text


def _get_pair_name_from_first_trade(first_trade: TradeExecution):
    return f"{first_trade.pair.base.token_symbol} - {first_trade.pair.quote.token_symbol}"