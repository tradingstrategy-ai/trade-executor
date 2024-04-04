"""Tools to visualise live trading/backtest outcome for strategies trading only one pair."""
import datetime
import logging

from typing import Optional, Union, List
from dataclasses import dataclass, field

import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots

from tradeexecutor.state.state import State
from tradeexecutor.state.types import PairInternalId
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.visual.technical_indicator import overlay_all_technical_indicators
from tradeexecutor.visual.utils import get_pair_base_quote_names

from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.charting.candle_chart import (
    make_candle_labels,
    VolumeBarMode,
    _get_secondary_y,
    _get_specs,
)

from tradeexecutor.visual.utils import (
    export_trades_as_dataframe,
    visualise_trades,
    get_start_and_end,
    get_all_text,
    get_num_detached_and_names,
    get_num_detached_and_names_no_indicators,
)


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


def get_start_and_end_full(candles: pd.DataFrame | GroupedCandleUniverse, start_at: pd.Timestamp | datetime.datetime | None, end_at: pd.Timestamp | datetime.datetime | None):

    start_at, end_at = get_start_and_end(start_at, end_at)
    
    if isinstance(candles, GroupedCandleUniverse):
        s, e = candles.get_timestamp_range()
        if not start_at or start_at < s:
            start_at = s
        if not end_at or end_at > e:
            end_at = e
    else:
        if not start_at or start_at < candles["timestamp"].min():
            start_at = candles["timestamp"].min()
        if not end_at or end_at > candles["timestamp"].max():
            end_at = candles["timestamp"].max()

    return start_at, end_at


def visualise_multiple_pairs(
    state: State,
    candle_universe: GroupedCandleUniverse | pd.DataFrame,
    execution_context: ExecutionContext,
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
    vertical_spacing=0.03,
    subplot_font_size=11,
    relative_sizing: list[float] = None,
    volume_axis_name: str = "Volume USD",
    candle_decimals: int = 4,
    show_trades: bool = True,
    detached_indicators: bool = True,
) -> go.Figure:
    """Visualise single-pair trade execution.

    .. note:: Volume has been disabled for now for multipair visualisation. Using volume_bar_modes or volume_axis_name currently has no effect.

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
        How to draw the volume bars. By default, volume is hidden.

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
        
    :param show_trades:
        If set, show trades on the chart.
        
    :return:
        Plotly figure object
    """

    assert isinstance(execution_context, ExecutionContext)

    logger.info("Visualising %s", state)

    if not show_trades:
        logger.warning("Trades will not be shown")

    # convert candles to raw dataframe
    if isinstance(candle_universe, GroupedCandleUniverse):
        if pair_ids is None:
            pair_ids = [
                int(pair_id) for pair_id in list(candle_universe.get_pair_ids())
            ]

        df = candle_universe.df
        candles = df.loc[df["pair_id"].isin(pair_ids)]
    else:
        candles = candle_universe

        if pair_ids is None:
            assert "pair_id" in candles.columns, "candles must have a pair_id column"
            pair_ids = list(candles["pair_id"].unique())

    pair_ids = [int(pair_id) for pair_id in pair_ids]

    if not (start_at and end_at):
        start_at, end_at = state.get_strategy_start_and_end()

    start_at, end_at = get_start_and_end_full(candles, start_at, end_at)

    logger.info(f"Visualising multipair strategy for range {start_at} - {end_at}")

    # volume disabled by default for multipair visualisation
    if not volume_bar_modes:
        volume_bar_modes = [VolumeBarMode.hidden] * len(pair_ids)
    else:
        assert len(volume_bar_modes) == len(
            pair_ids
        ), f"volume_bar_modes must be the same length as pair_ids ({len(pair_ids)})"

    pair_subplot_infos: list[PairSubplotInfo] = []
    current_candlestick_row = 1

    assert "pair_id" in candles.columns, "candles must have a pair_id column"

    for i, pair_id in enumerate(pair_ids):

        # Candles define our diagram X axis
        # Crop it to the trading range and correct pair
        c = candles.loc[(candles["pair_id"] == pair_id) & (candles["timestamp"].between(start_at, end_at))]

        pair_name, base_token, quote_token = get_pair_base_quote_names(state, pair_id)

        logger.info(f"Visualising pair {pair_name} for range {start_at} - {end_at}")

        candle_start_ts = c["timestamp"].min()
        candle_end_ts = c["timestamp"].max()
        logger.info(f"Candles for {pair_name} are {candle_start_ts} - {candle_end_ts}")

        if show_trades:
            trades_df = export_trades_as_dataframe(
                state.portfolio,
                pair_id,
                candle_start_ts,
                candle_end_ts,
            )
        else: 
            trades_df = None

        labels = make_candle_labels(
            c,
            base_token_name=base_token,
            quote_token_name=quote_token,
            candle_decimals=candle_decimals,
        )

        plots = [
            plot
            for plot in state.visualisation.plots.values()
            if plot.pair and plot.pair.internal_id == pair_id
        ]

        title_text, axes_text, volume_text = get_all_text(
            state.name, axes, title, pair_name, volume_axis_name
        )

        if technical_indicators:
            num_detached_indicators, subplot_names = get_num_detached_and_names(
                plots, execution_context, volume_bar_modes[i], volume_text, pair_name, detached_indicators,
            )
        else:
            num_detached_indicators, subplot_names = get_num_detached_and_names_no_indicators(execution_context, volume_bar_modes[i], volume_text, pair_name)

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
                candle_labels=labels,
                trades=trades_df,
                relative_sizing=_relative_sizing,
            )
        )

        current_candlestick_row += num_detached_indicators + 1

    # Create empty grid space

    relative_sizing = [
        item
        for pair_subplot_info in pair_subplot_infos
        for item in pair_subplot_info.relative_sizing
    ]

    subplot_names = [
        item
        for pair_subplot_info in pair_subplot_infos
        for item in pair_subplot_info.subplot_names
    ]

    specs = [
        item
        for pair_subplot_info in pair_subplot_infos
        for item in pair_subplot_info.specs
    ]

    num_rows = current_candlestick_row - 1

    # Disabled in live trading, so we do not crash if the strategy has managed to produce invalid visualisation data
    if execution_context.mode.is_live_trading():
        assert (
            len(relative_sizing) == len(subplot_names) == len(specs) == num_rows
        ), f"Sanity check. Should not happen. Expected {num_rows}, got {len(relative_sizing), len(subplot_names), len(specs)}"

    subplot = make_subplots(
        rows=num_rows,
        cols=1,
        row_heights=relative_sizing,
        row_titles=subplot_names,
        shared_xaxes=True,
        specs=specs,
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
            open=_candles["open"],
            high=_candles["high"],
            low=_candles["low"],
            close=_candles["close"],
            showlegend=False,
            text=text,
            hoverinfo="text",
        )

        subplot.add_trace(
            candlesticks,
            row=pair_subplot_info.candlestick_row,
            col=1,
        )

        if (
            "volume" in _candles.columns
            and pair_subplot_info.volume_bar_mode != VolumeBarMode.hidden
        ):
            volume_bars = go.Bar(
                x=_candles.index,
                y=_candles["volume"],
                showlegend=False,
                marker={
                    "color": "rgba(128,128,128,0.5)",
                },
            )

            subplot.add_trace(
                volume_bars,
                row=pair_subplot_info.get_volume_row(),
                col=1,
                secondary_y=True,
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
                detached_indicators=detached_indicators,
            )

        # Add trade markers if any trades have been made
        trades_df = pair_subplot_info.trades
        if show_trades and trades_df is not None and len(trades_df) > 0:
            visualise_trades(
                subplot,
                _candles,
                trades_df,
                pair_subplot_info.candlestick_row,
                1,
            )

    subplot.update_xaxes(rangeslider={"visible": False})

    subplot.update_annotations(font_size=subplot_font_size)

    subplot.update_layout(
        dict(
            showlegend=False,
            height=height,
            width=width,
            template=theme,
        )
    )
    
    if volume_bars:
        subplot.update_annotations(xshift=40)

    return subplot
