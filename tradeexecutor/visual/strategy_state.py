"""Visualise the strategy state as an image.

- Draw the latest price action start

- Plot indicators on this

- Make this available PNG for sharing

"""
import datetime
from typing import Optional

import pandas as pd

from tradeexecutor.state.state import State
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.visual.single_pair import visualise_single_pair

import plotly.graph_objects as go
import plotly.subplots as sp

from tradingstrategy.charting.candle_chart import VolumeBarMode


def draw_single_pair_strategy_state(
        state: State,
        universe: TradingStrategyUniverse,
        width=512,
        height=512,
        candle_count=64,
        start_at: Optional[datetime.datetime] = None,
        end_at: Optional[datetime.datetime] = None,
        technical_indicators=True,
) -> go.Figure:
    """Draw a mini price chart image.

    See also

    - `manual-visualisation-test.py`

    - :py:meth:`tradeeexecutor.strategy.pandas_runner.PandasTraderRunner.report_strategy_thinking`.

    :param candle_count:
        Draw N latest candles

    :param start_at:
        Draw by a given time range

    :param end_at:
        Draw by a given time range

    :return:
        The strategy state visualisation as Plotly figure
    """

    assert universe.is_single_pair_universe(), "This visualisation can be done only for single pair trading"

    if start_at is None and end_at is None:
        # Get
        target_pair_candles = universe.universe.candles.get_single_pair_data(sample_count=candle_count, raise_on_not_enough_data=False)
        start_at = target_pair_candles.iloc[0]["timestamp"]
        end_at = target_pair_candles.iloc[-1]["timestamp"]
    else:
        assert start_at, "Must have start_at with end_at"
        assert end_at, "Must have start_at with end_at"
        assert universe.universe.candles.get_pair_count() == 1
        target_pair_candles = universe.universe.candles.df.loc[pd.Timestamp(start_at):pd.Timestamp(end_at)]

    pair = universe.get_single_pair()
    title = f"{pair.base.token_symbol}/{pair.quote.token_symbol}"

    return visualise_single_pair_strategy_state(state, target_pair_candles, start_at, end_at, technical_indicators=technical_indicators, title=title)



def draw_multi_pair_strategy_state(
        state: State,
        universe: TradingStrategyUniverse,
        width=512,
        height=512,
        candle_count=64,
        start_at: Optional[datetime.datetime] = None,
        end_at: Optional[datetime.datetime] = None,
        technical_indicators=True,
) -> list[go.Figure]:
    """Draw mini price chart images for multiple pairs. Returns a single figure with multiple subplots.

    See also

    - `manual-visualisation-test.py`

    - :py:meth:`tradeeexecutor.strategy.pandas_runner.PandasTraderRunner.report_strategy_thinking`.

    :param state:
        The strategy state

    :param universe:
        The trading strategy universe

    :param width:
        The width of the image

    :param height:
        The height of the image

    :param candle_count:
        Draw N latest candles

    :param start_at:
        Draw by a given time range

    :param end_at:
        Draw by a given time range

    :param technical_indicators:
        Whether to draw technical indicators or not

    :return:
        The strategy state visualisation as a single Plotly figure with multiple subplots
    """

    assert universe.get_pair_count() <= 3, "This visualisation can be done only for less than 3 pairs"

    figures, titles = [], []

    for pair_id, data in universe.universe.candles.get_all_pairs():
        
        if start_at is None and end_at is None:
            # Get
            target_pair_candles = data

            # Do candle count clip
            if candle_count:
                target_pair_candles = target_pair_candles.iloc[-candle_count:]

            start_at = target_pair_candles.iloc[0]["timestamp"]
            end_at = target_pair_candles.iloc[-1]["timestamp"]
        else:
            assert start_at, "Must have start_at with end_at"
            assert end_at, "Must have start_at with end_at"
            target_pair_candles = data.loc[pd.Timestamp(start_at):pd.Timestamp(end_at)]

        pair = universe.universe.pairs.get_pair_by_id(pair_id)

        title = f"{pair.base_token_symbol}/{pair.quote_token_symbol}"
        titles.append(title)

        figure = visualise_single_pair_strategy_state(
            state, 
            target_pair_candles, 
            start_at, 
            end_at,
            pair_id=pair_id,
            technical_indicators=technical_indicators, 
            title=title
        )

        figures.append(figure)

    combined_figure = combine_images(figures, width, height, titles)

    return combined_figure


def combine_images(figures, width, height, titles):
    """Combine multiple Plotly Figures into one image."""

    combined_figure = sp.make_subplots(rows=len(figures), cols=1, shared_xaxes=True, vertical_spacing=0.1, row_titles=titles)

    for i, figure in enumerate(figures):
        combined_figure.add_trace(figure.data[0], row=i + 1, col=1)

    # Range slider is not very user friendly so just
    # disable it for now
    combined_figure.update_xaxes(rangeslider={"visible": False})
    
    combined_figure.update_annotations(font_size=12)

    # By default, Plotly seems to make a good size for the subplot figure
    # so we don't use height and width for now
    combined_figure.update_layout(
        # height=height,
        # width=width,
        margin=dict(l=50, r=30, t=50, b=50),
    )

    return combined_figure


def visualise_single_pair_strategy_state(
        state: State,
        target_pair_candles,
        start_at: Optional[datetime.datetime] = None,
        end_at: Optional[datetime.datetime] = None,
        pair_id: Optional[int] = None,
        height=512,
        technical_indicators=True,
        title=False,
) -> go.Figure:
    """Produces a visualisation of the strategy state for a single pair.
    
    :param state:
        The strategy state
    
    :param target_pair_candles:
        The candles for the pair
        
    :param start_at:
        Draw by a given time range
    
    :param end_at:
        Draw by a given time range
        
    :param height:
        The height of the image
        
    :param technical_indicators:
        Whether to draw technical indicators or not
    
    :return:
        The strategy state visualisation as Plotly figure
    """
    figure = visualise_single_pair(
        state,
        target_pair_candles,
        start_at=start_at,
        end_at=end_at,
        pair_id=pair_id,
        height=height,
        title=title,
        axes=False,
        technical_indicators=technical_indicators,
        volume_bar_mode=VolumeBarMode.hidden,  # TODO: Might be needed in the future strats
    )

    figure.update_layout(
        margin=dict(l=60, r=50, t=70, b=60),
    )

    return figure