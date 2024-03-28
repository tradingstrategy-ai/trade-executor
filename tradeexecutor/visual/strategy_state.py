"""Visualise the strategy state as an image.

- Draw the latest price action start

- Plot indicators on this

- Make this available PNG for sharing

"""
import datetime
import logging
import pandas as pd
from typing import Optional

from tradeexecutor.state.state import State
from tradeexecutor.state.types import PairInternalId
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.visual.single_pair import visualise_single_pair
from tradeexecutor.visual.multiple_pairs import visualise_multiple_pairs

import plotly.graph_objects as go
import plotly.subplots as sp

from tradingstrategy.charting.candle_chart import VolumeBarMode


logger = logging.getLogger(__name__)


def adjust_legend(f):
    def wrapped(*args, **kwargs):
        fig = f(*args, **kwargs)  # Call the original function to get the figure
        if isinstance(fig, go.Figure):
            # Apply the layout adjustment for the legend
            fig.update_layout(legend=dict(
                bgcolor='rgba(0,0,0,0)'  # Make legend background transparent
            ))
        else:
            raise TypeError("Expected a plotly.graph_objs.Figure object")
        return fig
    return wrapped


@adjust_legend
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
        target_pair_candles = universe.data_universe.candles.get_single_pair_data(sample_count=candle_count, raise_on_not_enough_data=False)
        start_at = target_pair_candles.iloc[0]["timestamp"]
        end_at = target_pair_candles.iloc[-1]["timestamp"]
    else:
        assert start_at, "Must have start_at with end_at"
        assert end_at, "Must have start_at with end_at"
        assert universe.data_universe.candles.get_pair_count() == 1
        target_pair_candles = universe.data_universe.candles.df.loc[pd.Timestamp(start_at):pd.Timestamp(end_at)]

    pair = universe.get_single_pair()
    title = f"{pair.base.token_symbol}/{pair.quote.token_symbol}"

    return visualise_single_pair_strategy_state(state, target_pair_candles, start_at, end_at, technical_indicators=technical_indicators, title=title, height=height)


@adjust_legend
def draw_multi_pair_strategy_state(
        state: State,
        universe: TradingStrategyUniverse,
        width: Optional[int] =1024,
        height: Optional[int] = 2048,
        candle_count: Optional[int] = 64,
        start_at: Optional[datetime.datetime] = None,
        end_at: Optional[datetime.datetime] = None,
        technical_indicators: Optional[bool] = True,
        pair_ids: Optional[list[PairInternalId]] = None,
        detached_indicators: Optional[bool] = True,
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

    :param pair_ids:
        The pair ids to draw

    :param detached_indicators:
        Whether to draw detached technical indicators. This will be ignored if technical_indicators is False.

    :return:
        The strategy state visualisation as a single Plotly figure with multiple subplots
    """

    data = universe.data_universe.candles.df

    if not pair_ids:
        pair_ids = universe.data_universe.pairs.get_all_pair_ids()

    if start_at is None and end_at is None:
            # Get
            candles = data

            # Do candle count clip
            if candle_count:
                pair_count = universe.get_pair_count()
                candles = candles.iloc[-candle_count * pair_count:]

            start_at = candles.iloc[0]["timestamp"]
            end_at = candles.iloc[-1]["timestamp"]
    else:
        assert start_at, "Must have start_at with end_at"
        assert end_at, "Must have start_at with end_at"
        logger.info("Own start_at and end_at provided for multipair live visualisation.")
        candles = data.loc[pd.Timestamp(start_at):pd.Timestamp(end_at)]

    return visualise_multiple_pairs(
        state,
        data,
        start_at,
        end_at,
        pair_ids,
        height=height,
        width=width,
        technical_indicators=technical_indicators,
        detached_indicators=detached_indicators,
    )


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