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


def draw_single_pair_strategy_state(
        state: State,
        universe: TradingStrategyUniverse,
        width=512,
        height=512,
        candle_count=32,
        start_at: Optional[datetime.datetime] = None,
        end_at: Optional[datetime.datetime] = None,
) -> go.Figure:
    """Draw a mini price chart image.

    :param candle_count:
        Draw N latest candles

    :param start_at:
        Draw by a given time range

    :param end_at:
        Draw by a given time range
    """

    assert universe.is_single_pair_universe(), "This visualisation can be done only for single pair trading"

    if start_at is None and end_at is None:
        # Get
        target_pair_candles = universe.universe.candles.get_single_pair_data(sample_count=candle_count)
        start_at = target_pair_candles.iloc[0]["timestamp"]
        end_at = target_pair_candles.iloc[-1]["timestamp"]
    else:
        assert start_at, "Must have start_at with end_at"
        assert end_at, "Must have start_at with end_at"
        assert universe.universe.candles.get_pair_count() == 1
        target_pair_candles = universe.universe.candles.df.loc[pd.Timestamp(start_at):pd.Timestamp(end_at)]

    figure = visualise_single_pair(
        state,
        target_pair_candles,
        start_at=start_at,
        end_at=end_at,
        height=height,
        title=False,
        axes=False,
    )

    return figure
