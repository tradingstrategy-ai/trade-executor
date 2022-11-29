"""Visualise the strategy state as an image.

- Draw the latest price action start

- Plot indicators on this

- Make this available PNG for sharing

"""
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
) -> go.Figure:
    """Draw a mini price chart image."""

    assert universe.is_single_pair_universe(), "This visualisation can be done only for single pair trading"

    target_pair_candles = universe.universe.candles.get_single_pair_data(sample_count=candle_count)

    figure = visualise_single_pair(
        state,
        universe.universe.candles,
        start_at=target_pair_candles.iloc[0]["timestamp"],
        end_at=target_pair_candles.iloc[-1]["timestamp"],
        height=height,
        draw_title=False,
        draw_axes=False,
    )

    return figure
