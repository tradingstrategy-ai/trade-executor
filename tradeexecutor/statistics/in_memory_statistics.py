"""Statistics and visualisation calculations not part of the persistent state.

- Regenerated on startup

- Regenerated after trades
"""

import datetime
import logging

from tradeexecutor.ethereum.wallet import perform_gas_level_checks
from tradeexecutor.state.state import State
from tradeexecutor.statistics.summary import calculate_summary_statistics
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.run_state import RunState
from tradeexecutor.strategy.sync_model import SyncModel
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.visual.image_output import render_plotly_figure_as_image_file
from tradeexecutor.visual.strategy_state import draw_single_pair_strategy_state, draw_multi_pair_strategy_state


logger = logging.getLogger(__name__)


def refresh_run_state(
    run_state: RunState,
    state: State,
    execution_context: ExecutionContext,
    visualisation=False,
    universe: TradingStrategyUniverse=None,
    sync_model: SyncModel | None = None,
    backtested_state: State | None = None,
    backtest_cut_off = datetime.timedelta(days=90),
    cycle_duration: CycleDuration = None,
):
    """Update in-memory RunState structures.

    - Redraw all visualisations and recalculate in-memory statistics

    - To be called

        - On startup

        - After trades have been executed

        - After position valuations have been updated
    """

    assert isinstance(run_state, RunState)

    # Strategy statistics
    # Even if the strategy has no action yet (deposits, trades)
    # we need to calculate these statistics, as this will
    # calculate the backtested metrics using in strategy summary tiles
    logger.info("refresh_run_state() - calculating summary statistics, visualisations are %s", visualisation)
    stats = calculate_summary_statistics(
        state,
        execution_context.mode,
        backtested_state=backtested_state,
        # key_metrics_backtest_cut_off=self.metadata.key_metrics_backtest_cut_off,
        key_metrics_backtest_cut_off=backtest_cut_off,
        cycle_duration=cycle_duration,
    )
    run_state.summary_statistics = stats

    # Frozen positions is needed for fault checking hooks
    run_state.frozen_positions = len(state.portfolio.frozen_positions)

    # Strategy charts
    if visualisation:
        assert universe, "Candle data must be available to update visualisations"
        logger.info("Updating the strategy technical charts")
        redraw_visualisations(run_state, state, universe, execution_context)
    else:
        logger.info("Visualisation disabled - technical charts are not updated")

    # Set gas level warning
    if sync_model is not None:
        hot_wallet = sync_model.get_hot_wallet()
        web3 = getattr(sync_model, "web3", None)  # TODO: Typing

        if web3 is not None:
            perform_gas_level_checks(
                web3,
                run_state,
                hot_wallet,
            )


def redraw_visualisations(
    run_state: RunState | None,
    state: State,
    universe: TradingStrategyUniverse,
    execution_context: ExecutionContext,
):
    """Redraw strategy thinking visualisations."""

    assert isinstance(run_state, RunState)

    if not run_state:
        # This strategy is not maintaining a run-state
        # Backtest, simulation, etc.
        logger.info("Could not update strategy thinking image data, self.run_state not available")
        return

    pair_count = universe.get_pair_count()

    logger.info("Refreshing strategy visualisations: %s, pair count is %d",
                run_state.visualisation,
                pair_count
                )

    if universe.is_empty():
        # TODO: Not sure how we end up here
        logger.info("Strategy universe is empty - nothing to report")
        return

    try:

        if pair_count == 1:

            small_figure = draw_single_pair_strategy_state(state, execution_context, universe, height=512)
            # Draw the inline plot and expose them tot he web server
            # TODO: SVGs here are not very readable, have them as a stop gap solution
            large_figure = draw_single_pair_strategy_state(state, execution_context, universe, height=1024)

            refresh_live_strategy_images(run_state, execution_context, small_figure, large_figure)

        elif 1 < pair_count <= 3:

            small_figure_combined = draw_multi_pair_strategy_state(state, execution_context, universe, height=1024)
            large_figure_combined = draw_multi_pair_strategy_state(state, execution_context, universe, height=2048)

            refresh_live_strategy_images(run_state, execution_context, small_figure_combined, large_figure_combined)

        elif 3 < pair_count <=5:

            small_figure_combined = draw_multi_pair_strategy_state(state, execution_context, universe, height=2048, detached_indicators = False)
            large_figure_combined = draw_multi_pair_strategy_state(state, execution_context, universe, height=3840, width = 2160, detached_indicators = False)

            refresh_live_strategy_images(run_state, execution_context, small_figure_combined, large_figure_combined)

        else:
            logger.warning("Charts not yet available for this strategy type. Pair count: %d", pair_count)

    except Exception as e:
        # Don't take trade executor down if visualisations fail
        logger.warning("Could not draw visualisations in refresh_visualisations() - exception ignored")
        logger.warning("Visualisation exception %s", e, exc_info=e)


def refresh_live_strategy_images(
    run_state: RunState,
    execution_context: ExecutionContext,
    small_figure,
    large_figure
):
    """Update the strategy thinking image data with small, small dark theme, large, and large dark theme images.

    :param small_image: 512 x 512 image
    :param large_image: 1920 x 1920 image
    """

    assert isinstance(run_state, RunState)
    assert isinstance(execution_context, ExecutionContext)

    small_image, small_image_dark = get_small_images(small_figure)
    large_image, large_image_dark = get_large_images(large_figure)

    # uncomment if you want light mode for Discord
    # small_figure.update_layout(template="plotly")
    # large_figure.update_layout(template="plotly")

    # don't need the dark images for png (only post light images to discord)
    small_image_png, _ = get_image_and_dark_image(small_figure, format="png", width=512, height=512)
    large_image_png, _ = get_image_and_dark_image(large_figure, format="png", width=1024, height=1024)

    run_state.visualisation.update_image_data(
        small_image,
        large_image,
        small_image_dark,
        large_image_dark,
        small_image_png,
        large_image_png,
    )

    # Workaround: Kaleido backend is crashing
    # https://github.com/tradingstrategy-ai/trade-executor/issues/699
    import plotly.io as pio
    scope = pio.kaleido.scope
    _shutdown_kaleido = getattr(scope, "_shutdown_kaleido", None)
    if _shutdown_kaleido is not None:
        # Removed in Kaleido 1.x?
        _shutdown_kaleido()


def get_small_images(small_figure):
    """Gets the png image of the figure and the dark theme png image. Images are 512 x 512."""
    return get_image_and_dark_image(small_figure, width=512, height=512)

def get_large_images(large_figure):
    """Gets the png image of the figure and the dark theme png image. Images are 1024 x 1024."""
    return get_image_and_dark_image(large_figure, width=1024, height=1024)

def get_image_and_dark_image(figure, width, height, format="svg"):
    """Renders the figure as a PNG image and a dark theme PNG image."""

    image = render_plotly_figure_as_image_file(figure, width=width, height=height, format=format)

    figure.update_layout(template="plotly_dark")
    image_dark = render_plotly_figure_as_image_file(figure, width=width, height=height, format=format)

    return image, image_dark