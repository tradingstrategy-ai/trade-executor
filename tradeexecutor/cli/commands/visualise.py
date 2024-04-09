"""Command to visualise a given strategy module locally.

.. code-block:: console
    trade-executor \
        visualise \
        --strategy-file=path/to/strategy-module.py \
        --trading-strategy-api-key=$TRADING_STRATEGY_API_KEY

"""

import logging
from pathlib import Path
from typing import Optional
from PIL import Image
from io import BytesIO

from . import shared_options
from .app import app
from ..bootstrap import prepare_executor_id
from ..log import setup_logging
from tradeexecutor.strategy.execution_context import standalone_backtest_execution_context
from tradeexecutor.backtest.backtest_module import run_backtest_for_module

from tradeexecutor.visual.strategy_state import draw_single_pair_strategy_state, draw_multi_pair_strategy_state
from tradeexecutor.statistics.in_memory_statistics import get_image_and_dark_image

logger = logging.getLogger(__name__)


@app.command()
def visualise(
    id: str = shared_options.id,
    name: Optional[str] = shared_options.name,
    strategy_file: Path = shared_options.strategy_file,

    # Backtest already requires an API key
    trading_strategy_api_key: str = shared_options.trading_strategy_api_key,
    log_level: str = shared_options.log_level,

    # Debugging and unit testing
    unit_testing: bool = shared_options.unit_testing,
):
    """Visualise a given strategy module.

    - Run a backtest on a strategy module.

    - Visualise the image in decide_trades() cycle.

    """
    global logger

    id = prepare_executor_id(id, strategy_file)

    if not log_level:
        log_level = logging.ERROR
    logger = setup_logging(log_level)

    execution_context = standalone_backtest_execution_context 
    state, universe, _ = run_backtest_for_module(
        strategy_file=strategy_file,
        trading_strategy_api_key=trading_strategy_api_key,
        execution_context=execution_context
    )

    pair_count = universe.get_pair_count()
    if pair_count == 1:
        small_figure = draw_single_pair_strategy_state(state, execution_context, universe, height=512)
    elif 1 < pair_count <= 3:
        small_figure = draw_multi_pair_strategy_state(state, execution_context, universe,  height=1024)
    elif 3 < pair_count <= 5:
        small_figure = draw_multi_pair_strategy_state(state, execution_context, universe, height=2048, detached_indicators=False)

    light_image_data, _ = get_image_and_dark_image(small_figure, format="png", width=1024, height=1024)

    # save image to file for debugging
    img = Image.open(BytesIO(light_image_data))
    img.save(f"state/{name}.png")
    img.show()
