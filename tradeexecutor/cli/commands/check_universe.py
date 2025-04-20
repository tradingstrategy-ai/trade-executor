"""check-universe command"""

import datetime
import logging
from pathlib import Path
from typing import Optional

import pandas as pd


from .app import app
from ..bootstrap import prepare_executor_id, prepare_cache
from ..log import setup_logging
from ..universe import setup_universe
from ...analysis.pair import display_strategy_universe
from ...strategy.bootstrap import import_strategy_file
from ...strategy.cycle import CycleDuration, snap_to_previous_tick
from ...strategy.execution_context import  console_command_execution_context
from ...strategy.pandas_trader.indicator import calculate_and_load_indicators_inline, MemoryIndicatorStorage
from ...utils.cpu import get_safe_max_workers_count
from . import shared_options


@app.command()
def check_universe(
    id: str = shared_options.id,
    strategy_file: Path = shared_options.strategy_file,
    trading_strategy_api_key: str = shared_options.trading_strategy_api_key,
    cache_path: Optional[Path] = shared_options.cache_path,
    max_data_delay_minutes: int = shared_options.max_data_delay_minutes,
    log_level: str = shared_options.log_level,
    max_workers: int | None = shared_options.max_workers,
):
    """Checks that the trading universe is healthy.

    Check that create_trading_universe() and create_indicators() functions in the strategy module work.
    """

    global logger

    id = prepare_executor_id(id, strategy_file)

    logger = setup_logging(log_level)
    if log_level != "disabled":
        assert logger.level <= logging.INFO, "Log level must be at least INFO to get output from this command"

    logger.info("Loading strategy file %s", strategy_file)

    strategy_factory = import_strategy_file(strategy_file)

    cache_path = prepare_cache(id, cache_path)

    assert trading_strategy_api_key, "TRADING_STRATEGY_API_KEY missing"

    execution_context = console_command_execution_context

    universe_init = setup_universe(
        trading_strategy_api_key=trading_strategy_api_key,
        cache_path=cache_path,
        max_data_delay_minutes=max_data_delay_minutes,
        strategy_factory=strategy_factory,
        execution_context=execution_context,
    )

    # Deconstruct strategy input
    universe_model = universe_init.universe_model
    universe_options = universe_init.universe_options
    max_data_delay = universe_init.max_data_delay
    run_description = universe_init.run_description

    ts = datetime.datetime.utcnow()
    logger.info("Performing universe data check for timestamp %s", ts)
    universe = universe_model.construct_universe(ts, execution_context.mode, universe_options)

    latest_candle_at = universe_model.check_data_age(ts, universe, max_data_delay)
    ago = datetime.datetime.utcnow() - latest_candle_at
    logger.info("Latest OHCLV candle is at: %s, %s ago", latest_candle_at, ago)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 140):
        universe_df = display_strategy_universe(universe)
        logger.info("Universe is:\n%s", str(universe_df))

    # Disable excessive logging for the following section
    logging.getLogger("tradeexecutor.strategy.pandas_trader.indicator").setLevel(logging.WARNING)

    # Poke create_indicators() if the strategy module defines one
    create_indicators = run_description.runner.create_indicators
    if create_indicators:
        parameters = run_description.runner.parameters
        cycle_duration: CycleDuration = parameters["cycle_duration"]
        clock = datetime.datetime.utcnow()
        strategy_cycle_timestamp = snap_to_previous_tick(
            clock,
            cycle_duration,
        )

        logger.info("Checking create_indicators(), using strategy cycle timestamp %s", strategy_cycle_timestamp)
        calculate_and_load_indicators_inline(
            create_indicators=create_indicators,
            strategy_universe=universe,
            parameters=parameters,
            execution_context=execution_context,
            storage=MemoryIndicatorStorage(universe.get_cache_key()),
            strategy_cycle_timestamp=strategy_cycle_timestamp,
            max_workers=max_workers or get_safe_max_workers_count,
        )
    else:
        logger.info("Strategy module lacks create_indicators()")

    logger.info("All ok")


