"""check-universe command"""

import datetime
import logging
from pathlib import Path
from typing import Optional
import typer


from packaging import version

from tradingstrategy.client import Client
from .app import app
from .pair_mapping import construct_identifier_from_pair
from ..bootstrap import prepare_executor_id, prepare_cache
from ..log import setup_logging
from ...strategy.bootstrap import import_strategy_file
from ...strategy.cycle import CycleDuration, snap_to_previous_tick
from ...strategy.description import StrategyExecutionDescription
from ...strategy.execution_context import ExecutionContext, ExecutionMode, preflight_execution_context
from ...strategy.pandas_trader.indicator import calculate_and_load_indicators, calculate_and_load_indicators_inline, MemoryIndicatorStorage
from ...strategy.parameters import dump_parameters
from ...strategy.run_state import RunState
from ...strategy.trading_strategy_universe import TradingStrategyUniverseModel
from ...strategy.universe_model import UniverseOptions
from ...utils.timer import timed_task
from . import shared_options


@app.command()
def check_universe(
    id: str = shared_options.id,
    strategy_file: Path = shared_options.strategy_file,
    trading_strategy_api_key: str = shared_options.trading_strategy_api_key,
    cache_path: Optional[Path] = shared_options.cache_path,
    max_data_delay_minutes: int = typer.Option(24*60, envvar="MAX_DATA_DELAY_MINUTES", help="How fresh the OHCLV data for our strategy must be before failing"),
    log_level: str = shared_options.log_level,
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

    client = Client.create_live_client(
        trading_strategy_api_key,
        cache_path=cache_path,
        settings_path=None,
    )
    client.clear_caches()

    execution_context = preflight_execution_context

    max_data_delay = datetime.timedelta(minutes=max_data_delay_minutes)

    run_description: StrategyExecutionDescription = strategy_factory(
        execution_model=None,
        execution_context=execution_context,
        timed_task_context_manager=timed_task,
        sync_model=None,
        valuation_model_factory=None,
        pricing_model_factory=None,
        approval_model=None,
        client=client,
        run_state=RunState(),
    )

    parameters = run_description.runner.parameters
    if parameters:
        logger.info(
            "Strategy parameters:\n%s",
            dump_parameters(parameters)
        )
        universe_options = UniverseOptions.from_strategy_parameters_class(
            parameters,
            execution_context,
        )
    else:
        universe_options = UniverseOptions()

    # Check that Parameters gives us period how much history we need
    engine_version = run_description.trading_strategy_engine_version
    if engine_version:
        if version.parse(engine_version) >= version.parse("0.5"):
            parameters = run_description.runner.parameters
            assert parameters, f"Engine version {engine_version}, but runner lacks decoded strategy parameters"
            assert "required_history_period" in parameters, f"Strategy lacks Parameters.required_history_period. We have {parameters}"

    # Deconstruct strategy input
    universe_model: TradingStrategyUniverseModel = run_description.universe_model

    ts = datetime.datetime.utcnow()
    logger.info("Performing universe data check for timestamp %s", ts)
    universe = universe_model.construct_universe(ts, execution_context.mode, universe_options)

    latest_candle_at = universe_model.check_data_age(ts, universe, max_data_delay)
    ago = datetime.datetime.utcnow() - latest_candle_at
    logger.info("Latest OHCLV candle is at: %s, %s ago", latest_candle_at, ago)

    # Display trading pairs
    logger.info("Trading pairs in the trading universe:")
    logger.info("-" * 80)
    for idx, pair in enumerate(universe.data_universe.pairs.iterate_pairs(), start=1):
        command_line_pair_id = construct_identifier_from_pair(pair)
        logger.info(f"Pair {idx}. {pair.get_ticker()}, identifier: {command_line_pair_id}")
    logger.info("-" * 80)

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
        )
    else:
        logger.info("Strategy module lacks create_indicators()")


