"""check-universe command"""

import datetime
from pathlib import Path
from typing import Optional
import typer


from packaging import version

from tradingstrategy.client import Client
from .app import app
from ..bootstrap import prepare_executor_id, prepare_cache
from ..log import setup_logging
from ...strategy.bootstrap import import_strategy_file
from ...strategy.description import StrategyExecutionDescription
from ...strategy.execution_context import ExecutionContext, ExecutionMode, preflight_execution_context
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

    Check that we can call create_trading_universe() in the strategy module and it loads data correctly.
    """

    global logger

    id = prepare_executor_id(id, strategy_file)

    logger = setup_logging(log_level)

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
        )
    else:
        universe_options = UniverseOptions()

    engine_version = run_description.trading_strategy_engine_version
    if engine_version:
        if version.parse(engine_version) >= version.parse("0.5"):
            parameters = run_description.runner.parameters
            assert "required_history_period" in parameters, f"Strategy lacks Parameters.required_history_period. We have {parameters}"

    # Deconstruct strategy input
    universe_model: TradingStrategyUniverseModel = run_description.universe_model

    ts = datetime.datetime.utcnow()
    logger.info("Performing universe data check for timestamp %s", ts)
    universe = universe_model.construct_universe(ts, execution_context.mode, universe_options)

    latest_candle_at = universe_model.check_data_age(ts, universe, max_data_delay)
    ago = datetime.datetime.utcnow() - latest_candle_at
    logger.info("Latest OHCLV candle is at: %s, %s ago", latest_candle_at, ago)
