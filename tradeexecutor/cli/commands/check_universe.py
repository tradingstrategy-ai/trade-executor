"""check-universe command"""

import datetime
from pathlib import Path
from typing import Optional

import typer

from tradingstrategy.client import Client
from .app import app
from ..bootstrap import prepare_executor_id, prepare_cache
from ..log import setup_logging
from ...strategy.bootstrap import import_strategy_file
from ...strategy.description import StrategyExecutionDescription
from ...strategy.execution_context import ExecutionContext, ExecutionMode
from ...strategy.run_state import RunState
from ...strategy.trading_strategy_universe import TradingStrategyUniverseModel
from ...strategy.universe_model import UniverseOptions
from ...utils.timer import timed_task


@app.command()
def check_universe(
    id: str = typer.Option(None, envvar="EXECUTOR_ID", help="Executor id used when programmatically referring to this instance. If not given, take the base of --strategy-file."),
    strategy_file: Path = typer.Option(..., envvar="STRATEGY_FILE"),
    trading_strategy_api_key: str = typer.Option(None, envvar="TRADING_STRATEGY_API_KEY", help="Trading Strategy API key"),
    cache_path: Optional[Path] = typer.Option(None, envvar="CACHE_PATH", help="Where to store downloaded datasets"),
    max_data_delay_minutes: int = typer.Option(24*60, envvar="MAX_DATA_DELAY_MINUTES", help="How fresh the OHCLV data for our strategy must be before failing"),
    log_level: str = typer.Option(None, envvar="LOG_LEVEL", help="The Python default logging level. The defaults are 'info' is live execution, 'warning' if backtesting. Set 'disabled' in testing."),
):
    """Checks that the trading universe is helthy for a given strategy."""

    global logger

    id = prepare_executor_id(id, strategy_file)

    logger = setup_logging(log_level)

    logger.info("Loading strategy file %s", strategy_file)

    strategy_factory = import_strategy_file(strategy_file)

    cache_path = prepare_cache(id, cache_path)

    assert trading_strategy_api_key, "TRADING_STRATEGY_API_KEY missing"

    client = Client.create_live_client(trading_strategy_api_key, cache_path=cache_path)
    client.clear_caches()

    execution_context = ExecutionContext(
        mode=ExecutionMode.preflight_check,
        timed_task_context_manager=timed_task
    )

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

    # Deconstruct strategy input
    universe_model: TradingStrategyUniverseModel = run_description.universe_model

    ts = datetime.datetime.utcnow()
    logger.info("Performing universe data check for timestamp %s", ts)
    universe = universe_model.construct_universe(ts, ExecutionMode.preflight_check, UniverseOptions())

    latest_candle_at = universe_model.check_data_age(ts, universe, max_data_delay)
    ago = datetime.datetime.utcnow() - latest_candle_at
    logger.info("Latest OHCLV candle is at: %s, %s ago", latest_candle_at, ago)
