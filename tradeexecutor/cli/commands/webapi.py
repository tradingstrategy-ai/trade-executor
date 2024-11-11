"""Web API command

Example::

    poetry run trade-executor webapi --strategy-file=strategies/enzyme-polygon-matic-usdc.py
"""

import datetime
import faulthandler
import logging
import os
import time
from decimal import Decimal 
from pathlib import Path
from queue import Queue
from typing import Optional

import typer
import waitress

from . import shared_options
from .app import app
from ..bootstrap import prepare_executor_id, prepare_cache, create_metadata, create_state_store
from ..log import setup_logging, setup_discord_logging, setup_logstash_logging, setup_file_logging
from ..version_info import VersionInfo
from ...state.state import State
from ...strategy.execution_model import AssetManagementMode
from ...strategy.run_state import RunState
from ...strategy.strategy_module import read_strategy_module, StrategyModuleInformation
from ...statistics.in_memory_statistics import refresh_run_state
from ...webhook.server import create_pyramid_app
from ...strategy.execution_context import ExecutionContext, ExecutionMode
from ...strategy.cycle import CycleDuration


logger = logging.getLogger(__name__)


@app.command()
def webapi(

    # Strategy assets
    id: str = shared_options.id,
    name: Optional[str] = shared_options.name,
    short_description: Optional[str] = typer.Option(None, envvar="SHORT_DESCRIPTION", help="Short description for metadata"),
    long_description: Optional[str] = typer.Option(None, envvar="LONG_DESCRIPTION", help="Long description for metadata"),
    badges: Optional[str] = typer.Option(None, envvar="BADGES", help="Comma separated list of badges to be displayed on the strategy tile"),
    icon_url: Optional[str] = typer.Option(None, envvar="ICON_URL", help="Strategy icon for web rendering and Discord avatar"),

    strategy_file: Path = shared_options.strategy_file,

    # Webhook server options
    http_port: int = typer.Option(3456, envvar="HTTP_PORT", help="Which HTTP port to listen. The default is 3456, the default port of Pyramid web server."),
    http_host: str = typer.Option("0.0.0.0", envvar="HTTP_HOST", help="The IP address to bind for the web server. By default listen to all IP addresses available in the run-time environment."),
    http_username: str = typer.Option(None, envvar="HTTP_USERNAME", help="Username for HTTP Basic Auth protection of webhooks"),
    http_password: str = typer.Option(None, envvar="HTTP_PASSWORD", help="Password for HTTP Basic Auth protection of webhooks"),

    # Logging
    file_log_level: Optional[str] = typer.Option("info", envvar="FILE_LOG_LEVEL", help="Log file log level. The default log file is logs/id.log."),

    # Logging
    log_level: str = shared_options.log_level,

    # Various file configurations
    state_file: Optional[Path] = shared_options.state_file,
    cache_path: Optional[Path] = shared_options.cache_path,
):
    """Launch Trade Executor instance."""
    global logger

    started_at = datetime.datetime.utcnow()

    # Guess id from the strategy file
    id = prepare_executor_id(id, strategy_file)

    # We always need a name-*-
    if not name:
        if strategy_file:
            name = os.path.basename(strategy_file)
        else:
            name = "Unnamed backtest"

    if not log_level:
        log_level = logging.INFO

    # Make sure unit tests run logs do not get polluted
    # Don't touch any log levels, but
    # make sure we have logger.trading() available when
    # log_level is "disabled"
    logger = setup_logging(log_level, in_memory_buffer=True)

    setup_file_logging(
        f"logs/{id}.log",
        file_log_level,
        http_logging=True,
    )

    if not state_file:
        state_file = f"state/{id}.json"

    cache_path = prepare_cache(id, cache_path, False)
    mod = read_strategy_module(strategy_file)

    if state_file:
        store = create_state_store(Path(state_file))

    fees = dict(
        management_fee=mod.management_fee,
        trading_strategy_protocol_fee=mod.trading_strategy_protocol_fee,
        strategy_developer_fee=mod.strategy_developer_fee,
    )

    metadata = create_metadata(
        name,
        short_description,
        long_description,
        icon_url,
        AssetManagementMode.dummy,
        chain_id=mod.get_default_chain_id(),
        vault=None,
        fees=fees,
    )

    # Start the queue that relays info from the web server to the strategy executor
    command_queue = Queue()

    run_state = RunState()
    run_state.version = VersionInfo.read_docker_version()
    run_state.executor_id = id

    # Set up read-only state sync
    if not store.is_pristine():
        run_state.read_only_state_copy = store.load()

    refresh_run_state(
        run_state,
        store.load(),
        ExecutionContext(mode=ExecutionMode.unit_testing),
        cycle_duration=mod.parameters.cycle_duration,
    )

    app = create_pyramid_app(
        http_username,
        http_password,
        command_queue,
        store,
        metadata,
        run_state,
    )
    waitress.serve(app, host=http_host, port=http_port)
