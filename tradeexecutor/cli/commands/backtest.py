"""backtest CLI command

To read generated cprofile reports:

.. code-block:: python

    import pstats
    p = pstats.Stats('backtest.cprof')
    p.strip_dirs().sort_stats('cumulative').print_stats(10)



"""

import datetime
import logging
import os
import time
from decimal import Decimal
from pathlib import Path
from queue import Queue
from typing import Optional

import pandas as pd
import typer

from eth_defi.gas import GasPriceMethod
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.testing.uniswap_v2_mock_client import UniswapV2MockClient
from tradingstrategy.timebucket import TimeBucket
from typer import Option

from . import shared_options
from .app import app, TRADE_EXECUTOR_VERSION
from .shared_options import required_option
from ..bootstrap import prepare_executor_id, prepare_cache, create_web3_config, create_state_store, \
    create_execution_and_sync_model, create_metadata, create_approval_model, create_client
from ..log import setup_logging, setup_discord_logging, setup_logstash_logging, setup_file_logging, \
    setup_custom_log_levels
from ..loop import ExecutionLoop
from ..result import display_backtesting_results
from ..version_info import VersionInfo
from ..watchdog import stop_watchdog
from ...backtest.backtest_module import run_backtest_for_module
from ...backtest.backtest_runner import setup_backtest, run_backtest, setup_backtest_for_universe
from ...backtest.tearsheet import export_backtest_report
from ...ethereum.enzyme.vault import EnzymeVaultSyncModel
from ...ethereum.uniswap_v2.uniswap_v2_routing import UniswapV2Routing
from ...state.state import State
from ...state.store import NoneStore, JSONFileStore
from ...strategy.approval import ApprovalType
from ...strategy.bootstrap import import_strategy_file
from ...strategy.cycle import CycleDuration
from ...strategy.default_routing_options import TradeRouting
from ...strategy.execution_context import ExecutionContext, ExecutionMode, standalone_backtest_execution_context
from ...strategy.execution_model import AssetManagementMode
from ...strategy.pandas_trader.indicator import IndicatorStorage, DiskIndicatorStorage
from ...strategy.routing import RoutingModel
from ...strategy.run_state import RunState
from ...strategy.strategy_cycle_trigger import StrategyCycleTrigger
from ...strategy.strategy_module import read_strategy_module, StrategyModuleInformation
from ...utils.timer import timed_task
from ...webhook.server import create_webhook_server


logger = logging.getLogger(__name__)


@app.command()
def backtest(
    id: str = shared_options.id,
    name: Optional[str] = shared_options.name,
    strategy_file: Path = shared_options.strategy_file,
    state_file: Path = shared_options.state_file,

    # Backtest already requires an API key
    trading_strategy_api_key: str = required_option(shared_options.trading_strategy_api_key),

    log_level: str = shared_options.log_level,

    # Debugging and unit testing
    unit_testing: bool = shared_options.unit_testing,

    # Unsorted options
    cache_path: Optional[Path] = shared_options.cache_path,

    notebook_report: Optional[Path] = shared_options.notebook_report,
    html_report: Optional[Path] = shared_options.html_report,

    python_profile_report: Optional[Path] = Option(None, envvar="PYTHON_PROFILE_REPORT", help="Write a Python cprof file to check where backtest spends time"),

    generate_report: Optional[bool] = Option(True, envvar="GENERATE_REPORT", help="Generate a HTML report file based on the template notebook. Disable to reduce unit test execution time."),

    max_workers: Optional[int] = Option(None, envvar="MAX_WORKERS", help="Number of workers to use for parallel processing"),

    extra_output: Optional[bool] = Option(False, envvar="EXTRA_OUTPUT", help="By default, info level is so verbose that running the backtest takes a long time. Give --extra-output to make sure you want to run info log level for a backtest."),
):
    """Backtest a given strategy module.

    - Run a backtest on a strategy module.

    - Writes the resulting state file report,
      as it is being used by the webhook server to read backtest results

    - Writes the resulting Jupyter Notebook report,
      as it is being used by the webhook server to display backtest results

    """
    global logger

    # Guess id from the strategy file
    id = prepare_executor_id(id, strategy_file)

    # We always need a name
    if not name:
        name = f"{id} backtest"

    if not extra_output:
        log_level = logging.WARNING

    # Make sure
    # - We do not flood console with the messages
    # - There are no warnings in the resulting notebook file
    if not log_level:
        log_level = logging.WARNING

    # Make sure unit tests run logs do not get polluted
    # Don't touch any log levels, but
    # make sure we have logger.trading() available when
    # log_level is "disabled"
    logger = setup_logging(log_level)

    backtest_result = state_file
    if not backtest_result:
        backtest_result = Path(f"state/{id}-backtest.json")

    # State file not overridden from the command line
    if not unit_testing:
        if backtest_result.exists():
            os.remove(backtest_result)

    if not cache_path:
        cache_path = prepare_cache(id, cache_path)

    if not html_report:
        html_report = Path(f"state/{id}-backtest.html")

    if not notebook_report:
        notebook_report = Path(f"state/{id}-backtest.ipynb")

    assert trading_strategy_api_key, "Cannot start the backtest without trading_strategy_api_key - please give command line option or give TRADING_STRATEGY_API_KEY env var"

    print(f"Starting backtesting for {strategy_file}")

    def loop():
        nonlocal trading_strategy_api_key
        nonlocal cache_path
        result = run_backtest_for_module(
            strategy_file=strategy_file,
            trading_strategy_api_key=trading_strategy_api_key,
            execution_context=standalone_backtest_execution_context,
            max_workers=max_workers,
            cache_path=cache_path,
            verbose=not unit_testing,
        )
        return result

    if python_profile_report:
        import cProfile
        profiler = cProfile.Profile()
        print("Preparing to profile the backtest execution")
        result = profiler.runcall(loop)
        print(f"Writing Python profile report {python_profile_report}")
        profiler.dump_stats(python_profile_report)
    else:
        result = loop()

    state = result.state
    universe = result.strategy_universe

    # We should not be able let unnamed backtests through
    assert state.name

    print(f"Writing backtest data the state file: {backtest_result.resolve()}")
    state.write_json_file(backtest_result)

    if generate_report:

        display_backtesting_results(state, strategy_universe=universe)

        if not unit_testing:
            state2 = State.read_json_file(backtest_result)
            assert state.name == state2.name  # Early prototype serialisation checks

        print(f"Exporting report")
        print(f"Notebook: {notebook_report.resolve()}")
        print(f"HTML: {html_report.resolve()}")
        export_backtest_report(
            state,
            universe,
            output_notebook=notebook_report,
            output_html=html_report,
        )
    else:
        print("Report generation skipped")
