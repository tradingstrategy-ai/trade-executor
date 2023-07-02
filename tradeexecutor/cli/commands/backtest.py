"""bacltest command

"""

import datetime
import logging
import os
import time
from decimal import Decimal
from pathlib import Path
from queue import Queue
from typing import Optional

import typer

from eth_defi.gas import GasPriceMethod
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.testing.uniswap_v2_mock_client import UniswapV2MockClient
from tradingstrategy.timebucket import TimeBucket
from typer import Option

from . import shared_options
from .app import app, TRADE_EXECUTOR_VERSION
from ..bootstrap import prepare_executor_id, prepare_cache, create_web3_config, create_state_store, \
    create_execution_and_sync_model, create_metadata, create_approval_model, create_client
from ..log import setup_logging, setup_discord_logging, setup_logstash_logging, setup_file_logging, \
    setup_custom_log_levels
from ..loop import ExecutionLoop
from ..result import display_backtesting_results
from ..version_info import VersionInfo
from ..watchdog import stop_watchdog
from ...backtest.backtest_runner import setup_backtest, run_backtest
from ...ethereum.enzyme.vault import EnzymeVaultSyncModel
from ...ethereum.uniswap_v2.uniswap_v2_routing import UniswapV2SimpleRoutingModel
from ...state.state import State
from ...state.store import NoneStore, JSONFileStore
from ...strategy.approval import ApprovalType
from ...strategy.bootstrap import import_strategy_file
from ...strategy.cycle import CycleDuration
from ...strategy.default_routing_options import TradeRouting
from ...strategy.execution_context import ExecutionContext, ExecutionMode
from ...strategy.execution_model import AssetManagementMode
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

    trading_strategy_api_key: str = shared_options.trading_strategy_api_key,

    log_level: str = shared_options.log_level,

    # Debugging and unit testing
    unit_testing: bool = shared_options.unit_testing,

    # Unsorted options
    backtest_result: Optional[Path] = shared_options.backtest_result,
    cache_path: Optional[Path] = shared_options.cache_path,

    notebook_report: Optional[Path] = Option(None, envvar="NOTEBOOK_REPORT", help="Jupyter Notebook file where the store the notebook results. If not given defaults to state/{executor-id}-backtest.ipynb."),
    html_report: Optional[Path] = Option(None, envvar="HTML_REPORT", help="HTML file where the store the notebook results. If not given defaults to state/{executor-id}-backtest.html."),
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

    # We always need a name-*-
    if not name:
        if strategy_file:
            name = os.path.basename(strategy_file)
        else:
            name = "Unnamed backtest"

    if not log_level:
        log_level = logging.WARNING

    # Make sure unit tests run logs do not get polluted
    # Don't touch any log levels, but
    # make sure we have logger.trading() available when
    # log_level is "disabled"
    logger = setup_logging(log_level)

    if not unit_testing:
        state_file = Path(f"state/{id}-backtest.json")
        if state_file.exists():
            os.remove(state_file)

    # Avoid polluting user caches during test runs,
    # so we use different default
    if not cache_path:
        if unit_testing:
            cache_path = Path("/tmp/trading-strategy-tests")

    cache_path = prepare_cache(id, cache_path)

    if not html_report:
        html_report = Path(f"state/{id}-backtest.html")

    if not notebook_report:
        notebook_report = Path(f"state/{id}-backtest.ipynb")

    if not backtest_result:
        backtest_result = Path(f"state/{id}-backtest.json")

    # TODO: This strategy file is reloaded again in ExecutionLoop.run()
    # We do an extra hop here, because we need to know chain_id associated with the strategy,
    # because there is an inversion of control issue for passing web3 connection around.
    # Clean this up in the future versions, by changing the order of initialzation.
    mod = read_strategy_module(strategy_file)

    client = Client.create_live_client(trading_strategy_api_key, cache_path=cache_path)

    backtest_setup = setup_backtest(
        strategy_file,
        start_at=mod.backtest_start,
        end_at=mod.backtest_end,
        initial_deposit=mod.initial_cash,
        strategy_module=mod,
    )

    state, universe, debug_data = run_backtest(
        backtest_setup,
        client=client,
    )

    display_backtesting_results(state)

    state.write_json_file(backtest_result)



