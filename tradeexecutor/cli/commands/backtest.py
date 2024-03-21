"""backtest CLI command

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

    # Backtest already requires an API key
    trading_strategy_api_key: str = required_option(shared_options.trading_strategy_api_key),

    log_level: str = shared_options.log_level,

    # Debugging and unit testing
    unit_testing: bool = shared_options.unit_testing,

    # Unsorted options
    backtest_result: Optional[Path] = shared_options.backtest_result,
    cache_path: Optional[Path] = shared_options.cache_path,

    notebook_report: Optional[Path] = shared_options.notebook_report,
    html_report: Optional[Path] = shared_options.html_report,
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

    # Make sure
    # - We do not flood console with the messages
    # - There are no warnings in the resulting notebook file
    if not log_level:
        log_level = logging.ERROR

    # Make sure unit tests run logs do not get polluted
    # Don't touch any log levels, but
    # make sure we have logger.trading() available when
    # log_level is "disabled"
    logger = setup_logging(log_level)

    if not unit_testing:
        state_file = Path(f"state/{id}-backtest.json")
        if state_file.exists():
            os.remove(state_file)

    if not cache_path:
        cache_path = prepare_cache(id, cache_path)

    if not html_report:
        html_report = Path(f"state/{id}-backtest.html")

    if not notebook_report:
        notebook_report = Path(f"state/{id}-backtest.ipynb")

    if not backtest_result:
        backtest_result = Path(f"state/{id}-backtest.json")

    print(f"Starting backtesting for {strategy_file}")

    mod = read_strategy_module(strategy_file)

    assert mod.is_version_greater_or_equal_than(0, 2, 0), f"trading_strategy_engine_version must be 0.2.0 or newer for {strategy_file}"

    client = Client.create_live_client(
        trading_strategy_api_key,
        cache_path=cache_path,
        settings_path=None,  # Disable settings file on backtest
    )

    if mod.name:
        # Overwrite from the module
        name = mod.name

    universe_options = mod.get_universe_options()

    logger.info("Loading backtesting universe data for %s", universe_options)

    universe = mod.create_trading_universe(
        datetime.datetime.utcnow(),
        client,
        standalone_backtest_execution_context,
        universe_options,
    )

    initial_cash = mod.initial_cash
    assert initial_cash is not None, f"Strategy module does not set initial_cash needed to backtest"

    assert mod.backtest_start, f"Strategy module does not set backtest_start"
    assert mod.backtest_end, f"Strategy module does not set backtest_end"

    # Don't start at T+0 because we have not any data for that day yet
    backtest_start_at = universe_options.start_at + mod.trading_strategy_cycle.to_timedelta()
    logger.info("Backtest starts at %s", backtest_start_at)

    indicator_storage = DiskIndicatorStorage(cache_path / "indicators", universe_key=universe.get_cache_key())

    backtest_setup = setup_backtest_for_universe(
        mod,
        start_at=backtest_start_at,
        end_at=mod.backtest_end,
        cycle_duration=mod.trading_strategy_cycle,
        initial_deposit=initial_cash,
        name=name,
        universe=universe,
        universe_options=universe_options,
        create_indicators=mod.create_indicators,
        parameters=mod.parameters,
        indicator_storage=indicator_storage,
    )

    assert backtest_setup.trading_strategy_engine_version
    assert backtest_setup.name

    state, universe, debug_data = run_backtest(
        backtest_setup,
        client=client,
    )

    # We should not be able let unnamed backtests through
    assert state.name

    display_backtesting_results(state, universe=universe)

    print(f"Writing backtest data the state file: {backtest_result.resolve()}")
    state.write_json_file(backtest_result)
    state2 = State.read_json_file(backtest_result)
    assert state.name == state2.name  # Early prototype serialisation checks

    print(f"Exporting report, notebook: {notebook_report}, HTML: {html_report}")
    export_backtest_report(
        state,
        universe,
        output_notebook=notebook_report,
        output_html=html_report,
    )

