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
    state_file: Optional[Path] = shared_options.state_file,
    cache_path: Optional[Path] = shared_options.cache_path,
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

    # TODO: This strategy file is reloaded again in ExecutionLoop.run()
    # We do an extra hop here, because we need to know chain_id associated with the strategy,
    # because there is an inversion of control issue for passing web3 connection around.
    # Clean this up in the future versions, by changing the order of initialzation.
    mod = read_strategy_module(strategy_file)

    execution_model, sync_model, valuation_model_factory, pricing_model_factory = create_execution_and_sync_model(
        asset_management_mode=asset_management_mode,
        private_key=private_key,
        web3config=web3config,
        confirmation_timeout=confirmation_timeout,
        confirmation_block_count=confirmation_block_count,
        max_slippage=max_slippage,
        min_gas_balance=min_gas_balance,
        vault_address=vault_address,
        vault_adapter_address=vault_adapter_address,
        vault_payment_forwarder_address=vault_payment_forwarder_address,
        routing_hint=mod.trade_routing,
    )

    if state_file:
        store = create_state_store(Path(state_file))
    else:
        # Backtests do not have persistent state
        if asset_management_mode == AssetManagementMode.backtest:
            logger.info("This backtest run won't create a state file")
            store = NoneStore(State())
        else:
            raise RuntimeError("Does not know how to set up a state file for this run")

    client = client = Client.create_live_client(trading_strategy_api_key, cache_path=cache_path)

    # We cannot have real-time triggered trades when doing backtesting
    strategy_cycle_trigger = StrategyCycleTrigger.cycle_offset

    # Running as a backtest
    execution_context = ExecutionContext(
        mode=ExecutionMode.backtesting,
        timed_task_context_manager=timed_task,
    )

    loop = ExecutionLoop(
        name=name,
        command_queue=command_queue,
        execution_model=execution_model,
        execution_context=execution_context,
        sync_model=sync_model,
        approval_model=approval_model,
        pricing_model_factory=pricing_model_factory,
        valuation_model_factory=valuation_model_factory,
        store=store,
        client=client,
        strategy_factory=strategy_factory,
        reset=reset_state,
        max_cycles=max_cycles,
        debug_dump_file=debug_dump_file,
        backtest_start=backtest_start,
        backtest_end=backtest_end,
        backtest_stop_loss_time_frame_override=backtest_stop_loss_time_frame_override,
        backtest_candle_time_frame_override=backtest_candle_time_frame_override,
        stop_loss_check_frequency=stop_loss_check_frequency,
        cycle_duration=cycle_duration,
        tick_offset=tick_offset,
        max_data_delay=max_data_delay,
        trade_immediately=trade_immediately,
        stats_refresh_frequency=stats_refresh_frequency,
        position_trigger_check_frequency=position_trigger_check_frequency,
        run_state=run_state,
        strategy_cycle_trigger=strategy_cycle_trigger,
        routing_model=routing_model,
        metadata=metadata,
    )

    state = loop.setup()
    loop.run_with_state(state)
    display_backtesting_results(store.state)
