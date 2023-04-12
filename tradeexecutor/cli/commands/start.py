"""start command"""

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
    create_trade_execution_model, create_metadata, create_approval_model
from ..log import setup_logging, setup_discord_logging, setup_logstash_logging, setup_file_logging, \
    setup_custom_log_levels
from ..loop import ExecutionLoop
from ..result import display_backtesting_results
from ..version_info import VersionInfo
from ...ethereum.uniswap_v2.uniswap_v2_routing import UniswapV2SimpleRoutingModel
from ...state.state import State
from ...state.store import NoneStore
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
def start(

    # Strategy assets
    id: str = shared_options.id,
    log_level: str = shared_options.log_level,
    name: Optional[str] = typer.Option(None, envvar="NAME", help="Executor name used in the web interface and notifications"),
    short_description: Optional[str] = typer.Option(None, envvar="SHORT_DESCRIPTION", help="Short description for metadata"),
    long_description: Optional[str] = typer.Option(None, envvar="LONG_DESCRIPTION", help="Long description for metadata"),
    icon_url: Optional[str] = typer.Option(None, envvar="ICON_URL", help="Strategy icon for web rendering and Discord avatar"),

    strategy_file: Path = shared_options.strategy_file,

    # Live trading or backtest
    asset_management_mode: AssetManagementMode = shared_options.asset_management_mode,
    vault_address: Optional[str] = shared_options.vault_address,
    vault_adapter_address: Optional[str] = shared_options.vault_adapter_address,
    trading_strategy_api_key: str = shared_options.trading_strategy_api_key,

    # Webhook server options
    http_enabled: bool = typer.Option(False, envvar="HTTP_ENABLED", help="Enable Webhook server"),
    http_port: int = typer.Option(3456, envvar="HTTP_PORT", help="Which HTTP port to listen. The default is 3456, the default port of Pyramid web server."),
    http_host: str = typer.Option("0.0.0.0", envvar="HTTP_HOST", help="The IP address to bind for the web server. By default listen to all IP addresses available in the run-time environment."),
    http_username: str = typer.Option(None, envvar="HTTP_USERNAME", help="Username for HTTP Basic Auth protection of webhooks"),
    http_password: str = typer.Option(None, envvar="HTTP_PASSWORD", help="Password for HTTP Basic Auth protection of webhooks"),
    http_wait_good_startup_seconds: int = typer.Option(60, envvar="HTTP_WAIT_GOOD_STARTUP_SECONDS", help="How long we wait befor switching the web server mode where an exception does not bring the web server down"),

    # Web3 connection options
    json_rpc_binance: Optional[str] = shared_options.json_rpc_binance,
    json_rpc_polygon: Optional[str] = shared_options.json_rpc_polygon,
    json_rpc_ethereum: Optional[str] = shared_options.json_rpc_ethereum,
    json_rpc_avalanche: Optional[str] = shared_options.json_rpc_avalanche,
    json_rpc_arbitrum: Optional[str] = shared_options.json_rpc_arbitrum,
    json_rpc_anvil: Optional[str] = shared_options.json_rpc_anvil,

    gas_price_method: Optional[GasPriceMethod] = typer.Option(None, envvar="GAS_PRICE_METHOD", help="How to set the gas price for Ethereum transactions. After the Berlin hardfork Ethereum mainnet introduced base + tip cost gas model. Leave out to autodetect."),
    confirmation_timeout: int = typer.Option(900, envvar="CONFIRMATION_TIMEOUT", help="How many seconds to wait for transaction batches to confirm"),
    confirmation_block_count: int = typer.Option(2, envvar="CONFIRMATION_BLOCK_COUNT", help="How many blocks we wait before we consider transaction receipt a final"),
    private_key: Optional[str] = shared_options.private_key,
    minimum_gas_balance: Optional[float] = typer.Option(0.1, envvar="MINUMUM_GAS_BALANCE", help="What is the minimum balance of gas token you need to have in your wallet. If the balance falls below this, abort by crashing and do not attempt to create transactions. Expressed in the native token e.g. ETH."),

    # Logging
    discord_webhook_url: Optional[str] = typer.Option(None, envvar="DISCORD_WEBHOOK_URL", help="Discord webhook URL for notifications"),
    logstash_server: Optional[str] = typer.Option(None, envvar="LOGSTASH_SERVER", help="LogStash server hostname where to send logs"),
    file_log_level: Optional[str] = typer.Option("info", envvar="FILE_LOG_LEVEL", help="Log file log level. The default log file is logs/id.log."),

    # Debugging and unit testing
    port_mortem_debugging: bool = typer.Option(False, "--post-mortem-debugging", envvar="POST_MORTEM_DEBUGGING", help="Launch ipdb debugger on a main loop crash to debug the exception"),
    clear_caches: bool = typer.Option(False, "--clear-caches", envvar="CLEAR_CACHES", help="Purge any dataset download caches before starting"),
    unit_testing: bool = typer.Option(False, "--unit-testing", envvar="UNIT_TESTING", help="The trade executor is called under the unit testing mode. No caches are purged."),
    reset_state: bool = typer.Option(False, envvar="RESET_STATE", help="Recreate the state file. Used for testing. Same as running trade-executor init command"),
    max_cycles: int = typer.Option(None, envvar="MAX_CYCLES", help="Max main loop cycles run in an automated testing mode"),
    debug_dump_file: Optional[Path] = typer.Option(None, envvar="DEBUG_DUMP_FILE", help="Write Python Pickle dump of all internal debugging states of the strategy run to this file"),

    # Backtesting
    backtest_start: Optional[datetime.datetime] = typer.Option(None, envvar="BACKTEST_START", help="Start timestamp of backesting"),
    backtest_end: Optional[datetime.datetime] = typer.Option(None, envvar="BACKTEST_END", help="End timestamp of backesting"),
    backtest_candle_time_frame_override: Optional[TimeBucket] = typer.Option(None, envvar="BACKTEST_CANDLE_TIME_FRAME_OVERRIDE", help="Force backtests to use different candle time frame"),
    backtest_stop_loss_time_frame_override: Optional[TimeBucket] = typer.Option(None, envvar="BACKTEST_STOP_LOSS_TIME_FRAME_OVERRIDE", help="Force backtests to use different candle time frame for stop losses"),

    # Test EMV backend
    test_evm_uniswap_v2_router: Optional[str] = typer.Option(None, envvar="TEST_EVM_UNISWAP_V2_ROUTER", help="Uniswap v2 instance paramater when doing live trading test against a local dev chain"),
    test_evm_uniswap_v2_factory: Optional[str] = typer.Option(None, envvar="TEST_EVM_UNISWAP_V2_FACTORY", help="Uniswap v2 instance paramater when doing live trading test against a local dev chain"),
    test_evm_uniswap_v2_init_code_hash: Optional[str] = typer.Option(None, envvar="TEST_EVM_UNISWAP_V2_INIT_CODE_HASH", help="Uniswap v2 instance paramater when doing live trading test against a local dev chain"),

    # Live trading configuration
    max_slippage: float = typer.Option(0.0025, envvar="MAX_SLIPPAGE", help="Max slippage allowed per trade before failing. The default is 0.0025 is 0.25%."),
    approval_type: ApprovalType = typer.Option("unchecked", envvar="APPROVAL_TYPE", help="Set a manual approval flow for trades"),
    stop_loss_check_frequency: Optional[TimeBucket] = typer.Option(None, envvar="STOP_LOSS_CYCLE_DURATION", help="Override live/backtest stop loss check frequency. If not given read from the strategy module."),
    cycle_offset_minutes: int = typer.Option(8, envvar="CYCLE_OFFSET_MINUTES", help="How many minutes we wait after the tick before executing the tick step"),
    stats_refresh_minutes: int = typer.Option(60.0, envvar="STATS_REFRESH_MINUTES", help="How often we refresh position statistics. Default to once in an hour."),
    cycle_duration: CycleDuration = typer.Option(None, envvar="CYCLE_DURATION", help="How long strategy tick cycles use to execute the strategy. While strategy modules offer their own cycle duration value, you can override it here."),
    position_trigger_check_minutes: int = typer.Option(3.0, envvar="POSITION_TRIGGER_CHECK_MINUTES", help="How often we check for take profit/stop loss triggers. Default to once in 3 minutes. Set 0 to disable."),
    max_data_delay_minutes: int = typer.Option(1*60, envvar="MAX_DATA_DELAY_MINUTES", help="If our data feed is delayed more than this minutes, abort the execution. Defaults to 1 hours. Used by both strategy cycle trigger types."),
    trade_immediately: bool = typer.Option(False, "--trade-immediately", envvar="TRADE_IMMEDIATELY", help="Perform the first rebalance immediately, do not wait for the next trading universe refresh"),
    strategy_cycle_trigger: StrategyCycleTrigger = typer.Option("cycle_offset", envvar="STRATEGY_CYCLE_TRIGGER", help="How do decide when to start executing the next live trading strategy cycle"),

    # Unsorted options
    state_file: Optional[Path] = shared_options.state_file,
    cache_path: Optional[Path] = typer.Option(None, envvar="CACHE_PATH", help="Where to store downloaded datasets. This must be specific to each executor so that there are no write conflicts if multiple executors run on the same server. If not given default to cache/{executor-id}"),
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
        if asset_management_mode == AssetManagementMode.backtest:
            log_level = logging.WARNING
        else:
            log_level = logging.INFO

    # Make sure unit tests run logs do not get polluted
    # Don't touch any log levels, but
    # make sure we have logger.trading() available when
    # log_level is "disabled"
    logger = setup_logging(log_level, in_memory_buffer=True)

    if discord_webhook_url:
        setup_discord_logging(
            name,
            webhook_url=discord_webhook_url,
            avatar_url=icon_url)

    if logstash_server:
        setup_logstash_logging(
            logstash_server,
            f"executor-{id}",  # Always prefix logged with executor id
            quiet=False,
        )

    setup_file_logging(
        f"logs/{id}.log",
        file_log_level,
    )

    try:

        if not state_file:
            if asset_management_mode != AssetManagementMode.backtest:
                state_file = f"state/{id}.json"

        # Avoid polluting user caches during test runs,
        # so we use different default
        if not cache_path:
            if unit_testing:
                cache_path = Path("/tmp/trading-strategy-tests")

        cache_path = prepare_cache(id, cache_path)

        confirmation_timeout = datetime.timedelta(seconds=confirmation_timeout)

        if asset_management_mode in (AssetManagementMode.hot_wallet, AssetManagementMode.dummy, AssetManagementMode.enzyme):
            web3config = create_web3_config(
                json_rpc_binance=json_rpc_binance,
                json_rpc_polygon=json_rpc_polygon,
                json_rpc_avalanche=json_rpc_avalanche,
                json_rpc_ethereum=json_rpc_ethereum,
                json_rpc_anvil=json_rpc_anvil,
                json_rpc_arbitrum=json_rpc_arbitrum,
                gas_price_method=gas_price_method,
            )

            if not web3config.has_any_connection():
                raise RuntimeError("Live trading requires that you pass JSON-RPC connection to one of the networks")
        else:
            web3config = None

        # TODO: This strategy file is reloaded again in ExecutionLoop.run()
        # We do an extra hop here, because we need to know chain_id associated with the strategy,
        # because there is an inversion of control issue for passing web3 connection around.
        # Clean this up in the future versions, by changing the order of initialzation.
        mod = read_strategy_module(strategy_file)

        if web3config is not None:

            if isinstance(mod, StrategyModuleInformation):
                # This path is not enabled for legacy strategy modules
                if mod.chain_id:
                    # Strategy tells what chain to use
                    web3config.set_default_chain(mod.chain_id)
                    web3config.check_default_chain_id()
                else:
                    # User has configured only one chain, use it
                    web3config.choose_single_chain()

            else:
                # Legacy unit testing path.
                # All chain_ids are 56 (BNB Chain)
                logger.warning("Legacy strategy module: makes assumption of BNB Chain")
                web3config.set_default_chain(ChainId.bsc)

        if minimum_gas_balance:
            minimum_gas_balance = Decimal(minimum_gas_balance)

        if unit_testing:
            # Do not let Ganache to wait for too long
            # because likely Ganache has simply crashed on background
            confirmation_timeout = datetime.timedelta(seconds=30)

        execution_model, sync_model, valuation_model_factory, pricing_model_factory = create_trade_execution_model(
            asset_management_mode,
            private_key,
            web3config,
            confirmation_timeout,
            confirmation_block_count,
            max_slippage,
            minimum_gas_balance,
            vault_address,
            vault_adapter_address,
        )

        approval_model = create_approval_model(approval_type)

        if state_file:
            store = create_state_store(Path(state_file))
        else:
            # Backtests do not have persistent state
            if asset_management_mode == AssetManagementMode.backtest:
                logger.info("This backtest run won't create a state file")
                store = NoneStore(State())
            else:
                raise RuntimeError("Does not know how to set up a state file for this run")

        metadata = create_metadata(name, short_description, long_description, icon_url)

        # Start the queue that relays info from the web server to the strategy executor
        command_queue = Queue()

        run_state = RunState()
        run_state.version = VersionInfo.read_docker_version()
        run_state.executor_id = id

        # Create our webhook server
        if http_enabled:
            server = create_webhook_server(
                http_host,
                http_port,
                http_username,
                http_password,
                command_queue,
                store,
                metadata,
                run_state,
            )
        else:
            logger.info("Web server disabled")
            server = None

        # Routing model comes usually from the strategy and hard-coded blockchain defaults,
        # but for local dev chains it is dynamically constructed from the deployed contracts
        routing_model: RoutingModel = None

        # Create our data client
        if test_evm_uniswap_v2_factory:
            # Running against a local dev chain
            client = UniswapV2MockClient(
                web3config.get_default(),
                test_evm_uniswap_v2_factory,
                test_evm_uniswap_v2_router,
                test_evm_uniswap_v2_init_code_hash,
            )

            if mod.trade_routing == TradeRouting.user_supplied_routing_model:
                routing_model = UniswapV2SimpleRoutingModel(
                    factory_router_map={test_evm_uniswap_v2_factory: (test_evm_uniswap_v2_router, test_evm_uniswap_v2_init_code_hash)},
                    allowed_intermediary_pairs={},
                    reserve_token_address=client.get_default_quote_token_address(),
                )

        elif trading_strategy_api_key:
            # Backtest / real trading
            client = Client.create_live_client(trading_strategy_api_key, cache_path=cache_path)
            if clear_caches:
                client.clear_caches()
        else:
            # This run does not need to dowwnload any data
            client = None

        # Currently, all actions require us to have a valid API key
        # might change in the future
        if not client:
            raise RuntimeError("Trading Strategy client instance is not available - needed to run backtests. Make sure trading_strategy_api_key is set.")

        tick_offset = datetime.timedelta(minutes=cycle_offset_minutes)

        if max_data_delay_minutes:
            max_data_delay = datetime.timedelta(minutes=max_data_delay_minutes)
        else:
            max_data_delay = None

        stats_refresh_frequency = datetime.timedelta(minutes=stats_refresh_minutes)
        position_trigger_check_frequency = datetime.timedelta(minutes=position_trigger_check_minutes)

        logger.info("Loading strategy file %s", strategy_file)
        strategy_factory = import_strategy_file(strategy_file)

        logger.trade("%s: trade execution starting", name)

        if backtest_start:
            # Running as a backtest
            execution_context = ExecutionContext(
                mode=ExecutionMode.backtesting,
                timed_task_context_manager=timed_task,
            )
        else:
            if unit_testing:
                execution_context = ExecutionContext(
                    mode=ExecutionMode.unit_testing_trading,
                    timed_task_context_manager=timed_task,
                )
            else:
                execution_context = ExecutionContext(
                    mode=ExecutionMode.real_trading,
                    timed_task_context_manager=timed_task,
                )
    except Exception as e:
        # Logging is set up is in this point, so we can log this exception that
        # caused the start up to fail
        logger.critical("Startup failed: %s", e)
        logger.exception(e)
        raise

    try:
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
        )
        loop.run()

        # Display summary stats for terminal backtest runs
        if asset_management_mode == AssetManagementMode.backtest and isinstance(store, NoneStore):
            display_backtesting_results(store.state)

    except KeyboardInterrupt as e:
        # CTRL+C shutdown
        logger.trade("Trade Executor %s shut down by CTRL+C requested: %s", name, e)
    except Exception as e:

        logger.error("trade-executor execution loop crashed")

        # Unwind the traceback and notify the webserver about the failure
        run_state.set_fail()

        # Debug exceptions in production
        if port_mortem_debugging:
            import ipdb
            ipdb.post_mortem()

        logger.exception(e)

        running_time = datetime.datetime.utcnow() - started_at

        if (server is None) or (running_time < datetime.timedelta(seconds=http_wait_good_startup_seconds)):
            # Only terminate the process if the webhook server is not running,
            # otherwise the user can read the crash status from /status endpoint
            logger.error("Raising the error and crashing away, running time was %s", running_time)
            raise
        else:
            # Execution is dea  d.
            # Sleep forever, let the webhook still serve the requests.
            logger.error("Main loop terminated. Entering to the web server wait mode.")
            time.sleep(3600*24*365)
    finally:
        if server:
            logger.info("Closing the web server")
            server.close()
