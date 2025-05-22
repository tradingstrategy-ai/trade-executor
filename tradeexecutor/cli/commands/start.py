"""start command

TODO: Restructure and move backtesting related functionality to a separate command
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

from eth_defi.confirmation import ConfirmationTimedOut
from eth_defi.gas import GasPriceMethod
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket

from . import shared_options
from .app import app
from ..bootstrap import prepare_executor_id, prepare_cache, create_web3_config, create_state_store, \
    create_execution_and_sync_model, create_metadata, create_approval_model, create_client
from ..log import setup_logging, setup_discord_logging, setup_logstash_logging, setup_file_logging, setup_telegram_logging, setup_sentry_logging
from ..loop import ExecutionLoop
from ..result import display_backtesting_results
from ..slippage import configure_max_slippage_tolerance
from ..version_info import VersionInfo
from ..watchdog import stop_watchdog
from ...ethereum.enzyme.vault import EnzymeVaultSyncModel
from ...ethereum.lagoon.vault import LagoonVaultSyncModel
from ...ethereum.velvet.execution import VelvetExecution
from ...ethereum.velvet.vault import VelvetVaultSyncModel
from ...state.state import State
from ...state.store import NoneStore, JSONFileStore
from ...strategy.approval import ApprovalType
from ...strategy.bootstrap import import_strategy_file
from ...strategy.cycle import CycleDuration
from ...strategy.execution_context import ExecutionContext, ExecutionMode
from ...strategy.execution_model import AssetManagementMode, ExecutionHaltableIssue
from ...strategy.parameters import dump_parameters
from ...strategy.routing import RoutingModel
from ...strategy.run_state import RunState
from ...strategy.strategy_cycle_trigger import StrategyCycleTrigger
from ...strategy.strategy_module import read_strategy_module, StrategyModuleInformation
from ...strategy.universe_model import UniverseOptions
from ...utils.timer import timed_task

try:
    from ...webhook.server import create_webhook_server
except ImportError as e:
    create_webhook_server = None


logger = logging.getLogger(__name__)


@app.command()
def start(

    # Strategy assets
    id: str = shared_options.id,
    name: Optional[str] = shared_options.name,
    short_description: Optional[str] = typer.Option(None, envvar="SHORT_DESCRIPTION", help="Short description for metadata"),
    long_description: Optional[str] = typer.Option(None, envvar="LONG_DESCRIPTION", help="Long description for metadata"),
    badges: Optional[str] = typer.Option(None, envvar="BADGES", help="Comma separated list of badges to be displayed on the strategy tile"),
    icon_url: Optional[str] = typer.Option(None, envvar="ICON_URL", help="Strategy icon for web rendering and Discord avatar"),

    strategy_file: Path = shared_options.strategy_file,

    # Live trading or backtest
    asset_management_mode: AssetManagementMode = shared_options.asset_management_mode,
    vault_address: Optional[str] = shared_options.vault_address,
    vault_adapter_address: Optional[str] = shared_options.vault_adapter_address,
    vault_payment_forwarder_address: Optional[str] = shared_options.vault_payment_forwarder,
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
    json_rpc_base: Optional[str] = shared_options.json_rpc_base,
    json_rpc_arbitrum: Optional[str] = shared_options.json_rpc_arbitrum,
    json_rpc_anvil: Optional[str] = shared_options.json_rpc_anvil,

    gas_price_method: Optional[GasPriceMethod] = typer.Option(None, envvar="GAS_PRICE_METHOD", help="How to set the gas price for Ethereum transactions. After the Berlin hardfork Ethereum mainnet introduced base + tip cost gas model. Leave out to autodetect."),
    confirmation_block_count: int = shared_options.confirmation_block_count,
    confirmation_timeout: int = shared_options.confirmation_timeout,
    private_key: Optional[str] = shared_options.private_key,
    min_gas_balance: Optional[float] = shared_options.min_gas_balance,
    gas_balance_warning_level: Optional[float] = typer.Option(25.0, envvar="GAS_BALANCE_WARNING_LEVEL", help="If hot wallet gas level falls below this amount of tokens, issue a low gas warning."),

    # Logging
    log_level: str = shared_options.log_level,
    discord_webhook_url: Optional[str] = shared_options.discord_webhook_url,
    logstash_server: Optional[str] = shared_options.logstash_server,
    file_log_level: Optional[str] = shared_options.file_log_level,
    telegram_api_key: Optional[str] = shared_options.telegram_api_key,
    telegram_chat_id: Optional[str] = shared_options.telegram_chat_id,
    sentry_dsn: Optional[str] = shared_options.sentry_dsn,

    # Debugging and unit testing
    port_mortem_debugging: bool = typer.Option(False, "--post-mortem-debugging", envvar="POST_MORTEM_DEBUGGING", help="Launch ipdb debugger on a main loop crash to debug the exception"),
    clear_caches: bool = typer.Option(False, "--clear-caches", envvar="CLEAR_CACHES", help="Purge any dataset download caches before starting"),
    unit_testing: bool = shared_options.unit_testing,
    reset_state: bool = typer.Option(False, envvar="RESET_STATE", help="Recreate the state file. Used for testing. Same as running trade-executor init command"),
    max_cycles: int = typer.Option(None, envvar="MAX_CYCLES", help="Max main loop cycles run in an automated testing mode"),
    debug_dump_file: Optional[Path] = typer.Option(None, envvar="DEBUG_DUMP_FILE", help="Write Python Pickle dump of all internal debugging states of the strategy run to this file"),

    # Backtesting
    backtest_start: Optional[datetime.datetime] = typer.Option(None, envvar="BACKTEST_START", help="Start timestamp of backesting"),
    backtest_end: Optional[datetime.datetime] = typer.Option(None, envvar="BACKTEST_END", help="End timestamp of backesting"),
    backtest_candle_time_frame_override: Optional[TimeBucket] = typer.Option(None, envvar="BACKTEST_CANDLE_TIME_FRAME_OVERRIDE", help="Force backtests to use different candle time frame"),
    backtest_stop_loss_time_frame_override: Optional[TimeBucket] = typer.Option(None, envvar="BACKTEST_STOP_LOSS_TIME_FRAME_OVERRIDE", help="Force backtests to use different candle time frame for stop losses"),

    # Test EVM backend when running e2e tests
    test_evm_uniswap_v2_router: Optional[str] = shared_options.test_evm_uniswap_v2_router,
    test_evm_uniswap_v2_factory: Optional[str] = shared_options.test_evm_uniswap_v2_factory,
    test_evm_uniswap_v2_init_code_hash: Optional[str] = shared_options.test_evm_uniswap_v2_init_code_hash,

    # Live trading configuration
    max_slippage: float = shared_options.max_slippage,
    approval_type: ApprovalType = typer.Option("unchecked", envvar="APPROVAL_TYPE", help="Set a manual approval flow for trades"),
    stop_loss_check_frequency: Optional[TimeBucket] = typer.Option(None, envvar="STOP_LOSS_CYCLE_DURATION", help="Override live/backtest stop loss check frequency. If not given read from the strategy module."),
    cycle_offset_minutes: int = typer.Option(8, envvar="CYCLE_OFFSET_MINUTES", help="How many minutes we wait after the tick before executing the tick step"),
    stats_refresh_minutes: int = typer.Option(60.0, envvar="STATS_REFRESH_MINUTES", help="How often we refresh position statistics. Default to once in an hour."),
    cycle_duration: CycleDuration = typer.Option(None, envvar="CYCLE_DURATION", help="How long strategy tick cycles use to execute the strategy. While strategy modules offer their own cycle duration value, you can override it here for unit testing."),
    position_trigger_check_minutes: int = typer.Option(3.0, envvar="POSITION_TRIGGER_CHECK_MINUTES", help="How often we check for take profit/stop loss triggers. Default to once in 3 minutes. Set 0 to disable."),
    max_data_delay_minutes: int = typer.Option(1*60, envvar="MAX_DATA_DELAY_MINUTES", help="If our data feed is delayed more than this minutes, abort the execution. Defaults to 1 hours. Used by both strategy cycle trigger types."),
    trade_immediately: bool = typer.Option(False, "--trade-immediately", envvar="TRADE_IMMEDIATELY", help="Perform the first rebalance immediately, do not wait for the next trading universe refresh"),
    strategy_cycle_trigger: StrategyCycleTrigger = typer.Option("cycle_offset", envvar="STRATEGY_CYCLE_TRIGGER", help="How do decide when to start executing the next live trading strategy cycle"),
    key_metrics_backtest_cut_off_days: float = typer.Option(90, envvar="KEY_METRIC_BACKTEST_CUT_OFF_DAYS", help="How many days live data is collected until key metrics are switched from backtest to live trading based"),
    check_accounts: bool = typer.Option(True, "--check-accounts", envvar="CHECK_ACCOUNTS", help="Do extra accounting checks to track mismatch balances"),
    sync_treasury_on_startup: bool = typer.Option(True, "--sync-treasury-on-startup", envvar="SYNC_TREASURY_ON_STARTUP", help="Sync treasury events before starting any trading"),
    visualisation: bool = typer.Option(True, "--visualisation", envvar="VISUALISATION", help="Disable generation of charts using Kaleido library. Helps with issues with broken installations"),

    run_single_cycle: bool = typer.Option(False, "--run-single-cycle", envvar="RUN_SINGLE_CYCLE", help="Run a single strategy decision cycle and exist, regardless of the current pending state."),

    simulate: bool = shared_options.simulate,

    # Various file configurations
    state_file: Optional[Path] = shared_options.state_file,
    backtest_result: Optional[Path] = shared_options.backtest_result,
    notebook_report: Optional[Path] = shared_options.notebook_report,
    html_report: Optional[Path] = shared_options.html_report,
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

    assert asset_management_mode, f"ASSET_MANAGEMENT_MODE must given, options are: {[member.name for member in AssetManagementMode]}"

    if not log_level:
        if asset_management_mode == AssetManagementMode.backtest:
            log_level = logging.WARNING
        else:
            log_level = logging.INFO

    # Make sure unit tests run logs do not get polluted
    # Don't touch any log levels, but
    # make sure we have logger.trading() available when
    # log_level is "disabled"
    logger = setup_logging(
        log_level,
        in_memory_buffer=True,
        enable_trade_high=True,
    )

    if backtest_start or backtest_end:
        # Disable legacy backtest method
        logger.error("start --backtest-start or --backtest-end are no longer supported.")
        logger.error("Please use separate backtest command instead of start command.")
        raise NotImplementedError()

    if not (unit_testing or simulate) and asset_management_mode.is_live_trading():
        if discord_webhook_url:
            # TODO: Move backtesting to its own console command
            setup_discord_logging(
                name,
                webhook_url=discord_webhook_url,
                avatar_url=icon_url)

        if telegram_api_key:
            setup_telegram_logging(
                telegram_api_key,
                telegram_chat_id,
            )

        if logstash_server:
            logger.info("Enabling Logstash logging to %s", logstash_server)
            setup_logstash_logging(
                logstash_server,
                f"executor-{id}",  # Always prefix logged with executor id
                quiet=False,
            )
        else:
            logger.info("Logstash logging disabled")

        if sentry_dsn:
            setup_sentry_logging(
                application_name=name,
                sentry_dsn=sentry_dsn,
            )

    setup_file_logging(
        f"logs/{id}.log",
        file_log_level,
        http_logging=True,
    )

    try:

        if not state_file:
            if asset_management_mode != AssetManagementMode.backtest:
                state_file = f"state/{id}.json"
            else:
                # Backtest generates a state file for the web frontend
                # TODO: Avoid legacy unit test issues
                if not unit_testing:
                    state_file = Path(f"state/{id}-backtest.json")
                    if state_file.exists():
                        os.remove(state_file)

        cache_path = prepare_cache(id, cache_path, unit_testing)

        if asset_management_mode.is_live_trading() or asset_management_mode == AssetManagementMode.dummy:
            web3config = create_web3_config(
                json_rpc_binance=json_rpc_binance,
                json_rpc_polygon=json_rpc_polygon,
                json_rpc_avalanche=json_rpc_avalanche,
                json_rpc_ethereum=json_rpc_ethereum,
                json_rpc_base=json_rpc_base,
                json_rpc_anvil=json_rpc_anvil,
                json_rpc_arbitrum=json_rpc_arbitrum,
                gas_price_method=gas_price_method,
                unit_testing=unit_testing,
                simulate=simulate,
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

        # Overwrite name, short and long descriptions from the strategy file
        # and ignore legacy env config
        if mod.name:
            name = mod.name
        if mod.short_description:
            short_description = mod.short_description
        if mod.long_description:
            long_description = mod.long_description
        if mod.icon:
            icon_url = mod.icon

        if web3config is not None:

            if isinstance(mod, StrategyModuleInformation):
                # This path is not enabled for legacy strategy modules
                if mod.get_default_chain_id():
                    # Strategy tells what chain to use
                    web3config.set_default_chain(mod.get_default_chain_id())
                    web3config.check_default_chain_id()
                else:
                    # User has configured only one chain, use it
                    web3config.choose_single_chain()

            else:
                # Legacy unit testing path.
                # All chain_ids are 56 (BNB Chain)
                logger.warning("Legacy strategy module: makes assumption of BNB Chain")
                web3config.set_default_chain(ChainId.bsc)

        if min_gas_balance:
            min_gas_balance = Decimal(min_gas_balance)

        if confirmation_timeout == 60 and web3config.get_default().eth.chain_id == 1:
            # TODO: Haack
            # Ethereum mainnet needs much longer confirmation timeout,
            # don't let the default 60 seconds slip in
            confirmation_timeout = 900

        confirmation_timeout = datetime.timedelta(seconds=confirmation_timeout)

        if unit_testing:
            # Do not let Ganache to wait for too long
            # because likely Ganache has simply crashed on background
            confirmation_timeout = datetime.timedelta(seconds=60)

        logger.info("Transaction confirmation timeout set to %s", confirmation_timeout)

        max_slippage = configure_max_slippage_tolerance(max_slippage, mod)

        assert web3config is not None, "Web3 RPC configuration failed?"

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

        approval_model = create_approval_model(approval_type)

        if state_file:
            store = create_state_store(Path(state_file), simulate=simulate)
        else:
            # Backtests do not have persistent state
            if asset_management_mode == AssetManagementMode.backtest:
                logger.info("This backtest run won't create a state file")
                store = NoneStore(State())
            else:
                raise RuntimeError("Does not know how to set up a state file for this run")

        # Pass vault metadata to HTTP API
        if asset_management_mode.is_vault():
            match sync_model:
                case EnzymeVaultSyncModel():
                    vault = sync_model.vault
                case VelvetVaultSyncModel():
                    vault = sync_model.vault
                case LagoonVaultSyncModel():
                    vault = sync_model.vault
                case _:
                    raise NotImplementedError(f"Vault not implemented: {asset_management_mode}")
        else:
            vault = None

        if http_enabled:
            # We need to have the results from the previous backtest run
            # to be used with the web frontend
            if asset_management_mode.is_live_trading() and not unit_testing:
                if not backtest_result:
                    backtest_result = Path(f"state/{id}-backtest.json")

                if not backtest_result.exists():
                    logger.warning(f"Previous backtest results are needed to show them on the web.\n" 
                                   f"The BACKTEST_RESULT file {backtest_result.absolute()} does not exist.")
                    backtest_result = None

        if not html_report and backtest_result:
            html_report = Path(f"state/{id}-backtest.html")

        if not notebook_report and backtest_result:
            notebook_report = Path(f"state/{id}-backtest.ipynb")

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
            asset_management_mode,
            chain_id=mod.get_default_chain_id(),
            vault=vault,
            backtest_result=backtest_result,
            backtest_notebook=notebook_report,
            backtest_html=html_report,
            key_metrics_backtest_cut_off_days=key_metrics_backtest_cut_off_days,
            badges=badges,
            tags=mod.tags,
            hot_wallet=sync_model.get_hot_wallet(),
            sort_priority=mod.sort_priority,
            fees=fees,
        )

        # Start the queue that relays info from the web server to the strategy executor
        command_queue = Queue()

        run_state = RunState()
        run_state.version = VersionInfo.read_docker_version()
        run_state.executor_id = id
        run_state.hot_wallet_gas_warning_level = Decimal(gas_balance_warning_level)

        # Set up read-only state sync
        if not store.is_pristine():
            run_state.read_only_state_copy = store.load()
        store.on_save = run_state.on_save_hook

        # Create our webhook server
        if http_enabled:

            assert create_webhook_server is not None, "Could not load tradeexecutor.webhook.server: check all extra packages have been installed"

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

        client, routing_model = create_client(
            mod=mod,
            web3config=web3config,
            trading_strategy_api_key=trading_strategy_api_key,
            cache_path=cache_path,
            test_evm_uniswap_v2_factory=test_evm_uniswap_v2_factory,
            test_evm_uniswap_v2_router=test_evm_uniswap_v2_router,
            test_evm_uniswap_v2_init_code_hash=test_evm_uniswap_v2_init_code_hash,
            clear_caches=clear_caches,
            asset_management_mode=asset_management_mode,
        )

        # Currently, all actions require us to have a valid API key
        # might change in the future
        if not client:
            raise RuntimeError("Trading Strategy API key needed. Make sure to give --trading-strategy-api-key or set TRADING_STRATEGY_API_KEY env.")

        tick_offset = datetime.timedelta(minutes=cycle_offset_minutes)

        if max_data_delay_minutes:
            max_data_delay = datetime.timedelta(minutes=max_data_delay_minutes)
            logger.info(f"Maximum price feed delay is {max_data_delay}")
        else:
            logger.info(f"Maximum price feed delay is not set")
            max_data_delay = None

        stats_refresh_frequency = datetime.timedelta(minutes=stats_refresh_minutes)
        position_trigger_check_frequency = datetime.timedelta(minutes=position_trigger_check_minutes)

        logger.info("Loading strategy file %s", strategy_file)
        strategy_factory = import_strategy_file(strategy_file)

        logger.trade("%s (%s): trade execution starting", name, id)

        universe_options = None
        if backtest_start:

            assert asset_management_mode == AssetManagementMode.backtest, f"Expected backtest mode, got {asset_management_mode}"

            # We cannot have real-time triggered trades when doing backtestin
            strategy_cycle_trigger = StrategyCycleTrigger.cycle_offset

            # Running as a backtest
            execution_context = ExecutionContext(
                mode=ExecutionMode.backtesting,
                timed_task_context_manager=timed_task,
                engine_version=mod.trading_strategy_engine_version,
            )
        else:
            if unit_testing:
                execution_context = ExecutionContext(
                    mode=ExecutionMode.unit_testing_trading,
                    timed_task_context_manager=timed_task,
                    engine_version=mod.trading_strategy_engine_version,
                )
            else:
                # Live trading
                execution_context = ExecutionContext(
                    mode=ExecutionMode.real_trading,
                    timed_task_context_manager=timed_task,
                    engine_version=mod.trading_strategy_engine_version,
                )

            if mod.parameters:
                universe_options = UniverseOptions.from_strategy_parameters_class(mod.parameters, execution_context)
                logger.info("UniverseOptions set to: %s", universe_options)

        if mod.parameters:
            logger.trade(
                "Starting with strategy parameters:\n%s",
                dump_parameters(mod.parameters)
            )

        logger.info(
            "Starting %s, with execution mode: %s, unit testing is %s, version is %s",
            name,
            execution_context.mode.name,
            unit_testing,
            execution_context.engine_version,
        )

    except Exception as e:
        # Logging is set up is in this point, so we can log this exception that
        # caused the start up to fail
        logger.critical("Startup failed: %s", e)
        logger.exception(e)
        raise

    # Allow to do Python thread dump using a signal with
    # docker-compose kill command
    if not unit_testing:
        faulthandler.enable()

    if simulate:
        if not run_single_cycle:
            raise RuntimeError("Simulation mode is only supported with --run-single-cycle at the moment")
        
        logger.info("Simulating single cycle")

    # Force run a single cycle
    if run_single_cycle:
        max_cycles = 1
        sync_treasury_on_startup = True
        trade_immediately = True
    else:
        logger.info("No immediate trade set up, max cycles is %s", max_cycles)

    # Trip wire for Velvet integration, as Velvet needs its special Enso path
    if asset_management_mode == AssetManagementMode.velvet:
        assert isinstance(execution_model, VelvetExecution), f"Got: {execution_model}"
        assert isinstance(sync_model, VelvetVaultSyncModel), f"Got: {sync_model}"
        assert routing_model is None, f"Got: {routing_model}"

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
        check_accounts=check_accounts,
        sync_treasury_on_startup=sync_treasury_on_startup,
        create_indicators=mod.create_indicators,
        parameters=mod.parameters,
        visualisation=visualisation,
        max_price_impact=mod.get_max_price_impact(),
        universe_options=universe_options,
    )

    # Crash gracefully at the start up if our main loop cannot set itself up
    try:
        state = loop.setup()
    except Exception as e:
        logger.error("trade-executor crashed on initialisation: %s", e)
        raise e

    try:
        loop.run_with_state(state)

        # Display summary stats for terminal backtest runs
        if backtest_start:
            # TODO: Hack. Refactor Backtest to its own command / class,
            # Confusion how state should be passed from the execution loop to here
            if hasattr(store, "state"):
                display_backtesting_results(store.state)
            else:
                state = store.load()
                display_backtesting_results(state)

            if isinstance(store, JSONFileStore):
                logger.info("Wrote backtest result to %s", store.path.absolute())

        if run_single_cycle:
            if len(state.portfolio.frozen_positions) > 0:
                logger.error("After single cycle run, we have frozen positions")
                for p in state.portfolio.frozen_positions.values():
                    logger.error("Position: %s", p)

    except KeyboardInterrupt as e:

        # CTRL+C shutdown or watch dog crash
        # Watchdog detected a process has hung: Watched worker live_cycle did not report back in time. Threshold seconds 4500.0, but it has been 4503.594225645065 seconds. Shutting down.
        logger.error("trade-executor %s killed by watchdog or CTRL+C requested: %s. Shutting down.", id, e)
        logger.error("If you are running manually press CTRL+C again to quit")

        # Unwind the traceback and notify the webserver about the failure
        run_state.set_fail()

        logger.exception(e)

        stop_watchdog()

        # Dump all threads to stderr in order to see the stuck threads
        faulthandler.dump_traceback()

        # Spend the rest of the time idling
        time.sleep(3600*24*365)

    except Exception as e:

        logger.error("trade-executor %s execution loop crashed", id)

        # Unwind the traceback and notify the webserver about the failure
        run_state.set_fail()

        stop_watchdog()

        logger.exception(e)

        # Save state on known good exceptions we know are causing headache
        # for automated execution due to unstability of the blockchains.
        # eth_defi.confirmation.ConfirmationTimedOut
        if isinstance(e, (ExecutionHaltableIssue, ConfirmationTimedOut)):
            logger.error("Saving state with aborted execution: %s", e)
            store.sync(state)

        # Debug exceptions in production
        if port_mortem_debugging:
            import ipdb
            ipdb.post_mortem()

        running_time = datetime.datetime.utcnow() - started_at

        if (server is None) or (running_time < datetime.timedelta(seconds=http_wait_good_startup_seconds)):
            # Only terminate the process if the webhook server is not running,
            # otherwise the user can read the crash status from /status endpoint
            logger.error(
                "Raising the error and crashing away, running time was %s Run-time version was:\n%s",
                running_time,
                run_state.version
            )
            raise
        else:

            if run_single_cycle:
                logger.error(
                    "run_single_cycle active - exiting after a crash"
                )
                raise
            else:

                # Execution is dead.
                # Sleep forever, let the webhook still serve the requests.
                logger.error(
                    "Main loop terminated. Entering to the web server wait mode. Run-time version was:\n%s",
                    run_state.version,
                )
                time.sleep(3600*24*365)
    finally:
        if server:
            logger.info("Closing the web server")
            server.close()
