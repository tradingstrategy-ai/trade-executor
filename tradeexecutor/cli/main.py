"""Command-line entry point for the daemon build on the top of Typer."""
import datetime
import logging
import os.path
from decimal import Decimal
from pathlib import Path
from queue import Queue
from typing import Optional, Callable, Tuple
from importlib.metadata import version

import typer
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket

from web3 import Web3

from eth_defi.balances import fetch_erc20_balances_by_token_list
from eth_defi.gas import GasPriceMethod
from eth_defi.token import fetch_erc20_details
from eth_defi.hotwallet import HotWallet

from tradeexecutor.backtest.backtest_execution import BacktestExecutionModel
from tradeexecutor.backtest.backtest_pricing import backtest_pricing_factory
from tradeexecutor.backtest.backtest_sync import BacktestSyncer
from tradeexecutor.backtest.backtest_valuation import backtest_valuation_factory
from tradeexecutor.backtest.simulated_wallet import SimulatedWallet
from tradeexecutor.cli.result import display_backtesting_results
from tradeexecutor.cli.testtrade import make_test_trade
from tradeexecutor.ethereum.web3config import Web3Config
from tradeexecutor.state.metadata import Metadata
from tradeexecutor.state.state import State
from tradeexecutor.strategy.description import StrategyExecutionDescription
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.cli.approval import CLIApprovalModel
from tradeexecutor.cli.loop import ExecutionLoop
from tradeexecutor.ethereum.hot_wallet_sync import EthereumHotWalletReserveSyncer
from tradeexecutor.ethereum.uniswap_v2_execution import UniswapV2ExecutionModel
from tradeexecutor.ethereum.uniswap_v2_valuation import uniswap_v2_sell_valuation_factory
from tradeexecutor.monkeypatch.dataclasses_json import patch_dataclasses_json
from tradeexecutor.ethereum.uniswap_v2_live_pricing import uniswap_v2_live_pricing_factory
from tradeexecutor.state.store import JSONFileStore, StateStore, NoneStore
from tradeexecutor.strategy.approval import ApprovalType, UncheckedApprovalModel, ApprovalModel
from tradeexecutor.strategy.bootstrap import import_strategy_file, make_factory_from_strategy_mod
from tradeexecutor.strategy.dummy import DummyExecutionModel
from tradeexecutor.strategy.execution_context import ExecutionMode, ExecutionContext
from tradeexecutor.strategy.execution_model import TradeExecutionType, ExecutionModel
from tradeexecutor.cli.log import setup_logging, setup_discord_logging, setup_logstash_logging
from tradeexecutor.strategy.strategy_module import read_strategy_module, StrategyModuleInformation
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverseModel
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.utils.fullname import get_object_full_name
from tradeexecutor.utils.timer import timed_task
from tradeexecutor.webhook.server import create_webhook_server
from tradingstrategy.client import Client

app = typer.Typer()


TRADE_EXECUTOR_VERSION = version('trade-executor')


logger: Optional[logging.Logger] = None


def validate_executor_id(id: str):
    """Check that given executor id is good.

    No spaces.

    - Will be used in filenames

    - Will be used in URLs

    :raise AssertionError:
        If the user gives us non-id like id
    """

    assert id, f"EXECUTOR_ID must be given so that executor instances can be identified"
    assert " " not in id, f"Bad EXECUTOR_ID: {id}"


def create_web3_config(
    json_rpc_binance,
    json_rpc_polygon,
    json_rpc_avalanche,
    json_rpc_ethereum,
    gas_price_method: Optional[GasPriceMethod]=None,
) -> Optional[Web3Config]:
    """Create Web3 connection to the live node we are executing against.

    :return web3:
        Connect to any passed JSON RPC URL

    """
    web3config = Web3Config.setup_from_environment(
        gas_price_method,
        json_rpc_ethereum=json_rpc_ethereum,
        json_rpc_binance=json_rpc_binance,
        json_rpc_polygon=json_rpc_polygon,
        json_rpc_avalanche=json_rpc_avalanche,
    )
    return web3config


def create_trade_execution_model(
        execution_type: TradeExecutionType,
        private_key: str,
        web3config: Web3Config,
        confirmation_timeout: datetime.timedelta,
        confirmation_block_count: int,
        max_slippage: float,
        min_balance_threshold: Optional[Decimal],
):
    """Set up the execution mode for the command line client."""

    if execution_type == TradeExecutionType.dummy:
        return DummyExecutionModel()
    elif execution_type == TradeExecutionType.uniswap_v2_hot_wallet:
        assert private_key, "Private key is needed for live trading"
        web3 = web3config.get_default()
        hot_wallet = HotWallet.from_private_key(private_key)
        sync_method = EthereumHotWalletReserveSyncer(web3, hot_wallet.address)
        execution_model = UniswapV2ExecutionModel(
            web3,
            hot_wallet,
            confirmation_timeout=confirmation_timeout,
            confirmation_block_count=confirmation_block_count,
            max_slippage=max_slippage,
            min_balance_threshold=min_balance_threshold,
        )
        valuation_model_factory = uniswap_v2_sell_valuation_factory
        pricing_model_factory = uniswap_v2_live_pricing_factory
        return execution_model, sync_method, valuation_model_factory, pricing_model_factory
    elif execution_type == TradeExecutionType.backtest:
        logger.info("TODO: Command line backtests are always executed with initial deposit of $10,000")
        wallet = SimulatedWallet()
        execution_model = BacktestExecutionModel(wallet, max_slippage=0.01, stop_loss_data_available=True)
        sync_method = BacktestSyncer(wallet, Decimal(10_000))
        pricing_model_factory = backtest_pricing_factory
        valuation_model_factory = backtest_valuation_factory
        return execution_model, sync_method, valuation_model_factory, pricing_model_factory
    else:
        raise NotImplementedError()


def create_approval_model(approval_type: ApprovalType) -> ApprovalModel:
    if approval_type == ApprovalType.unchecked:
        return UncheckedApprovalModel()
    elif approval_type == ApprovalType.cli:
        return CLIApprovalModel()
    else:
        raise NotImplementedError()


def create_state_store(state_file: Path) -> StateStore:
    store = JSONFileStore(state_file)
    return store


def prepare_cache(executor_id: str, cache_path: Optional[Path]) -> Path:
    """Fail early if the cache path is not writable.

    Otherwise Docker might spit misleading "Device or resource busy" message.
    """

    assert executor_id

    if not cache_path:
        cache_path = Path("cache").joinpath(executor_id)

    logger.info("Dataset cache is %s", cache_path)

    os.makedirs(cache_path, exist_ok=True)

    with open(cache_path.joinpath("cache.pid"), "wt") as out:
        print(os.getpid(), file=out)

    return cache_path


def create_metadata(name, short_description, long_description, icon_url) -> Metadata:
    """Create metadata object from the configuration variables."""
    return Metadata(
        name,
        short_description,
        long_description,
        icon_url,
        datetime.datetime.utcnow(),
    )


def prepare_executor_id(id: Optional[str], strategy_file: Path) -> str:
    """Autodetect exeuctor id."""

    if id:
        # Explicitly passed
        pass
    else:
        # Guess id from the strategy file
        if strategy_file:
            id = Path(strategy_file).stem
            pass
        else:
            raise RuntimeError("EXECUTOR_ID or STRATEGY_FILE must be given")

    validate_executor_id(id)

    return id


def monkey_patch():
    """Apply all monkey patches."""
    patch_dataclasses_json()


# Run this during the module loading so that it is
# applied to all subcommands
monkey_patch()


# Typer documentation https://typer.tiangolo.com/
@app.command()
def start(

    # Strategy assets
    id: str = typer.Option(None, envvar="EXECUTOR_ID", help="Executor id used when programmatically referring to this instance. If not given, take the base of --strategy-file."),
    log_level: str = typer.Option(None, envvar="LOG_LEVEL", help="The Python default logging level. The defaults are 'info' is live execution, 'warning' if backtesting."),
    name: Optional[str] = typer.Option(None, envvar="NAME", help="Executor name used in the web interface and notifications"),
    short_description: Optional[str] = typer.Option(None, envvar="SHORT_DESCRIPTION", help="Short description for metadata"),
    long_description: Optional[str] = typer.Option(None, envvar="LONG_DESCRIPTION", help="Long description for metadata"),
    icon_url: Optional[str] = typer.Option(None, envvar="ICON_URL", help="Strategy icon for web rendering and Discord avatar"),
    strategy_file: Path = typer.Option(..., envvar="STRATEGY_FILE", help="Python strategy file to run"),

    # Live trading or backtest
    execution_type: TradeExecutionType = typer.Option(..., envvar="EXECUTION_TYPE"),
    trading_strategy_api_key: str = typer.Option(None, envvar="TRADING_STRATEGY_API_KEY", help="Trading Strategy API key"),

    # Webhook server options
    http_enabled: bool = typer.Option(False, envvar="HTTP_ENABLED", help="Enable Webhook server"),
    http_port: int = typer.Option(3456, envvar="HTTP_PORT", help="Which HTTP port to listen. The default is 3456, the default port of Pyramid web server."),
    http_host: str = typer.Option("0.0.0.0", envvar="HTTP_HOST", help="The IP address to bind for the web server. By default listen to all IP addresses available in the run-time environment."),
    http_username: str = typer.Option("webhook", envvar="HTTP_USERNAME", help="Username for HTTP Basic Auth protection of webhooks"),
    http_password: str = typer.Option(None, envvar="HTTP_PASSWORD", help="Password for HTTP Basic Auth protection of webhooks"),

    # Web3 connection options
    json_rpc_binance: str = typer.Option(None, envvar="JSON_RPC_BINANCE", help="BNB Chain JSON-RPC node URL we connect to"),
    json_rpc_polygon: str = typer.Option(None, envvar="JSON_RPC_POLYGON", help="Polygon JSON-RPC node URL we connect to"),
    json_rpc_ethereum: str = typer.Option(None, envvar="JSON_RPC_ETHEREUM", help="Ethereum JSON-RPC node URL we connect to"),
    json_rpc_avalanche: str = typer.Option(None, envvar="JSON_RPC_AVALANCHE", help="Avalanche C-chain JSON-RPC node URL we connect to"),
    gas_price_method: Optional[GasPriceMethod] = typer.Option(None, envvar="GAS_PRICE_METHOD", help="How to set the gas price for Ethereum transactions. After the Berlin hardfork Ethereum mainnet introduced base + tip cost gas model. Leave out to autodetect."),
    confirmation_timeout: int = typer.Option(900, envvar="CONFIRMATION_TIMEOUT", help="How many seconds to wait for transaction batches to confirm"),
    confirmation_block_count: int = typer.Option(8, envvar="CONFIRMATION_BLOCK_COUNT", help="How many blocks we wait before we consider transaction receipt a final"),
    private_key: Optional[str] = typer.Option(None, envvar="PRIVATE_KEY", help="Ethereum private key to be used as a hot wallet/broadcast wallet"),
    minimum_gas_balance: Optional[float] = typer.Option(0.1, envvar="MINUMUM_GAS_BALANCE", help="What is the minimum balance of gas token you need to have in your wallet. If the balance falls below this, abort by crashing and do not attempt to create transactions. Expressed in the native token e.g. ETH."),

    # Logging
    discord_webhook_url: Optional[str] = typer.Option(None, envvar="DISCORD_WEBHOOK_URL", help="Discord webhook URL for notifications"),
    logstash_server: Optional[str] = typer.Option(None, envvar="LOGSTASH_SERVER", help="LogStash server hostname where to send logs"),

    # Debugging and unit testing
    port_mortem_debugging: bool = typer.Option(False, "--post-mortem-debugging", envvar="POST_MORTEM_DEBUGGING", help="Launch ipdb debugger on a main loop crash to debug the exception"),
    clear_caches: bool = typer.Option(False, "--clear-caches", envvar="CLEAR_CACHES", help="Purge any dataset download caches before starting"),
    unit_testing: bool = typer.Option(False, "--unit-testing", envvar="UNIT_TESTING", help="The trade executor is called under the unit testing mode. No caches are purged."),
    reset_state: bool = typer.Option(False, envvar="RESET_STATE"),
    max_cycles: int = typer.Option(None, envvar="MAX_CYCLES", help="Max main loop cycles run in an automated testing mode"),
    debug_dump_file: Optional[Path] = typer.Option(None, envvar="DEBUG_DUMP_FILE", help="Write Python Pickle dump of all internal debugging states of the strategy run to this file"),

    # Backtesting
    backtest_start: Optional[datetime.datetime] = typer.Option(None, envvar="BACKTEST_START", help="Start timestamp of backesting"),
    backtest_end: Optional[datetime.datetime] = typer.Option(None, envvar="BACKTEST_END", help="End timestamp of backesting"),
    backtest_candle_time_frame_override: Optional[TimeBucket] = typer.Option(None, envvar="BACKTEST_CANDLE_TIME_FRAME_OVERRIDE", help="Force backtests to use different candle time frame"),
    backtest_stop_loss_time_frame_override: Optional[TimeBucket] = typer.Option(None, envvar="BACKTEST_STOP_LOSS_TIME_FRAME_OVERRIDE", help="Force backtests to use different candle time frame for stop losses"),

    # Live trading configuration
    max_slippage: float = typer.Option(0.0025, envvar="MAX_SLIPPAGE", help="Max slippage allowed per trade before failing. The default is 0.0025 is 0.25%."),
    approval_type: ApprovalType = typer.Option("unchecked", envvar="APPROVAL_TYPE", help="Set a manual approval flow for trades"),
    stop_loss_check_frequency: Optional[TimeBucket] = typer.Option(None, envvar="STOP_LOSS_CYCLE_DURATION", help="Override live/backtest stop loss check frequency. If not given read from the strategy module."),
    cycle_offset_minutes: int = typer.Option(8, envvar="CYCLE_OFFSET_MINUTES", help="How many minutes we wait after the tick before executing the tick step"),
    stats_refresh_minutes: int = typer.Option(60.0, envvar="STATS_REFRESH_MINUTES", help="How often we refresh position statistics. Default to once in an hour."),
    cycle_duration: CycleDuration = typer.Option(None, envvar="CYCLE_DURATION", help="How long strategy tick cycles use to execute the strategy. While strategy modules offer their own cycle duration value, you can override it here."),
    position_trigger_check_minutes: int = typer.Option(3.0, envvar="POSITION_TRIGGER_CHECK_MINUTES", help="How often we check for take profit/stop loss triggers. Default to once in 3 minutes. Set 0 to disable."),
    max_data_delay_minutes: int = typer.Option(3*60, envvar="MAX_DATA_DELAY_MINUTES", help="If our data feed is delayed more than this minutes, abort the execution. Defaukts to 3 hours."),
    trade_immediately: bool = typer.Option(False, "--trade-immediately", envvar="TRADE_IMMEDIATELY", help="Perform the first rebalance immediately, do not wait for the next trading universe refresh"),

    # Unsorted options
    state_file: Optional[Path] = typer.Option(None, envvar="STATE_FILE", help="JSON file where we serialise the execution state. If not given defaults to state/{executor-id}.json"),
    cache_path: Optional[Path] = typer.Option(None, envvar="CACHE_PATH", help="Where to store downloaded datasets. This must be specific to each executor so that there are no write conflicts if multiple executors run on the same server. If not given default to cache/{executor-id}"),
    ):
    """Launch Trade Executor instance."""
    global logger

    # Guess id from the strategy file
    id = prepare_executor_id(id, strategy_file)

    # We always need a name
    if not name:
        if strategy_file:
            name = os.path.basename(strategy_file)
        else:
            name = "Unnamed backtest"

    if log_level:
        log_level = log_level.upper()
    else:
        if execution_type == TradeExecutionType.backtest:
            log_level = logging.WARNING
        else:
            log_level = logging.INFO

    logger = setup_logging(log_level)

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

    if not state_file:
        if execution_type != TradeExecutionType.backtest:
            state_file = f"state/{id}.json"

    cache_path = prepare_cache(id, cache_path)

    confirmation_timeout = datetime.timedelta(seconds=confirmation_timeout)

    if execution_type == TradeExecutionType.uniswap_v2_hot_wallet:
        web3config = create_web3_config(
            json_rpc_binance=json_rpc_binance,
            json_rpc_polygon=json_rpc_polygon,
            json_rpc_avalanche=json_rpc_avalanche,
            json_rpc_ethereum=json_rpc_ethereum,
            gas_price_method=None,
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
            web3config.set_default_chain(mod.chain_id)
            web3config.check_default_chain_id()
        else:
            # Legacy unit testing path.
            # All chain_ids are 56 (BNB Chain)
            logger.warning("Legacy strategy module: makes assumption of BNB Chain")
            web3config.set_default_chain(ChainId.bsc)

    if minimum_gas_balance:
        minimum_gas_balance = Decimal(minimum_gas_balance)

    execution_model, sync_method, valuation_model_factory, pricing_model_factory = create_trade_execution_model(
        execution_type,
        private_key,
        web3config,
        confirmation_timeout,
        confirmation_block_count,
        max_slippage,
        minimum_gas_balance,
    )

    approval_model = create_approval_model(approval_type)

    if state_file:
        store = create_state_store(Path(state_file))
    else:
        # Backtests never have persistent state
        if execution_type == TradeExecutionType.backtest:
            logger.info("This backtest run won't create a state file")
            store = NoneStore(State())
        else:
            raise RuntimeError("Does not know how to set up a state file for this run")

    metadata = create_metadata(name, short_description, long_description, icon_url)

    # Start the queue that relays info from the web server to the strategy executor
    command_queue = Queue()

    # Create our webhook server
    if http_enabled:
        server = create_webhook_server(http_host, http_port, http_username, http_password, command_queue, store, metadata)
    else:
        logger.info("Web server disabled")
        server = None

    # Create our data client
    if trading_strategy_api_key:
        client = Client.create_live_client(trading_strategy_api_key, cache_path=cache_path)
        if clear_caches:
            client.clear_caches()
    else:
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

    logger.trade("Trade Executor version %s starting strategy %s", TRADE_EXECUTOR_VERSION, name)

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

    try:
        loop = ExecutionLoop(
            name=name,
            command_queue=command_queue,
            execution_model=execution_model,
            execution_context=execution_context,
            sync_method=sync_method,
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
        )
        loop.run()

        # Display summary stats for terminal backtest runs
        if execution_type == TradeExecutionType.backtest and isinstance(store, NoneStore):
            display_backtesting_results(store.state)

    except KeyboardInterrupt as e:
        # CTRL+C shutdown
        logger.trade("Trade Executor %s shut down by CTRL+C requested: %s", name, e)
    except Exception as e:
        # Debug exceptions in production
        if port_mortem_debugging:
            import ipdb
            ipdb.post_mortem()
        logger.exception(e)
        raise
    finally:
        if server:
            server.close()


@app.command()
def check_universe(
    id: str = typer.Option(None, envvar="EXECUTOR_ID", help="Executor id used when programmatically referring to this instance. If not given, take the base of --strategy-file."),
    strategy_file: Path = typer.Option(..., envvar="STRATEGY_FILE"),
    trading_strategy_api_key: str = typer.Option(None, envvar="TRADING_STRATEGY_API_KEY", help="Trading Strategy API key"),
    cache_path: Optional[Path] = typer.Option(None, envvar="CACHE_PATH", help="Where to store downloaded datasets"),
    max_data_delay_minutes: int = typer.Option(24*60, envvar="MAX_DATA_DELAY_MINUTES", help="How fresh the OHCLV data for our strategy must be before failing"),
):
    """Checks that the trading universe is helthy for a given strategy."""

    global logger

    id = prepare_executor_id(id, strategy_file)

    logger = setup_logging()

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
        sync_method=None,
        valuation_model_factory=None,
        pricing_model_factory=None,
        approval_model=None,
        client=client,
    )

    # Deconstruct strategy input
    universe_model: TradingStrategyUniverseModel = run_description.universe_model

    ts = datetime.datetime.utcnow()
    logger.info("Performing universe data check for timestamp %s", ts)
    universe = universe_model.construct_universe(ts, ExecutionMode.preflight_check, UniverseOptions())

    latest_candle_at = universe_model.check_data_age(ts, universe, max_data_delay)
    ago = datetime.datetime.utcnow() - latest_candle_at
    logger.info("Latest OHCLV candle is at: %s, %s ago", latest_candle_at, ago)


@app.command()
def check_wallet(
    id: str = typer.Option(None, envvar="EXECUTOR_ID", help="Executor id used when programmatically referring to this instance. If not given, take the base of --strategy-file."),

    strategy_file: Path = typer.Option(..., envvar="STRATEGY_FILE"),
    private_key: str = typer.Option(None, envvar="PRIVATE_KEY"),
    trading_strategy_api_key: str = typer.Option(None, envvar="TRADING_STRATEGY_API_KEY", help="Trading Strategy API key"),
    cache_path: Optional[Path] = typer.Option("cache/", envvar="CACHE_PATH", help="Where to store downloaded datasets"),

    # Get minimum gas balance from the env
    minimum_gas_balance: Optional[float] = typer.Option(0.1, envvar="MINUMUM_GAS_BALANCE", help="What is the minimum balance of gas token you need to have in your wallet. If the balance falls below this, abort by crashing and do not attempt to create transactions. Expressed in the native token e.g. ETH."),

    # Web3 connection options
    json_rpc_binance: str = typer.Option(None, envvar="JSON_RPC_BINANCE", help="BNB Chain JSON-RPC node URL we connect to"),
    json_rpc_polygon: str = typer.Option(None, envvar="JSON_RPC_POLYGON", help="Polygon JSON-RPC node URL we connect to"),
    json_rpc_ethereum: str = typer.Option(None, envvar="JSON_RPC_ETHEREUM", help="Ethereum JSON-RPC node URL we connect to"),
    json_rpc_avalanche: str = typer.Option(None, envvar="JSON_RPC_AVALANCHE", help="Avalanche C-chain JSON-RPC node URL we connect to"),
):
    """Print out the token balances of the hot wallet.

    Check that our hot wallet has cash deposits and gas token deposits.
    """

    # To run this from command line with .env file you can do
    # set -o allexport ; source ~/pancake-eth-usd-sma-final.env ; set +o allexport ;  trade-executor check-wallet

    global logger

    id = prepare_executor_id(id, strategy_file)

    logger = setup_logging()

    mod = read_strategy_module(strategy_file)

    cache_path = prepare_cache(id, cache_path)

    client = Client.create_live_client(trading_strategy_api_key, cache_path=cache_path)

    execution_context = ExecutionContext(
        mode=ExecutionMode.preflight_check,
        timed_task_context_manager=timed_task
    )

    web3config = create_web3_config(
        json_rpc_binance,
        json_rpc_polygon,
        json_rpc_avalanche,
        json_rpc_ethereum,
        GasPriceMethod.london,
    )

    assert web3config, "No RPC endpoints given. A working JSON-RPC connection is needed for check-wallet"

    # Check that we are connected to the chain strategy assumes
    web3config.set_default_chain(mod.chain_id)
    web3config.check_default_chain_id()

    execution_model, sync_method, valuation_model_factory, pricing_model_factory = create_trade_execution_model(
        execution_type=TradeExecutionType.uniswap_v2_hot_wallet,
        private_key=private_key,
        web3config=web3config,
        confirmation_timeout=60,
        confirmation_block_count=6,
        max_slippage=0.01,
        min_balance_threshold=minimum_gas_balance,
    )

    hot_wallet = HotWallet.from_private_key(private_key)

    # Set up the strategy engine
    factory = make_factory_from_strategy_mod(mod)
    run_description: StrategyExecutionDescription = factory(
        execution_model=execution_model,
        execution_context=execution_context,
        timed_task_context_manager=execution_context.timed_task_context_manager,
        sync_method=sync_method,
        valuation_model_factory=valuation_model_factory,
        pricing_model_factory=pricing_model_factory,
        approval_model=UncheckedApprovalModel(),
        client=client,
    )

    # We construct the trading universe to know what's our reserve asset
    universe_model: TradingStrategyUniverseModel = run_description.universe_model
    ts = datetime.datetime.utcnow()
    universe = universe_model.construct_universe(
        ts,
        ExecutionMode.preflight_check,
        UniverseOptions())

    # Get all tokens from the universe
    reserve_assets = universe.reserve_assets
    web3 = web3config.get_default()
    tokens = [Web3.toChecksumAddress(a.address) for a in reserve_assets]

    logger.info("RPC details")

    # Check the chain is online
    logger.info(f"  Chain id is {web3.eth.chain_id:,}")
    logger.info(f"  Latest block is {web3.eth.block_number:,}")

    # Check balances
    logger.info("Balance details")
    logger.info("  Hot wallet is %s", hot_wallet.address)
    gas_balance = web3.eth.get_balance(hot_wallet.address) / 10**18
    logger.info("  We have %f tokens for gas left", gas_balance)
    logger.info("  The gas error limit is %f tokens", minimum_gas_balance)
    balances = fetch_erc20_balances_by_token_list(web3, hot_wallet.address, tokens)

    for asset in reserve_assets:
        logger.info("Reserve asset: %s", asset.token_symbol)

    for address, balance in balances.items():
        details = fetch_erc20_details(web3, address)
        logger.info("  Balance of %s (%s): %s %s", details.name, details.address, details.convert_to_decimals(balance), details.symbol)

    # Check that the routing looks sane
    # E.g. there is no mismatch between strategy reserve token, wallet and pair universe
    runner = run_description.runner
    routing_state, pricing_model, valuation_method = runner.setup_routing(universe)
    routing_model = runner.routing_model

    logger.info("Execution details")
    logger.info("  Execution model is %s", get_object_full_name(execution_model))
    logger.info("  Routing model is %s", get_object_full_name(routing_model))
    logger.info("  Token pricing model is %s", get_object_full_name(pricing_model))
    logger.info("  Position valuation model is %s", get_object_full_name(valuation_method))

    # Check we have enough gas
    execution_model.preflight_check()

    # Check our routes
    routing_model.perform_preflight_checks_and_logging(routing_state, universe.universe.pairs)

    web3config.close()


@app.command()
def hello():
    """Check that the application loads without doing anything."""
    print("Hello blockchain")


@app.command()
def perform_test_trade(
    id: str = typer.Option(None, envvar="EXECUTOR_ID", help="Executor id used when programmatically referring to this instance. If not given, take the base of --strategy-file."),

    strategy_file: Path = typer.Option(..., envvar="STRATEGY_FILE"),
    private_key: str = typer.Option(None, envvar="PRIVATE_KEY"),
    trading_strategy_api_key: str = typer.Option(None, envvar="TRADING_STRATEGY_API_KEY", help="Trading Strategy API key"),
    state_file: Optional[Path] = typer.Option(None, envvar="STATE_FILE", help="JSON file where we serialise the execution state. If not given defaults to state/{executor-id}.json"),
    cache_path: Optional[Path] = typer.Option("cache/", envvar="CACHE_PATH", help="Where to store downloaded datasets"),

    # Web3 connection options
    json_rpc_binance: str = typer.Option(None, envvar="JSON_RPC_BINANCE", help="BNB Chain JSON-RPC node URL we connect to"),
    json_rpc_polygon: str = typer.Option(None, envvar="JSON_RPC_POLYGON", help="Polygon JSON-RPC node URL we connect to"),
    json_rpc_ethereum: str = typer.Option(None, envvar="JSON_RPC_ETHEREUM", help="Ethereum JSON-RPC node URL we connect to"),
    json_rpc_avalanche: str = typer.Option(None, envvar="JSON_RPC_AVALANCHE", help="Avalanche C-chain JSON-RPC node URL we connect to"),
):
    """Perform a small test swap.

    Tests that the private wallet and the exchange can trade by making 1 USD trade using
    the routing configuration from the strategy.

    The trade will be recorded on the state as a position.
    """
    global logger

    id = prepare_executor_id(id, strategy_file)

    logger = setup_logging(log_level=logging.INFO)

    mod = read_strategy_module(strategy_file)

    cache_path = prepare_cache(id, cache_path)

    client = Client.create_live_client(trading_strategy_api_key, cache_path=cache_path)

    execution_context = ExecutionContext(
        mode=ExecutionMode.preflight_check,
        timed_task_context_manager=timed_task
    )

    web3config = create_web3_config(
        json_rpc_binance=json_rpc_binance,
        json_rpc_polygon=json_rpc_polygon,
        json_rpc_avalanche=json_rpc_avalanche,
        json_rpc_ethereum=json_rpc_ethereum,
        gas_price_method=None,
    )

    assert web3config, "No RPC endpoints given. A working JSON-RPC connection is needed for check-wallet"

    # Check that we are connected to the chain strategy assumes
    web3config.set_default_chain(mod.chain_id)
    web3config.check_default_chain_id()

    execution_model, sync_method, valuation_model_factory, pricing_model_factory = create_trade_execution_model(
        execution_type=TradeExecutionType.uniswap_v2_hot_wallet,
        private_key=private_key,
        web3config=web3config,
        confirmation_timeout=60,
        confirmation_block_count=6,
        max_slippage=2.50,
        min_balance_threshold=0,
    )

    if not state_file:
        state_file = f"state/{id}.json"

    store = create_state_store(Path(state_file))

    if store.is_pristine():
        state = store.create()
    else:
        state = store.load()

    # Set up the strategy engine
    factory = make_factory_from_strategy_mod(mod)
    run_description: StrategyExecutionDescription = factory(
        execution_model=execution_model,
        execution_context=execution_context,
        timed_task_context_manager=execution_context.timed_task_context_manager,
        sync_method=sync_method,
        valuation_model_factory=valuation_model_factory,
        pricing_model_factory=pricing_model_factory,
        approval_model=UncheckedApprovalModel(),
        client=client,
    )

    # We construct the trading universe to know what's our reserve asset
    universe_model: TradingStrategyUniverseModel = run_description.universe_model
    ts = datetime.datetime.utcnow()
    universe = universe_model.construct_universe(
        ts,
        ExecutionMode.preflight_check,
        UniverseOptions())

    runner = run_description.runner
    routing_state, pricing_model, valuation_method = runner.setup_routing(universe)

    make_test_trade(
        execution_model,
        pricing_model,
        sync_method,
        state,
        universe,
        runner.routing_model,
        routing_state,
    )

    # Store the test trade data in the strategy history
    store.sync(state)




