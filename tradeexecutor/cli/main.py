"""Command-line entry point for the daemon build on the top of Typer."""
import datetime
import logging
from pathlib import Path
from queue import Queue
from typing import Optional
import pkg_resources

import typer

from web3.middleware import geth_poa_middleware
from web3 import Web3, HTTPProvider

from eth_defi.balances import fetch_erc20_balances_by_token_list
from eth_defi.gas import GasPriceMethod, node_default_gas_price_strategy
from eth_defi.token import fetch_erc20_details
from eth_defi.hotwallet import HotWallet

from tradeexecutor.state.metadata import Metadata
from tradeexecutor.strategy.description import StrategyExecutionDescription
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.cli.approval import CLIApprovalModel
from tradeexecutor.cli.loop import ExecutionLoop
from tradeexecutor.ethereum.hot_wallet_sync import EthereumHotWalletReserveSyncer
from tradeexecutor.ethereum.uniswap_v2_execution import UniswapV2ExecutionModel
from tradeexecutor.ethereum.uniswap_v2_valuation import uniswap_v2_sell_valuation_factory
from tradeexecutor.monkeypatch.dataclasses_json import patch_dataclasses_json
from tradeexecutor.ethereum.uniswap_v2_live_pricing import uniswap_v2_live_pricing_factory
from tradeexecutor.state.store import JSONFileStore, StateStore
from tradeexecutor.strategy.approval import ApprovalType, UncheckedApprovalModel, ApprovalModel
from tradeexecutor.strategy.bootstrap import import_strategy_file
from tradeexecutor.strategy.dummy import DummyExecutionModel
from tradeexecutor.strategy.execution_context import ExecutionMode, ExecutionContext
from tradeexecutor.strategy.execution_model import TradeExecutionType
from tradeexecutor.cli.log import setup_logging, setup_discord_logging, setup_logstash_logging
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverseModel
from tradeexecutor.utils.timer import timed_task
from tradeexecutor.utils.url import redact_url_password
from tradeexecutor.webhook.server import create_webhook_server
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client

app = typer.Typer()

version = pkg_resources.get_distribution('trade-executor').version

logger: Optional[logging.Logger] = None


def check_good_id(id: str):
    """
    :raise AssertionError:
        If the user gives us non-id like id
    """

    assert id, f"EXECUTOR_ID must be given so that executor instances can be identified"
    assert " " not in id, f"Bad EXECUTOR_ID: {id}"


def create_trade_execution_model(
        execution_type: TradeExecutionType,
        json_rpc: str,
        private_key: str,
        gas_price_method: Optional[GasPriceMethod],
        confirmation_timeout: datetime.timedelta,
        confirmation_block_count: int,
        max_slippage: float,
):

    if not gas_price_method:
        raise RuntimeError("GAS_PRICE_METHOD missing")

    if execution_type == TradeExecutionType.dummy:
        return DummyExecutionModel()
    elif execution_type == TradeExecutionType.uniswap_v2_hot_wallet:
        assert private_key, "Private key is needed"
        assert json_rpc, "JSON-RPC endpoint is needed"
        web3 = create_web3(json_rpc, gas_price_method)

        hot_wallet = HotWallet.from_private_key(private_key)
        sync_method = EthereumHotWalletReserveSyncer(web3, hot_wallet.address)
        execution_model = UniswapV2ExecutionModel(
            web3,
            hot_wallet,
            confirmation_timeout=confirmation_timeout,
            confirmation_block_count=confirmation_block_count,
            max_slippage=max_slippage)
        valuation_model_factory = uniswap_v2_sell_valuation_factory
        pricing_model_factory = uniswap_v2_live_pricing_factory
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


def create_web3(url, gas_price_method: Optional[GasPriceMethod] = None) -> Web3:

    assert gas_price_method

    web3 = Web3(HTTPProvider(url))

    chain_id = web3.eth.chain_id

    logger.info("Connected to chain id: %d, using gas price method %s", chain_id, gas_price_method.name)

    # London is the default method
    if gas_price_method == GasPriceMethod.legacy:
        logger.info("Setting up gas price middleware for Web3")
        web3.eth.set_gas_price_strategy(node_default_gas_price_strategy)

    # Set POA middleware if needed
    if chain_id in (ChainId.bsc.value, ChainId.polygon.value):
        logger.info("Using proof-of-authority web3 middleware")
        web3.middleware_onion.inject(geth_poa_middleware, layer=0)

    return web3


def create_metadata(name, short_description, long_description, icon_url) -> Metadata:
    return Metadata(
        name,
        short_description,
        long_description,
        icon_url,
        datetime.datetime.utcnow(),
    )


def monkey_patch():
    """Apply all monkey patches."""
    patch_dataclasses_json()


# Typer documentation https://typer.tiangolo.com/
@app.command()
def start(
    id: str = typer.Option(None, envvar="EXECUTOR_ID", help="Executor id used when programmatically referring to this instance"),
    name: Optional[str] = typer.Option("Unnamed Trade Executor", envvar="NAME", help="Executor name used in the web interface and notifications"),
    short_description: Optional[str] = typer.Option(None, envvar="SHORT_DESCRIPTION", help="Short description for metadata"),
    long_description: Optional[str] = typer.Option(None, envvar="LONG_DESCRIPTION", help="Long description for metadata"),
    icon_url: Optional[str] = typer.Option(None, envvar="ICON_URL", help="Strategy icon for web rendering and Discord avatar"),
    private_key: Optional[str] = typer.Option(None, envvar="PRIVATE_KEY", help="Ethereum private key to be used as a hot wallet/broadcast wallet"),
    strategy_file: Path = typer.Option(..., envvar="STRATEGY_FILE", help="Python strategy file to run"),
    http_enabled: bool = typer.Option(False, envvar="HTTP_ENABLED", help="Enable Webhook server"),
    http_port: int = typer.Option(19000, envvar="HTTP_PORT"),
    http_host: str = typer.Option("127.0.0.1", envvar="HTTP_HOST"),
    http_username: str = typer.Option("webhook", envvar="HTTP_USERNAME"),
    http_password: str = typer.Option(None, envvar="HTTP_PASSWORD"),
    json_rpc: str = typer.Option(None, envvar="JSON_RPC", help="Ethereum JSON-RPC node URL we connect to for execution"),
    gas_price_method: Optional[GasPriceMethod] = typer.Option(None, envvar="GAS_PRICE_METHOD", help="How to set the gas price for Ethereum transactions"),
    confirmation_timeout: int = typer.Option(900, envvar="CONFIRMATION_TIMEOUT", help="How many seconds to wait for transaction batches to confirm"),
    confirmation_block_count: int = typer.Option(8, envvar="CONFIRMATION_BLOCK_COUNT", help="How many blocks we wait before we consider transaction receipt a final"),
    execution_type: TradeExecutionType = typer.Option(..., envvar="EXECUTION_TYPE"),
    max_slippage: float = typer.Option(0.0025, envvar="MAX_SLIPPAGE", help="Max slippage allowed per trade before failing. The default is 0.0025 is 0.25%."),
    approval_type: ApprovalType = typer.Option(..., envvar="APPROVAL_TYPE"),
    state_file: Optional[Path] = typer.Option("strategy-state.json", envvar="STATE_FILE"),
    trading_strategy_api_key: str = typer.Option(None, envvar="TRADING_STRATEGY_API_KEY", help="Trading Strategy API key"),
    cache_path: Optional[Path] = typer.Option(None, envvar="CACHE_PATH", help="Where to store downloaded datasets"),
    reset_state: bool = typer.Option(False, "--reset-state", envvar="RESET_STATE"),
    max_cycles: int = typer.Option(None, envvar="MAX_CYCLES", help="Max main loop cycles run in an automated testing mode"),
    debug_dump_file: Optional[Path] = typer.Option(None, envvar="DEBUG_DUMP_FILE", help="Write Python Pickle dump of all internal debugging states of the strategy run to this file"),
    backtest_start: Optional[datetime.datetime] = typer.Option(None, envvar="BACKTEST_START", help="Start timestamp of backesting"),
    backtest_end: Optional[datetime.datetime] = typer.Option(None, envvar="BACKTEST_END", help="End timestamp of backesting"),
    tick_size: CycleDuration = typer.Option(None, envvar="TICK_SIZE", help="How large tick use to execute the strategy"),
    tick_offset_minutes: int = typer.Option(0, envvar="TICK_OFFSET_MINUTES", help="How many minutes we wait after the tick before executing the tick step"),
    stats_refresh_minutes: int = typer.Option(60, envvar="STATS_REFRESH_MINUTES", help="How often we refresh position statistics. Default to once in an hour."),
    max_data_delay_minutes: int = typer.Option(None, envvar="MAX_DATA_DELAY_MINUTES", help="If our data feed is delayed more than this minutes, abort the execution"),
    discord_webhook_url: Optional[str] = typer.Option(None, envvar="DISCORD_WEBHOOK_URL", help="Discord webhook URL for notifications"),
    logstash_server: Optional[str] = typer.Option(None, envvar="LOGSTASH_SERVER", help="LogStash server hostname where to send logs"),
    trade_immediately: bool = typer.Option(False, "--trade-immediately", envvar="TRADE_IMMEDIATELY", help="Perform the first rebalance immediately, do not wait for the next trading universe refresh"),
    port_mortem_debugging: bool = typer.Option(False, "--post-mortem-debugging", envvar="POST_MORTEM_DEBUGGING", help="Launch ipdb debugger on a main loop crash to debug the exception"),
    clear_caches: bool = typer.Option(False, "--clear-caches", envvar="CLEAR_CACHES", help="Purge any dataset download caches before starting"),
    unit_testing: bool = typer.Option(False, "--unit-testing", envvar="UNIT_TESTING", help="The trade executor is called under the unit testing mode. No caches are purged."),
    ):
    """Launch Trade Executor instance."""
    global logger

    check_good_id(id)

    logger = setup_logging()

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

    monkey_patch()

    confirmation_timeout = datetime.timedelta(seconds=confirmation_timeout)

    execution_model, sync_method, valuation_model_factory, pricing_model_factory = create_trade_execution_model(
        execution_type,
        json_rpc,
        private_key,
        gas_price_method,
        confirmation_timeout,
        confirmation_block_count,
        max_slippage,
    )

    approval_model = create_approval_model(approval_type)

    store = create_state_store(state_file)

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
    tick_offset = datetime.timedelta(minutes=tick_offset_minutes)

    if max_data_delay_minutes:
        max_data_delay = datetime.timedelta(minutes=max_data_delay_minutes)
    else:
        max_data_delay = None

    stats_refresh_frequency = datetime.timedelta(minutes=stats_refresh_minutes)

    logger.info("Loading strategy file %s", strategy_file)
    strategy_factory = import_strategy_file(strategy_file)

    logger.trade("Trade Executor version %s starting strategy %s", version, name)

    if backtest_start:
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
            cycle_duration=tick_size,
            tick_offset=tick_offset,
            max_data_delay=max_data_delay,
            trade_immediately=trade_immediately,
            stats_refresh_frequency=stats_refresh_frequency,
        )
        loop.run()
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
    strategy_file: Path = typer.Option(..., envvar="STRATEGY_FILE"),
    trading_strategy_api_key: str = typer.Option(None, envvar="TRADING_STRATEGY_API_KEY", help="Trading Strategy API key"),
    cache_path: Optional[Path] = typer.Option(None, envvar="CACHE_PATH", help="Where to store downloaded datasets"),
    max_data_delay_minutes: int = typer.Option(None, envvar="MAX_DATA_DELAY_MINUTES", help="If our data feed is delayed more than this minutes, abort the execution"),
):
    """Checks that the trading universe is helthy for a given strategy."""

    global logger

    logger = setup_logging()

    logger.info("Loading strategy file %s", strategy_file)
    strategy_factory = import_strategy_file(strategy_file)

    assert trading_strategy_api_key, "TRADING_STRATEGY_API_KEY missing"
    assert max_data_delay_minutes, "MAX_DATA_DELAY_MINUTES missing"

    client = Client.create_live_client(trading_strategy_api_key, cache_path=cache_path)

    client.clear_caches()

    max_data_delay = datetime.timedelta(minutes=max_data_delay_minutes)

    run_description: StrategyExecutionDescription = strategy_factory(
        execution_model=None,
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
    universe = universe_model.construct_universe(ts, ExecutionMode.data_preload)

    universe_model.check_data_age(ts, universe, max_data_delay)

    logger.info("All ok!")


@app.command()
def check_wallet(
    strategy_file: Path = typer.Option(..., envvar="STRATEGY_FILE"),
    private_key: str = typer.Option(None, envvar="PRIVATE_KEY"),
    json_rpc: str = typer.Option(None, envvar="JSON_RPC", help="Ethereum JSON-RPC node URL we connect to for execution"),
    trading_strategy_api_key: str = typer.Option(None, envvar="TRADING_STRATEGY_API_KEY", help="Trading Strategy API key"),
    cache_path: Optional[Path] = typer.Option(None, envvar="CACHE_PATH", help="Where to store downloaded datasets"),
):
    """Prints out the token balances of the hot wallet.

    TODO: Add balances also for non-reserve assets. This would need to mapping of the trading universe.
    """
    global logger

    logger = setup_logging()

    logger.info("Loading strategy file %s", strategy_file)
    strategy_factory = import_strategy_file(strategy_file)
    client = Client.create_live_client(trading_strategy_api_key, cache_path=cache_path)

    run_description: StrategyExecutionDescription = strategy_factory(
        execution_model=None,
        timed_task_context_manager=timed_task,
        sync_method=None,
        valuation_model_factory=None,
        pricing_model_factory=None,
        approval_model=None,
        client=client,
    )

    hot_wallet = HotWallet.from_private_key(private_key)

    universe_model: TradingStrategyUniverseModel = run_description.universe_model
    ts = datetime.datetime.utcnow()
    universe = universe_model.construct_universe(ts, live=False)

    reserve_assets = universe.reserve_assets

    logger.info("Connecting to %s", redact_url_password(json_rpc))

    web3 = create_web3(json_rpc)

    logger.info("Reserve assets are: %s", reserve_assets)
    tokens = [Web3.toChecksumAddress(a.address) for a in reserve_assets]

    logger.info("Checking JSON-RPC connection")

    logger.info(f"Latest block is {web3.eth.block_number:,}")

    logger.info("Hot wallet is %s", hot_wallet.address)
    logger.info("We have %f gas money left", web3.eth.get_balance(hot_wallet.address) / 10**18)
    balances = fetch_erc20_balances_by_token_list(web3, hot_wallet.address, tokens)
    for address, balance in balances.items():
        details = fetch_erc20_details(web3, address)
        logger.info("Balance of %s: %s %s", details.name, details.convert_to_decimals(balance), details.symbol)


