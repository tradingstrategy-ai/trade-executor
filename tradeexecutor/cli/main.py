"""Command-line interface main entry point build on the top of typer.

https://typer.tiangolo.com/
"""
import datetime
from pathlib import Path
from queue import Queue
from typing import Optional
import pkg_resources

import typer
from web3.middleware import geth_poa_middleware

from eth_hentai.balances import fetch_erc20_balances_by_token_list
from eth_hentai.gas import GasPriceMethod, node_default_gas_price_strategy
from eth_hentai.token import fetch_erc20_details
from tradeexecutor.strategy.description import StrategyExecutionDescription
from tradeexecutor.strategy.tick import TickSize
from web3 import Web3, HTTPProvider

from eth_hentai.hotwallet import HotWallet
from eth_hentai.uniswap_v2.deployment import fetch_deployment
from tradeexecutor.cli.approval import CLIApprovalModel
from tradeexecutor.cli.loop import run_main_loop
from tradeexecutor.ethereum.hot_wallet_sync import EthereumHotWalletReserveSyncer

from tradeexecutor.ethereum.uniswap_v2_execution import UniswapV2ExecutionModel
from tradeexecutor.ethereum.uniswap_v2_live_pricing import uniswap_v2_live_pricing_factory
from tradeexecutor.ethereum.uniswap_v2_revaluation import UniswapV2PoolRevaluator
from tradeexecutor.state.store import JSONFileStore, StateStore
from tradeexecutor.strategy.approval import ApprovalType, UncheckedApprovalModel, ApprovalModel
from tradeexecutor.strategy.bootstrap import import_strategy_file
from tradeexecutor.strategy.dummy import DummyExecutionModel
from tradeexecutor.strategy.execution_type import TradeExecutionType
from tradeexecutor.cli.log import setup_logging, setup_discord_logging
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverseModel
from tradeexecutor.utils.timer import timed_task
from tradeexecutor.webhook.server import create_webhook_server
from tradingstrategy.client import Client

app = typer.Typer()


version = pkg_resources.get_distribution('tradeexecutor').version


def create_trade_execution_model(
        execution_type: TradeExecutionType,
        factory_address: str,
        router_address: str,
        uniswap_init_code_hash,
        json_rpc: str,
        private_key: str,
        gas_price_method: Optional[GasPriceMethod],
):
    if execution_type == TradeExecutionType.dummy:
        return DummyExecutionModel()
    elif execution_type == TradeExecutionType.uniswap_v2_hot_wallet:
        assert private_key, "Private key is needed"
        assert factory_address, "Uniswap v2 factory address needed"
        assert router_address, "Uniswap v2 factory router needed"
        assert json_rpc, "JSON-RPC endpoint is needed"
        web3 = create_web3(json_rpc)

        # London is the default method
        if gas_price_method == GasPriceMethod.legacy:
            web3.eth.set_gas_price_strategy(node_default_gas_price_strategy)
            # Also assume BSC, set POA middleware
            web3.middleware_onion.inject(geth_poa_middleware, layer=0)

        hot_wallet = HotWallet.from_private_key(private_key)
        uniswap = fetch_deployment(web3, factory_address, router_address, init_code_hash=uniswap_init_code_hash)
        sync_method = EthereumHotWalletReserveSyncer(web3, hot_wallet.address)
        execution_model = UniswapV2ExecutionModel(uniswap, hot_wallet)
        revaluation_method = UniswapV2PoolRevaluator(uniswap)
        pricing_model_factory = uniswap_v2_live_pricing_factory
        return execution_model, sync_method, revaluation_method, pricing_model_factory
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


def create_web3(url) -> Web3:
    return Web3(HTTPProvider(url))


# Typer documentation https://typer.tiangolo.com/
@app.command()
def start(
    name: Optional[str] = typer.Option("Unnamed Trade Executor", envvar="NAME", help="Executor name used in logging and notifications"),
    private_key: Optional[str] = typer.Option(None, envvar="PRIVATE_KEY", help="Ethereum private key to be used as a hot wallet/broadcast wallet"),
    strategy_file: Path = typer.Option(..., envvar="STRATEGY_FILE", help="Python strategy file to run"),
    http_enabled: bool = typer.Option(True, envvar="HTTP_ENABLED", help="Enable Webhook server"),
    http_port: int = typer.Option(19000, envvar="HTTP_PORT"),
    http_host: str = typer.Option("0.0.0.0", envvar="HTTP_HOST"),
    http_username: str = typer.Option("webhook", envvar="HTTP_USERNAME"),
    http_password: str = typer.Option(None, envvar="HTTP_PASSWORD"),
    json_rpc: str = typer.Option(None, envvar="JSON_RPC", help="Ethereum JSON-RPC node URL we connect to for execution"),
    gas_price_method: Optional[GasPriceMethod] = typer.Option(None, envvar="GAS_PRICE_METHOD", help="How to set the gas price for Ethereum transactions"),
    execution_type: TradeExecutionType = typer.Option(..., envvar="EXECUTION_TYPE"),
    approval_type: ApprovalType = typer.Option(..., envvar="APPROVAL_TYPE"),
    uniswap_v2_factory_address: str = typer.Option(None, envvar="UNISWAP_V2_FACTORY_ADDRESS"),
    uniswap_v2_router_address: str = typer.Option(None, envvar="UNISWAP_V2_ROUTER_ADDRESS"),
    uniswap_init_code_hash: str = typer.Option(None, envvar="UNISWAP_V2_INIT_CODE_HASH"),
    state_file: Optional[Path] = typer.Option("strategy-state.json", envvar="STATE_FILE"),
    trading_strategy_api_key: str = typer.Option(None, envvar="TRADING_STRATEGY_API_KEY", help="Trading Strategy API key"),
    cache_path: Optional[Path] = typer.Option(None, envvar="CACHE_PATH", help="Where to store downloaded datasets"),
    reset_state: bool = typer.Option(False, "--reset-state", envvar="RESET_STATE"),
    max_cycles: int = typer.Option(None, envvar="MAX_CYCLES", help="Max main loop cycles run in an automated testing mode"),
    debug_dump_file: Optional[Path] = typer.Option(None, envvar="DEBUG_DUMP_FILE", help="Write Python Pickle dump of all internal debugging states of the strategy run to this file"),
    backtest_start: Optional[datetime.datetime] = typer.Option(None, envvar="BACKTEST_START", help="Start timestamp of backesting"),
    backtest_end: Optional[datetime.datetime] = typer.Option(None, envvar="BACKTEST_END", help="End timestamp of backesting"),
    tick_size: TickSize = typer.Option(None, envvar="TICK_SIZE", help="How large tick use to execute the strategy"),
    tick_offset_minutes: int = typer.Option(0, envvar="TICK_OFFSET_MINUTES", help="How many minutes we wait after the tick before executing the tick step"),
    max_data_delay_minutes: int = typer.Option(None, envvar="MAX_DATA_DELAY_MINUTES", help="If our data feed is delayed more than this minutes, abort the execution"),
    discord_webhook_url: Optional[str] = typer.Option(None, envvar="DISCORD_WEBHOOK_URL", help="Discord webhook URL for notifications"),
    discord_avatar_url: Optional[str] = typer.Option(None, envvar="DISCORD_AVATAR_URL", help="Discord avatar image URL for notifications"),
    trade_immediately: bool = typer.Option(False, "--trade-immediately", envvar="TRADE_IMMEDIATELY", help="Perform the first rebalance immediately, do not wait for the next trading universe refresh"),
    port_mortem_debugging: bool = typer.Option(False, "--post-mortem-debugging", envvar="POST_MORTEM_DEBUGGING", help="Launch ipdb debugger on a main loop crash to debug the exception"),
    clear_caches: bool = typer.Option(False, "--clear-caches", envvar="CLEAR_CACHES", help="Purge any dataset download caches before starting"),
    ):
    """Launch Trade Executor instance."""

    logger = setup_logging()

    if discord_webhook_url:
        setup_discord_logging(
            name,
            webhook_url=discord_webhook_url,
            avatar_url=discord_avatar_url)

    logger.trade("Trade Executor version %s starting strategy %s", version, name)

    execution_model, sync_method, revaluation_method, pricing_model_factory = create_trade_execution_model(
        execution_type,
        uniswap_v2_factory_address,
        uniswap_v2_router_address,
        uniswap_init_code_hash,
        json_rpc,
        private_key,
        gas_price_method,
    )

    approval_model = create_approval_model(approval_type)

    store = create_state_store(state_file)

    # Start the queue that relays info from the web server to the strategy executor
    command_queue = Queue()

    # Create our webhook server
    if http_enabled:
        server = create_webhook_server(http_host, http_port, http_username, http_password, command_queue)
    else:
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

    logger.info("Loading strategy file %s", strategy_file)
    strategy_factory = import_strategy_file(strategy_file)

    try:
        run_main_loop(
            name=name,
            command_queue=command_queue,
            execution_model=execution_model,
            sync_method=sync_method,
            approval_model=approval_model,
            pricing_model_factory=pricing_model_factory,
            revaluation_method=revaluation_method,
            store=store,
            client=client,
            strategy_factory=strategy_factory,
            reset=reset_state,
            max_cycles=max_cycles,
            debug_dump_file=debug_dump_file,
            backtest_start=backtest_start,
            backtest_end=backtest_end,
            tick_size=tick_size,
            tick_offset=tick_offset,
            max_data_delay=max_data_delay,
            trade_immediately=trade_immediately,
        )
    except Exception as e:
        # Debug exceptions in production
        if port_mortem_debugging:
            import ipdb
            ipdb.post_mortem()
        logger.exception(e)
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

    logger = setup_logging()

    logger.info("Loading strategy file %s", strategy_file)
    strategy_factory = import_strategy_file(strategy_file)

    assert trading_strategy_api_key, "TRADING_STRATEGY_API_KEY missing"
    assert max_data_delay_minutes, "MAX_DATA_DELAY_MINUTES missing"

    client = Client.create_live_client(trading_strategy_api_key, cache_path=cache_path)

    max_data_delay = datetime.timedelta(minutes=max_data_delay_minutes)

    run_description: StrategyExecutionDescription = strategy_factory(
        execution_model=None,
        timed_task_context_manager=timed_task,
        sync_method=None,
        revaluation_method=None,
        pricing_model_factory=None,
        approval_model=None,
        client=client,
    )

    # Deconstruct strategy input
    universe_model: TradingStrategyUniverseModel = run_description.universe_model

    ts = datetime.datetime.utcnow()
    logger.info("Performing universe data check for timestamp %s", ts)
    universe = universe_model.construct_universe(ts, live=True)

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

    logger = setup_logging()

    logger.info("Loading strategy file %s", strategy_file)
    strategy_factory = import_strategy_file(strategy_file)
    client = Client.create_live_client(trading_strategy_api_key, cache_path=cache_path)

    run_description: StrategyExecutionDescription = strategy_factory(
        execution_model=None,
        timed_task_context_manager=timed_task,
        sync_method=None,
        revaluation_method=None,
        pricing_model_factory=None,
        approval_model=None,
        client=client,
    )

    hot_wallet = HotWallet.from_private_key(private_key)

    # Deconstruct strategy input
    universe_model: TradingStrategyUniverseModel = run_description.universe_model

    ts = datetime.datetime.utcnow()
    universe = universe_model.construct_universe(ts, live=False)
    reserve_assets = universe.reserve_assets
    web3 = create_web3(json_rpc)
    tokens = [web3.toChecksumAddress(a.address) for a in reserve_assets]
    balances = fetch_erc20_balances_by_token_list(web3, hot_wallet.address, tokens)
    logger.info("Balances of %s", hot_wallet.address)
    for address, balance in balances.items():
        details = fetch_erc20_details(web3, address)
        logger.info("%s: %s %s", details.name, details.convert_to_decimals(balance), details.symbol)



