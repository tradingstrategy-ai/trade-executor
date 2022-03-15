"""Command-line interface main entry point build on the top of typer.

https://typer.tiangolo.com/
"""
import datetime
from pathlib import Path
from queue import Queue
from typing import Optional
import pkg_resources

import typer
from tradeexecutor.strategy.tick import TickSize
from web3 import Web3, HTTPProvider
import pandas as pd

from eth_hentai.hotwallet import HotWallet
from eth_hentai.uniswap_v2 import fetch_deployment
from tradeexecutor.cli.approval import CLIApprovalModel
from tradeexecutor.cli.loop import run_main_loop
from tradeexecutor.ethereum.hot_wallet_sync import EthereumHotWalletReserveSyncer

from tradeexecutor.ethereum.uniswap_v2_execution import UniswapV2ExecutionModel
from tradeexecutor.ethereum.uniswap_v2_live_pricing import UniswapV2LivePricing, uniswap_v2_live_pricing_factory
from tradeexecutor.ethereum.uniswap_v2_revaluation import UniswapV2PoolRevaluator
from tradeexecutor.state.store import JSONFileStore, StateStore
from tradeexecutor.strategy.approval import ApprovalType, UncheckedApprovalModel, ApprovalModel
from tradeexecutor.strategy.bootstrap import import_strategy_file
from tradeexecutor.strategy.dummy import DummyExecutionModel
from tradeexecutor.strategy.execution_type import TradeExecutionType
from tradeexecutor.cli.logging import setup_logging
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
):
    if execution_type == TradeExecutionType.dummy:
        return DummyExecutionModel()
    elif execution_type == TradeExecutionType.uniswap_v2_hot_wallet:
        assert private_key, "Private key is needed"
        assert factory_address, "Uniswap v2 factory address needed"
        assert router_address, "Uniswap v2 factory router needed"
        assert json_rpc, "JSON-RPC endpoint is needed"
        web3 = create_web3(json_rpc)
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
def run(
    private_key: Optional[str] = typer.Option(None, envvar="PRIVATE_KEY"),
    strategy_file: Path = typer.Option(..., envvar="STRATEGY_FILE"),
    http_enabled: bool = typer.Option(True, envvar="HTTP_ENABLED", help="Enable Webhook server"),
    http_port: int = typer.Option(19000, envvar="HTTP_PORT"),
    http_host: str = typer.Option("0.0.0.0", envvar="HTTP_HOST"),
    http_username: str = typer.Option("webhook", envvar="HTTP_USERNAME"),
    http_password: str = typer.Option(None, envvar="HTTP_PASSWORD"),
    json_rpc: str = typer.Option(None, envvar="JSON_RPC"),
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
    tick_hours: TickSize = typer.Option(None, envvar="TICK_HOURS", help="How large tick use to execute the strategy"),
    ):

    logger = setup_logging()
    logger.info("Trade Executor version %s starting", version)

    execution_model, sync_method, revaluation_method, pricing_model_factory = create_trade_execution_model(
        execution_type,
        uniswap_v2_factory_address,
        uniswap_v2_router_address,
        uniswap_init_code_hash,
        json_rpc,
        private_key)

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
    else:
        client = None

    strategy_factory = import_strategy_file(strategy_file)

    try:
        run_main_loop(
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
            tick_hours=tick_hours,
        )
    finally:
        if server:
            server.close()
