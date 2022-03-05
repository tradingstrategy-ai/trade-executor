"""Command-line interface main entry point build on the top of typer.

https://typer.tiangolo.com/
"""
from pathlib import Path
from queue import Queue
from typing import Optional
import pkg_resources

import typer
from web3 import Web3, HTTPProvider

from eth_hentai.hotwallet import HotWallet
from eth_hentai.uniswap_v2 import fetch_deployment
from tradeexecutor.cli.approval import CLIApprovalModel
from tradeexecutor.cli.loop import run_main_loop
from tradeexecutor.ethereum.hot_wallet_sync import EthereumHotWalletReserveSyncer

from tradeexecutor.ethereum.uniswap_v2_execution import UniswapV2ExecutionModel
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
        uniswap = fetch_deployment(web3, factory_address, router_address)
        sync_model = EthereumHotWalletReserveSyncer(web3, hot_wallet.address)
        return UniswapV2ExecutionModel(web3, uniswap, hot_wallet), sync_model
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
    state_file: Optional[Path] = typer.Option("strategy-state.json", envvar="STATE_FILE"),
    trading_strategy_api_key: str = typer.Option(None, envvar="TRADING_STRATEGY_API_KEY", help="Trading Strategy API key"),
    reset_state: bool = typer.Option(False, "--reset-state"),
    max_cycles: int = typer.Option(None, env_var="MAX_CYCLES")
    ):

    logger = setup_logging()
    logger.info("Trade Executor version %s starting", version)

    execution_model, sync_model = create_trade_execution_model(
        execution_type,
        uniswap_v2_factory_address,
        uniswap_v2_router_address,
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
        client = Client.create_live_client(trading_strategy_api_key)
    else:
        client = None

    strategy_factory = import_strategy_file(strategy_file)

    try:
        run_main_loop(
            command_queue,
            execution_model
            sync_model,
            approval_model,
            store,
            client,
            strategy_factory,
            reset=reset_state,
            max_cycles=max_cycles)
    finally:
        if server:
            server.close()
