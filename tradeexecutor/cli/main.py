"""Command-line interface main entry point build on the top of typer.

https://typer.tiangolo.com/
"""
from pathlib import Path
from queue import Queue
from typing import Optional
import pkg_resources

import typer

from tradeexecutor.cli.loop import run_main_loop
from tradeexecutor.state.statestoremodel import StateStoreModel
from tradeexecutor.state.file import FileStore
from tradeexecutor.strategy.importer import import_strategy_file
from tradeexecutor.trade.dummy import DummyExecutionModel
from tradeexecutor.trade.executionmodel import TradeExecutionModel
from tradeexecutor.cli.logging import setup_logging
from tradeexecutor.trade.hotwallet import HotWalletExecutionModel
from tradeexecutor.webhook.server import create_webhook_server

app = typer.Typer()


version = pkg_resources.get_distribution('tradeexecutor').version


def create_trade_execution_model(execution_model: TradeExecutionModel, private_key: str):
    if execution_models == TradeExecutionModel.dummy:
        return DummyExecutionModel()
    elif execution_model == TradeExecutionModel.hot_wallet:
        assert private_key, "Private key is needed"
        return HotWalletExecutionModel(private_key)
    else:
        raise NotImplementedError()


def create_state_model(state_model: StateStoreModel, state_file: Path):
    if state_model == StateStoreModel.file:
        assert state_file, "State file required"
        return FileStore(state_file)
    else:
        raise NotImplementedError()


@app.command()
def run(
    private_key: Optional[str] = typer.Option(None, envvar="PRIVATE_KEY"),
    strategy_file: Path = typer.Option(..., envvar="STRATEGY_FILE"),
    http_enabled: bool = typer.Option(True, envvar="HTTP_ENABLED", help="Enable Webhook server"),
    http_port: int = typer.Option(19000, envvar="HTTP_PORT"),
    http_host: str = typer.Option("0.0.0.0", envvar="HTTP_HOST"),
    http_username: str = typer.Option("webhook", envvar="HTTP_USERNAME"),
    http_password: str = typer.Option(None, envvar="HTTP_PASSWORD"),
    execution_model: TradeExecutionModel = typer.Option(..., envvar="EXECUTION_MODEL"),
    state_model: StateStoreModel = typer.Option(..., envvar="STATE_MODEL"),
    state_file: Optional[Path] = typer.Option("strategy-state.json", envvar="STATE_FILE"),
    ):

    logger = setup_logging()
    logger.info("Trade Executor version %s starting", version)

    execution_model = create_trade_execution_model(execution_model, private_key)

    state_model = create_state_model(state_model, state_file)

    strategy_runner = import_strategy_file(strategy_file)

    execution_model.preflight_check()
    state_model.preflight_check()
    strategy_runner.preflight_check()

    # Start the queue that relays info from the web server to the strategy executor
    command_queue = Queue()

    if http_enabled:
        server = create_webhook_server(http_host, http_port, http_username, http_password, command_queue)
    else:
        server = None

    try:
        run_main_loop(command_queue, execution_model, state_model, strategy_runner)
    finally:
        if server:
            server.close()




