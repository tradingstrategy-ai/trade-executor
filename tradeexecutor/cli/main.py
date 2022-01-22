"""Command-line interface main entry point build on the top of typer.

https://typer.tiangolo.com/
"""
from pathlib import Path
from queue import Queue
from typing import Optional
import pkg_resources

import typer

from tradeexecutor.cli.loop import run_main_loop
from tradeexecutor.state.statemodel import StateModel
from tradeexecutor.trade.executionmodel import TradeInstructionExecutionModel
from tradeexecutor.cli.logging import setup_logging


# https://typer.tiangolo.com/tutorial/package/
from tradeexecutor.webhook.server import create_webhook_server

app = typer.Typer()


version = pkg_resources.get_distribution('tradeexecutor').version


@app.command()
def run(
    private_key: Optional[str] = typer.Option(None, envvar="PRIVATE_KEY"),
    strategy_file: Path = typer.Option(..., envvar="STRATEGY_FILE"),
    http_enabled: bool = typer.Option(True, envvar="HTTP_ENABLED", help="Enable Webhook server"),
    http_port: int = typer.Option(19000, envvar="HTTP_PORT"),
    http_host: str = typer.Option("0.0.0.0", envvar="HTTP_HOST"),
    http_username: str = typer.Option("webhook", envvar="HTTP_USERNAME"),
    http_password: str = typer.Option(None, envvar="HTTP_PASSWORD"),
    execution_model: TradeInstructionExecutionModel = typer.Option(..., envvar="EXECUTION_MODEL"),
    state_model: StateModel = typer.Option(..., envvar="STATE_MODEL"),
    state_file: Optional[Path] = typer.Option("strategy-state.json", envvar="STATE_FILE"),
    ):

    logger = setup_logging()
    logger.info("Trade Executor version %s starting", version)

    # Start the queue that relays info from the web server to the strategy executor
    command_queue = Queue()

    if http_enabled:
        server = create_webhook_server(http_host, http_port, http_username, http_password, command_queue)

    else:
        server = None

    try:
        run_main_loop()
    finally:
        if server:
            server.close()




