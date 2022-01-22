"""Command-line interface main entry point build on the top of typer.

https://typer.tiangolo.com/
"""
from pathlib import Path
from typing import Optional

import typer

from tradeexecutor.state.statemodel import StateModel
from tradeexecutor.trade.executionmodel import TradeInstructionExecutionModel


# https://typer.tiangolo.com/tutorial/package/
app = typer.Typer()


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
    pass




