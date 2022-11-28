"""Define Typer app root."""

import typer

from importlib.metadata import version

from tradeexecutor.cli.init import monkey_patch

app = typer.Typer()


TRADE_EXECUTOR_VERSION = version('trade-executor')

# Run this during the module loading so that it is
# applied to all subcommands
monkey_patch()
