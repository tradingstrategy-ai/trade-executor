"""Define Typer app root."""

import typer
import shutil
from importlib.metadata import version

from tradeexecutor.cli.init import monkey_patch


# https://github.com/tiangolo/typer/issues/511#issuecomment-1331692007
app = typer.Typer(context_settings={
    "max_content_width": shutil.get_terminal_size().columns
})


TRADE_EXECUTOR_VERSION = version('trade-executor')

# Run this during the module loading so that it is
# applied to all subcommands
monkey_patch()
