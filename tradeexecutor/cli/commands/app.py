"""Define Typer app root."""

import typer
import shutil

from tradeexecutor.cli.bootstrap import monkey_patch


# https://github.com/tiangolo/typer/issues/511#issuecomment-1331692007
app = typer.Typer(
    name="Trade Executor",
    context_settings={
        "max_content_width": shutil.get_terminal_size().columns
    },
    # Typer swallows nested exceptions
    # https://github.com/tiangolo/typer/issues/129
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
)


#: We do not use Python package versions, but Docker image files to track the deployed version
#:
#: See `version_info.py` for more information.
#: The variable is just left here as a place holder if we ever
#: move to package versioning again.
TRADE_EXECUTOR_VERSION = NotImplemented

# Run this during the module loading so that it is
# applied to all subcommands
monkey_patch()
