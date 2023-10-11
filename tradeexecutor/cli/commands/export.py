
"""version CLi command."""
import os
from typing import Iterable

from typer import Context, Typer
from typer.core import TyperOption
from typer.main import get_command_from_info

from tradeexecutor.cli.commands.app import app


def walk_all_typer_options(app: Typer) -> Iterable[TyperOption]:
    """Get all options of all commands.

    Command groups not supported.
    """
    for info in app.registered_commands:
        cmd = get_command_from_info(info)
        for param in cmd.params:
            yield param


@app.command()
def export():
    """Export the configuration of the trade executor.

    The main purpose of this command is to be able
    to transfer the trade executor state anf configuration to a debugging
    environment. It will print out the current environment
    variable configuration to stdout.

    The export command only works on Dockerised trade-executor
    instances where all options are passed as environment variables.

    Export all environment variables configured for this executor.
    This includes associated private keys, so it is not safe to
    give this export to anyone.

    The export is in bash shell script source format.
    """

    env_var_set = set()

    for param in walk_all_typer_options(app):
        if not param.envvar:
            print(f"# Cannot export {param.name} as it lacks environment variable")
        else:
            env_var_set.add(param.envvar)

    for env in env_var_set:
        print(f"export {env}={os.environ.get(env, '')}")



