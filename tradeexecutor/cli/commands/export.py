
"""Export CLI command."""
import datetime
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
        cmd = get_command_from_info(info, pretty_exceptions_short=True, rich_markup_mode="markdown")
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

    Besides the settings export, you need to copy over the state file
    and Python strategy module, and you have
    encapsulated everything a trade executor takes as inputs.
    """

    env_var_set = set()

    print(f"# Trade executor settings export, created {datetime.datetime.utcnow()} UTC")
    print("# ")
    print("# Save to a local file and then import with Bash source command")
    print("# ")

    for param in walk_all_typer_options(app):
        if not param.envvar:
            print(f"# Cannot export {param.name} as it lacks environment variable")
        else:
            env_var_set.add(param.envvar)

    env_vars = [e for e in env_var_set]
    env_vars.sort()

    # TODO: We do not deal with the case if env var contains "
    for env in env_vars:

        val = os.environ.get(env, '')

        if val:
            print(f"""export {env}="{val}" """)
        else:
            print(f"""# export {env}="{val}" """)



