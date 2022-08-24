"""Environment variable management."""
from dataclasses import dataclass
from typing import List

from click import Context
from typer.main import get_command

from tradeexecutor.cli.main import app


@dataclass
class EnvVarDescription:
    name: str
    help: str
    type: str


def get_available_env_vars() -> List[EnvVarDescription]:
    """Get list of environment variable configuration options for trade-executor.

    :return:
        List of environment variable names
    """
    command = get_command(app)
    start = command.commands["start"]
    ctx = Context(start)
    params = start.get_params(ctx)
    result = []
    for p in params:
        envvar = p.envvar
        if envvar:
            # Option --help does not have envvar, etc.
            result.append(
                EnvVarDescription(
                    envvar,
                    p.help,
                    p.type,
                )
            )

    return result
