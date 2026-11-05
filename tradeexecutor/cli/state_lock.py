"""Command-boundary writer locking, preserving Typer's command signatures."""

import inspect
from collections.abc import Callable
from functools import wraps
from pathlib import Path

from typer.models import OptionInfo

from tradeexecutor.cli.bootstrap import locked_state_file, prepare_executor_id


def with_state_lock(
    *,
    default_suffix: str = "",
    read_only: Callable[[dict], bool] | None = None,
) -> Callable:
    """Lock a command's state before its body, including setup and error handlers.

    Commands explicitly opt out for modes which never persist state. Helpers
    called by a command must use its store instead of reacquiring ownership.
    """
    def decorate(command: Callable) -> Callable:
        signature = inspect.signature(command)

        @wraps(command)
        def locked_command(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            # Direct Python callers can leave Typer OptionInfo defaults unbound.
            options = {
                name: value.default if isinstance(value, OptionInfo) else value
                for name, value in bound.arguments.items()
            }
            path = options["state_file"]
            if not path:
                executor_id = prepare_executor_id(options.get("id"), options.get("strategy_file"))
                path = Path(f"state/{executor_id}{default_suffix}.json")
            enabled = not (read_only and read_only(options))
            with locked_state_file(path, enabled=enabled) as canonical:
                bound.arguments["state_file"] = canonical
                return command(*bound.args, **bound.kwargs)

        return locked_command

    return decorate
