"""Hold state-file ownership around an entire Typer command invocation.

The decorator resolves the command's state path before its body can load state,
create backups or submit transactions. It preserves the function signature used
by Typer to discover options and passes the canonical path into the command.
Commands declare non-writing modes explicitly; delegated helpers reuse the
caller's ownership rather than acquiring another lock.
"""

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
    """Decorate a Typer command to hold state ownership through its cleanup.

    Apply inside the shared RPC-option decorator, when present. The command
    must accept ``state_file`` and, for an inferred path, ``id`` or
    ``strategy_file``. Helpers called by the command must not reacquire the
    lock. Command option descriptions remain the source of CLI argument help.

    :param default_suffix:
        Suffix before ``.json`` in an inferred ``state/{executor_id}.json``
        path. Backtest outputs use ``-backtest``. Explicit paths are unchanged.
    :param read_only:
        Optional predicate receiving bound command options, including defaults.
        A true result skips locking while still passing the canonical state
        path. Use only for modes that neither persist state nor submit real
        transactions. Without a predicate, every invocation acquires the lock.
    :return:
        A decorator preserving the command signature and forwarding its result.
        Invocations raise :class:`click.ClickException` on lock contention.
    """
    def decorate(command: Callable) -> Callable:
        """Wrap a command without changing the signature Typer inspects.

        :param command:
            Command function accepting a ``state_file`` argument.
        :return:
            A callable that acquires ownership before running the command.
        """
        signature = inspect.signature(command)

        @wraps(command)
        def locked_command(*args, **kwargs):
            """Resolve state ownership and invoke the wrapped command.

            :param args:
                Positional arguments bound against the original signature.
            :param kwargs:
                Keyword command options, normally already parsed by Typer.
            :return:
                The wrapped command's result, after its cleanup has completed.
            :raises click.ClickException:
                Another writer owns the selected state file.
            """
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
