"""Process-lifetime advisory locks for executor state writers on Linux."""

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from io import TextIOWrapper
from pathlib import Path


class StateFileLocked(RuntimeError):
    """Another process owns the executor state file."""


_active_locks: set[TextIOWrapper] = set()


def _close_inherited_locks() -> None:
    """Children must close inherited handles without unlocking their parent."""
    for handle in _active_locks:
        handle.close()
    _active_locks.clear()


os.register_at_fork(after_in_child=_close_inherited_locks)


@contextmanager
def state_file_lock(path: Path | str) -> Iterator[Path]:
    """Own a state's sibling lock until exit, or fail immediately on contention.

    Use the yielded canonical path for all state operations. The lock file is
    deliberately retained: unlinking it could allow writers to lock different
    inodes. Kernel ownership, rather than file existence or a PID, is authoritative.
    """
    path = Path(path).resolve()
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a") as handle:
        # Register before acquisition so a fork cannot retain an untracked handle.
        _active_locks.add(handle)
        acquired = False
        try:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise StateFileLocked(
                    f"State file {path} is in use (lock: {lock_path}). "
                    "Stop the other executor or maintenance command before retrying."
                ) from exc
            acquired = True
            yield path
        finally:
            # An inherited context may unwind in a forked child after the hook
            # has closed its handle. It must not unlock the parent's lock.
            try:
                if acquired and not handle.closed:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
                _active_locks.discard(handle)
