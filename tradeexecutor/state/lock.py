"""Exclusive writer ownership for executor state files on Linux.

Atomic file replacement prevents partial JSON reads but cannot prevent two
writers overwriting each other's changes. :func:`state_file_lock` uses
:func:`fcntl.flock` on a separate file beside the state JSON, so ownership
survives replacement of the JSON file. Callers must hold the context across
the entire read, modify and save operation.

Ownership ends when the context exits or the owning process dies. The lock
file remains on disk and is reused. Forked workers close inherited descriptors
without releasing the parent's ownership. All writers must participate and
share a filesystem that supports these advisory locks.
"""

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from io import TextIOWrapper
from pathlib import Path


class StateFileLocked(RuntimeError):
    """A writer already holds the advisory lock for the requested state file.

    Raised immediately on contention, including a second acquisition from the
    same process. The exception message identifies the state and lock paths.
    """


_active_locks: set[TextIOWrapper] = set()


def _close_inherited_locks() -> None:
    """Drop a forked child's references to the parent's active lock handles.

    Registered with :func:`os.register_at_fork` because close-on-exec does not
    prevent descriptor inheritance across a fork. Closing child handles lets
    ownership end if the parent dies while a worker survives. Explicitly
    unlocking here would also release the live parent's lock.
    """
    for handle in _active_locks:
        handle.close()
    _active_locks.clear()


os.register_at_fork(after_in_child=_close_inherited_locks)


@contextmanager
def state_file_lock(path: Path | str) -> Iterator[Path]:
    """Acquire an exclusive, non-blocking lock for an executor state file.

    Enter this context before loading state and keep it open through the final
    save and any transaction submission. Ownership lasts for the ``with``
    block and is released on normal exit, exceptions or process termination.
    Other writers must use the same locking mechanism.

    The lock is taken on ``<resolved state filename>.lock``, for example
    ``state/hyper-ai.json.lock``. This separate file keeps its inode when the
    JSON is atomically replaced. It is never deleted: an unlocked file left
    after a crash is safe to reuse, whereas deleting it could let writers lock
    different inodes for the same state.

    Example::

        from tradeexecutor.state.store import JSONFileStore

        with state_file_lock("state/my-strategy.json") as state_path:
            store = JSONFileStore(state_path)
            state = store.load()
            # Apply state changes while ownership is held.
            store.sync(state)

    :param path:
        State JSON path. Relative paths and symlinks are resolved before
        selecting the adjacent lock file. The parent directory must exist and
        permit opening the lock file for writing; the JSON need not exist yet.
    :return:
        A context manager yielding the canonical absolute :class:`pathlib.Path`.
        Use this path for all state reads, backups and writes in the context.
    :raises StateFileLocked:
        Another writer holds the lock. Acquisition does not wait or retry.
    :raises OSError:
        Opening or operating on the lock file fails for another filesystem
        reason, such as a missing directory or insufficient permissions.
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
