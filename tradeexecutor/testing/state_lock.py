"""Real-process fixtures for executor state-lock contention and crash recovery.

The fixture starts an isolated process group and waits for confirmed lock
ownership before yielding. An optional forked worker acknowledges that its
inherited lock handles have been closed, allowing tests to kill the owner
without a readiness race. Fixture cleanup kills the whole isolated group.

The same module can run in disposable Compose containers with only the shared
state directory mounted for writing::

    poetry run python -m tradeexecutor.testing.state_lock /state/example.json
    poetry run python -m tradeexecutor.testing.state_lock /state/example.json --try-lock

These entry points acquire locks only; they never load state or submit trades.
"""

import argparse
import json
import os
import select
import signal
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from tradeexecutor.state.lock import StateFileLocked, state_file_lock


@contextmanager
def hold_state_lock_in_subprocess(
    path: Path, *, fork_worker: bool = False,
) -> Iterator[tuple[subprocess.Popen, int | None]]:
    """Start a separate lock owner and wait until it is ready for assertions.

    Real processes exercise kernel exclusion and owner-death behaviour that a
    mocked lock cannot establish. Cleanup kills the isolated process group,
    including workers surviving an interrupted owner or failed readiness check.

    :param path:
        State path passed to the subprocess. Its parent directory must exist;
        no state JSON is loaded or created.
    :param fork_worker:
        Whether to fork a worker that remains alive independently of the owner.
        Readiness is reported after the worker closes inherited lock handles.
    :return:
        A context manager yielding ``(owner_process, worker_pid)``. The PID is
        ``None`` when no worker was requested. The owner holds the lock on entry.
    :raises TimeoutError:
        Ownership is not confirmed within the readiness deadline.
    :raises RuntimeError:
        The subprocess exits before reporting ownership.
    """
    command = [sys.executable, "-m", "tradeexecutor.testing.state_lock", str(path)]
    if fork_worker:
        command.append("--fork-worker")
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    try:
        ready, _, _ = select.select([process.stdout], [], [], 20)
        if not ready:
            raise TimeoutError("State lock subprocess did not report readiness")
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f"State lock subprocess failed: {process.stderr.read()}")
        worker_pid = json.loads(line)["worker_pid"]
        yield process, worker_pid
    finally:
        # The isolated process group also covers a worker whose owner died
        # before reporting readiness, so a failed test cannot strand it.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)
        process.stdout.close()
        process.stderr.close()


def main() -> int:
    """Run the lock-fixture subprocess selected by command-line arguments.

    The positional argument selects the state path. ``--fork-worker`` starts
    the surviving-worker scenario; ``--try-lock`` releases ownership and exits
    after acquisition. Otherwise the owner waits for process termination.
    Successful acquisition emits one JSON readiness record on stdout; contention
    emits the operator error on stderr without reading or changing the JSON.

    :return:
        Exit status 0 for a successful probe, or 1 for lock contention. Argument
        parsing and other filesystem failures retain their normal exceptions.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state_file", type=Path)
    parser.add_argument("--fork-worker", action="store_true")
    parser.add_argument("--try-lock", action="store_true")
    args = parser.parse_args()
    try:
        with state_file_lock(args.state_file):
            worker_pid = None
            if args.fork_worker:
                ready_read, ready_write = os.pipe()
                worker_pid = os.fork()
                if worker_pid == 0:
                    os.close(ready_read)
                    # The at-fork hook has closed the inherited lock by now.
                    os.write(ready_write, b"1")
                    os.close(ready_write)
                    while True:
                        signal.pause()
                os.close(ready_write)
                try:
                    assert os.read(ready_read, 1) == b"1", "Worker exited before readiness"
                finally:
                    os.close(ready_read)
            sys.stdout.write(json.dumps({"worker_pid": worker_pid}) + "\n")
            sys.stdout.flush()
            if not args.try_lock:
                while True:
                    signal.pause()
    except StateFileLocked as exc:
        sys.stderr.write(str(exc) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
