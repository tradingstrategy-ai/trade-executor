"""Small real-process harness for state locking tests and Compose smoke checks."""

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
    """Wait for real ownership, then clean up the owner and any forked worker."""
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
    """Hold a lock until terminated, or attempt one non-blocking acquisition."""
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
