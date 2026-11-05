"""Regression coverage for state-file ownership and owner-death recovery.

Real competing processes exercise kernel exclusion, while canonical-path and
atomic-replacement checks protect the state-store integration. Forked-worker
coverage verifies both that a live parent's lock survives and that its death
does not strand ownership in an unrelated worker. No chain access is required.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tradeexecutor.state.lock import StateFileLocked, state_file_lock
from tradeexecutor.state.state import State
from tradeexecutor.state.store import JSONFileStore
from tradeexecutor.testing.state_lock import hold_state_lock_in_subprocess


def test_state_file_lock_lifetime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Protect complete writer sessions without preventing snapshot reads.

    1. Lock a canonical state and verify contention through path aliases.
    2. Save an atomic replacement and verify readers and independent writers work.
    3. Verify release after an exception without replacing the lock inode.

    :param tmp_path:
        Pytest directory for isolated state, alias and lock files.
    :param monkeypatch:
        Pytest helper used to exercise relative paths from that directory.
    """
    # 1. Lock a canonical state and verify contention through path aliases.
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "executor.json"
    alias = tmp_path / "alias.json"
    alias.symlink_to(path)
    store = JSONFileStore(path)
    state = State()
    state.name = "Lock test"
    store.sync(state)
    lock_path = path.with_name(path.name + ".lock")
    with pytest.raises(ValueError, match="interrupted"):
        with state_file_lock(Path("executor.json")) as canonical:
            assert canonical == path
            inode = lock_path.stat().st_ino
            with pytest.raises(StateFileLocked):
                with state_file_lock(alias):
                    pytest.fail("Alias bypassed writer ownership")

            # 2. Save an atomic replacement and verify readers and independent writers work.
            store.sync(state)
            assert store.load().name == state.name
            with state_file_lock(tmp_path / "other.json"):
                pass
            contender = subprocess.run(
                [sys.executable, "-m", "tradeexecutor.testing.state_lock", str(alias), "--try-lock"],
                capture_output=True, text=True, timeout=20,
            )
            assert contender.returncode == 1
            assert str(lock_path) in contender.stderr
            raise ValueError("interrupted")

    # 3. Verify release after an exception without replacing the lock inode.
    with state_file_lock(alias) as canonical:
        assert canonical == path
        assert lock_path.stat().st_ino == inode
    with state_file_lock(path):
        pass
    assert lock_path.exists()
    assert not alias.with_name(alias.name + ".lock").exists()


@pytest.mark.parametrize("fork_worker", [False, True])
def test_state_file_lock_owner_death(tmp_path: Path, fork_worker: bool) -> None:
    """Recover ownership even when a forked worker survives a killed writer.

    1. Start a real owner, optionally with an inherited worker, and verify exclusion.
    2. Kill the owner and confirm the worker can remain alive.
    3. Reacquire the surviving lock file immediately without manual cleanup.

    :param tmp_path:
        Pytest directory containing the lock used by both competing processes.
    :param fork_worker:
        Include a surviving forked worker when true; otherwise test owner death
        without descriptor inheritance.
    """
    # 1. Start a real owner, optionally with an inherited worker, and verify exclusion.
    path = tmp_path / "executor.json"
    with hold_state_lock_in_subprocess(path, fork_worker=fork_worker) as (owner, worker_pid):
        inode = path.with_name(path.name + ".lock").stat().st_ino
        with pytest.raises(StateFileLocked):
            with state_file_lock(path):
                pytest.fail("A live owner's child released its lock")

        # 2. Kill the owner and confirm the worker can remain alive.
        owner.kill()
        owner.wait(timeout=10)
        if worker_pid is not None:
            os.kill(worker_pid, 0)

        # 3. Reacquire the surviving lock file immediately without manual cleanup.
        with state_file_lock(path):
            assert path.with_name(path.name + ".lock").stat().st_ino == inode
