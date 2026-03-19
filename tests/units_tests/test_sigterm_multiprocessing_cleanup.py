"""Regression tests for multiprocessing SIGTERM cleanup helpers."""

import pickle
import signal
from types import SimpleNamespace

import pytest

from tradeexecutor.backtest import grid_search
from tradeexecutor.strategy.pandas_trader import indicator
from tradeexecutor.utils import multiprocessing_signal


class DummyProcess:
    """Minimal child-process stub for shutdown handler tests."""

    def __init__(self) -> None:
        self.kill_calls = 0

    def kill(self) -> None:
        self.kill_calls += 1


class DummyPool:
    """Minimal process-pool stub for shutdown handler tests."""

    def __init__(self, processes: dict[int, DummyProcess] | None = None) -> None:
        self._processes = processes or {}
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1


@pytest.mark.timeout(300)
def test_indicator_setup_and_sigterm_cleanup_restore_internal_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test indicator multiprocessing cleanup uses the registered process pool.

    1. Register a dummy process pool through the normal indicator setup helper.
    2. Trigger the SIGTERM cleanup handler with a pool that has child-process stubs.
    3. Confirm the pool is shut down, children are killed, and the global pool reference is cleared.
    """
    dummy_processes = {
        1: DummyProcess(),
        2: DummyProcess(),
    }
    dummy_pool = DummyPool(dummy_processes)
    dummy_executor = SimpleNamespace(_executor=dummy_pool)

    # 1. Register a dummy process pool through the normal indicator setup helper.
    monkeypatch.setattr(indicator.signal, "signal", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(indicator, "_process_pool", None)
    indicator.setup_indicator_multiprocessing(dummy_executor)

    # 2. Trigger the SIGTERM cleanup handler with a pool that has child-process stubs.
    with pytest.raises(SystemExit):
        indicator._handle_sigterm()

    # 3. Confirm the pool is shut down, children are killed, and the global pool reference is cleared.
    assert dummy_pool.shutdown_calls == 1
    assert all(process.kill_calls == 1 for process in dummy_processes.values())
    assert indicator._process_pool is None


@pytest.mark.timeout(300)
def test_indicator_sigterm_cleanup_tolerates_missing_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test indicator multiprocessing cleanup is safe after the pool has already gone away.

    1. Clear the indicator module's process-pool global to mimic late notebook shutdown.
    2. Invoke the SIGTERM handler after the executor has already been torn down.
    3. Confirm the handler exits cleanly instead of crashing on a missing pool.
    """
    # 1. Clear the indicator module's process-pool global to mimic late notebook shutdown.
    monkeypatch.setattr(indicator, "_process_pool", None)

    # 2. Invoke the SIGTERM handler after the executor has already been torn down.
    with pytest.raises(SystemExit):
        indicator._handle_sigterm()

    # 3. Confirm the handler exits cleanly instead of crashing on a missing pool.
    assert indicator._process_pool is None


@pytest.mark.timeout(300)
def test_grid_search_sigterm_cleanup_uses_registered_process_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test grid-search multiprocessing cleanup uses the registered process pool.

    1. Store a dummy process pool in the grid-search module as if multiprocess execution had started.
    2. Invoke the grid-search SIGTERM cleanup handler while child-process stubs are present.
    3. Confirm the pool is shut down, children are killed, and the global pool reference is cleared.
    """
    dummy_processes = {
        1: DummyProcess(),
        2: DummyProcess(),
    }
    dummy_pool = DummyPool(dummy_processes)

    # 1. Store a dummy process pool in the grid-search module as if multiprocess execution had started.
    monkeypatch.setattr(grid_search, "_process_pool", dummy_pool)

    # 2. Invoke the grid-search SIGTERM cleanup handler while child-process stubs are present.
    with pytest.raises(SystemExit):
        grid_search._handle_sigterm()

    # 3. Confirm the pool is shut down, children are killed, and the global pool reference is cleared.
    assert dummy_pool.shutdown_calls == 1
    assert all(process.kill_calls == 1 for process in dummy_processes.values())
    assert grid_search._process_pool is None


@pytest.mark.timeout(300)
def test_grid_search_sigterm_cleanup_tolerates_missing_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test grid-search multiprocessing cleanup is safe when teardown runs late.

    1. Clear the grid-search module's process-pool global to mimic a late shutdown callback.
    2. Invoke the SIGTERM handler after the executor reference has already been cleared.
    3. Confirm the handler exits cleanly instead of raising an attribute error.
    """
    # 1. Clear the grid-search module's process-pool global to mimic a late shutdown callback.
    monkeypatch.setattr(grid_search, "_process_pool", None)

    # 2. Invoke the SIGTERM handler after the executor reference has already been cleared.
    with pytest.raises(SystemExit):
        grid_search._handle_sigterm()

    # 3. Confirm the handler exits cleanly instead of raising an attribute error.
    assert grid_search._process_pool is None


@pytest.mark.timeout(300)
def test_worker_signal_helper_ignores_sigint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test worker signal helper suppresses child SIGINT tracebacks.

    1. Patch :py:func:`signal.signal` so we can observe registrations.
    2. Run the shared worker helper directly.
    3. Confirm child workers are configured to ignore ``SIGINT``.
    """
    registrations = []

    def capture(sig, handler):
        registrations.append((sig, handler))

    monkeypatch.setattr(multiprocessing_signal.signal, "signal", capture)

    multiprocessing_signal.suppress_worker_sigint_tracebacks()

    assert registrations == [(signal.SIGINT, signal.SIG_IGN)]


@pytest.mark.timeout(300)
def test_indicator_process_init_suppresses_sigint_and_loads_universe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test indicator worker initialiser installs signal handling before work starts."""
    registrations = []

    def capture(sig, handler):
        registrations.append((sig, handler))

    monkeypatch.setattr(multiprocessing_signal.signal, "signal", capture)
    monkeypatch.setattr(indicator, "_universe", None)

    payload = {"source": "indicator"}
    indicator._process_init(pickle.dumps(payload))

    assert registrations == [(signal.SIGINT, signal.SIG_IGN)]
    assert indicator._universe == payload


@pytest.mark.timeout(300)
def test_grid_search_process_init_suppresses_sigint_and_loads_universe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test grid-search worker initialiser installs signal handling before work starts."""
    registrations = []

    def capture(sig, handler):
        registrations.append((sig, handler))

    monkeypatch.setattr(multiprocessing_signal.signal, "signal", capture)
    monkeypatch.setattr(grid_search, "_universe", None)

    payload = {"source": "grid-search"}
    grid_search._process_init(pickle.dumps(payload))

    assert registrations == [(signal.SIGINT, signal.SIG_IGN)]
    assert grid_search._universe == payload
