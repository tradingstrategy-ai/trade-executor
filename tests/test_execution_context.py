import pytest

import tradeexecutor.strategy.execution_context as execution_context_module
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode, is_non_interactive_notebook_execution


class _FakeParentProcess:
    def __init__(self, cmdline: list[str]):
        self._cmdline = cmdline

    def cmdline(self) -> list[str]:
        return self._cmdline


class _FakeCurrentProcess:
    def __init__(self, parents: list[_FakeParentProcess]):
        self._parents = parents

    def parents(self) -> list[_FakeParentProcess]:
        return self._parents


def test_progress_bars_disabled_for_jupyter_execute(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify static notebook execution does not emit TQDM progress bars.

    1. Patch parent process detection to look like ``jupyter execute``.
    2. Create a backtest execution context with progress bars enabled.
    3. Confirm the context suppresses progress bars.
    """

    # 1. Patch parent process detection to look like ``jupyter execute``.
    process = _FakeCurrentProcess([
        _FakeParentProcess(["/usr/bin/jupyter", "execute", "backtest.ipynb"]),
    ])
    monkeypatch.setattr(execution_context_module.psutil, "Process", lambda: process)

    # 2. Create a backtest execution context with progress bars enabled.
    execution_context = ExecutionContext(mode=ExecutionMode.backtesting)

    # 3. Confirm the context suppresses progress bars.
    assert is_non_interactive_notebook_execution()
    assert not execution_context.is_progress_bar_enabled()


def test_progress_bars_enabled_outside_static_notebook_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify normal execution keeps TQDM progress bars available.

    1. Patch parent process detection to look like a regular shell run.
    2. Create a backtest execution context with progress bars enabled.
    3. Confirm the context keeps progress bars enabled.
    """

    # 1. Patch parent process detection to look like a regular shell run.
    process = _FakeCurrentProcess([
        _FakeParentProcess(["bash"]),
        _FakeParentProcess(["poetry", "run", "pytest"]),
    ])
    monkeypatch.setattr(execution_context_module.psutil, "Process", lambda: process)

    # 2. Create a backtest execution context with progress bars enabled.
    execution_context = ExecutionContext(mode=ExecutionMode.backtesting)

    # 3. Confirm the context keeps progress bars enabled.
    assert not is_non_interactive_notebook_execution()
    assert execution_context.is_progress_bar_enabled()
