"""Black-box command coverage for writer ownership and read-only access."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from tradeexecutor.cli.commands import start as start_command
from tradeexecutor.cli.commands import correct_accounts as correct_accounts_command
from tradeexecutor.cli.commands import blacklist as blacklist_command
from tradeexecutor.cli.commands import perform_test_trade as test_trade_command
from tradeexecutor.cli.main import app
from tradeexecutor.state.state import State
from tradeexecutor.state.store import JSONFileStore
from tradeexecutor.testing.state_lock import hold_state_lock_in_subprocess


def test_cli_state_writer_conflicts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse commands before state reads or side effects, whilst allowing readers.

    1. Prepare a state and hold its lock in another process.
    2. Invoke daemon, local, backtest and delegated writers and verify clean refusal.
    3. Read positions during ownership, then prune successfully after release.
    """
    # 1. Prepare a state and hold its lock in another process.
    # Isolate CLI defaults from local credentials; no RPC or trading is needed.
    for key in tuple(os.environ):
        if key not in {"PATH", "PYTHONPATH"}:
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "executor.json"
    state = State()
    state.name = "Lock test"
    JSONFileStore(path).sync(state)
    state_directory = tmp_path / "state"
    state_directory.mkdir()
    for name in ("lock-test.json", "lock-test-backtest.json"):
        (state_directory / name).symlink_to(path)
    original = path.read_bytes()
    original_compressed = path.with_name(path.name + ".br").read_bytes()
    # Avoid unrelated log file creation before start's lock boundary.
    monkeypatch.setattr(start_command, "setup_file_logging", Mock())
    runner = CliRunner()
    common = ["--state-file", str(path), "--strategy-file", "unused.py"]
    with hold_state_lock_in_subprocess(path):
        # 2. Invoke daemon, local, backtest and delegated writers and verify clean refusal.
        # Fail on working-state loads; filesystem checks also catch backup creation.
        original_load = JSONFileStore.load
        blocked_load = Mock(side_effect=AssertionError("Loaded state before acquiring ownership"))
        monkeypatch.setattr(JSONFileStore, "load", blocked_load)
        invocations = [
            ["start", *common, "--asset-management-mode", "hot_wallet", "--http-enabled", "--unit-testing", "--log-level", "disabled"],
            ["prune-state", *common],
            ["close-position", *common],
            ["check-position-triggers", *common],
            ["lagoon-settle", *common],
            ["claim-hypercore-vault-dust", *common],
            ["vault-test-trade", "--state-file", str(path), "--id", "lock-test", "--auto-real", "--asset-management-mode", "lagoon"],
            ["backtest", *common, "--trading-strategy-api-key", "unused"],
            ["start", "--id", "lock-test", "--strategy-file", "unused.py", "--asset-management-mode", "hot_wallet", "--unit-testing", "--log-level", "disabled"],
            ["backtest", "--id", "lock-test", "--strategy-file", "unused.py", "--trading-strategy-api-key", "unused"],
        ]
        for arguments in invocations:
            result = runner.invoke(app, arguments)
            assert result.exit_code == 1, (arguments, result.output, result.exception)
            assert "Stop the other executor or maintenance command" in result.output
            # Typer may wrap long paths inside its Rich error panel.
            compact_output = "".join(result.output.split()).replace("│", "")
            assert str(path) in compact_output
            assert str(path) + ".lock" in compact_output
            assert path.read_bytes() == original
            assert path.with_name(path.name + ".br").read_bytes() == original_compressed
            assert not list(tmp_path.glob("*.backup*.json"))
            assert not list(tmp_path.glob("*prune-backup*"))
        blocked_load.assert_not_called()
        monkeypatch.setattr(JSONFileStore, "load", original_load)

        repair_script = Path(__file__).resolve().parents[2] / "scripts/hyper-ai/repair-closed-position-profitability.py"
        preview = subprocess.run(
            [sys.executable, str(repair_script), str(path)],
            capture_output=True, text=True, timeout=30,
        )
        assert preview.returncode == 0, preview.stderr
        assert "Preview only" in preview.stdout
        write = subprocess.run(
            [sys.executable, str(repair_script), str(path), "--write"],
            capture_output=True, text=True, timeout=30,
        )
        assert write.returncode == 1
        assert "Stop the other executor" in write.stderr
        assert "Traceback" not in write.stderr
        cleanup_script = repair_script.parents[1] / "hyperliquid/clean-closed-vault-positions.py"
        cleanup = subprocess.run(
            [sys.executable, str(cleanup_script)],
            env={**os.environ, "STATE_FILE": str(path)},
            capture_output=True, text=True, timeout=30,
        )
        assert cleanup.returncode == 1
        assert "Stop the other executor" in cleanup.stderr
        assert "Traceback" not in cleanup.stderr

        # 3. Read positions during ownership, then prune successfully after release.
        result = runner.invoke(app, ["show-positions", "--state-file", str(path)])
        assert result.exit_code == 0, result.exception
        assert "Lock test" in result.output
    result = runner.invoke(app, ["prune-state", *common, "--unit-testing"])
    assert result.exit_code == 0, result.exception
    assert "Pruning completed successfully" in result.output


def test_cli_non_writing_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep conditional read-only modes usable when another process owns state.

    1. Hold the state lock and replace costly command setup with a sentinel.
    2. Verify preview, simulation and listing reach setup instead of refusing.
    3. Verify their corresponding writing modes refuse before setup.
    """
    # 1. Hold the state lock and replace costly command setup with a sentinel.
    # Preserve Python subprocess imports while removing local CLI settings.
    for key in tuple(os.environ):
        if key not in {"PATH", "PYTHONPATH"}:
            monkeypatch.delenv(key)
    path = tmp_path / "executor.json"
    runner = CliRunner()
    # These commands all begin with ID preparation. Stop there to avoid RPC,
    # dataset downloads or transaction setup; the real lock wrapper still runs.
    setup = Mock(side_effect=RuntimeError("read-only setup reached"))
    for module in (correct_accounts_command, blacklist_command, test_trade_command):
        monkeypatch.setattr(module, "prepare_executor_id", setup)
    common = ["--state-file", str(path), "--strategy-file", "unused.py"]
    with hold_state_lock_in_subprocess(path):
        # 2. Verify preview, simulation and listing reach setup instead of refusing.
        for command, extra in [
            ("correct-accounts", ["--dry-run"]),
            ("perform-test-trade", ["--simulate", "--single-pair"]),
            ("blacklist", []),
        ]:
            result = runner.invoke(app, [command, *common, *extra])
            assert isinstance(result.exception, RuntimeError), result.output
            assert str(result.exception) == "read-only setup reached"
        assert setup.call_count == 3

        # 3. Verify their corresponding writing modes refuse before setup.
        for command, extra in [
            ("correct-accounts", []),
            ("perform-test-trade", []),
            ("blacklist", ["--add-token", "0x0000000000000000000000000000000000000001"]),
        ]:
            result = runner.invoke(app, [command, *common, *extra])
            assert result.exit_code == 1, result.output
            assert "Stop the other executor" in result.output
        assert setup.call_count == 3
