"""Exercise state ownership through the real CLI and maintenance entry points.

A separate process owns the lock while commands are invoked through Typer.
Checks cover refusal before state reads or backups, continued snapshot access,
conditional non-writing modes and successful maintenance after release. Script
configuration and in-call ownership are checked with financial execution mocked
so the tests cannot submit real transactions.
"""

import os
import runpy
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
from tradeexecutor.ethereum.vault import hyperliquid_cleanup
from tradeexecutor.state.state import State
from tradeexecutor.state.lock import StateFileLocked, state_file_lock
from tradeexecutor.state.store import JSONFileStore
from tradeexecutor.testing.state_lock import hold_state_lock_in_subprocess


def test_cli_state_writer_conflicts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse commands before state reads or side effects, whilst allowing readers.

    1. Prepare a state and hold its lock in another process.
    2. Invoke daemon, local, backtest and delegated writers and verify clean refusal.
    3. Read positions during ownership, then run maintenance after release.

    :param tmp_path:
        Pytest directory for state files, path aliases and maintenance outputs.
    :param monkeypatch:
        Pytest helper isolating deployment settings and replacing operations
        that would perform unrelated logging or financial execution.
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

        # 3. Read positions during ownership, then run maintenance after release.
        result = runner.invoke(app, ["show-positions", "--state-file", str(path)])
        assert result.exit_code == 0, result.exception
        assert "Lock test" in result.output
    result = runner.invoke(app, ["prune-state", *common, "--unit-testing"])
    assert result.exit_code == 0, result.exception
    assert "Pruning completed successfully" in result.output
    # Execute the real script entry point but mock financial operations: only
    # configuration parsing, canonical paths and lock ownership are under test.
    expected_settings = {
        "state_file": path,
        "strategy_file": Path("unused-strategy.py"),
        "private_key": "not-a-signing-key",
        "json_rpc_hyperliquid": "https://rpc.invalid",
        "vault_address": "0x" + "11" * 20,
        "vault_adapter_address": "0x" + "22" * 20,
        "trading_strategy_api_key": "unused-api-key",
        "network": "testnet",
        "auto_approve": True,
        "cache_path": tmp_path / "cleanup-cache",
        "unit_testing": True,
        "log_level": "warning",
    }

    def fake_execute(**settings: object) -> None:
        """Assert that script execution receives settings while ownership is held.

        :param settings:
            Parsed keyword settings passed by the real script entry point.
            They must match the expected configuration, including its state path.
        """
        assert settings == expected_settings
        with pytest.raises(StateFileLocked):
            with state_file_lock(path):
                pytest.fail("The cleanup driver ran without writer ownership")

    execute = Mock(side_effect=fake_execute)
    monkeypatch.setattr(hyperliquid_cleanup, "run_hyperliquid_cleanup", execute)
    for key, value in {
        "STATE_FILE": str(state_directory / "lock-test.json"),
        "STRATEGY_FILE": "unused-strategy.py",
        "PRIVATE_KEY": "not-a-signing-key",
        "JSON_RPC_HYPERLIQUID": "https://rpc.invalid",
        "VAULT_ADDRESS": "0x" + "11" * 20,
        "VAULT_ADAPTER_ADDRESS": "0x" + "22" * 20,
        "TRADING_STRATEGY_API_KEY": "unused-api-key",
        "NETWORK": "testnet",
        "AUTO_APPROVE": "yes",
        "CACHE_PATH": str(tmp_path / "cleanup-cache"),
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "warning",
    }.items():
        monkeypatch.setenv(key, value)
    runpy.run_path(str(cleanup_script), run_name="__main__")
    execute.assert_called_once()
    with state_file_lock(path):
        pass


def test_cli_non_writing_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep conditional read-only modes usable when another process owns state.

    1. Hold the state lock and replace costly command setup with a sentinel.
    2. Verify preview, simulation and listing reach setup instead of refusing.
    3. Verify their corresponding writing modes refuse before setup.

    :param tmp_path:
        Pytest directory for the state path held by the competing writer.
    :param monkeypatch:
        Pytest helper isolating CLI settings and replacing expensive setup with
        a sentinel exception after the lock wrapper has made its decision.
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
