"""CLI coverage for Lagoon settlement safety options."""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from tradeexecutor.cli.commands import start as start_module
from tradeexecutor.cli.loop import ExecutionLoop
from tradeexecutor.cli.main import app
from tradeexecutor.state.state import State
from tradeexecutor.state.lock import StateFileLocked, state_file_lock
from tradeexecutor.strategy.dummy import DummyExecutionModel
from tradeexecutor.strategy.sync_model import DummySyncModel


@pytest.mark.parametrize("interrupt", [False, True])
def test_cli_start_enables_lagoon_frozen_position_safety_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt: bool,
) -> None:
    """CLI start enables the Lagoon frozen-position safety option by default.

    1. Patch CLI start-up so it runs offline with a fake Lagoon sync model.
    2. Run `start` without overriding the Lagoon frozen-position safety option.
    3. Verify the safety flag and lock ownership during execution and crash sleep.
    4. Verify writer ownership is released when the command exits.
    """
    # 1. Patch CLI start-up so it runs offline with a fake Lagoon sync model.
    # DummySyncModel stands in for LagoonVaultSyncModel so the real isinstance
    # branch sets the safety flag; mocking Web3 avoids all RPC connections.
    strategy_path = Path(os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "test_only", "quickswap_dummy.py"))
    state_file = tmp_path / "lagoon_safety.json"
    cache_path = tmp_path / "cache"
    fake_sync_model = DummySyncModel()
    fake_sync_model.abort_lagoon_settlement_on_frozen_positions = False
    fake_web3_config = mock.Mock()
    fake_web3_config.has_any_connection.return_value = True
    fake_web3_config.get_default.return_value = SimpleNamespace(eth=SimpleNamespace(chain_id=137))

    def fake_create_execution_and_sync_model(**kwargs):
        return DummyExecutionModel(SimpleNamespace()), fake_sync_model, None, None

    def fake_setup(self: ExecutionLoop) -> State:
        state = self.init_state()
        self.runner = SimpleNamespace(check_accounts=lambda universe, state: None)
        self.universe_model = SimpleNamespace()
        return state

    def fake_run_with_state(self: ExecutionLoop, state: State) -> None:
        """Check ownership; state is unused but required by the execution interface."""
        # 3. Verify the safety flag and lock ownership during execution and crash sleep.
        assert self.sync_model.abort_lagoon_settlement_on_frozen_positions is True
        with pytest.raises(StateFileLocked):
            with state_file_lock(state_file):
                pytest.fail("start released ownership before execution finished")
        if interrupt:
            raise KeyboardInterrupt("Exercise the crash-status sleep")

    def fake_sleep(seconds: float) -> None:
        """Check the crash-status wait without leaving the test sleeping."""
        assert seconds == 3600 * 24 * 365
        with pytest.raises(StateFileLocked):
            with state_file_lock(state_file):
                pytest.fail("start released ownership during crash-status sleep")
        raise RuntimeError("Stop crash-status sleep")

    monkeypatch.setattr(start_module, "LagoonVaultSyncModel", DummySyncModel)
    monkeypatch.setattr(start_module, "create_web3_config", lambda **kwargs: fake_web3_config)
    monkeypatch.setattr(start_module, "configure_default_chain", lambda web3config, mod: None)
    monkeypatch.setattr(start_module, "create_execution_and_sync_model", fake_create_execution_and_sync_model)
    monkeypatch.setattr(start_module, "create_client", lambda **kwargs: (object(), None))
    monkeypatch.setattr(ExecutionLoop, "setup", fake_setup)
    monkeypatch.setattr(ExecutionLoop, "run_with_state", fake_run_with_state)
    # Avoid the indefinite crash wait and unrelated thread-stack output.
    monkeypatch.setattr(start_module.time, "sleep", fake_sleep)
    monkeypatch.setattr(start_module.faulthandler, "dump_traceback", mock.Mock())

    environment = {
        "STRATEGY_FILE": strategy_path.as_posix(),
        "STATE_FILE": state_file.as_posix(),
        "RESET_STATE": "true",
        "ASSET_MANAGEMENT_MODE": "dummy",
        "CACHE_PATH": cache_path.as_posix(),
        "MAX_CYCLES": "0",
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
        "CHECK_ACCOUNTS": "false",
        "SYNC_TREASURY_ON_STARTUP": "false",
    }

    # 2. Run `start` without overriding the Lagoon frozen-position safety option.
    with mock.patch.dict("os.environ", environment, clear=True):
        if interrupt:
            with pytest.raises(RuntimeError, match="Stop crash-status sleep"):
                app(["start"], standalone_mode=False)
        else:
            app(["start"], standalone_mode=False)

    # 4. Verify writer ownership is released when the command exits.
    with state_file_lock(state_file):
        pass
