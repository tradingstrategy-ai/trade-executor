"""CLI coverage for Lagoon settlement safety options."""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from tradeexecutor.cli.loop import ExecutionLoop
from tradeexecutor.cli.main import app
from tradeexecutor.state.state import State
from tradeexecutor.strategy.dummy import DummyExecutionModel
from tradeexecutor.strategy.sync_model import DummySyncModel


class FakeWeb3Config:
    """Provide the minimal Web3 configuration needed for CLI start-up tests."""

    def __init__(self) -> None:
        self._default = SimpleNamespace(eth=SimpleNamespace(chain_id=137))

    def has_any_connection(self) -> bool:
        return True

    def get_default(self):
        return self._default


def test_cli_start_propagates_lagoon_frozen_position_safety_option(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI start propagates the Lagoon frozen-position safety option.

    1. Patch CLI start-up so it runs offline with a fake Lagoon sync model.
    2. Run `start` with the Lagoon frozen-position safety option enabled.
    3. Verify the fake Lagoon sync model receives the propagated safety flag.
    """
    from tradeexecutor.cli.commands import start as start_module

    class FakeLagoonVaultSyncModel(DummySyncModel):
        def __init__(self) -> None:
            self.abort_lagoon_settlement_on_frozen_positions = False

    strategy_path = Path(os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "test_only", "quickswap_dummy.py"))
    state_file = tmp_path / "lagoon_safety.json"
    cache_path = tmp_path / "cache"
    fake_sync_model = FakeLagoonVaultSyncModel()

    def fake_create_execution_and_sync_model(**kwargs):
        return DummyExecutionModel(SimpleNamespace()), fake_sync_model, None, None

    def fake_setup(self: ExecutionLoop) -> State:
        # 1. Patch CLI start-up so it runs offline with a fake Lagoon sync model.
        state = self.init_state()
        self.runner = SimpleNamespace(check_accounts=lambda universe, state: None)
        self.universe_model = SimpleNamespace()
        return state

    def fake_run_with_state(self: ExecutionLoop, state: State) -> None:
        # 3. Verify the fake Lagoon sync model receives the propagated safety flag.
        assert self.sync_model.abort_lagoon_settlement_on_frozen_positions is True

    monkeypatch.setattr(start_module, "LagoonVaultSyncModel", FakeLagoonVaultSyncModel)
    monkeypatch.setattr(start_module, "create_web3_config", lambda **kwargs: FakeWeb3Config())
    monkeypatch.setattr(start_module, "configure_default_chain", lambda web3config, mod: None)
    monkeypatch.setattr(start_module, "create_execution_and_sync_model", fake_create_execution_and_sync_model)
    monkeypatch.setattr(start_module, "create_client", lambda **kwargs: (object(), None))
    monkeypatch.setattr(ExecutionLoop, "setup", fake_setup)
    monkeypatch.setattr(ExecutionLoop, "run_with_state", fake_run_with_state)

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
        "ABORT_LAGOON_SETTLEMENT_ON_FROZEN_POSITIONS": "true",
    }

    # 2. Run `start` with the Lagoon frozen-position safety option enabled.
    with mock.patch.dict("os.environ", environment, clear=True):
        app(["start"], standalone_mode=False)
