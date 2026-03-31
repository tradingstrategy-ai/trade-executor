"""CLI coverage for stats refresh post-valuation scheduling."""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import pytest

from tradeexecutor.cli import loop as loop_module
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


def test_cli_stats_refresh_post_valuation_runs_once_for_lagoon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stats refresh can drive one Lagoon post-valuation settlement in unit tests.

    1. Patch CLI start-up so it runs offline with a fake Lagoon sync model.
    2. Run `start` with a tiny stats refresh interval and the unit-testing shutdown hook enabled.
    3. Verify one stats refresh and one post-valuation settlement were persisted in state.
    """
    from tradeexecutor.cli.commands import start as start_module

    strategy_path = Path(os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "test_only", "quickswap_dummy.py"))
    state_file = tmp_path / "stats_refresh_post_valuation_lagoon.json"
    cache_path = tmp_path / "cache"

    class FakeLagoonVaultSyncModel(DummySyncModel):
        def __init__(self) -> None:
            self.sync_treasury_calls: list[bool] = []
            self.abort_lagoon_settlement_on_frozen_positions = False

        def has_async_deposits(self):
            return True

        def sync_treasury(
            self,
            strategy_cycle_ts,
            state: State,
            supported_reserves=None,
            end_block=None,
            post_valuation=False,
        ):
            self.sync_treasury_calls.append(post_valuation)
            return []

    fake_sync_model = FakeLagoonVaultSyncModel()

    def fake_create_execution_and_sync_model(**kwargs):
        return DummyExecutionModel(SimpleNamespace()), fake_sync_model, None, None

    def fake_setup(self: ExecutionLoop) -> State:
        state = self.init_state()
        self.runner = SimpleNamespace(check_accounts=lambda universe, state: None)
        self.universe_model = SimpleNamespace()
        return state

    def fake_warm_up_live_trading(self: ExecutionLoop):
        return SimpleNamespace(reserve_assets=[])

    def fake_refresh_live_run_state(
        self: ExecutionLoop,
        state: State,
        visualisation: bool = False,
        universe=None,
        cycle_duration=None,
    ) -> None:
        return None

    def fake_update_position_valuations(
        self: ExecutionLoop,
        clock,
        state: State,
        universe,
        execution_mode=None,
        interest=True,
    ) -> None:
        return None

    def fake_tick(
        self: ExecutionLoop,
        unrounded_timestamp,
        cycle_duration,
        state: State,
        cycle: int,
        live: bool,
        existing_universe=None,
        strategy_cycle_timestamp=None,
        extra_debug_data=None,
        indicators=None,
    ):
        return existing_universe

    monkeypatch.setattr(start_module, "LagoonVaultSyncModel", FakeLagoonVaultSyncModel)
    monkeypatch.setattr(start_module, "create_web3_config", lambda **kwargs: FakeWeb3Config())
    monkeypatch.setattr(start_module, "configure_default_chain", lambda web3config, mod: None)
    monkeypatch.setattr(start_module, "create_execution_and_sync_model", fake_create_execution_and_sync_model)
    monkeypatch.setattr(start_module, "create_client", lambda **kwargs: (object(), None))
    monkeypatch.setattr(ExecutionLoop, "setup", fake_setup)
    monkeypatch.setattr(ExecutionLoop, "warm_up_live_trading", fake_warm_up_live_trading)
    monkeypatch.setattr(ExecutionLoop, "refresh_live_run_state", fake_refresh_live_run_state)
    monkeypatch.setattr(ExecutionLoop, "update_position_valuations", fake_update_position_valuations)
    monkeypatch.setattr(ExecutionLoop, "tick", fake_tick)
    monkeypatch.setattr(loop_module, "display_strategy_universe", lambda universe: pd.DataFrame())

    environment = {
        "STRATEGY_FILE": strategy_path.as_posix(),
        "STATE_FILE": state_file.as_posix(),
        "RESET_STATE": "true",
        "ASSET_MANAGEMENT_MODE": "dummy",
        "CACHE_PATH": cache_path.as_posix(),
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
        "CHECK_ACCOUNTS": "false",
        "SYNC_TREASURY_ON_STARTUP": "false",
        "STATS_REFRESH_MINUTES": "0.0001",
        "STATS_REFRESH_POST_VALUATION": "true",
        "STATS_REFRESH_UNIT_TESTING": "true",
        "POSITION_TRIGGER_CHECK_MINUTES": "0",
        "CYCLE_DURATION": "1d",
    }

    # 2. Run `start` with a tiny stats refresh interval and the unit-testing shutdown hook enabled.
    with mock.patch.dict("os.environ", environment, clear=True):
        app(["start"], standalone_mode=False)

    state = State.from_json(state_file.read_text())

    # 3. Verify one stats refresh and one post-valuation settlement were persisted in state.
    assert fake_sync_model.sync_treasury_calls == [True]
    assert state.uptime.stats_refresh_completed == 1
    assert state.uptime.post_valuation_settlements_completed == 1


def test_cli_stats_refresh_post_valuation_ignored_for_non_lagoon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stats refresh unit testing still exits cleanly when post-valuation is disabled by sync model.

    1. Patch CLI start-up so it runs offline with a non-Lagoon sync model.
    2. Run `start` with the same tiny stats refresh interval and unit-testing shutdown hook.
    3. Verify the stats refresh counter increments once and settlement counter stays at zero.
    """
    from tradeexecutor.cli.commands import start as start_module

    strategy_path = Path(os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "test_only", "quickswap_dummy.py"))
    state_file = tmp_path / "stats_refresh_post_valuation_dummy.json"
    cache_path = tmp_path / "cache"

    fake_sync_model = DummySyncModel()

    def fake_create_execution_and_sync_model(**kwargs):
        return DummyExecutionModel(SimpleNamespace()), fake_sync_model, None, None

    def fake_setup(self: ExecutionLoop) -> State:
        state = self.init_state()
        self.runner = SimpleNamespace(check_accounts=lambda universe, state: None)
        self.universe_model = SimpleNamespace()
        return state

    def fake_warm_up_live_trading(self: ExecutionLoop):
        return SimpleNamespace(reserve_assets=[])

    def fake_refresh_live_run_state(
        self: ExecutionLoop,
        state: State,
        visualisation: bool = False,
        universe=None,
        cycle_duration=None,
    ) -> None:
        return None

    def fake_update_position_valuations(
        self: ExecutionLoop,
        clock,
        state: State,
        universe,
        execution_mode=None,
        interest=True,
    ) -> None:
        return None

    def fake_tick(
        self: ExecutionLoop,
        unrounded_timestamp,
        cycle_duration,
        state: State,
        cycle: int,
        live: bool,
        existing_universe=None,
        strategy_cycle_timestamp=None,
        extra_debug_data=None,
        indicators=None,
    ):
        return existing_universe

    monkeypatch.setattr(start_module, "create_web3_config", lambda **kwargs: FakeWeb3Config())
    monkeypatch.setattr(start_module, "configure_default_chain", lambda web3config, mod: None)
    monkeypatch.setattr(start_module, "create_execution_and_sync_model", fake_create_execution_and_sync_model)
    monkeypatch.setattr(start_module, "create_client", lambda **kwargs: (object(), None))
    monkeypatch.setattr(ExecutionLoop, "setup", fake_setup)
    monkeypatch.setattr(ExecutionLoop, "warm_up_live_trading", fake_warm_up_live_trading)
    monkeypatch.setattr(ExecutionLoop, "refresh_live_run_state", fake_refresh_live_run_state)
    monkeypatch.setattr(ExecutionLoop, "update_position_valuations", fake_update_position_valuations)
    monkeypatch.setattr(ExecutionLoop, "tick", fake_tick)
    monkeypatch.setattr(loop_module, "display_strategy_universe", lambda universe: pd.DataFrame())

    environment = {
        "STRATEGY_FILE": strategy_path.as_posix(),
        "STATE_FILE": state_file.as_posix(),
        "RESET_STATE": "true",
        "ASSET_MANAGEMENT_MODE": "dummy",
        "CACHE_PATH": cache_path.as_posix(),
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
        "CHECK_ACCOUNTS": "false",
        "SYNC_TREASURY_ON_STARTUP": "false",
        "STATS_REFRESH_MINUTES": "0.0001",
        "STATS_REFRESH_POST_VALUATION": "true",
        "STATS_REFRESH_UNIT_TESTING": "true",
        "POSITION_TRIGGER_CHECK_MINUTES": "0",
        "CYCLE_DURATION": "1d",
    }

    # 2. Run `start` with the same tiny stats refresh interval and unit-testing shutdown hook.
    with mock.patch.dict("os.environ", environment, clear=True):
        app(["start"], standalone_mode=False)

    state = State.from_json(state_file.read_text())

    # 3. Verify the stats refresh counter increments once and settlement counter stays at zero.
    assert state.uptime.stats_refresh_completed == 1
    assert state.uptime.post_valuation_settlements_completed == 0
