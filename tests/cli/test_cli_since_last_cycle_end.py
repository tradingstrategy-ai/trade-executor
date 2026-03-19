"""Test CLI live scheduling for rolling cycle-end offsets."""

import datetime
import os
import pickle
import time
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


@pytest.fixture()
def strategy_path() -> Path:
    """Return the test-only strategy path used by the CLI integration test."""
    return Path(os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "test_only", "quickswap_dummy.py"))


@pytest.mark.timeout(20)
def test_cli_since_last_cycle_end_offsets_from_previous_cycle_end(
    tmp_path: Path,
    strategy_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI live loop should schedule the next cycle from the previous cycle end time.

    1. Patch the CLI start-up dependencies so `start` can run fully offline with the real scheduler.
    2. Run the `start` command for a few 1 second cycles using `since_last_cycle_end`.
    3. Verify each next logical cycle timestamp matches the previous persisted cycle end plus 1 second.
    """

    debug_dump_file = tmp_path / "since_last_cycle_end.debug.pickle"
    state_file = tmp_path / "since_last_cycle_end.json"
    cache_path = tmp_path / "cache"

    def fake_create_execution_and_sync_model(**kwargs):
        return DummyExecutionModel(SimpleNamespace()), DummySyncModel(), None, None

    def fake_setup(self: ExecutionLoop) -> State:
        state = self.init_state()
        self.runner = SimpleNamespace(check_accounts=lambda universe, state: None)
        self.universe_model = SimpleNamespace()
        return state

    def fake_warm_up_live_trading(self: ExecutionLoop):
        return SimpleNamespace()

    def fake_refresh_live_run_state(
        self: ExecutionLoop,
        state: State,
        visualisation: bool = False,
        universe=None,
        cycle_duration=None,
    ) -> None:
        return None

    def fake_tick(
        self: ExecutionLoop,
        unrounded_timestamp: datetime.datetime,
        cycle_duration,
        state: State,
        cycle: int,
        live: bool,
        existing_universe=None,
        strategy_cycle_timestamp: datetime.datetime | None = None,
        extra_debug_data: dict | None = None,
        indicators=None,
    ):
        assert live
        assert strategy_cycle_timestamp is not None

        debug_details = {
            "cycle": cycle,
            "unrounded_timestamp": unrounded_timestamp,
            "timestamp": strategy_cycle_timestamp,
            "strategy_cycle_trigger": self.strategy_cycle_trigger.value,
        }

        if extra_debug_data:
            debug_details.update(extra_debug_data)

        time.sleep(0.2)
        ended_at = loop_module.native_datetime_utc_now()
        state.record_cycle_end(cycle, now_=ended_at, live=True)
        self.store.sync(state)
        self.debug_dump_state[cycle] = debug_details

        if self.debug_dump_file is not None:
            with open(self.debug_dump_file, "wb") as out:
                pickle.dump(self.debug_dump_state, out)

        return existing_universe

    # 1. Patch the CLI start-up dependencies so `start` can run fully offline with the real scheduler.
    monkeypatch.setattr("tradeexecutor.cli.commands.start.create_web3_config", lambda **kwargs: FakeWeb3Config())
    monkeypatch.setattr("tradeexecutor.cli.commands.start.configure_default_chain", lambda web3config, mod: None)
    monkeypatch.setattr("tradeexecutor.cli.commands.start.create_execution_and_sync_model", fake_create_execution_and_sync_model)
    monkeypatch.setattr("tradeexecutor.cli.commands.start.create_client", lambda **kwargs: (object(), None))
    monkeypatch.setattr(ExecutionLoop, "setup", fake_setup)
    monkeypatch.setattr(ExecutionLoop, "warm_up_live_trading", fake_warm_up_live_trading)
    monkeypatch.setattr(ExecutionLoop, "refresh_live_run_state", fake_refresh_live_run_state)
    monkeypatch.setattr(ExecutionLoop, "tick", fake_tick)
    monkeypatch.setattr(loop_module, "display_strategy_universe", lambda universe: pd.DataFrame())

    environment = {
        "STRATEGY_FILE": strategy_path.as_posix(),
        "STATE_FILE": state_file.as_posix(),
        "RESET_STATE": "true",
        "ASSET_MANAGEMENT_MODE": "dummy",
        "STRATEGY_CYCLE_TRIGGER": "since_last_cycle_end",
        "CACHE_PATH": cache_path.as_posix(),
        "DEBUG_DUMP_FILE": debug_dump_file.as_posix(),
        "CYCLE_DURATION": "1s",
        "MAX_CYCLES": "4",
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
        "CHECK_ACCOUNTS": "false",
        "SYNC_TREASURY_ON_STARTUP": "false",
        "STATS_REFRESH_MINUTES": "0",
        "POSITION_TRIGGER_CHECK_MINUTES": "0",
    }

    # 2. Run the `start` command for a few 1 second cycles using `since_last_cycle_end`.
    with mock.patch.dict("os.environ", environment, clear=True):
        app(["start"], standalone_mode=False)

    with open(debug_dump_file, "rb") as inp:
        debug_dump = pickle.load(inp)

    state = State.from_json(state_file.read_text())

    cycle_1_timestamp = debug_dump[1]["timestamp"]
    cycle_2_timestamp = debug_dump[2]["timestamp"]
    cycle_3_timestamp = debug_dump[3]["timestamp"]

    cycle_1_ended_at = datetime.datetime.fromisoformat(state.other_data.data[1]["decision_cycle_ended_at"])
    cycle_2_ended_at = datetime.datetime.fromisoformat(state.other_data.data[2]["decision_cycle_ended_at"])
    cycle_3_ended_at = datetime.datetime.fromisoformat(state.other_data.data[3]["decision_cycle_ended_at"])

    # 3. Verify each next logical cycle timestamp matches the previous persisted cycle end plus 1 second.
    assert len(debug_dump) == 3
    assert len(state.uptime.cycles_completed_at) == 3
    assert cycle_2_timestamp == cycle_1_ended_at + datetime.timedelta(seconds=1)
    assert cycle_3_timestamp == cycle_2_ended_at + datetime.timedelta(seconds=1)
    assert cycle_2_timestamp.microsecond != 0
    assert cycle_3_timestamp.microsecond != 0
    assert cycle_2_timestamp > cycle_1_timestamp + datetime.timedelta(seconds=1)
    assert cycle_3_timestamp > cycle_2_timestamp
    assert 1 in state.uptime.cycles_completed_at
    assert 2 in state.uptime.cycles_completed_at
    assert 3 in state.uptime.cycles_completed_at
