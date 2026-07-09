"""CLI coverage for stats refresh post-valuation scheduling.

Post-valuation treasury settlement during the background stats refresh runs
automatically for all Lagoon-style vaults (sync models with async deposits).
There is no environment variable to configure.
"""

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


class FakeLagoonVaultSyncModel(DummySyncModel):
    """Lagoon-like sync model stub recording treasury sync calls in a shared event log."""

    def __init__(self, calls: list[tuple] | None = None) -> None:
        self.sync_treasury_calls: list[bool] = []
        self.calls = calls if calls is not None else []
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
        self.calls.append(("sync_treasury", post_valuation))
        # A reconciliation-only pass legitimately returns no balance updates
        return []


def make_cli_environment(strategy_path: Path, state_file: Path, cache_path: Path) -> dict:
    """CLI environment for a stats-refresh unit test run."""
    return {
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
        "STATS_REFRESH_UNIT_TESTING": "true",
        "POSITION_TRIGGER_CHECK_MINUTES": "0",
        "CYCLE_DURATION": "1d",
    }


def patch_cli_startup(
    monkeypatch: pytest.MonkeyPatch,
    fake_sync_model: DummySyncModel,
    calls: list[tuple],
) -> None:
    """Patch CLI start-up so the live loop runs offline with the given sync model."""
    from tradeexecutor.cli.commands import start as start_module

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
        skip_statistics=False,
    ) -> None:
        calls.append(("update_position_valuations", skip_statistics))
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


def test_cli_stats_refresh_post_valuation_runs_once_for_lagoon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stats refresh writes statistics once, after Lagoon post-valuation settlement.

    A statistics point written before treasury settlement combines a stale reserve
    balance with a fresh position valuation and records a spurious NAV spike when an
    external system (e.g. FreqTrade on GMX) has moved cash in or out of the vault.
    Settlement runs automatically for any Lagoon-style vault (async deposits detected
    from the sync model, no environment variable): the refresh must revalue first
    (statistics held back), settle treasury, then write statistics once.

    1. Patch CLI start-up so it runs offline with a fake Lagoon sync model, recording the call order.
    2. Run `start` with a tiny stats refresh interval and the unit-testing shutdown hook enabled.
    3. Verify the call order is: revalue without statistics, treasury settlement, revalue with statistics.
    4. Verify one stats refresh and one post-valuation settlement were persisted in state.
    """
    strategy_path = Path(os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "test_only", "quickswap_dummy.py"))
    state_file = tmp_path / "stats_refresh_post_valuation_lagoon.json"
    cache_path = tmp_path / "cache"

    # Shared event log capturing the order of valuation and settlement calls
    calls: list[tuple] = []
    fake_sync_model = FakeLagoonVaultSyncModel(calls)

    # 1. Patch CLI start-up so it runs offline with a fake Lagoon sync model.
    patch_cli_startup(monkeypatch, fake_sync_model, calls)

    # 2. Run `start` with a tiny stats refresh interval and the unit-testing shutdown hook enabled.
    environment = make_cli_environment(strategy_path, state_file, cache_path)
    with mock.patch.dict("os.environ", environment, clear=True):
        app(["start"], standalone_mode=False)

    state = State.from_json(state_file.read_text())

    # 3. Verify the call order is: revalue without statistics, treasury settlement, revalue with statistics.
    assert calls == [
        ("update_position_valuations", True),
        ("sync_treasury", True),
        ("update_position_valuations", False),
    ]

    # 4. Verify one stats refresh and one post-valuation settlement were persisted in state.
    assert fake_sync_model.sync_treasury_calls == [True]
    assert state.uptime.stats_refresh_completed == 1
    assert state.uptime.post_valuation_settlements_completed == 1


def test_cli_stats_refresh_post_valuation_skipped_for_non_lagoon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stats refresh skips post-valuation settlement for non-Lagoon strategies.

    Settlement is driven by the sync model's async deposit capability, so a
    strategy without async deposits (e.g. hot wallet) must keep the plain refresh
    path: one statistics write, never held back, and no treasury settlement.

    1. Patch CLI start-up so it runs offline with a non-Lagoon sync model.
    2. Run `start` with the same tiny stats refresh interval and unit-testing shutdown hook.
    3. Verify a single statistics write happened with statistics never held back and no treasury settlement.
    4. Verify the stats refresh counter increments once and the settlement counter stays at zero.
    """
    strategy_path = Path(os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "test_only", "quickswap_dummy.py"))
    state_file = tmp_path / "stats_refresh_post_valuation_non_lagoon.json"
    cache_path = tmp_path / "cache"

    calls: list[tuple] = []
    fake_sync_model = DummySyncModel()

    # 1. Patch CLI start-up so it runs offline with a non-Lagoon sync model.
    patch_cli_startup(monkeypatch, fake_sync_model, calls)

    # 2. Run `start` with the same tiny stats refresh interval and unit-testing shutdown hook.
    environment = make_cli_environment(strategy_path, state_file, cache_path)
    with mock.patch.dict("os.environ", environment, clear=True):
        app(["start"], standalone_mode=False)

    state = State.from_json(state_file.read_text())

    # 3. Verify a single statistics write happened with statistics never held back and no treasury settlement.
    assert calls == [
        ("update_position_valuations", False),
    ]

    # 4. Verify the stats refresh counter increments once and the settlement counter stays at zero.
    assert state.uptime.stats_refresh_completed == 1
    assert state.uptime.post_valuation_settlements_completed == 0
