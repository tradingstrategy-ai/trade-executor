"""Trade-ui startup treasury sync ordering tests."""

import datetime
from types import SimpleNamespace

from tradeexecutor.cli.commands.trade_ui import _sync_trade_ui_startup_treasury
from tradeexecutor.state.state import State


def test_trade_ui_revalues_before_lagoon_post_valuation() -> None:
    """Trade-ui refreshes async vault settlements and valuations before Lagoon NAV posting.

    1. Build fake execution, runner, sync model and store objects that record call order.
    2. Run the trade-ui startup treasury helper.
    3. Verify Lagoon treasury sync happens only after settlement resolution and revaluation.
    4. Run the helper in simulate mode and verify it does not write the state file.
    """

    calls: list[str] = []
    ts = datetime.datetime(2026, 6, 19, 11, 6, 9)
    state = State()
    reserve_asset = SimpleNamespace(token_symbol="USDC")
    universe = SimpleNamespace(reserve_assets=[reserve_asset])
    pricing_model = object()
    valuation_method = object()

    class FakeExecutionModel:
        def initialize(self) -> None:
            calls.append("initialize")

        def resolve_pending_vault_settlements(self, *, state: State, ts: datetime.datetime, pricing_model):
            calls.append("resolve_pending_vault_settlements")
            assert pricing_model is not None
            return []

    class FakeRunner:
        def revalue_state(self, revalue_ts: datetime.datetime, revalue_state: State, method) -> None:
            calls.append("revalue_state")
            assert revalue_ts == ts
            assert revalue_state is state
            assert method is valuation_method

    class FakeSyncModel:
        def sync_treasury(
            self,
            strategy_cycle_ts: datetime.datetime,
            synced_state: State,
            supported_reserves: list,
            post_valuation: bool = False,
        ):
            calls.append("sync_treasury")
            assert strategy_cycle_ts == ts
            assert synced_state is state
            assert supported_reserves == [reserve_asset]
            assert post_valuation is True
            return []

    class FakeStore:
        def sync(self, synced_state: State) -> None:
            calls.append("store_sync")
            assert synced_state is state

    logger = SimpleNamespace(info=lambda *args, **kwargs: None)

    # 2. Run the trade-ui startup treasury helper.
    _sync_trade_ui_startup_treasury(
        ts=ts,
        state=state,
        universe=universe,
        execution_model=FakeExecutionModel(),
        sync_model=FakeSyncModel(),
        runner=FakeRunner(),
        valuation_method=valuation_method,
        pricing_model=pricing_model,
        store=FakeStore(),
        simulate=False,
        logger=logger,
    )

    # 3. Verify Lagoon treasury sync happens only after settlement resolution and revaluation.
    assert calls == [
        "initialize",
        "resolve_pending_vault_settlements",
        "revalue_state",
        "sync_treasury",
        "store_sync",
    ]

    calls.clear()

    # 4. Run in simulate mode and verify the state file is not written.
    _sync_trade_ui_startup_treasury(
        ts=ts,
        state=state,
        universe=universe,
        execution_model=FakeExecutionModel(),
        sync_model=FakeSyncModel(),
        runner=FakeRunner(),
        valuation_method=valuation_method,
        pricing_model=pricing_model,
        store=FakeStore(),
        simulate=True,
        logger=logger,
    )

    assert calls == [
        "initialize",
        "resolve_pending_vault_settlements",
        "revalue_state",
        "sync_treasury",
    ]
