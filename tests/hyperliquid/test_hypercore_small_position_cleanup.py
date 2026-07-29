"""Test cleanup planning for undersized HyperCore-native vault positions."""

import datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tradingstrategy.chain import ChainId

from tradeexecutor.ethereum.vault.hypercore_routing import (
    HYPERCORE_SMALL_POSITION_CLEANUP_RETRY_SAFETY_MARGINS_RAW,
    HYPERCORE_WITHDRAWAL_SAFETY_MARGIN_RAW,
    HypercoreWithdrawalPreflightError,
    HypercoreVaultRouting,
)
from tradeexecutor.ethereum.vault.hypercore_small_position_cleanup import (
    HypercoreSmallPositionCandidate,
    discover_hypercore_small_positions,
    get_hypercore_minimum_allocation,
    get_hypercore_small_position_top_up_reserve,
    plan_hypercore_small_position_cleanup,
    run_hypercore_small_position_cleanup,
)
from tradeexecutor.ethereum.vault.hypercore_vault import create_hypercore_vault_pair
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeType


NOW = datetime.datetime(2026, 7, 28, 12, 0, 0)


def _make_position(live_equity: Decimal) -> tuple[State, TradingPosition]:
    """Create an open HyperCore position with a chosen current USD value."""

    reserve_asset = AssetIdentifier(
        chain_id=ChainId.hyperliquid.value,
        address="0x2000000000000000000000000000000000000000",
        token_symbol="USDC",
        decimals=6,
    )
    pair = create_hypercore_vault_pair(
        quote=reserve_asset,
        vault_address="0x1111111111111111111111111111111111111111",
    )
    state = State()
    state.portfolio.initialise_reserves(reserve_asset, reserve_token_price=1.0)
    state.portfolio.adjust_reserves(reserve_asset, Decimal("100"), "Initial reserve")
    position, opening_trade, _created = state.create_trade(
        strategy_cycle_at=NOW,
        pair=pair,
        quantity=None,
        reserve=Decimal("10"),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
    )
    state.mark_trade_success(
        executed_at=NOW,
        trade=opening_trade,
        executed_price=1.0,
        executed_amount=Decimal("10"),
        executed_reserve=Decimal("10"),
        lp_fees=0,
        native_token_price=0.0,
        force=True,
    )
    position.revalue_base_asset(NOW, float(live_equity / Decimal("10")))
    return state, position


def test_cleanup_plans_top_up_for_sub_floor_hypercore_position():
    """A sub-floor HyperCore position schedules a temporary top-up before a later redemption.

    1. Create a 3.45 USDC HyperCore position below the strategy's 50 USDC allocation floor.
    2. Discover it and plan the cleanup.
    3. Verify the plan tops up enough for every retry without creating a lock-up-blocked close.
    """

    # 1. Create a 3.45 USDC HyperCore position below the strategy's 50 USDC allocation floor.
    state, position = _make_position(Decimal("3.45"))

    # 2. Discover it and plan the cleanup.
    candidates = discover_hypercore_small_positions(state, Decimal("50"))
    assert len(candidates) == 1
    assert candidates[0].top_up_reserve == Decimal("11.55")
    trades = plan_hypercore_small_position_cleanup(state, candidates[0], NOW)

    # 3. Verify the plan tops up enough for every retry without creating a lock-up-blocked close.
    assert len(trades) == 1
    assert trades[0].is_buy()
    assert trades[0].planned_reserve == Decimal("11.55")
    assert position.get_quantity(planned=True) > position.get_quantity()
    assert trades[0].other_data["hypercore_small_position_cleanup"] is True
    assert get_hypercore_small_position_top_up_reserve(Decimal("2")) == Decimal("13")


def test_cleanup_uses_strategy_minimum_allocation_and_avoids_unneeded_top_up():
    """A redeemable position is selected from strategy parameters without a top-up.

    1. Read HyperAI's minimum-allocation style parameter and verify an explicit zero disables cleanup.
    2. Create a 25 USDC HyperCore position below that allocation but above the redeemable floor.
    3. Verify cleanup plans one full-close trade and does not add temporary capital.
    """

    # 1. Read HyperAI's minimum-allocation style parameter and verify an explicit zero disables cleanup.
    minimum_allocation = get_hypercore_minimum_allocation(
        {"individual_rebalance_min_threshold_usd": 50}
    )
    assert minimum_allocation == Decimal("50")
    assert get_hypercore_minimum_allocation(
        {
            "individual_rebalance_min_threshold_usd": 0,
            "minimum_rebalance_trade_threshold": 50,
        }
    ) is None

    # 2. Create a 25 USDC HyperCore position below that allocation but above the redeemable floor.
    state, position = _make_position(Decimal("25"))
    candidates = discover_hypercore_small_positions(state, minimum_allocation)
    assert len(candidates) == 1
    assert candidates[0].top_up_reserve is None
    trades = plan_hypercore_small_position_cleanup(state, candidates[0], NOW)

    # 3. Verify cleanup plans one full-close trade and does not add temporary capital.
    assert len(trades) == 1
    assert trades[0].is_sell()
    assert trades[0].planned_quantity == -position.get_quantity()
    assert trades[0].other_data["hypercore_small_position_cleanup"] is True
    position.other_data["hypercore_small_position_cleanup_pending_redeem"] = True
    pending_candidates = discover_hypercore_small_positions(state, Decimal("5"))
    assert pending_candidates[0].is_pending_cleanup is True


def test_cleanup_uses_progressively_larger_silent_noop_retry_margins():
    """A cleanup close receives an adaptive retry margin ladder.

    1. Create the normal and cleanup trade markers used by the HyperCore router.
    2. Ask the routing helper for their silent-no-op retry margins.
    3. Verify only cleanup closes receive the progressively larger retry ladder.
    """

    # 1. Create the normal and cleanup trade markers used by the HyperCore router.
    routing = object.__new__(HypercoreVaultRouting)
    normal_trade = SimpleNamespace(other_data={})
    cleanup_trade = SimpleNamespace(other_data={"hypercore_small_position_cleanup": True})

    # 2. Ask the routing helper for their silent-no-op retry margins.
    normal_margins = routing._get_phase1_noop_retry_safety_margins(normal_trade)
    cleanup_margins = routing._get_phase1_noop_retry_safety_margins(cleanup_trade)

    # 3. Verify only cleanup closes receive the progressively larger retry ladder.
    assert normal_margins == (HYPERCORE_WITHDRAWAL_SAFETY_MARGIN_RAW,)
    assert cleanup_margins == HYPERCORE_SMALL_POSITION_CLEANUP_RETRY_SAFETY_MARGINS_RAW


def test_cleanup_retry_keeps_material_residual_tracked_for_another_pass():
    """A high-margin retry does not write off a material HyperCore residual.

    1. Create normal and cleanup retry trades for a HyperCore vault pair.
    2. Ask the router whether their residual must remain tracked.
    3. Verify only the 3 USDC cleanup retry requires a later redemption run.
    """

    # 1. Create normal and cleanup retry trades for a HyperCore vault pair.
    _state, position = _make_position(Decimal("25"))
    routing = object.__new__(HypercoreVaultRouting)
    normal_trade = SimpleNamespace(other_data={}, pair=position.pair)
    cleanup_trade = SimpleNamespace(
        other_data={
            "hypercore_small_position_cleanup": True,
            "hypercore_phase1_retry_safety_margin_raw": 3_000_000,
        },
        pair=position.pair,
    )

    # 2. Ask the router whether their residual must remain tracked.
    normal_has_residual = routing._cleanup_retry_leaves_material_residual(normal_trade)
    cleanup_has_residual = routing._cleanup_retry_leaves_material_residual(cleanup_trade)

    # 3. Verify only the 3 USDC cleanup retry requires a later redemption run.
    assert normal_has_residual is False
    assert cleanup_has_residual is True


def test_cleanup_top_up_waits_for_vault_lock_up_before_redeeming():
    """A cleanup top-up is persisted and a locked position is deferred.

    1. Arrange an unlocked small candidate that needs a top-up and a locked pending candidate.
    2. Run the cleaner with mocks because this unit test must not broadcast to HyperCore.
    3. Verify only the top-up executes and the pending-redeem marker is persisted.
    """

    # 1. Arrange an unlocked small candidate that needs a top-up and a locked pending candidate.
    # Mocks isolate pass orchestration from transaction broadcasts and live HyperCore reads.
    top_up_candidate = HypercoreSmallPositionCandidate(
        position_id=57,
        vault_name="Crypto Plaza",
        vault_address="0x1111111111111111111111111111111111111111",
        live_equity=Decimal("3.45"),
        minimum_allocation=Decimal("5"),
        top_up_reserve=Decimal("11.55"),
        is_pending_cleanup=False,
    )
    locked_candidate = HypercoreSmallPositionCandidate(
        position_id=58,
        vault_name="Crypto Plaza",
        vault_address="0x2222222222222222222222222222222222222222",
        live_equity=Decimal("15"),
        minimum_allocation=Decimal("5"),
        top_up_reserve=None,
        is_pending_cleanup=True,
    )
    state = MagicMock()
    state.portfolio.get_default_reserve_position.return_value.quantity = Decimal("100")
    top_up_position = MagicMock()
    top_up_position.other_data = {}
    state.portfolio.open_positions = {57: top_up_position}
    top_up_trade = MagicMock()
    top_up_trade.is_success.return_value = True
    store = MagicMock()

    # 2. Run the cleaner with mocks because this unit test must not broadcast to HyperCore.
    with (
        patch(
            "tradeexecutor.ethereum.vault.hypercore_small_position_cleanup.plan_hypercore_small_position_cleanup",
            return_value=[top_up_trade],
        ),
        patch(
            "tradeexecutor.ethereum.vault.hypercore_small_position_cleanup.fetch_user_vault_equity",
            side_effect=[
                SimpleNamespace(
                    equity=Decimal("3.45"),
                    is_lockup_expired=True,
                    locked_until=NOW,
                ),
                SimpleNamespace(
                    equity=Decimal("15"),
                    is_lockup_expired=False,
                    locked_until=NOW + datetime.timedelta(days=1),
                ),
            ],
        ),
    ):
        report = run_hypercore_small_position_cleanup(
            state=state,
            timestamp=NOW,
            candidates=[top_up_candidate, locked_candidate],
            execution_model=MagicMock(),
            routing_model=MagicMock(),
            routing_state=MagicMock(),
            session=MagicMock(),
            safe_address="0xSafe",
            store=store,
        )

    # 3. Verify only the top-up executes and the pending-redeem marker is persisted.
    assert report.executed_trades == [top_up_trade]
    assert report.closed_position_ids == []
    assert top_up_position.other_data["hypercore_small_position_cleanup_pending_redeem"] is True
    assert store.sync.call_count == 2


def test_cleanup_saves_preflight_failure_as_a_repairable_failed_trade():
    """A withdrawal preflight failure is persisted without aborting account correction.

    1. Arrange a redeemable candidate whose close preflight raises the standard exception.
    2. Run the cleaner with mocked execution because this unit test must not broadcast to HyperCore.
    3. Verify the attempted trade and failure state are saved, with no false local close.
    """

    # 1. Arrange a redeemable candidate whose close preflight raises the standard exception.
    candidate = HypercoreSmallPositionCandidate(
        position_id=57,
        vault_name="Crypto Plaza",
        vault_address="0x1111111111111111111111111111111111111111",
        live_equity=Decimal("25"),
        minimum_allocation=Decimal("50"),
        top_up_reserve=None,
        is_pending_cleanup=False,
    )
    failed_close_trade = MagicMock()
    store = MagicMock()
    execution_model = MagicMock()
    execution_model.execute_trades.side_effect = HypercoreWithdrawalPreflightError("lock-up changed")
    state = MagicMock()
    state.portfolio.get_default_reserve_position.return_value.quantity = Decimal("100")
    failed_close_trade.is_started.return_value = True

    # 2. Run the cleaner with mocked execution because this unit test must not broadcast to HyperCore.
    with patch(
        "tradeexecutor.ethereum.vault.hypercore_small_position_cleanup.plan_hypercore_small_position_cleanup",
        return_value=[failed_close_trade],
    ), patch(
        "tradeexecutor.ethereum.vault.hypercore_small_position_cleanup.fetch_user_vault_equity",
        return_value=SimpleNamespace(
            equity=Decimal("25"),
            is_lockup_expired=True,
            locked_until=NOW,
        ),
    ), patch(
        "tradeexecutor.ethereum.vault.hypercore_small_position_cleanup.freeze_position_on_failed_trade",
    ) as freeze_position:
        report = run_hypercore_small_position_cleanup(
            state=state,
            timestamp=NOW,
            candidates=[candidate],
            execution_model=execution_model,
            routing_model=MagicMock(),
            routing_state=MagicMock(),
            session=MagicMock(),
            safe_address="0xSafe",
            store=store,
        )

    # 3. Verify the attempted trade and failure state are saved, with no false local close.
    assert report.executed_trades == [failed_close_trade]
    assert report.closed_position_ids == []
    state.mark_trade_failed.assert_called_once_with(NOW, failed_close_trade)
    freeze_position.assert_called_once_with(NOW, state, [failed_close_trade])
    assert store.sync.call_count == 1
