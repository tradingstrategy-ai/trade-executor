"""Test cleanup planning for undersized HyperCore-native vault positions."""

import datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from tradingstrategy.chain import ChainId

from tradeexecutor.ethereum.vault.hypercore_routing import (
    HYPERCORE_SMALL_POSITION_CLEANUP_INITIAL_SAFETY_MARGIN_RAW,
    HYPERCORE_SMALL_POSITION_CLEANUP_RETRY_SAFETY_MARGINS_RAW,
    HYPERCORE_WITHDRAWAL_SAFETY_MARGIN_RAW,
    HypercoreWithdrawalPreflightError,
    HypercoreVaultRouting,
)
from tradeexecutor.ethereum.vault.hypercore_small_position_cleanup import (
    HYPERCORE_SMALL_POSITION_MINIMUM_REDEEMABLE_EQUITY,
    HypercoreSmallPositionCandidate,
    _close_confirmed_hypercore_residual,
    discover_hypercore_small_positions,
    get_hypercore_minimum_allocation,
    plan_hypercore_small_position_cleanup,
    run_hypercore_small_position_cleanup,
)
from tradeexecutor.ethereum.vault.hypercore_vault import create_hypercore_vault_pair
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeType
from tradeexecutor.strategy.dust import (
    HYPERCORE_SMALL_POSITION_CLEANUP_PENDING_REDEEM,
)


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


def test_cleanup_plans_direct_redemption_below_deposit_minimum():
    """A sub-5 USDC HyperCore position is redeemed without a temporary top-up.

    1. Create a 3.45 USDC HyperCore position below the strategy's 50 USDC allocation floor.
    2. Discover it and plan the cleanup.
    3. Verify the plan is a full close and does not add capital or extend the lock-up.
    """

    # 1. Create a 3.45 USDC HyperCore position below the strategy's 50 USDC allocation floor.
    state, position = _make_position(Decimal("3.45"))

    # 2. Discover it and plan the cleanup.
    candidates = discover_hypercore_small_positions(state, Decimal("50"))
    assert len(candidates) == 1
    trade = plan_hypercore_small_position_cleanup(state, candidates[0], NOW)

    # 3. Verify the plan is a full close and does not add capital or extend the lock-up.
    assert trade.is_sell()
    assert trade.planned_quantity == -position.get_quantity()
    assert trade.planned_reserve == pytest.approx(Decimal("3.45"))
    assert trade.other_data["hypercore_small_position_cleanup"] is True


def test_cleanup_dry_run_supersedes_planned_close_without_mutating_state():
    """Dry-run validates cleanup when an old planned close consumes the position quantity.

    1. Create a live undersized position and a planned full-close trade.
    2. Run cleanup in dry-run mode against the otherwise eligible position.
    3. Verify the isolated rehearsal expires the old trade and prepares a replacement.
    4. Verify the supplied state and state store remain untouched.

    Live HyperCore reads and routing are mocked so the unit test cannot sign or
    broadcast a mainnet transaction.
    """

    # 1. Create a live undersized position and a planned full-close trade.
    state, position = _make_position(Decimal("3.45"))
    reserve_asset = state.portfolio.get_default_reserve_position().asset
    _position, old_close_trade, _created = state.portfolio.create_trade(
        strategy_cycle_at=NOW,
        pair=position.pair,
        quantity=-position.get_quantity(),
        reserve=None,
        assumed_price=position.last_token_price,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
        position=position,
        closing=True,
    )
    candidates = discover_hypercore_small_positions(state, Decimal("5"))
    assert len(candidates) == 1
    state_before = state.to_json_safe()
    execution_model = MagicMock(max_slippage=None)
    routing_model = MagicMock()
    routing_state = MagicMock()
    store = MagicMock()

    # 2. Run cleanup in dry-run mode against the otherwise eligible position.
    with patch(
        "tradeexecutor.ethereum.vault.hypercore_small_position_cleanup.fetch_user_vault_equity",
        return_value=SimpleNamespace(
            equity=Decimal("3.45"),
            is_lockup_expired=True,
            locked_until=NOW,
        ),
    ):
        report = run_hypercore_small_position_cleanup(
            state=state,
            timestamp=NOW,
            candidates=candidates,
            execution_model=execution_model,
            routing_model=routing_model,
            routing_state=routing_state,
            session=MagicMock(),
            safe_address="0xSafe",
            store=store,
            dry_run=True,
        )

    # 3. Verify the isolated rehearsal expires the old trade and prepares a replacement.
    prepared_state = routing_model.setup_trades.call_args.kwargs["state"]
    prepared_position = prepared_state.portfolio.open_positions[position.position_id]
    prepared_old_trade = prepared_position.trades[old_close_trade.trade_id]
    prepared_new_trade = prepared_position.get_last_trade()
    assert prepared_old_trade.expired_at == NOW
    assert prepared_new_trade.trade_id != old_close_trade.trade_id
    assert prepared_new_trade.is_started()
    assert prepared_new_trade.planned_quantity == -position.get_quantity()
    assert report.simulated_state is not None
    simulated_position = report.simulated_state.portfolio.open_positions[
        position.position_id
    ]
    assert simulated_position.trades[old_close_trade.trade_id].expired_at == NOW
    assert simulated_position.get_quantity(planned=True) == position.get_quantity()
    assert report.executed_trades == []
    assert report.closed_position_ids == []

    # 4. Verify the supplied state and state store remain untouched.
    assert old_close_trade.is_planned()
    assert state.to_json_safe() == state_before
    store.sync.assert_not_called()
    execution_model.validate_confirmation_configuration.assert_called_once_with()
    execution_model.execute_trades.assert_not_called()


def test_cleanup_uses_strategy_minimum_allocation_and_pending_marker():
    """A redeemable position is selected from strategy parameters and pending state.

    1. Read HyperAI's minimum-allocation style parameter and verify an explicit zero disables cleanup.
    2. Create a 25 USDC HyperCore position below that allocation.
    3. Verify cleanup plans one full-close trade and preserves pending-cleanup discovery.
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

    # 2. Create a 25 USDC HyperCore position below that allocation.
    state, position = _make_position(Decimal("25"))
    candidates = discover_hypercore_small_positions(state, minimum_allocation)
    assert len(candidates) == 1
    trade = plan_hypercore_small_position_cleanup(state, candidates[0], NOW)

    # 3. Verify cleanup plans one full-close trade and preserves pending-cleanup discovery.
    assert trade.is_sell()
    assert trade.planned_quantity == -position.get_quantity()
    assert trade.other_data["hypercore_small_position_cleanup"] is True
    position.other_data[HYPERCORE_SMALL_POSITION_CLEANUP_PENDING_REDEEM] = True
    pending_candidates = discover_hypercore_small_positions(state, Decimal("5"))
    assert pending_candidates[0].is_pending_cleanup is True


def test_cleanup_uses_progressively_larger_silent_noop_retry_margins():
    """A cleanup close receives an adaptive retry ladder without a deposit floor.

    1. Create the normal and cleanup trade markers used by the HyperCore router.
    2. Ask the routing helper for their silent-no-op retry margins.
    3. Verify cleanup starts with a small adaptive margin and accepts a positive sub-5 retry.
    4. Verify only cleanup closes receive the progressively larger retry ladder.
    """

    # 1. Create the normal and cleanup trade markers used by the HyperCore router.
    routing = object.__new__(HypercoreVaultRouting)
    normal_trade = SimpleNamespace(other_data={})
    cleanup_trade = SimpleNamespace(other_data={"hypercore_small_position_cleanup": True})

    # 2. Ask the routing helper for their silent-no-op retry margins.
    normal_margins = routing._get_phase1_noop_retry_safety_margins(normal_trade)
    cleanup_margins = routing._get_phase1_noop_retry_safety_margins(cleanup_trade)

    # 3. Verify cleanup starts with a small adaptive margin and accepts a positive sub-5 retry.
    assert routing._get_full_close_safety_margin_raw(
        cleanup_trade,
        3_450_000,
    ) == HYPERCORE_SMALL_POSITION_CLEANUP_INITIAL_SAFETY_MARGIN_RAW
    with pytest.raises(
        HypercoreWithdrawalPreflightError,
        match="required to verify every withdrawal phase",
    ):
        routing._get_full_close_safety_margin_raw(
            cleanup_trade,
            50_000,
        )
    assert routing._get_full_close_safety_margin_raw(
        normal_trade,
        3_450_000,
    ) == HYPERCORE_WITHDRAWAL_SAFETY_MARGIN_RAW
    retry_raw = routing._get_phase1_noop_retry_raw(
        current_vault_equity=Decimal("3.45"),
        previous_raw=3_350_000,
        safety_margin_raw=250_000,
    )
    assert retry_raw == 3_200_000
    assert routing._get_phase1_noop_retry_raw(
        current_vault_equity=Decimal("3.45"),
        previous_raw=3_350_000,
        safety_margin_raw=1_000_000,
    ) == 2_450_000
    assert routing._get_phase1_noop_retry_raw(
        current_vault_equity=Decimal("1.15"),
        previous_raw=1_050_000,
        safety_margin_raw=1_000_000,
    ) is None

    # 4. Verify only cleanup closes receive the progressively larger retry ladder.
    assert normal_margins == (HYPERCORE_WITHDRAWAL_SAFETY_MARGIN_RAW,)
    assert cleanup_margins == HYPERCORE_SMALL_POSITION_CLEANUP_RETRY_SAFETY_MARGINS_RAW


def test_cleanup_keeps_verifiable_residual_and_closes_protocol_dust():
    """Retain a verifiable HyperCore residual and write off only protocol dust.

    1. Create two tracked HyperCore positions after successful cleanup withdrawals.
    2. Re-read residuals above and at the minimum verifiable equity.
    3. Verify the larger residual stays pending while protocol dust closes locally.
    """

    # 1. Create two tracked HyperCore positions after successful cleanup withdrawals.
    pending_state, pending_position = _make_position(Decimal("0.31"))
    dust_state, dust_position = _make_position(
        HYPERCORE_SMALL_POSITION_MINIMUM_REDEEMABLE_EQUITY
    )
    pending_position.get_first_trade().executed_quantity = Decimal("0.31")
    dust_position.get_first_trade().executed_quantity = (
        HYPERCORE_SMALL_POSITION_MINIMUM_REDEEMABLE_EQUITY
    )
    assert pending_position.can_be_closed() is True
    pending_candidate = HypercoreSmallPositionCandidate(
        position_id=pending_position.position_id,
        vault_name="Pending Fund",
        vault_address=pending_position.pair.pool_address,
        live_equity=Decimal("0.31"),
        minimum_allocation=Decimal("5"),
        is_pending_cleanup=False,
    )
    dust_candidate = HypercoreSmallPositionCandidate(
        position_id=dust_position.position_id,
        vault_name="Dust Fund",
        vault_address=dust_position.pair.pool_address,
        live_equity=HYPERCORE_SMALL_POSITION_MINIMUM_REDEEMABLE_EQUITY,
        minimum_allocation=Decimal("5"),
        is_pending_cleanup=False,
    )

    # 2. Re-read residuals above and at the minimum verifiable equity.
    with patch(
        "tradeexecutor.ethereum.vault.hypercore_small_position_cleanup.fetch_user_vault_equity",
        side_effect=[
            SimpleNamespace(equity=Decimal("0.31")),
            SimpleNamespace(
                equity=HYPERCORE_SMALL_POSITION_MINIMUM_REDEEMABLE_EQUITY
            ),
        ],
    ):
        pending_residual = _close_confirmed_hypercore_residual(
            state=pending_state,
            candidate=pending_candidate,
            session=MagicMock(),
            safe_address="0xSafe",
        )
        dust_residual = _close_confirmed_hypercore_residual(
            state=dust_state,
            candidate=dust_candidate,
            session=MagicMock(),
            safe_address="0xSafe",
        )

    # 3. Verify the larger residual stays pending while protocol dust closes locally.
    assert pending_residual == Decimal("0.31")
    assert (
        pending_position.other_data[
            HYPERCORE_SMALL_POSITION_CLEANUP_PENDING_REDEEM
        ]
        is True
    )
    assert pending_position.can_be_closed() is False
    assert pending_position.position_id in pending_state.portfolio.open_positions
    assert dust_residual is None
    assert dust_position.position_id in dust_state.portfolio.closed_positions


def test_cleanup_directly_redeems_dust_and_defers_locked_position():
    """Redeem viable dust, close protocol dust locally, and defer locked positions.

    1. Arrange unlocked redeemable and protocol-dust candidates plus a locked candidate.
    2. Run the cleaner with mocks because this unit test must not broadcast to HyperCore.
    3. Verify only the viable redemption executes and the protocol dust closes locally.
    """

    # 1. Arrange unlocked redeemable and protocol-dust candidates plus a locked candidate.
    # Mocks isolate pass orchestration from transaction broadcasts and live HyperCore reads.
    dust_candidate = HypercoreSmallPositionCandidate(
        position_id=57,
        vault_name="Crypto Plaza",
        vault_address="0x1111111111111111111111111111111111111111",
        live_equity=Decimal("3.45"),
        minimum_allocation=Decimal("5"),
        is_pending_cleanup=False,
    )
    locked_candidate = HypercoreSmallPositionCandidate(
        position_id=58,
        vault_name="Crypto Plaza",
        vault_address="0x2222222222222222222222222222222222222222",
        live_equity=Decimal("15"),
        minimum_allocation=Decimal("5"),
        is_pending_cleanup=True,
    )
    protocol_dust_candidate = HypercoreSmallPositionCandidate(
        position_id=59,
        vault_name="Tiny Fund",
        vault_address="0x3333333333333333333333333333333333333333",
        live_equity=HYPERCORE_SMALL_POSITION_MINIMUM_REDEEMABLE_EQUITY,
        minimum_allocation=Decimal("5"),
        is_pending_cleanup=False,
    )
    state = MagicMock()
    protocol_dust_position = MagicMock()
    state.portfolio.open_positions.get.return_value = protocol_dust_position
    close_trade = MagicMock()
    close_trade.is_success.return_value = True
    store = MagicMock()
    execution_model = MagicMock()

    # 2. Run the cleaner with mocks because this unit test must not broadcast to HyperCore.
    with (
        patch(
            "tradeexecutor.ethereum.vault.hypercore_small_position_cleanup.plan_hypercore_small_position_cleanup",
            return_value=close_trade,
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
                SimpleNamespace(
                    equity=HYPERCORE_SMALL_POSITION_MINIMUM_REDEEMABLE_EQUITY,
                    is_lockup_expired=True,
                    locked_until=NOW,
                ),
            ],
        ),
        patch(
            "tradeexecutor.ethereum.vault.hypercore_small_position_cleanup._close_confirmed_hypercore_residual",
            return_value=None,
        ),
        patch(
            "tradeexecutor.ethereum.vault.hypercore_small_position_cleanup.close_position_with_empty_trade",
        ) as close_locally,
    ):
        report = run_hypercore_small_position_cleanup(
            state=state,
            timestamp=NOW,
            candidates=[
                dust_candidate,
                locked_candidate,
                protocol_dust_candidate,
            ],
            execution_model=execution_model,
            routing_model=MagicMock(),
            routing_state=MagicMock(),
            session=MagicMock(),
            safe_address="0xSafe",
            store=store,
        )

    # 3. Verify only the viable redemption executes and the protocol dust closes locally.
    assert report.executed_trades == [close_trade]
    assert report.closed_position_ids == [57, 59]
    execution_model.execute_trades.assert_called_once()
    close_locally.assert_called_once_with(state.portfolio, protocol_dust_position)
    assert store.sync.call_count == 3


def test_cleanup_dry_run_validates_local_closure_without_mutating_state():
    """Dry-run validates local zero-equity closure on an isolated state copy.

    1. Arrange a tracked HyperCore position whose live equity has become zero.
    2. Run the cleaner in dry-run mode with the live balance mocked.
    3. Verify the supplied state and state store remain untouched.

    The live equity read is mocked because this test covers local state
    simulation rather than HyperCore API behaviour.
    """

    # 1. Arrange a tracked HyperCore position whose live equity has become zero.
    state, position = _make_position(Decimal("3.45"))
    zero_candidate = HypercoreSmallPositionCandidate(
        position_id=position.position_id,
        vault_name="Loop Fund",
        vault_address=position.pair.pool_address,
        live_equity=Decimal(0),
        minimum_allocation=Decimal("5"),
        is_pending_cleanup=False,
    )
    execution_model = MagicMock()
    store = MagicMock()
    state_before = state.to_json_safe()

    # 2. Run the cleaner in dry-run mode with the live balance mocked.
    with patch(
        "tradeexecutor.ethereum.vault.hypercore_small_position_cleanup.fetch_user_vault_equity",
        return_value=None,
    ):
        report = run_hypercore_small_position_cleanup(
            state=state,
            timestamp=NOW,
            candidates=[zero_candidate],
            execution_model=execution_model,
            routing_model=MagicMock(),
            routing_state=MagicMock(),
            session=MagicMock(),
            safe_address="0xSafe",
            store=store,
            dry_run=True,
        )

    # 3. Verify the supplied state and state store remain untouched.
    assert report.executed_trades == []
    assert report.closed_position_ids == []
    assert state.to_json_safe() == state_before
    execution_model.execute_trades.assert_not_called()
    store.sync.assert_not_called()


def test_cleanup_saves_preflight_failure_without_leaving_a_planned_trade():
    """Persist started failures and expire preflight failures before broadcast.

    1. Arrange two redeemable candidates whose close preflight raises before and after trade start.
    2. Run the cleaner with mocked execution because this unit test must not broadcast to HyperCore.
    3. Verify the started trade is repairable and the planned trade expires for rediscovery.
    """

    # 1. Arrange two redeemable candidates whose close preflight raises before and after trade start.
    started_candidate = HypercoreSmallPositionCandidate(
        position_id=57,
        vault_name="Crypto Plaza",
        vault_address="0x1111111111111111111111111111111111111111",
        live_equity=Decimal("25"),
        minimum_allocation=Decimal("50"),
        is_pending_cleanup=False,
    )
    planned_candidate = HypercoreSmallPositionCandidate(
        position_id=58,
        vault_name="Loop Fund",
        vault_address="0x2222222222222222222222222222222222222222",
        live_equity=Decimal("25"),
        minimum_allocation=Decimal("50"),
        is_pending_cleanup=False,
    )
    failed_close_trade = MagicMock()
    planned_close_trade = MagicMock()
    store = MagicMock()
    execution_model = MagicMock()
    execution_model.execute_trades.side_effect = HypercoreWithdrawalPreflightError("lock-up changed")
    state = MagicMock()
    state.portfolio.get_default_reserve_position.return_value.quantity = Decimal("100")
    failed_close_trade.is_started.return_value = True
    planned_close_trade.is_started.return_value = False
    planned_close_trade.is_planned.return_value = True

    # 2. Run the cleaner with mocked execution because this unit test must not broadcast to HyperCore.
    with patch(
        "tradeexecutor.ethereum.vault.hypercore_small_position_cleanup.plan_hypercore_small_position_cleanup",
        side_effect=[failed_close_trade, planned_close_trade],
    ), patch(
        "tradeexecutor.ethereum.vault.hypercore_small_position_cleanup.fetch_user_vault_equity",
        side_effect=[
            SimpleNamespace(
                equity=Decimal("25"),
                is_lockup_expired=True,
                locked_until=NOW,
            ),
            SimpleNamespace(
                equity=Decimal("25"),
                is_lockup_expired=True,
                locked_until=NOW,
            ),
        ],
    ), patch(
        "tradeexecutor.ethereum.vault.hypercore_small_position_cleanup.freeze_position_on_failed_trade",
    ) as freeze_position:
        report = run_hypercore_small_position_cleanup(
            state=state,
            timestamp=NOW,
            candidates=[started_candidate, planned_candidate],
            execution_model=execution_model,
            routing_model=MagicMock(),
            routing_state=MagicMock(),
            session=MagicMock(),
            safe_address="0xSafe",
            store=store,
        )

    # 3. Verify the started trade is repairable and the planned trade expires for rediscovery.
    assert report.executed_trades == [failed_close_trade, planned_close_trade]
    assert report.closed_position_ids == []
    state.mark_trade_failed.assert_called_once_with(NOW, failed_close_trade)
    freeze_position.assert_called_once_with(NOW, state, [failed_close_trade])
    planned_close_trade.mark_expired.assert_called_once_with(NOW)
    assert store.sync.call_count == 2
