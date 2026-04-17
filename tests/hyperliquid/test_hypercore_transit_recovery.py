"""Test Safe-level HyperCore spot/perp transit recovery helpers."""

import datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tradeexecutor.cli.commands import correct_accounts as correct_accounts_command
from tradeexecutor.ethereum.vault import hypercore_transit_recovery
from tradeexecutor.ethereum.vault.hypercore_transit_recovery import (
    HYPERCORE_TRANSIT_RECOVERY_DUST_USDC,
    HypercoreTransitBalanceSnapshot,
    HypercoreTransitRecoveryAction,
    plan_hypercore_transit_recovery_actions,
)
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.strategy.account_correction import (
    AccountingBalanceCheck,
    AccountingCorrectionCause,
    apply_accounting_correction,
)
from tradeexecutor.strategy.execution_model import AssetManagementMode


SAFE_ADDRESS = "0xB136581dFB3efA76Ae71293C1A70942f0726E8fD"


def _snapshot(
    *,
    spot_free_usdc: Decimal = Decimal(0),
    perp_withdrawable: Decimal = Decimal(0),
    perp_position_count: int = 0,
) -> HypercoreTransitBalanceSnapshot:
    """Build a transit balance snapshot for planner tests."""
    return HypercoreTransitBalanceSnapshot(
        safe_address=SAFE_ADDRESS,
        evm_usdc_balance=Decimal("1"),
        spot_total_usdc=spot_free_usdc,
        spot_free_usdc=spot_free_usdc,
        perp_withdrawable=perp_withdrawable,
        perp_account_value=perp_withdrawable,
        perp_position_count=perp_position_count,
    )


def test_hypercore_transit_plan_leaves_perp_and_spot_dust() -> None:
    """Perp and spot recovery leaves fixed dust on both HyperCore balances.

    1. Build a snapshot with meaningful Safe-level spot and perp USDC.
    2. Plan HyperCore transit recovery actions.
    3. Assert the planner moves only balances above the fixed dust amount.
    """
    # Step 1: Build a snapshot with meaningful Safe-level spot and perp USDC.
    snapshot = _snapshot(
        spot_free_usdc=Decimal("33.611598"),
        perp_withdrawable=Decimal("768.875892"),
    )

    # Step 2: Plan HyperCore transit recovery actions.
    actions = plan_hypercore_transit_recovery_actions(snapshot)

    # Step 3: Assert the planner moves only balances above the fixed dust amount.
    assert [action.action_kind for action in actions] == [
        "perp_to_spot",
        "spot_to_evm",
    ]
    assert actions[0].amount == Decimal("768.375892")
    assert actions[1].amount == Decimal("801.487490")


def test_hypercore_transit_plan_handles_spot_only_balance() -> None:
    """Spot-only recovery bridges spot USDC while leaving spot dust.

    1. Build a snapshot with only Safe-level spot USDC.
    2. Plan HyperCore transit recovery actions.
    3. Assert only the spot-to-EVM leg is planned.
    """
    # Step 1: Build a snapshot with only Safe-level spot USDC.
    snapshot = _snapshot(spot_free_usdc=Decimal("10"))

    # Step 2: Plan HyperCore transit recovery actions.
    actions = plan_hypercore_transit_recovery_actions(snapshot)

    # Step 3: Assert only the spot-to-EVM leg is planned.
    assert [action.action_kind for action in actions] == ["spot_to_evm"]
    assert actions[0].amount == Decimal("9.50")


def test_hypercore_transit_plan_handles_perp_only_balance() -> None:
    """Perp-only recovery plans perp-to-spot and then bridges the post-transfer spot balance.

    1. Build a snapshot with only Safe-level perp USDC.
    2. Plan HyperCore transit recovery actions.
    3. Assert the planner leaves fixed dust in both perp and spot.
    """
    # Step 1: Build a snapshot with only Safe-level perp USDC.
    snapshot = _snapshot(perp_withdrawable=Decimal("10"))

    # Step 2: Plan HyperCore transit recovery actions.
    actions = plan_hypercore_transit_recovery_actions(snapshot)

    # Step 3: Assert the planner leaves fixed dust in both perp and spot.
    assert [action.action_kind for action in actions] == [
        "perp_to_spot",
        "spot_to_evm",
    ]
    assert actions[0].amount == Decimal("9.50")
    assert actions[1].amount == Decimal("9.00")


def test_hypercore_transit_plan_ignores_dust_only_balances() -> None:
    """Dust-only spot and perp balances do not produce recovery actions.

    1. Build a snapshot at the fixed dust threshold for both spot and perp.
    2. Plan HyperCore transit recovery actions.
    3. Assert no zero, negative, or dust-only action is emitted.
    """
    # Step 1: Build a snapshot at the fixed dust threshold for both spot and perp.
    snapshot = _snapshot(
        spot_free_usdc=HYPERCORE_TRANSIT_RECOVERY_DUST_USDC,
        perp_withdrawable=HYPERCORE_TRANSIT_RECOVERY_DUST_USDC,
    )

    # Step 2: Plan HyperCore transit recovery actions.
    actions = plan_hypercore_transit_recovery_actions(snapshot)

    # Step 3: Assert no zero, negative, or dust-only action is emitted.
    assert actions == []


def test_hypercore_transit_plan_rejects_active_perp_positions() -> None:
    """Active Safe-level perp positions abort recovery before broadcasting.

    1. Build a snapshot with an active HyperCore perp position.
    2. Attempt to plan HyperCore transit recovery actions.
    3. Assert planning raises a manual-review error.
    """
    # Step 1: Build a snapshot with an active HyperCore perp position.
    snapshot = _snapshot(perp_withdrawable=Decimal("10"), perp_position_count=1)

    # Step 2: Attempt to plan HyperCore transit recovery actions.
    # Step 3: Assert planning raises a manual-review error.
    with pytest.raises(RuntimeError, match="active HyperCore perp position"):
        plan_hypercore_transit_recovery_actions(snapshot)


def test_hypercore_transit_execution_uses_dust_safe_spot_amount(monkeypatch) -> None:
    """Execution broadcasts perp-to-spot and spot-to-EVM using dust-preserving amounts.

    1. Mock live snapshots before each execution phase.
    2. Execute a two-leg recovery plan with mocked CoreWriter calls and waits.
    3. Assert the spot bridge amount is below the full spot balance and leaves dust.

    CoreWriter broadcasts are mocked because the test verifies recovery accounting,
    not contract execution on a live HyperEVM node.
    """
    # Step 1: Mock live snapshots before each execution phase.
    snapshots = [
        _snapshot(spot_free_usdc=Decimal("0"), perp_withdrawable=Decimal("5")),
        _snapshot(spot_free_usdc=Decimal("4.50"), perp_withdrawable=Decimal("0.50")),
    ]
    monkeypatch.setattr(
        hypercore_transit_recovery,
        "fetch_hypercore_transit_balances",
        lambda **kwargs: snapshots.pop(0),
    )

    reserve_token = MagicMock()
    reserve_token.convert_to_raw.side_effect = lambda amount: int(
        (Decimal(amount) * Decimal(10**6)).to_integral_value()
    )

    sent_calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        hypercore_transit_recovery,
        "build_hypercore_transfer_usd_class_call",
        lambda lagoon_vault, hypercore_usdc_amount, to_perp: (
            "perp_to_spot",
            hypercore_usdc_amount,
        ),
    )
    monkeypatch.setattr(
        hypercore_transit_recovery,
        "build_hypercore_send_asset_to_evm_call",
        lambda lagoon_vault, evm_usdc_amount: ("spot_to_evm", evm_usdc_amount),
    )
    monkeypatch.setattr(
        hypercore_transit_recovery,
        "broadcast_bound_call",
        lambda web3, hot_wallet, bound_func, gas_limit=650000: sent_calls.append(bound_func) or "0x1",
    )
    monkeypatch.setattr(
        hypercore_transit_recovery,
        "wait_for_spot_free_balance",
        lambda session, user, baseline_balance, expected_increase: Decimal("4.50"),
    )
    monkeypatch.setattr(
        hypercore_transit_recovery,
        "wait_for_evm_usdc_balance",
        lambda token, address, baseline_balance, expected_increase: Decimal("5.00"),
    )

    actions = [
        HypercoreTransitRecoveryAction(
            action_kind="perp_to_spot",
            amount=Decimal("4.50"),
            reason="test",
        ),
        HypercoreTransitRecoveryAction(
            action_kind="spot_to_evm",
            amount=Decimal("4.00"),
            reason="test",
        ),
    ]

    # Step 2: Execute a two-leg recovery plan with mocked CoreWriter calls and waits.
    executed = hypercore_transit_recovery.execute_hypercore_transit_recovery_actions(
        web3=MagicMock(),
        hot_wallet=MagicMock(),
        lagoon_vault=SimpleNamespace(safe_address=SAFE_ADDRESS),
        session=object(),
        reserve_token=reserve_token,
        actions=actions,
    )

    # Step 3: Assert the spot bridge amount is below the full spot balance and leaves dust.
    assert executed == ["perp_to_spot", "spot_to_evm"]
    assert sent_calls == [
        ("perp_to_spot", 4_500_000),
        ("spot_to_evm", 4_000_000),
    ]


def test_correct_accounts_recovery_helper_executes_for_closed_hypercore_position(monkeypatch) -> None:
    """Correct-accounts recovery helper executes when a closed Hypercore position exists.

    1. Build a fake Lagoon sync model and state with one closed Hypercore position.
    2. Mock HyperCore snapshot planning and execution.
    3. Assert recovery is broadcast through the shared executor.

    The Lagoon sync model is mocked because this is a command hook test, not a
    Lagoon vault deployment test.
    """
    # Step 1: Build a fake Lagoon sync model and state with one closed Hypercore position.
    class FakeLagoonVaultSyncModel:
        def __init__(self) -> None:
            self.vault = SimpleNamespace(
                safe_address=SAFE_ADDRESS,
                underlying_token=MagicMock(),
            )

        def get_token_storage_address(self) -> str:
            return SAFE_ADDRESS

    state = SimpleNamespace(
        portfolio=SimpleNamespace(
            closed_positions={
                1: SimpleNamespace(
                    pair=SimpleNamespace(is_hyperliquid_vault=lambda: True)
                )
            }
        )
    )
    sync_model = FakeLagoonVaultSyncModel()
    hot_wallet = MagicMock()
    correct_accounts_command.logger = MagicMock()
    monkeypatch.setattr(
        correct_accounts_command,
        "LagoonVaultSyncModel",
        FakeLagoonVaultSyncModel,
    )

    action = HypercoreTransitRecoveryAction(
        action_kind="spot_to_evm",
        amount=Decimal("9.50"),
        reason="test",
    )
    monkeypatch.setattr(
        hypercore_transit_recovery,
        "fetch_hypercore_transit_balances",
        lambda **kwargs: _snapshot(spot_free_usdc=Decimal("10")),
    )
    monkeypatch.setattr(
        hypercore_transit_recovery,
        "plan_hypercore_transit_recovery_actions",
        lambda snapshot: [action],
    )
    executed_arguments = {}
    monkeypatch.setattr(
        hypercore_transit_recovery,
        "execute_hypercore_transit_recovery_actions",
        lambda **kwargs: executed_arguments.update(kwargs) or ["spot_to_evm"],
    )

    # Step 2: Mock HyperCore snapshot planning and execution.
    executed = correct_accounts_command._recover_hypercore_transit_balances(
        asset_management_mode=AssetManagementMode.lagoon,
        sync_model=sync_model,
        web3=SimpleNamespace(eth=SimpleNamespace(chain_id=999)),
        hot_wallet=hot_wallet,
        state=state,
        skip_hypercore_transit_recovery=False,
    )

    # Step 3: Assert recovery is broadcast through the shared executor.
    assert executed == ["spot_to_evm"]
    assert executed_arguments["actions"] == [action]
    hot_wallet.sync_nonce.assert_called_once()


def test_correct_accounts_recovery_helper_can_be_skipped(monkeypatch) -> None:
    """Correct-accounts skip flag bypasses HyperCore recovery broadcasts.

    1. Build a fake state with one closed Hypercore position.
    2. Run the recovery helper with the skip flag enabled.
    3. Assert no HyperCore session or broadcast is attempted.
    """
    # Step 1: Build a fake state with one closed Hypercore position.
    state = SimpleNamespace(
        portfolio=SimpleNamespace(
            closed_positions={
                1: SimpleNamespace(
                    pair=SimpleNamespace(is_hyperliquid_vault=lambda: True)
                )
            }
        )
    )
    correct_accounts_command.logger = MagicMock()
    monkeypatch.setattr(
        hypercore_transit_recovery,
        "execute_hypercore_transit_recovery_actions",
        MagicMock(side_effect=AssertionError("should not broadcast")),
    )

    # Step 2: Run the recovery helper with the skip flag enabled.
    executed = correct_accounts_command._recover_hypercore_transit_balances(
        asset_management_mode=AssetManagementMode.lagoon,
        sync_model=MagicMock(),
        web3=SimpleNamespace(eth=SimpleNamespace(chain_id=999)),
        hot_wallet=MagicMock(),
        state=state,
        skip_hypercore_transit_recovery=True,
    )

    # Step 3: Assert no HyperCore session or broadcast is attempted.
    assert executed == []


def test_apply_accounting_correction_records_expected_old_balance() -> None:
    """Accounting correction audit event records the pre-correction ledger balance.

    1. Build a state reserve position with an expected ledger balance.
    2. Apply an accounting correction to a different actual balance.
    3. Assert the balance update old_balance is the expected amount.
    """
    # Step 1: Build a state reserve position with an expected ledger balance.
    asset = AssetIdentifier(
        chain_id=999,
        address="0x0000000000000000000000000000000000000001",
        token_symbol="USDC",
        decimals=6,
    )
    state = State()
    reserve = state.portfolio.initialise_reserves(asset, reserve_token_price=1.0)
    reserve.quantity = Decimal("100")
    correction = AccountingBalanceCheck(
        type=AccountingCorrectionCause.unknown_cause,
        holding_address=SAFE_ADDRESS,
        asset=asset,
        positions={reserve},
        expected_amount=Decimal("100"),
        actual_amount=Decimal("125"),
        dust_epsilon=Decimal("0.01"),
        relative_epsilon=0.0,
        block_number=123,
        timestamp=datetime.datetime(2026, 4, 17, 12, 0, 0),
        usd_value=25.0,
        reserve_asset=True,
        mismatch=True,
        price=Decimal("1"),
        price_at=datetime.datetime(2026, 4, 17, 12, 0, 0),
    )

    # Step 2: Apply an accounting correction to a different actual balance.
    event = apply_accounting_correction(
        state=state,
        correction=correction,
        strategy_cycle_included_at=None,
    )

    # Step 3: Assert the balance update old_balance is the expected amount.
    assert event.old_balance == Decimal("100")
    assert reserve.quantity == Decimal("125")
