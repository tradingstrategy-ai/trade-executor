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


def test_hypercore_transit_plan_recovers_perp_excess_before_existing_spot() -> None:
    """Default recovery returns all newly recovered perp USDC before spot cleanup.

    1. Build a snapshot with meaningful Safe-level spot and perp USDC.
    2. Plan HyperCore transit recovery actions.
    3. Assert the planner bridges the full perp excess, then cleans old spot USDC.
    """
    # Step 1: Build a snapshot with meaningful Safe-level spot and perp USDC.
    snapshot = _snapshot(
        spot_free_usdc=Decimal("33.611598"),
        perp_withdrawable=Decimal("768.875892"),
    )

    # Step 2: Plan HyperCore transit recovery actions.
    actions = plan_hypercore_transit_recovery_actions(snapshot)

    # Step 3: Assert the full perp excess returns before old spot cleanup.
    assert [action.action_kind for action in actions] == [
        "perp_to_spot",
        "spot_to_evm",
        "spot_to_evm",
    ]
    assert actions[0].amount == Decimal("768.375892")
    assert actions[1].amount == Decimal("768.375892")
    assert actions[2].amount == Decimal("33.111598")


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


def test_hypercore_transit_plan_keeps_bridge_fee_margin_for_perp_only_balance() -> None:
    """Perp-only recovery leaves only HyperCore's bridge-fee margin in spot.

    1. Build a snapshot with only Safe-level perp USDC.
    2. Plan HyperCore transit recovery actions.
    3. Assert the planner retains perp dust and its required bridge-fee margin.
    """
    # Step 1: Build a snapshot with only Safe-level perp USDC.
    snapshot = _snapshot(perp_withdrawable=Decimal("10"))

    # Step 2: Plan HyperCore transit recovery actions.
    actions = plan_hypercore_transit_recovery_actions(snapshot)

    # Step 3: Assert the planner retains perp dust and the required fee margin.
    assert [action.action_kind for action in actions] == [
        "perp_to_spot",
        "spot_to_evm",
    ]
    assert actions[0].amount == Decimal("9.50")
    assert actions[1].amount == Decimal("9.49")


def test_hypercore_transit_plan_keeps_unbridgeable_perp_dust_in_perp() -> None:
    """Planner does not strand a dust-sized perp-to-spot first leg.

    1. Build a perp-only snapshot whose recovered excess cannot cover bridge headroom.
    2. Plan HyperCore transit recovery actions.
    3. Assert neither half of an unexecutable two-leg recovery is emitted.
    """
    # Step 1: 0.021 USDC excess becomes an unbridgeable 0.011 USDC after the fee margin.
    snapshot = _snapshot(perp_withdrawable=Decimal("0.521"))

    # Step 2: Plan recovery before any Safe/CoreWriter transaction is created.
    actions = plan_hypercore_transit_recovery_actions(snapshot)

    # Step 3: Retain the tiny perp balance instead of stranding it in spot.
    assert actions == []


def test_hypercore_transit_plan_recovers_incident_amount_by_default() -> None:
    """Default #1486 recovery preserves pre-existing HyperCore balances.

    1. Build the incident snapshot: 48.884068 USDC above 0.50 perp dust and
       0.217882 USDC already in spot.
    2. Plan the default HyperCore transit recovery.
    3. Assert both transfer legs use exactly that amount, not a dust sweep.
    """
    # Step 1: Build the #1486 balances recorded after the failed vault deposit.
    snapshot = _snapshot(
        spot_free_usdc=Decimal("0.217882"),
        perp_withdrawable=Decimal("49.384068"),
    )

    # Step 2: Plan default recovery; no incident-specific command option is needed.
    actions = plan_hypercore_transit_recovery_actions(snapshot)

    # Step 3: The original spot/perp balances remain untouched by the default plan.
    assert [action.action_kind for action in actions] == [
        "perp_to_spot",
        "spot_to_evm",
    ]
    assert [action.amount for action in actions] == [
        Decimal("48.884068"),
        Decimal("48.884068"),
    ]


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


def test_hypercore_spot_to_evm_execution_keeps_bridge_fee_margin(monkeypatch) -> None:
    """Spot-to-EVM execution leaves HyperCore's mandatory fee headroom.

    1. Mock a spot account which contains only newly recovered perp USDC.
    2. Execute the requested full bridge amount through the real fee calculation.
    3. Assert the CoreWriter call keeps exactly the 0.01 USDC bridge-fee margin.

    The account reader, transaction builder, broadcaster, and confirmation wait
    are mocked because this verifies the client-side fee-safety calculation,
    not a live Safe/module transaction.
    """
    # Step 1: Model a 4.50 USDC perp recovery with no pre-existing spot headroom.
    monkeypatch.setattr(
        hypercore_transit_recovery,
        "fetch_hypercore_transit_balances",
        lambda **kwargs: _snapshot(spot_free_usdc=Decimal("4.50")),
    )
    reserve_token = MagicMock()
    reserve_token.convert_to_raw.side_effect = lambda amount: int(
        (Decimal(amount) * Decimal(10**6)).to_integral_value()
    )
    sent_calls: list[tuple[str, int]] = []
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
        "wait_for_evm_usdc_balance",
        lambda token, address, baseline_balance, expected_increase: expected_increase,
    )

    # Step 2: Request the full recovered amount; execution applies the protocol margin.
    hypercore_transit_recovery.execute_spot_to_evm(
        web3=MagicMock(),
        hot_wallet=MagicMock(),
        lagoon_vault=SimpleNamespace(safe_address=SAFE_ADDRESS),
        session=object(),
        reserve_token=reserve_token,
        amount=Decimal("4.50"),
    )

    # Step 3: The builder receives 4.49 USDC, retaining exactly 0.01 USDC in spot.
    assert sent_calls == [("spot_to_evm", 4_490_000)]


def test_correct_accounts_recovery_helper_executes_for_open_hypercore_position(monkeypatch) -> None:
    """Correct-accounts recovery runs for an open HyperCore position by default.

    1. Build a fake Lagoon sync model and state with one open HyperCore position.
    2. Mock HyperCore snapshot planning and execution.
    3. Assert recovery is broadcast through the shared executor.

    The Lagoon sync model is mocked because this is a command hook test, not a
    Lagoon vault deployment test.
    """
    # Step 1: Build a fake Lagoon sync model and state with one open HyperCore position.
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
            open_positions={
                1: SimpleNamespace(
                    pair=SimpleNamespace(is_hyperliquid_vault=lambda: True)
                )
            },
            frozen_positions={},
            closed_positions={},
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
        lambda snapshot, **kwargs: [action],
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
        dry_run=False,
    )

    # Step 3: Assert recovery is broadcast through the shared executor.
    assert executed == ["spot_to_evm"]
    assert executed_arguments["actions"] == [action]
    hot_wallet.sync_nonce.assert_called_once()


def test_correct_accounts_detects_frozen_hypercore_position() -> None:
    """Frozen HyperCore positions enable the default Safe-level recovery check.

    1. Build a state with only a frozen HyperCore vault position.
    2. Check whether the correct-accounts gate detects HyperCore strategy usage.
    3. Assert the Safe-level transit recovery remains enabled for that strategy.
    """
    # Step 1: A frozen position can retain the only state evidence after a failed trade.
    state = SimpleNamespace(
        portfolio=SimpleNamespace(
            open_positions={},
            frozen_positions={
                1: SimpleNamespace(
                    pair=SimpleNamespace(is_hyperliquid_vault=lambda: True)
                )
            },
            closed_positions={},
        )
    )

    # Step 2: Inspect the state-only gate before any live HyperCore API read.
    has_hypercore_position = correct_accounts_command._has_hypercore_vault_positions(state)

    # Step 3: Frozen strategies use the same default recovery path as open ones.
    assert has_hypercore_position is True


def test_correct_accounts_dry_run_plans_transit_recovery_without_signer(monkeypatch) -> None:
    """Dry-run exposes #1486-style transit recovery without broadcasting it.

    1. Build a Lagoon state with a closed HyperCore position and stranded perp USDC.
    2. Run the correct-accounts recovery hook in dry-run mode without a hot-wallet signer.
    3. Verify it reports the planned recovery while never invoking the broadcaster.

    The HyperCore snapshot and broadcaster are mocked because this test checks
    the command's no-side-effect contract rather than a live Safe transaction.
    """
    # Step 1: Model the legacy-state condition: closed positions exist even
    # though the state may predate the #1486 failure marker.
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
    monkeypatch.setattr(
        correct_accounts_command,
        "LagoonVaultSyncModel",
        FakeLagoonVaultSyncModel,
    )
    monkeypatch.setattr(
        hypercore_transit_recovery,
        "fetch_hypercore_transit_balances",
        lambda **kwargs: _snapshot(perp_withdrawable=Decimal("49.384068")),
    )
    broadcast = MagicMock(side_effect=AssertionError("dry run must not broadcast"))
    monkeypatch.setattr(
        hypercore_transit_recovery,
        "execute_hypercore_transit_recovery_actions",
        broadcast,
    )

    # Step 2: Dry-run must require only read access, not the Safe's private key.
    planned = correct_accounts_command._recover_hypercore_transit_balances(
        asset_management_mode=AssetManagementMode.lagoon,
        sync_model=FakeLagoonVaultSyncModel(),
        web3=SimpleNamespace(eth=SimpleNamespace(chain_id=999)),
        hot_wallet=None,
        state=state,
        skip_hypercore_transit_recovery=False,
        dry_run=True,
    )

    # Step 3: The operator can inspect the exact incident recovery with no mutation.
    assert planned == ["perp_to_spot", "spot_to_evm"]
    broadcast.assert_not_called()


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
        dry_run=False,
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
