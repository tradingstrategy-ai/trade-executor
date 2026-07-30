"""Test P5: Stranded USDC marker on failed Hypercore deposits."""

from unittest.mock import MagicMock, patch

import pytest

from tradeexecutor.ethereum.vault.hypercore_routing import HypercoreVaultRouting
from tradeexecutor.state.repair import HypercoreTransitRecoveryRequired, repair_trades


def test_mark_stranded_usdc_stores_info():
    """_mark_stranded_usdc records recovery info in trade.other_data."""
    routing = MagicMock(spec=HypercoreVaultRouting)
    routing.safe_address = "0xABC123"

    trade = MagicMock()
    trade.other_data = {}
    trade.trade_id = 42

    HypercoreVaultRouting._mark_stranded_usdc(
        routing,
        trade=trade,
        raw_amount=50_000_000,
        location="hypercore_spot",
    )

    info = trade.other_data["hypercore_stranded_usdc"]
    assert info["amount_raw"] == 50_000_000
    assert info["amount_human"] == "50"
    assert info["location"] == "hypercore_spot"
    assert info["safe_address"] == "0xABC123"
    assert "check-hypercore-user.py" in info["recovery"]
    trade.add_note.assert_called_once()
    assert "stranded" in trade.add_note.call_args[0][0].lower()


def test_mark_stranded_usdc_creates_other_data():
    """Works even if trade.other_data is None."""
    routing = MagicMock(spec=HypercoreVaultRouting)
    routing.safe_address = "0xABC123"

    trade = MagicMock()
    trade.other_data = None
    trade.trade_id = 1

    HypercoreVaultRouting._mark_stranded_usdc(
        routing,
        trade=trade,
        raw_amount=10_000_000,
        location="hypercore_spot_or_perp",
    )

    assert trade.other_data is not None
    assert "hypercore_stranded_usdc" in trade.other_data


def test_phase1_at_risk_marker_is_written_before_broadcast():
    """Phase-1 deposit protection is persisted before an uncertain broadcast.

    1. Create a mocked live HyperCore routing and buy trade.
    2. Mark the phase-1 amount as at risk before broadcast.
    3. Verify generic failed-buy accounting will retain the allocation.
    """
    # 1. Use a mock because this marker must exist even if no RPC receipt arrives.
    routing = MagicMock(spec=HypercoreVaultRouting)
    routing.safe_address = "0xABC123"
    trade = MagicMock()
    trade.other_data = {}

    # 2. Persist the conservative phase-1 checkpoint.
    HypercoreVaultRouting._mark_deposit_capital_at_risk(
        routing,
        trade=trade,
        raw_amount=48_884_068,
    )

    # 3. A crash can no longer look like an unused reserve allocation.
    marker = trade.other_data["hypercore_deposit_capital_at_risk"]
    assert marker["amount_raw"] == 48_884_068
    assert marker["phase"] == "phase1_broadcast_pending"
    assert trade.other_data["retain_reserve_allocation_on_failure"] is True


def test_state_only_repair_refuses_unreconciled_hypercore_deposit():
    """State-only repair must not refund a deposit whose location is unknown.

    1. Create a failed trade carrying the HyperCore at-risk marker.
    2. Present it to the state-only repair workflow.
    3. Verify repair stops and instructs the operator to reconcile live balances.
    """
    # 1. This mock represents a crash before HyperCore receipt classification.
    state = MagicMock()
    state.portfolio.frozen_positions.values.return_value = []
    trade = MagicMock()
    trade.trade_id = 1486
    trade.other_data = {"hypercore_deposit_capital_at_risk": {"amount_raw": 48_884_068}}

    # 2. State-only repair has no RPC/Info API evidence to resolve the location.
    with patch("tradeexecutor.state.repair.find_trades_to_be_repaired", return_value=[trade]):
        # 3. It must fail closed rather than manufacture a counter-trade refund.
        with pytest.raises(HypercoreTransitRecoveryRequired, match="check-hypercore-user.py"):
            repair_trades(state, attempt_repair=True, interactive=False)
