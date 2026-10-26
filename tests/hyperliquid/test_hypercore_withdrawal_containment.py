"""Containment for the October Tortoise withdrawal incident."""

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tradeexecutor.cli.commands.start import should_save_failed_execution
from tradeexecutor.strategy.execution_model import ExecutionHaltableIssue
from tradeexecutor.ethereum.vault.hypercore_routing import HypercoreVaultRouting
from tradeexecutor.state.trade import has_unresolved_hypercore_accounting


def test_saved_second_stage_requires_reconciliation() -> None:
    """Protect the checkpoint that predates the final failure marker.

    1. Check every phase with possible external effects.
    2. Check that planned, skipped and resolved phases remain non-blocking.
    """
    # 1. Check every phase with possible external effects.
    for phase in ("broadcast_pending", "broadcast_ambiguous", "receipt_confirmed", "verification_ambiguous"):
        trade = SimpleNamespace(other_data={"hypercore_second_stage_status": phase})
        assert has_unresolved_hypercore_accounting(trade)
    # 2. Check that planned, skipped and resolved phases remain non-blocking.
    for phase in ("planned", "skipped_pending_protocol_validation", "verified", "reconciled"):
        trade = SimpleNamespace(other_data={"hypercore_second_stage_status": phase})
        assert not has_unresolved_hypercore_accounting(trade)


def test_optional_withdrawal_keeps_verified_first_leg() -> None:
    """Do not request the rejected $1.50 remainder.

    1. Mock fresh equity reads to reproduce the incident without network calls.
    2. Verify no second request is planned and the residual remains visible.
    """
    # 1. Mock fresh equity reads to reproduce the incident without network calls.
    routing = object.__new__(HypercoreVaultRouting)
    routing.lagoon_vault = MagicMock()
    routing._session = MagicMock()
    trade = SimpleNamespace(other_data={}, trade_id=1825)
    equity = SimpleNamespace(equity=Decimal("19.836496"), is_lockup_expired=True)
    with patch("tradeexecutor.ethereum.vault.hypercore_routing.fetch_user_vault_equity", return_value=equity):
        residual, request = routing._plan_same_cycle_residual_withdrawal(trade, "0xvault")
    # 2. Verify no second request is planned and the residual remains visible.
    assert residual == equity.equity
    assert request is None
    assert trade.other_data["hypercore_second_stage_status"] == "skipped_pending_protocol_validation"


def test_scheduler_wrapped_halt_is_saveable() -> None:
    """Save known execution halts through wrappers, without saving arbitrary errors.

    1. Construct a wrapped halt that itself has a lower-level cause.
    2. Verify it requests persistence while an ordinary error does not.
    """
    # 1. Construct a wrapped halt that itself has a lower-level cause.
    low = ValueError("balance verification")
    halt = ExecutionHaltableIssue("withdrawal stopped")
    halt.__cause__ = low
    wrapper = RuntimeError("scheduler stopped")
    wrapper.__cause__ = halt
    # 2. Verify it requests persistence while an ordinary error does not.
    assert should_save_failed_execution(wrapper)
    assert not should_save_failed_execution(low)
