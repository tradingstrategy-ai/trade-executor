"""Containment for the October Tortoise withdrawal incident."""

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tradeexecutor.cli.commands.start import should_save_failed_execution
from tradeexecutor.strategy.dust import HYPERCORE_ACCEPTED_RESIDUAL_USD, HYPERCORE_FULL_CLOSE_MIN_REMAINDER_USD
from tradeexecutor.strategy.execution_model import ExecutionHaltableIssue
from tradeexecutor.ethereum.vault.hypercore_routing import HypercoreVaultRouting, vault_kept_requested_funds
from tradeexecutor.ethereum.vault.hypercore_withdrawal_recovery import clear_rejected_first_stage_withdrawals
from tradeexecutor.state.trade import TradeFlag, TradeStatus, has_unresolved_hypercore_accounting


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


def test_full_close_leaves_minimum_remainder() -> None:
    """Leave the HyperCore minimum deposit plus the drift floor in a closed vault.

    The 28 October 2026 Tortoise close requested a withdrawal leaving $1.50 and
    was rejected with ``Vault deposits must be at least $5.``

    1. Size the 28 October close, a large close and a cleanup withdrawal.
    2. Verify the accepted residual boundary covers the deliberate remainder.
    """
    # 1. Size the 28 October close, a large close and a cleanup withdrawal.
    routing = object.__new__(HypercoreVaultRouting)
    trade = SimpleNamespace(other_data={})
    assert routing._get_full_close_safety_margin_raw(trade, 19_836_430) == 6_500_000
    assert routing._get_full_close_safety_margin_raw(trade, 20_000_000_000) == 100_000_000
    cleanup = SimpleNamespace(other_data={"hypercore_small_position_cleanup": True})
    assert routing._get_full_close_safety_margin_raw(cleanup, 4_000_000) == 100_000
    # 2. Verify the accepted residual boundary covers the deliberate remainder.
    assert HYPERCORE_FULL_CLOSE_MIN_REMAINDER_USD == Decimal("6.50")
    assert HYPERCORE_ACCEPTED_RESIDUAL_USD == Decimal("8.00")


def test_tiny_close_is_expired_before_execution() -> None:
    """Expire a close that cannot withdraw above the minimum remainder instead of halting.

    1. Mock the equity read because the preflight needs no network to decide.
    2. Verify a 6.60 USDC position is expired and a 19.84 USDC position proceeds.
    3. Verify partial sells are not read at all.
    """
    # 1. Mock the equity read because the preflight needs no network to decide.
    routing = object.__new__(HypercoreVaultRouting)
    routing.lagoon_vault = MagicMock()
    routing._session = MagicMock()
    routing.simulate = False
    pair = SimpleNamespace(pool_address="0xvault")
    close = SimpleNamespace(other_data={}, pair=pair, flags={TradeFlag.close}, is_sell=lambda: True, is_buy=lambda: False)
    # 2. Verify a 6.60 USDC position is expired and a 19.84 USDC position proceeds.
    with patch("tradeexecutor.ethereum.vault.hypercore_routing.fetch_user_vault_equity", return_value=SimpleNamespace(equity=Decimal("6.60"))):
        assert "minimum remainder" in routing.check_trade_before_execution(close)
    with patch("tradeexecutor.ethereum.vault.hypercore_routing.fetch_user_vault_equity", return_value=SimpleNamespace(equity=Decimal("19.83643"))):
        assert routing.check_trade_before_execution(close) is None
    # 3. Verify partial sells are not read at all.
    partial = SimpleNamespace(other_data={}, pair=pair, flags=set(), is_sell=lambda: True, is_buy=lambda: False)
    with patch("tradeexecutor.ethereum.vault.hypercore_routing.fetch_user_vault_equity", side_effect=AssertionError("read")):
        assert routing.check_trade_before_execution(partial) is None


def test_repair_clears_rejected_phase1_marker() -> None:
    """Clear a false stranded marker only when the vault kept the requested funds.

    1. Build the 28 October failed sell and a sell whose vault paid the request.
    2. Mock the equity read, because the decision depends only on the snapshot comparison.
    3. Verify only the rejected sell is cleared and becomes repairable.
    """
    # 1. Build the 28 October failed sell and a sell whose vault paid the request.
    def make_trade(trade_id: int, equity_now: Decimal) -> SimpleNamespace:
        return SimpleNamespace(
            trade_id=trade_id,
            other_data={
                "hypercore_stranded_usdc": {"amount_raw": 18_336_430, "location": "hypercore_perp"},
                "retain_reserve_allocation_on_failure": True,
                "hypercore_phase1_vault_equity_usdc": "19.83643",
                "hypercore_first_stage_requested_raw": 18_336_430,
            },
            pair=SimpleNamespace(pool_address=str(equity_now), is_hyperliquid_vault=lambda: True),
            is_sell=lambda: True,
            get_status=lambda: TradeStatus.failed,
            add_note=MagicMock(),
        )
    rejected = make_trade(1833, Decimal("19.836442"))
    paid = make_trade(1834, Decimal("1.500012"))
    assert vault_kept_requested_funds(Decimal("19.83643"), Decimal("19.836442"), 18_336_430)
    assert not vault_kept_requested_funds(Decimal("19.83643"), Decimal("1.500012"), 18_336_430)
    # 2. Mock the equity read, because the decision depends only on the snapshot comparison.
    def fetch(session, user, vault_address, bypass_cache):
        return SimpleNamespace(equity=Decimal(vault_address))
    with patch("tradeexecutor.ethereum.vault.hypercore_withdrawal_recovery.fetch_user_vault_equity", side_effect=fetch):
        cleared = clear_rejected_first_stage_withdrawals([rejected, paid], safe_address="0xsafe", session=MagicMock())
    # 3. Verify only the rejected sell is cleared and becomes repairable.
    assert cleared == [rejected]
    assert not has_unresolved_hypercore_accounting(rejected)
    assert "retain_reserve_allocation_on_failure" not in rejected.other_data
    assert rejected.other_data["hypercore_rejected_withdrawal_cleared"]["equity_now"] == "19.836442"
    assert has_unresolved_hypercore_accounting(paid)
