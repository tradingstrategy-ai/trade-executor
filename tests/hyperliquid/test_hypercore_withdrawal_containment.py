"""Containment for the October Tortoise withdrawal incident."""

import datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tradeexecutor.cli.commands.start import should_save_failed_execution
from tradeexecutor.strategy.dust import HYPERCORE_ACCEPTED_RESIDUAL_USD, HYPERCORE_FULL_CLOSE_MIN_REMAINDER_USD
from tradeexecutor.strategy.execution_model import ExecutionHaltableIssue
from tradeexecutor.ethereum.vault.hypercore_routing import HypercoreVaultRouting, vault_kept_requested_funds
from tradeexecutor.ethereum.vault.hypercore_withdrawal_recovery import clear_rejected_first_stage_withdrawals, fetch_vault_withdrawals_since
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

    1. Mock the equity and liquidity reads because the preflight needs no network to decide.
    2. Verify an 8.00 USDC position or a 6.60 USDC liquidity cap expires, and 8.01 USDC proceeds.
    3. Verify partial sells are not read at all.
    """
    # 1. Mock the equity and liquidity reads because the preflight needs no network to decide.
    routing = object.__new__(HypercoreVaultRouting)
    routing.lagoon_vault = MagicMock()
    routing._session = MagicMock()
    routing.simulate = False
    pair = SimpleNamespace(pool_address="0xvault")
    close = SimpleNamespace(other_data={}, pair=pair, flags={TradeFlag.close}, is_sell=lambda: True, is_buy=lambda: False)
    # 2. Verify an 8.00 USDC position or a 6.60 USDC liquidity cap expires, and 8.01 USDC proceeds.
    def check(equity: str, max_withdrawable: str) -> str | None:
        with (
            patch("tradeexecutor.ethereum.vault.hypercore_routing.fetch_user_vault_equity", return_value=SimpleNamespace(equity=Decimal(equity))),
            patch("tradeexecutor.ethereum.vault.hypercore_routing.HyperliquidVault.fetch_info", return_value=SimpleNamespace(max_withdrawable=Decimal(max_withdrawable))),
        ):
            return routing.check_trade_before_execution(close)
    assert "minimum remainder" in check("8.00", "1000")
    assert "minimum remainder" in check("13.20", "6.60")
    assert check("8.01", "1000") is None
    # 3. Verify partial sells are not read at all.
    partial = SimpleNamespace(other_data={}, pair=pair, flags=set(), is_sell=lambda: True, is_buy=lambda: False)
    with patch("tradeexecutor.ethereum.vault.hypercore_routing.fetch_user_vault_equity", side_effect=AssertionError("read")):
        assert routing.check_trade_before_execution(partial) is None


def test_repair_clears_rejected_phase1_marker() -> None:
    """Clear a false stranded marker only when the vault kept the requested funds.

    1. Build the 28 October failed sell, a sell whose vault paid the request,
       a phase-2 failure and a late phase-1 payout hidden by a NAV gain.
    2. Mock the equity and ledger reads, because the decision depends only on that evidence.
    3. Verify only the rejected phase-1 sell is cleared and becomes repairable.
    """
    # 1. Build the 28 October failed sell, a paid sell and a phase-2 failure.
    def make_trade(trade_id: int, equity_now: Decimal, phase: str = "phase1_perp_wait") -> SimpleNamespace:
        return SimpleNamespace(
            trade_id=trade_id,
            other_data={
                "hypercore_stranded_usdc": {"amount_raw": 18_336_430, "location": "hypercore_perp"},
                "retain_reserve_allocation_on_failure": True,
                "hypercore_phase1_vault_equity_usdc": "19.83643",
                "hypercore_first_stage_requested_raw": 18_336_430,
                "hypercore_failure_diagnosis": f"HyperCore {phase}: diagnosis",
            },
            pair=SimpleNamespace(pool_address=str(equity_now), is_hyperliquid_vault=lambda: True),
            is_sell=lambda: True,
            started_at=datetime.datetime(2026, 10, 28, 2, 53),
            get_status=lambda: TradeStatus.failed,
            add_note=MagicMock(),
        )
    rejected = make_trade(1833, Decimal("19.836442"))
    paid = make_trade(1834, Decimal("1.500012"))
    phase2 = make_trade(1835, Decimal("19.836442"), phase="phase2_spot_wait")
    late_payout = make_trade(1836, Decimal("19.836443"))
    assert vault_kept_requested_funds(Decimal("19.83643"), Decimal("19.836442"), 18_336_430)
    assert not vault_kept_requested_funds(Decimal("19.83643"), Decimal("1.500012"), 18_336_430)
    # A 100 USDC payout from 10,000 USDC followed by a 60 USDC NAV gain is not "kept".
    assert not vault_kept_requested_funds(Decimal("10000"), Decimal("9960"), 100_000_000)
    # A request inside the drift margin cannot prove anything either way.
    assert not vault_kept_requested_funds(Decimal("8.01"), Decimal("8.01"), 1_510_000)
    # 2. Mock the equity and ledger reads, because the decision depends only on that evidence.
    def fetch(session, user, vault_address, bypass_cache):
        return SimpleNamespace(equity=Decimal(vault_address))
    def ledger(safe_address, vault_address, since):
        return [{"type": "vaultWithdraw", "vault": vault_address}] if vault_address == "19.836443" else []
    with (
        patch("tradeexecutor.ethereum.vault.hypercore_withdrawal_recovery.fetch_user_vault_equity", side_effect=fetch),
        patch("tradeexecutor.ethereum.vault.hypercore_withdrawal_recovery.fetch_vault_withdrawals_since", side_effect=ledger),
    ):
        cleared = clear_rejected_first_stage_withdrawals([rejected, paid, phase2, late_payout], safe_address="0xsafe", session=MagicMock())
    # 3. Verify only the rejected sell is cleared and becomes repairable.
    assert cleared == [rejected]
    assert not has_unresolved_hypercore_accounting(rejected)
    assert "retain_reserve_allocation_on_failure" not in rejected.other_data
    assert rejected.other_data["hypercore_rejected_withdrawal_cleared"]["equity_now"] == "19.836442"
    assert has_unresolved_hypercore_accounting(paid)
    assert has_unresolved_hypercore_accounting(phase2)
    assert has_unresolved_hypercore_accounting(late_payout)


def test_retry_and_ledger_evidence_cover_small_and_paged_cases() -> None:
    """Refuse an unverifiable retry and read every ledger page before clearing a marker.

    1. Size a no-op retry for a normal close and for cleanup after equity falls to 6.705 USDC.
    2. Mock two ledger pages, because the API caps time-range responses at 500 rows.
    3. Verify the paid withdrawal on the second page is found.
    """
    # 1. Size a no-op retry for a normal close and for cleanup after equity falls to 6.705 USDC.
    routing = object.__new__(HypercoreVaultRouting)
    assert routing._get_phase1_noop_retry_raw(Decimal("6.705"), 1_510_000, 6_500_000, minimum_raw=1_500_000) is None
    assert routing._get_phase1_noop_retry_raw(Decimal("6.705"), 1_510_000, 6_500_000) == 205_000
    # 2. Mock two ledger pages, because the API caps time-range responses at 500 rows.
    other = {"time": 1, "delta": {"type": "deposit"}}
    paid = {"time": 2, "delta": {"type": "vaultWithdraw", "vault": "0xVault"}}
    pages = [[other] * 500, [paid]]
    responses = [MagicMock(json=MagicMock(return_value=page)) for page in pages]
    with patch("tradeexecutor.ethereum.vault.hypercore_withdrawal_recovery.requests.post", side_effect=responses) as post:
        withdrawals = fetch_vault_withdrawals_since("0xsafe", "0xvault", datetime.datetime(2026, 10, 28))
    # 3. Verify the paid withdrawal on the second page is found.
    assert withdrawals == [paid["delta"]]
    assert post.call_count == 2
    assert post.call_args_list[1].kwargs["json"]["startTime"] == 2
