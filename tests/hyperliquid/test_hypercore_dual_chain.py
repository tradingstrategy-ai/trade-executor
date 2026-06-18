"""Test P6: Dual-chain confirmation after Hypercore withdrawals.

Tests that _settle_withdrawal() captures baseline vault equity before
the withdrawal and compares it against equity after USDC arrives on EVM.
"""

import datetime
import itertools
import logging
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from eth_defi.hyperliquid.api import UserVaultEquity
from hexbytes import HexBytes


VAULT_ADDR = "0xdfc24b077bc1425ad1dea75bcb6f8158e10df303"
SAFE_ADDR = "0xABC123"
LOCKED_UNTIL = datetime.datetime(2030, 1, 1)


def _make_equity(equity: Decimal) -> UserVaultEquity:
    return UserVaultEquity(
        vault_address=VAULT_ADDR,
        equity=equity,
        locked_until=LOCKED_UNTIL,
    )


def _make_routing(simulate=False):
    """Create a minimal HypercoreVaultRouting with mocked dependencies."""
    from tradeexecutor.ethereum.vault.hypercore_routing import HypercoreVaultRouting

    routing = object.__new__(HypercoreVaultRouting)
    routing.web3 = MagicMock()
    routing.lagoon_vault = MagicMock()
    routing.lagoon_vault.safe_address = SAFE_ADDR
    routing.deployer = MagicMock()
    routing.chain_id = 999
    routing.is_testnet = False
    routing.simulate = simulate
    routing.reserve_token_address = "0xusdc"
    routing._session = MagicMock()
    return routing


def _make_trade(planned_reserve=Decimal("50.0"), is_buy=False):
    """Create a mock trade."""
    trade = MagicMock()
    trade.is_buy.return_value = is_buy
    trade.is_vault.return_value = True
    trade.get_planned_reserve.return_value = planned_reserve
    trade.planned_quantity = -planned_reserve if not is_buy else planned_reserve
    trade.planned_price = 1.0
    trade.slippage_tolerance = 0.20
    trade.closing = False
    trade.flags = set()
    trade.trade_id = 1
    trade.position_id = 1
    trade.blockchain_transactions = [MagicMock(tx_hash="0xabc")]
    trade.other_data = {}
    trade.pair = MagicMock()
    trade.pair.pool_address = VAULT_ADDR
    trade.pair.other_data = {"vault_protocol": "hypercore"}
    return trade


def _monotonic_time():
    """Return a callable that yields monotonically increasing values.

    Patching ``time.time`` with a fixed side_effect list is fragile because
    the patch is global (``time`` is a module singleton).  Python's logging
    module calls ``time.time()`` for every ``LogRecord``, consuming values
    intended for the code under test.  A callable avoids StopIteration.
    """
    counter = itertools.count()
    return lambda: float(next(counter))


@patch("tradeexecutor.ethereum.vault.hypercore_routing.get_block_timestamp")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.fetch_user_vault_equity")
def test_withdrawal_dual_chain_verified(mock_fetch_equity, mock_block_ts):
    """After USDC arrives on EVM, equity decrease on HyperCore is compared to baseline."""
    routing = _make_routing()

    # Equity before withdrawal: 500, after: 450 (decreased by 50, matching withdrawal)
    mock_fetch_equity.side_effect = [
        _make_equity(Decimal("500.0")),   # baseline snapshot
        _make_equity(Decimal("450.0")),   # post-withdrawal check
    ]

    trade = _make_trade(planned_reserve=Decimal("50.0"))
    state = MagicMock()
    mock_block_ts.return_value = datetime.datetime(2025, 1, 1)

    from hexbytes import HexBytes
    receipts = {HexBytes("0xabc"): {"status": 1, "blockNumber": 100}}

    phase2_tx = MagicMock(tx_hash="0xdef")
    phase3_tx = MagicMock(tx_hash="0x123")

    with patch.object(routing, "_fetch_safe_evm_usdc_balance", side_effect=[100_000_000, 150_000_000]):
        with patch.object(routing, "_fetch_safe_perp_withdrawable_balance", return_value=Decimal("0")):
            with patch.object(routing, "_fetch_safe_spot_free_usdc_balance", return_value=Decimal("0")):
                with patch.object(routing, "_wait_for_perp_withdrawable_balance", return_value=Decimal("50")):
                    with patch.object(routing, "_wait_for_spot_free_usdc_balance", return_value=Decimal("50")):
                        with patch.object(routing, "_broadcast_withdrawal_phase2", return_value=(phase2_tx, {"status": 1, "blockNumber": 101})):
                            with patch.object(routing, "_broadcast_withdrawal_phase3", return_value=(phase3_tx, {"status": 1, "blockNumber": 102})):
                                with patch("tradeexecutor.ethereum.vault.hypercore_routing.time.sleep"):
                                    with patch("tradeexecutor.ethereum.vault.hypercore_routing.time.time", side_effect=_monotonic_time()):
                                        routing._settle_withdrawal(
                                            routing.web3, state, trade, receipts,
                                            stop_on_execution_failure=False,
                                        )

    state.mark_trade_success.assert_called_once()
    # Verify both equity calls were made (baseline + post-withdrawal)
    assert mock_fetch_equity.call_count == 2


@patch("tradeexecutor.ethereum.vault.hypercore_routing.get_block_timestamp")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.fetch_user_vault_equity")
def test_withdrawal_dual_chain_mismatch_logs_warning(mock_fetch_equity, mock_block_ts, caplog):
    """Equity does NOT decrease as expected — warning logged but trade still succeeds."""
    routing = _make_routing()

    # Equity stays at 500 (HyperCore didn't process the withdrawal)
    mock_fetch_equity.side_effect = [
        _make_equity(Decimal("500.0")),   # baseline
        _make_equity(Decimal("500.0")),   # post-withdrawal: no decrease!
    ]

    trade = _make_trade(planned_reserve=Decimal("50.0"))
    state = MagicMock()
    mock_block_ts.return_value = datetime.datetime(2025, 1, 1)

    from hexbytes import HexBytes
    receipts = {HexBytes("0xabc"): {"status": 1, "blockNumber": 100}}

    with caplog.at_level(logging.WARNING):
        with patch.object(routing, "_fetch_safe_evm_usdc_balance", side_effect=[100_000_000, 150_000_000]):
            with patch.object(routing, "_fetch_safe_perp_withdrawable_balance", return_value=Decimal("0")):
                with patch.object(routing, "_fetch_safe_spot_free_usdc_balance", return_value=Decimal("0")):
                    with patch.object(routing, "_wait_for_perp_withdrawable_balance", return_value=Decimal("50")):
                        with patch.object(routing, "_wait_for_spot_free_usdc_balance", return_value=Decimal("50")):
                            with patch.object(routing, "_broadcast_withdrawal_phase2", return_value=(MagicMock(tx_hash="0xdef"), {"status": 1, "blockNumber": 101})):
                                with patch.object(routing, "_broadcast_withdrawal_phase3", return_value=(MagicMock(tx_hash="0x123"), {"status": 1, "blockNumber": 102})):
                                    with patch("tradeexecutor.ethereum.vault.hypercore_routing.time.sleep"):
                                        with patch("tradeexecutor.ethereum.vault.hypercore_routing.time.time", side_effect=_monotonic_time()):
                                            routing._settle_withdrawal(
                                                routing.web3, state, trade, receipts,
                                                stop_on_execution_failure=False,
                                            )

    # Trade still succeeds (EVM verification passed)
    state.mark_trade_success.assert_called_once()
    # But a mismatch warning was logged
    assert any("mismatch" in r.message for r in caplog.records)


@patch("tradeexecutor.ethereum.vault.hypercore_routing.get_block_timestamp")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.fetch_user_vault_equity")
def test_withdrawal_equity_check_non_fatal_on_api_failure(mock_fetch_equity, mock_block_ts, caplog):
    """P6 post-withdrawal equity check failure does not crash the trade."""
    routing = _make_routing()

    # Baseline succeeds, post-withdrawal check fails
    mock_fetch_equity.side_effect = [
        _make_equity(Decimal("500.0")),   # baseline OK
        Exception("API timeout"),          # post-withdrawal fails
    ]

    trade = _make_trade(planned_reserve=Decimal("50.0"))
    state = MagicMock()
    mock_block_ts.return_value = datetime.datetime(2025, 1, 1)

    from hexbytes import HexBytes
    receipts = {HexBytes("0xabc"): {"status": 1, "blockNumber": 100}}

    with caplog.at_level(logging.WARNING):
        with patch.object(routing, "_fetch_safe_evm_usdc_balance", side_effect=[100_000_000, 150_000_000]):
            with patch.object(routing, "_fetch_safe_perp_withdrawable_balance", return_value=Decimal("0")):
                with patch.object(routing, "_fetch_safe_spot_free_usdc_balance", return_value=Decimal("0")):
                    with patch.object(routing, "_wait_for_perp_withdrawable_balance", return_value=Decimal("50")):
                        with patch.object(routing, "_wait_for_spot_free_usdc_balance", return_value=Decimal("50")):
                            with patch.object(routing, "_broadcast_withdrawal_phase2", return_value=(MagicMock(tx_hash="0xdef"), {"status": 1, "blockNumber": 101})):
                                with patch.object(routing, "_broadcast_withdrawal_phase3", return_value=(MagicMock(tx_hash="0x123"), {"status": 1, "blockNumber": 102})):
                                    with patch("tradeexecutor.ethereum.vault.hypercore_routing.time.sleep"):
                                        with patch("tradeexecutor.ethereum.vault.hypercore_routing.time.time", side_effect=_monotonic_time()):
                                            routing._settle_withdrawal(
                                                routing.web3, state, trade, receipts,
                                                stop_on_execution_failure=False,
                                            )

    # Trade still succeeds despite P6 check failure
    state.mark_trade_success.assert_called_once()
    assert any("Could not verify vault equity" in r.message for r in caplog.records)


# --- P2: _wait_for_usdc_arrival tests ---


def test_wait_for_usdc_arrival_succeeds():
    """P2: USDC arrives after 2 polls — returns actual increase."""

    routing = _make_routing()
    # Poll 1: no increase, Poll 2: +50 USDC
    routing._fetch_safe_evm_usdc_balance = MagicMock(
        side_effect=[100_000_000, 150_000_000],
    )

    with patch("tradeexecutor.ethereum.vault.hypercore_routing.time.sleep"):
        with patch("tradeexecutor.ethereum.vault.hypercore_routing.time.time", side_effect=_monotonic_time()):
            result = routing._wait_for_usdc_arrival(
                baseline_balance_raw=100_000_000,
                expected_increase_raw=50_000_000,
                timeout=30.0,
                poll_interval=2.0,
            )

    assert result == 50_000_000


def test_wait_for_usdc_arrival_accepts_follow_up_phase_tolerance():
    """Accept EVM arrival when the bridged amount is within the temporary later-phase tolerance.

    1. Create a routing object and mock the Safe EVM balance just inside the temporary tolerance.
    2. Wait for the EVM balance increase using the withdrawal verifier.
    3. Verify the slightly short increase is still accepted.
    """
    routing = _make_routing()

    # 1. Create a routing object and mock the Safe EVM balance just inside the temporary tolerance.
    routing._fetch_safe_evm_usdc_balance = MagicMock(
        side_effect=[149_981_000],
    )

    # 2. Wait for the EVM balance increase using the withdrawal verifier.
    with patch("tradeexecutor.ethereum.vault.hypercore_routing.time.sleep"):
        with patch("tradeexecutor.ethereum.vault.hypercore_routing.time.time", side_effect=_monotonic_time()):
            result = routing._wait_for_usdc_arrival(
                baseline_balance_raw=100_000_000,
                expected_increase_raw=50_000_000,
                timeout=30.0,
                poll_interval=2.0,
            )

    # 3. Verify the slightly short increase is still accepted.
    assert result == 49_981_000


def test_wait_for_usdc_arrival_timeout():
    """P2: USDC never arrives — raises HypercoreWithdrawalVerificationError."""
    import pytest
    from tradeexecutor.ethereum.vault.hypercore_routing import (
        HypercoreVaultRouting,
        HypercoreWithdrawalVerificationError,
    )

    routing = _make_routing()
    # Balance never increases
    routing._fetch_safe_evm_usdc_balance = MagicMock(return_value=100_000_000)

    # time.time returns: 0 (deadline=30), then 31 on the remaining check → timeout
    call_count = [0]
    def fake_time():
        call_count[0] += 1
        # First call sets deadline, subsequent calls simulate time passing
        if call_count[0] <= 2:
            return 0.0
        return 999.0  # past deadline

    with patch("tradeexecutor.ethereum.vault.hypercore_routing.time.sleep"):
        with patch("tradeexecutor.ethereum.vault.hypercore_routing.time.time", side_effect=fake_time):
            with pytest.raises(HypercoreWithdrawalVerificationError) as exc_info:
                routing._wait_for_usdc_arrival(
                    baseline_balance_raw=100_000_000,
                    expected_increase_raw=50_000_000,
                    timeout=30.0,
                    poll_interval=2.0,
                )

    assert "did not arrive" in str(exc_info.value)
    assert "50000000" in str(exc_info.value)


def test_wait_for_spot_free_usdc_balance_accepts_follow_up_phase_tolerance():
    """Accept spot balance when the moved amount is within the temporary later-phase tolerance.

    1. Create a routing object and mock the spot free balance just inside the temporary tolerance.
    2. Wait for the spot free USDC balance using the withdrawal verifier.
    3. Verify the slightly short balance is still accepted.
    """
    routing = _make_routing()

    # 1. Create a routing object and mock the spot free balance just inside the temporary tolerance.
    routing._fetch_safe_spot_free_usdc_balance = MagicMock(
        side_effect=[Decimal("1.980001")],
    )

    # 2. Wait for the spot free USDC balance using the withdrawal verifier.
    with patch("tradeexecutor.ethereum.vault.hypercore_routing.time.sleep"):
        with patch("tradeexecutor.ethereum.vault.hypercore_routing.time.time", side_effect=_monotonic_time()):
            result = routing._wait_for_spot_free_usdc_balance(
                baseline_balance=Decimal("1.0"),
                expected_increase_raw=1_000_000,
                timeout=30.0,
                poll_interval=2.0,
            )

    # 3. Verify the slightly short balance is still accepted.
    assert result == Decimal("1.980001")


def test_wait_for_perp_withdrawable_balance_accepts_relative_tolerance():
    """Accept perp withdrawable balance when the shortfall stays within relative tolerance.

    1. Create a routing object and mock a large perp withdrawable balance increase with a small relative shortfall.
    2. Wait for the perp withdrawable balance using the withdrawal verifier.
    3. Verify the slightly short increase is still accepted.
    """
    routing = _make_routing()

    # 1. Create a routing object and mock a large perp withdrawable balance increase with a small relative shortfall.
    routing._fetch_safe_perp_withdrawable_balance = MagicMock(
        side_effect=[Decimal("629.998483")],
    )

    # 2. Wait for the perp withdrawable balance using the withdrawal verifier.
    with patch("tradeexecutor.ethereum.vault.hypercore_routing.time.sleep"):
        with patch("tradeexecutor.ethereum.vault.hypercore_routing.time.time", side_effect=_monotonic_time()):
            result = routing._wait_for_perp_withdrawable_balance(
                baseline_balance=Decimal("59.559287"),
                expected_increase_raw=570_690_753,
                timeout=30.0,
                poll_interval=2.0,
            )

    # 3. Verify the slightly short increase is still accepted.
    assert result == Decimal("629.998483")


def test_wait_for_perp_withdrawable_balance_rejects_large_shortfall():
    """Reject perp withdrawable balance when the shortfall exceeds relative tolerance.

    1. Create a routing object and mock a large perp withdrawable balance increase with a material shortfall.
    2. Wait for the perp withdrawable balance using the withdrawal verifier.
    3. Verify the helper times out and raises a withdrawal verification error.
    """
    import pytest
    from tradeexecutor.ethereum.vault.hypercore_routing import (
        HypercoreWithdrawalVerificationError,
    )

    routing = _make_routing()

    # 1. Create a routing object and mock a large perp withdrawable balance increase with a material shortfall.
    routing._fetch_safe_perp_withdrawable_balance = MagicMock(
        return_value=Decimal("620.0"),
    )

    call_count = [0]

    def fake_time():
        call_count[0] += 1
        if call_count[0] <= 2:
            return 0.0
        return 999.0

    # 2. Wait for the perp withdrawable balance using the withdrawal verifier.
    with patch("tradeexecutor.ethereum.vault.hypercore_routing.time.sleep"):
        with patch("tradeexecutor.ethereum.vault.hypercore_routing.time.time", side_effect=fake_time):
            # 3. Verify the helper times out and raises a withdrawal verification error.
            with pytest.raises(HypercoreWithdrawalVerificationError) as exc_info:
                routing._wait_for_perp_withdrawable_balance(
                    baseline_balance=Decimal("59.559287"),
                    expected_increase_raw=570_690_753,
                    timeout=30.0,
                    poll_interval=2.0,
                )

    assert "did not reach" in str(exc_info.value)


def test_wait_for_perp_withdrawable_balance_accepts_trade_slippage_and_performance_fee():
    """Accept phase-1 perp shortfalls covered by either the slippage or the performance-fee tolerance.

    The HyperCore leader performance fee (deducted on/before withdrawal) can
    leave the net perp arrival well below the gross request — more than the 1%
    slippage tolerance absorbs. The phase-1 wait must accept the net amount when
    the worst-case performance fee covers the shortfall (2026-06-13 IKAGI #1022).

    1. Accept a fee-shaped shortfall covered by a wide trade slippage tolerance.
    2. Accept the IKAGI shortfall under a realistic 1% slippage but a 10% performance-fee tolerance.
    3. Confirm the same shortfall is rejected when neither tolerance covers it.
    """
    from eth_defi.hyperliquid.vault import estimate_max_withdrawal_commission
    from tradeexecutor.ethereum.vault.hypercore_routing import (
        HypercoreWithdrawalVerificationError,
    )

    # 1. Accept a fee-shaped shortfall covered by a wide trade slippage tolerance.
    routing = _make_routing()
    routing._fetch_safe_perp_withdrawable_balance = MagicMock(return_value=Decimal("123.0"))
    with patch("tradeexecutor.ethereum.vault.hypercore_routing.time.sleep"):
        with patch("tradeexecutor.ethereum.vault.hypercore_routing.time.time", side_effect=_monotonic_time()):
            result = routing._wait_for_perp_withdrawable_balance(
                baseline_balance=Decimal("0"),
                expected_increase_raw=130_000_000,
                relative_tolerance=Decimal("0.20"),
                timeout=30.0,
                poll_interval=2.0,
            )
    assert result == Decimal("123.0")

    # 2. Accept the IKAGI shortfall under a realistic 1% slippage but a 10% performance-fee tolerance.
    routing = _make_routing()
    routing._fetch_safe_perp_withdrawable_balance = MagicMock(return_value=Decimal("42.64074"))
    performance_fee_tolerance = estimate_max_withdrawal_commission(Decimal("45.232559"), Decimal("0.10"))
    with patch("tradeexecutor.ethereum.vault.hypercore_routing.time.sleep"):
        with patch("tradeexecutor.ethereum.vault.hypercore_routing.time.time", side_effect=_monotonic_time()):
            result = routing._wait_for_perp_withdrawable_balance(
                baseline_balance=Decimal("0.5"),
                expected_increase_raw=45_232_559,
                relative_tolerance=Decimal("0.01"),
                performance_fee_tolerance=performance_fee_tolerance,
                timeout=30.0,
                poll_interval=2.0,
            )
    assert result == Decimal("42.64074")

    # 3. Confirm the same shortfall is rejected when neither tolerance covers it.
    routing = _make_routing()
    routing._fetch_safe_perp_withdrawable_balance = MagicMock(return_value=Decimal("42.64074"))
    with patch("tradeexecutor.ethereum.vault.hypercore_routing.time.sleep"):
        with patch("tradeexecutor.ethereum.vault.hypercore_routing.time.time", side_effect=_monotonic_time()):
            with pytest.raises(HypercoreWithdrawalVerificationError):
                routing._wait_for_perp_withdrawable_balance(
                    baseline_balance=Decimal("0.5"),
                    expected_increase_raw=45_232_559,
                    relative_tolerance=Decimal("0.01"),
                    timeout=30.0,
                    poll_interval=2.0,
                )


def test_resolve_vault_performance_fee_prefers_pair_metadata_then_live_then_default():
    """Resolve the per-vault performance fee from pair metadata, then live API, then default.

    The fee differs per vault (leader vaults ~10%, protocol/HLP vaults 0%), so it must
    come from per-vault data rather than a fixed platform constant.

    1. A performance fee on the trading pair (incl. an explicit 0%) is used as-is, with no live read.
    2. With no pair fee, fall back to the live vaultDetails commission (incl. an explicit 0%).
    3. With neither a pair fee nor a reported live commission, fall back to the 10% default.
    """
    from tradeexecutor.ethereum.vault.hypercore_routing import (
        HYPERCORE_DEFAULT_PERFORMANCE_FEE,
    )

    unset = object()
    routing = _make_routing()
    # Routing needs a session to query vault details; a mock is enough here.
    routing._get_session = MagicMock(return_value=MagicMock())

    def resolve(pair_fee, commission_rate):
        trade = _make_trade()
        trade.pair.other_data = {"vault_protocol": "hypercore"}
        if pair_fee is not unset:
            trade.pair.other_data["vault_performance_fee"] = pair_fee
        with patch("tradeexecutor.ethereum.vault.hypercore_routing.HyperliquidVault") as mock_vault_cls:
            mock_vault_cls.return_value.fetch_metadata.return_value = MagicMock(commission_rate=commission_rate)
            rate = routing._resolve_vault_performance_fee(trade, VAULT_ADDR)
            return rate, mock_vault_cls

    # 1. A pair-metadata fee is used as-is and skips the live read (incl. explicit zero).
    rate, vault_cls = resolve(pair_fee=0.20, commission_rate=Decimal("0.10"))
    assert rate == Decimal("0.20")
    vault_cls.assert_not_called()
    rate, _ = resolve(pair_fee=0.0, commission_rate=Decimal("0.10"))
    assert rate == Decimal("0")

    # 2. With no pair fee, fall back to the live vaultDetails commission (incl. explicit zero).
    assert resolve(pair_fee=unset, commission_rate=Decimal("0.15"))[0] == Decimal("0.15")
    assert resolve(pair_fee=unset, commission_rate=Decimal("0"))[0] == Decimal("0")

    # 3. With neither a pair fee nor a reported live commission, fall back to the 10% default.
    assert resolve(pair_fee=unset, commission_rate=None)[0] == HYPERCORE_DEFAULT_PERFORMANCE_FEE


@patch("tradeexecutor.ethereum.vault.hypercore_routing.report_failure")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.HyperliquidVault")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.get_block_timestamp")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.fetch_user_vault_equity")
def test_settle_withdrawal_survives_performance_fee_shortfall_end_to_end(
    mock_fetch_equity,
    mock_block_ts,
    mock_vault_cls,
    mock_report_failure,
):
    """End-to-end regression for the 2026-06-13 IKAGI #1022 phase-1 crash.

    This drives the real ``_settle_withdrawal`` with the real
    ``_wait_for_perp_withdrawable_balance`` and the real
    ``_resolve_vault_performance_fee`` wiring (only the network reads are mocked),
    reproducing the production numbers: a 45.23 USDC gross partial sell whose
    net perp arrival is only 42.64 USDC because HyperCore deducted the ~10%
    leader performance fee, under a realistic 1% trade slippage tolerance.

    Unlike the helper-level tests, nothing passes ``performance_fee_tolerance``
    by hand here, so this fails if the settlement wiring that computes and
    forwards the fee tolerance is removed.

    1. Build the IKAGI partial-sell trade with 1% slippage, the per-vault 10% fee on the pair, and stored phase-1 baselines.
    2. Mock HyperCore reads: net perp arrival is fee-reduced.
    3. Settle and verify the trade is marked success and report_failure is never called.
    """
    from hexbytes import HexBytes
    from tradeexecutor.ethereum.vault.hypercore_routing import usdc_to_raw

    routing = _make_routing()

    # 1. Build the IKAGI partial-sell trade with 1% slippage, the per-vault 10% fee on the
    #    pair (the authoritative production source), and stored phase-1 baselines.
    trade = _make_trade(planned_reserve=Decimal("45.232559"))
    trade.trade_id = 1022
    trade.slippage_tolerance = 0.01  # Production value; far below the ~10% fee.
    trade.pair.other_data = {"vault_protocol": "hypercore", "vault_performance_fee": 0.10}
    trade.other_data = {
        "hypercore_phase1_perp_baseline_usdc": "0.5",
        "hypercore_phase1_vault_equity_usdc": "837.84627",
    }
    state = MagicMock()
    state.portfolio.get_position_by_id.return_value.get_quantity.return_value = Decimal("837.84627")
    mock_block_ts.return_value = datetime.datetime(2026, 6, 13, 11, 25, 0)
    # Post-phase-1 equity after the fee-reduced redemption.
    mock_fetch_equity.return_value = _make_equity(Decimal("794.984554"))
    receipts = {HexBytes("0xabc"): {"status": 1, "blockNumber": 100}}

    # 2. Mock HyperCore reads. The fee comes from the pair metadata above; the live
    #    vaultDetails read is only a fallback (mocked to 10% for safety, not exercised here).
    mock_vault_cls.return_value.fetch_metadata.return_value = MagicMock(commission_rate=Decimal("0.10"))

    with (
        patch.object(routing, "_fetch_safe_evm_usdc_balance", return_value=13_117_483),
        # Net perp arrival is 42.64 USDC — 2.64 below the gross request, a ~5.8%
        # shortfall that the old 1% tolerance rejected. The REAL perp wait runs.
        patch.object(routing, "_fetch_safe_perp_withdrawable_balance", return_value=Decimal("42.64074")),
        patch.object(routing, "_fetch_safe_spot_free_usdc_balance", return_value=Decimal("0")),
        patch.object(routing, "_broadcast_withdrawal_phase2", return_value=(MagicMock(tx_hash="0xdef"), {"status": 1, "blockNumber": 101})),
        patch.object(routing, "_wait_for_spot_free_usdc_balance", return_value=Decimal("42.14074")),
        patch.object(routing, "_broadcast_withdrawal_phase3", return_value=(MagicMock(tx_hash="0x123"), {"status": 1, "blockNumber": 102})),
        patch.object(routing, "_wait_for_usdc_arrival", return_value=usdc_to_raw(Decimal("42.14"))),
        patch("tradeexecutor.ethereum.vault.hypercore_routing.time.sleep"),
        patch("tradeexecutor.ethereum.vault.hypercore_routing.time.time", side_effect=_monotonic_time()),
    ):
        routing._settle_withdrawal(
            routing.web3,
            state,
            trade,
            receipts,
            stop_on_execution_failure=False,
        )

    # 3. Settle and verify the trade is marked success and report_failure is never called.
    mock_report_failure.assert_not_called()
    state.mark_trade_success.assert_called_once()


def test_withdrawal_already_reflected_in_vault_equity_is_detected():
    """Detect a phase 1 withdrawal that already reduced vault equity, including fee-reduced decreases.

    1. Recognise a clean equity decrease that matches the expected residual.
    2. Reject the IKAGI fee-reduced decrease (42.86 vs 45.23 gross) under the bare 1% tolerance.
    3. Accept the same fee-reduced decrease once the 10% performance-fee tolerance is applied.
    4. Honour the trade slippage tolerance so the fallback never rejects a shortfall the perp wait accepts.
    """
    from eth_defi.hyperliquid.vault import estimate_max_withdrawal_commission

    routing = _make_routing()

    # 1. Recognise a clean equity decrease that matches the expected residual.
    assert routing._is_withdrawal_already_reflected_in_vault_equity(
        position_quantity_before=Decimal("630.007301"),
        current_vault_equity=Decimal("77.748241"),
        expected_increase_raw=552_259_060,
    ) is True

    # 2. Reject the IKAGI fee-reduced decrease (42.86 vs 45.23 gross) under the bare 1% tolerance.
    #    The leader performance fee shrank the equity decrease below the gross request.
    assert routing._is_withdrawal_already_reflected_in_vault_equity(
        position_quantity_before=Decimal("837.84627"),
        current_vault_equity=Decimal("794.984554"),
        expected_increase_raw=45_232_559,
    ) is False

    # 3. Accept the same fee-reduced decrease once the 10% performance-fee tolerance is applied.
    performance_fee_tolerance = estimate_max_withdrawal_commission(Decimal("45.232559"), Decimal("0.10"))
    assert routing._is_withdrawal_already_reflected_in_vault_equity(
        position_quantity_before=Decimal("837.84627"),
        current_vault_equity=Decimal("794.984554"),
        expected_increase_raw=45_232_559,
        performance_fee_tolerance=performance_fee_tolerance,
    ) is True

    # 4. A 4% equity-decrease shortfall (no performance fee) is rejected at 1% but accepted at a
    #    matching 5% slippage tolerance — the fallback mirrors the primary perp wait.
    assert routing._is_withdrawal_already_reflected_in_vault_equity(
        position_quantity_before=Decimal("100.0"),
        current_vault_equity=Decimal("4.0"),  # decrease 96 vs 100 gross = 4% short
        expected_increase_raw=100_000_000,
    ) is False
    assert routing._is_withdrawal_already_reflected_in_vault_equity(
        position_quantity_before=Decimal("100.0"),
        current_vault_equity=Decimal("4.0"),
        expected_increase_raw=100_000_000,
        relative_tolerance=Decimal("0.05"),
    ) is True


@patch("tradeexecutor.ethereum.vault.hypercore_routing.get_block_timestamp")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.fetch_user_vault_equity")
def test_withdrawal_phase1_timeout_uses_vault_equity_fallback(
    mock_fetch_equity,
    mock_block_ts,
):
    """Continue withdrawal settlement when phase 1 already shows as residual vault equity.

    1. Simulate a timeout while waiting for perp withdrawable balance after phase 1.
    2. Return a vault equity snapshot that already matches the expected post-withdrawal residual.
    3. Verify settlement continues to phase 2 and marks the trade successful instead of freezing it.
    """
    from hexbytes import HexBytes

    from tradeexecutor.ethereum.vault.hypercore_routing import (
        HypercoreWithdrawalVerificationError,
    )

    routing = _make_routing()
    trade = _make_trade(planned_reserve=Decimal("552.259060"))
    state = MagicMock()
    state.portfolio.get_position_by_id.return_value.get_quantity.return_value = Decimal("630.007301")
    mock_block_ts.return_value = datetime.datetime(2025, 1, 1)
    mock_fetch_equity.side_effect = [
        _make_equity(Decimal("77.748241")),
        _make_equity(Decimal("77.748241")),
        _make_equity(Decimal("77.748241")),
    ]
    receipts = {HexBytes("0xabc"): {"status": 1, "blockNumber": 100}}

    phase2_tx = MagicMock(tx_hash="0xdef")
    phase3_tx = MagicMock(tx_hash="0x123")

    with patch.object(routing, "_fetch_safe_evm_usdc_balance", side_effect=[820_762_276, 1_373_021_336]):
        with patch.object(routing, "_fetch_safe_perp_withdrawable_balance", return_value=Decimal("761.147434")):
            with patch.object(routing, "_fetch_safe_spot_free_usdc_balance", return_value=Decimal("0.009157")):
                with patch.object(
                    routing,
                    "_wait_for_perp_withdrawable_balance",
                    side_effect=HypercoreWithdrawalVerificationError("phase 1 timed out"),
                ):
                    with patch.object(routing, "_wait_for_spot_free_usdc_balance", return_value=Decimal("552.268217")):
                        with patch.object(routing, "_broadcast_withdrawal_phase2", return_value=(phase2_tx, {"status": 1, "blockNumber": 101})):
                            with patch.object(routing, "_broadcast_withdrawal_phase3", return_value=(phase3_tx, {"status": 1, "blockNumber": 102})):
                                with patch.object(routing, "_wait_for_usdc_arrival", return_value=552_259_060):
                                    with patch("tradeexecutor.ethereum.vault.hypercore_routing.time.sleep"):
                                        with patch("tradeexecutor.ethereum.vault.hypercore_routing.time.time", side_effect=_monotonic_time()):
                                            routing._settle_withdrawal(
                                                routing.web3,
                                                state,
                                                trade,
                                                receipts,
                                                stop_on_execution_failure=False,
                                            )

    state.mark_trade_success.assert_called_once()


@patch("tradeexecutor.ethereum.vault.hypercore_routing.report_failure")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.get_block_timestamp")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.fetch_user_vault_equity")
def test_withdrawal_uses_pre_phase1_perp_baseline_for_fast_settlement(
    mock_fetch_equity,
    mock_block_ts,
    mock_report_failure,
):
    """Use the pre-phase-1 perp baseline when vaultTransfer settles before receipt handling.

    1. Simulate the 2026-04-16 HyperAI partial withdrawal where phase 1 has already increased perp by settlement time.
    2. Store the setup-time perp and vault-equity baselines in ``trade.other_data``.
    3. Verify settlement continues to phase 2 instead of waiting for a second perp increase.
    """
    from hexbytes import HexBytes

    from tradeexecutor.ethereum.vault.hypercore_routing import raw_to_usdc

    routing = _make_routing()
    trade = _make_trade(planned_reserve=Decimal("9.223899"))
    trade.other_data = {
        "hypercore_phase1_perp_baseline_usdc": "759.651993",
        "hypercore_phase1_vault_equity_usdc": "497.830378",
    }
    state = MagicMock()
    state.portfolio.get_position_by_id.return_value.get_quantity.return_value = Decimal("497.562464")
    mock_block_ts.return_value = datetime.datetime(2026, 4, 16, 14, 35, 33)
    mock_fetch_equity.return_value = _make_equity(Decimal("488.606479"))
    receipts = {HexBytes("0xabc"): {"status": 1, "blockNumber": 32630331}}

    phase2_tx = MagicMock(tx_hash="0xdef")
    phase3_tx = MagicMock(tx_hash="0x123")
    captured_phase2_raw = []

    def capture_phase2(raw_amount: int):
        captured_phase2_raw.append(raw_amount)
        return phase2_tx, {"status": 1, "blockNumber": 32630332}

    # 1. Simulate the HyperAI fast-settlement shape: current perp already includes phase 1.
    # 2. Store setup-time baselines in trade.other_data.
    # 3. Verify settlement continues to phase 2 without a false failure.
    with (
        patch.object(routing, "_fetch_safe_evm_usdc_balance", return_value=92_058_215),
        patch.object(routing, "_fetch_safe_perp_withdrawable_balance", return_value=Decimal("768.875892")),
        patch.object(routing, "_fetch_safe_spot_free_usdc_balance", return_value=Decimal("33.611598")),
        patch.object(routing, "_broadcast_withdrawal_phase2", side_effect=capture_phase2),
        patch.object(routing, "_wait_for_spot_free_usdc_balance", return_value=Decimal("42.835497")),
        patch.object(routing, "_broadcast_withdrawal_phase3", return_value=(phase3_tx, {"status": 1, "blockNumber": 32630333})),
        patch.object(routing, "_wait_for_usdc_arrival", return_value=9_223_899),
    ):
        routing._settle_withdrawal(
            routing.web3,
            state,
            trade,
            receipts,
            stop_on_execution_failure=False,
        )

    assert captured_phase2_raw == [9_223_899]
    assert state.mark_trade_success.call_args.kwargs["executed_reserve"] == raw_to_usdc(9_223_899)
    state.mark_trade_success.assert_called_once()
    mock_report_failure.assert_not_called()


@patch("tradeexecutor.ethereum.vault.hypercore_routing.get_block_timestamp")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.fetch_user_vault_equity")
def test_withdrawal_phase3_uses_fee_adjusted_amount(
    mock_fetch_equity,
    mock_block_ts,
):
    """Phase 3 should bridge the fee-adjusted spot amount and verify against that smaller amount.

    1. Simulate a successful phased withdrawal where spot free USDC is slightly below the requested amount.
    2. Verify phase 3 bridges the fee-adjusted amount instead of the pre-fee desired amount.
    3. Verify EVM-arrival confirmation and trade settlement use the same adjusted amount.
    """
    from hexbytes import HexBytes

    from tradeexecutor.ethereum.vault.hypercore_routing import raw_to_usdc, usdc_to_raw
    from eth_defi.hyperliquid.core_writer import compute_spot_to_evm_withdrawal_amount

    routing = _make_routing()
    trade = _make_trade(planned_reserve=Decimal("50.0"))
    state = MagicMock()
    mock_block_ts.return_value = datetime.datetime(2025, 1, 1)
    mock_fetch_equity.side_effect = [
        _make_equity(Decimal("500.0")),
        _make_equity(Decimal("450.029")),
    ]
    receipts = {HexBytes("0xabc"): {"status": 1, "blockNumber": 100}}

    spot_balance = Decimal("49.981")
    expected_phase3_raw = usdc_to_raw(
        compute_spot_to_evm_withdrawal_amount(
            spot_balance=spot_balance,
            desired_amount=Decimal("50.0"),
        )
    )

    with patch.object(routing, "_fetch_safe_evm_usdc_balance", return_value=100_000_000):
        with patch.object(routing, "_fetch_safe_perp_withdrawable_balance", return_value=Decimal("0")):
            with patch.object(routing, "_fetch_safe_spot_free_usdc_balance", return_value=Decimal("0")):
                with patch.object(routing, "_wait_for_perp_withdrawable_balance", return_value=Decimal("50")):
                    with patch.object(routing, "_wait_for_spot_free_usdc_balance", return_value=spot_balance):
                        with patch.object(routing, "_broadcast_withdrawal_phase2", return_value=(MagicMock(tx_hash="0xdef"), {"status": 1, "blockNumber": 101})):
                            with patch.object(routing, "_broadcast_withdrawal_phase3", return_value=(MagicMock(tx_hash="0x123"), {"status": 1, "blockNumber": 102})) as phase3:
                                with patch.object(routing, "_wait_for_usdc_arrival", return_value=expected_phase3_raw) as wait_evm:
                                    routing._settle_withdrawal(
                                        routing.web3,
                                        state,
                                        trade,
                                        receipts,
                                        stop_on_execution_failure=False,
                                    )

    phase3.assert_called_once_with(expected_phase3_raw)
    assert wait_evm.call_args.kwargs["expected_increase_raw"] == expected_phase3_raw
    assert state.mark_trade_success.call_args.kwargs["executed_reserve"] == raw_to_usdc(expected_phase3_raw)


@patch("tradeexecutor.ethereum.vault.hypercore_routing.get_block_timestamp")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.fetch_user_vault_equity")
def test_withdrawal_uses_gross_vault_decrease_for_quantity_and_net_usdc_for_reserve(
    mock_fetch_equity,
    mock_block_ts,
):
    """Account HyperCore withdrawal fees as execution price slippage, not fewer vault units sold.

    1. Simulate a withdrawal where HyperCore removes 130 USDC of vault equity but only 123 USDC reaches perp/EVM.
    2. Settle the phased withdrawal using the net amount for phases 2 and 3.
    3. Verify executed quantity follows the gross vault decrease while executed reserve follows net USDC received.
    """
    routing = _make_routing()
    trade = _make_trade(planned_reserve=Decimal("130.0"))
    trade.planned_price = 2.0
    trade.planned_quantity = Decimal("-65")
    state = MagicMock()
    state.portfolio.get_position_by_id.return_value.get_quantity.return_value = Decimal("200")
    state.portfolio.find_position_for_trade.return_value.get_quantity.return_value = Decimal("200")
    mock_block_ts.return_value = datetime.datetime(2025, 1, 1)
    mock_fetch_equity.side_effect = [
        _make_equity(Decimal("1000.0")),
        _make_equity(Decimal("870.0")),
    ]
    receipts = {HexBytes("0xabc"): {"status": 1, "blockNumber": 100}}

    phase2_tx = MagicMock(tx_hash="0xdef")
    phase3_tx = MagicMock(tx_hash="0x123")
    captured_phase2_raw = []
    captured_phase3_raw = []

    def capture_phase2(raw_amount: int):
        captured_phase2_raw.append(raw_amount)
        return phase2_tx, {"status": 1, "blockNumber": 101}

    def capture_phase3(raw_amount: int):
        captured_phase3_raw.append(raw_amount)
        return phase3_tx, {"status": 1, "blockNumber": 102}

    # 1. Simulate a withdrawal where HyperCore removes 130 USDC of vault equity but only 123 USDC reaches perp/EVM.
    # 2. Settle the phased withdrawal using the net amount for phases 2 and 3.
    # 3. Verify executed quantity follows the gross vault decrease while executed reserve follows net USDC received.
    with (
        patch.object(routing, "_fetch_safe_evm_usdc_balance", return_value=100_000_000),
        patch.object(routing, "_fetch_safe_perp_withdrawable_balance", return_value=Decimal("0")),
        patch.object(routing, "_fetch_safe_spot_free_usdc_balance", return_value=Decimal("0")),
        patch.object(routing, "_wait_for_perp_withdrawable_balance", return_value=Decimal("123.0")),
        patch.object(routing, "_broadcast_withdrawal_phase2", side_effect=capture_phase2),
        patch.object(routing, "_wait_for_spot_free_usdc_balance", return_value=Decimal("123.0")),
        patch.object(routing, "_broadcast_withdrawal_phase3", side_effect=capture_phase3),
        patch.object(routing, "_wait_for_usdc_arrival", return_value=122_990_000),
    ):
        routing._settle_withdrawal(
            routing.web3,
            state,
            trade,
            receipts,
            stop_on_execution_failure=False,
        )

    assert captured_phase2_raw == [123_000_000]
    assert captured_phase3_raw == [122_990_000]
    success_kwargs = state.mark_trade_success.call_args.kwargs
    assert success_kwargs["executed_amount"] == Decimal("-65")
    assert success_kwargs["executed_reserve"] == Decimal("122.99")
    assert success_kwargs["executed_price"] == pytest.approx(122.99 / 65)


@patch("tradeexecutor.ethereum.vault.hypercore_routing.get_block_timestamp")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.fetch_user_vault_equity")
def test_withdrawal_uses_phase1_request_for_quantity_when_post_equity_check_fails(
    mock_fetch_equity,
    mock_block_ts,
):
    """Use phase-1 gross request for quantity when post-withdrawal equity cannot be read.

    1. Simulate a withdrawal where 130 USDC is requested from the vault and only 123 USDC reaches perp/EVM.
    2. Make the post-withdrawal vault equity read fail after the baseline snapshot succeeds.
    3. Verify executed quantity still follows the phase-1 gross request while executed reserve follows net USDC received.
    """
    routing = _make_routing()
    trade = _make_trade(planned_reserve=Decimal("130.0"))
    trade.planned_price = 2.0
    trade.planned_quantity = Decimal("-65")
    state = MagicMock()
    state.portfolio.get_position_by_id.return_value.get_quantity.return_value = Decimal("200")
    state.portfolio.find_position_for_trade.return_value.get_quantity.return_value = Decimal("200")
    mock_block_ts.return_value = datetime.datetime(2025, 1, 1)
    mock_fetch_equity.side_effect = [
        _make_equity(Decimal("1000.0")),
        RuntimeError("HyperCore API down"),
    ]
    receipts = {HexBytes("0xabc"): {"status": 1, "blockNumber": 100}}

    phase2_tx = MagicMock(tx_hash="0xdef")
    phase3_tx = MagicMock(tx_hash="0x123")
    captured_phase2_raw = []
    captured_phase3_raw = []

    def capture_phase2(raw_amount: int):
        captured_phase2_raw.append(raw_amount)
        return phase2_tx, {"status": 1, "blockNumber": 101}

    def capture_phase3(raw_amount: int):
        captured_phase3_raw.append(raw_amount)
        return phase3_tx, {"status": 1, "blockNumber": 102}

    # 1. Simulate a withdrawal where 130 USDC is requested from the vault and only 123 USDC reaches perp/EVM.
    # 2. Make the post-withdrawal vault equity read fail after the baseline snapshot succeeds.
    # 3. Verify executed quantity still follows the phase-1 gross request while executed reserve follows net USDC received.
    with (
        patch.object(routing, "_fetch_safe_evm_usdc_balance", return_value=100_000_000),
        patch.object(routing, "_fetch_safe_perp_withdrawable_balance", return_value=Decimal("0")),
        patch.object(routing, "_fetch_safe_spot_free_usdc_balance", return_value=Decimal("0")),
        patch.object(routing, "_wait_for_perp_withdrawable_balance", return_value=Decimal("123.0")),
        patch.object(routing, "_broadcast_withdrawal_phase2", side_effect=capture_phase2),
        patch.object(routing, "_wait_for_spot_free_usdc_balance", return_value=Decimal("123.0")),
        patch.object(routing, "_broadcast_withdrawal_phase3", side_effect=capture_phase3),
        patch.object(routing, "_wait_for_usdc_arrival", return_value=122_990_000),
    ):
        routing._settle_withdrawal(
            routing.web3,
            state,
            trade,
            receipts,
            stop_on_execution_failure=False,
        )

    assert captured_phase2_raw == [123_000_000]
    assert captured_phase3_raw == [122_990_000]
    success_kwargs = state.mark_trade_success.call_args.kwargs
    assert success_kwargs["executed_amount"] == Decimal("-65")
    assert success_kwargs["executed_reserve"] == Decimal("122.99")
    assert success_kwargs["executed_price"] == pytest.approx(122.99 / 65)


@patch("tradeexecutor.ethereum.vault.hypercore_routing.get_block_timestamp")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.fetch_user_vault_equity")
def test_withdrawal_caps_gross_quantity_to_planned_sell_value(
    mock_fetch_equity,
    mock_block_ts,
):
    """Cap equity-derived gross redemption to the planned sell value.

    1. Simulate a withdrawal where vault equity drops by more than the intended 130 USDC sell value.
    2. Settle the phased withdrawal using the net amount for phases 2 and 3.
    3. Verify executed quantity does not exceed the planned partial sell quantity.
    """
    routing = _make_routing()
    trade = _make_trade(planned_reserve=Decimal("130.0"))
    trade.planned_price = 2.0
    trade.planned_quantity = Decimal("-65")
    state = MagicMock()
    state.portfolio.get_position_by_id.return_value.get_quantity.return_value = Decimal("200")
    state.portfolio.find_position_for_trade.return_value.get_quantity.return_value = Decimal("200")
    mock_block_ts.return_value = datetime.datetime(2025, 1, 1)
    mock_fetch_equity.side_effect = [
        _make_equity(Decimal("1000.0")),
        _make_equity(Decimal("850.0")),
    ]
    receipts = {HexBytes("0xabc"): {"status": 1, "blockNumber": 100}}

    phase2_tx = MagicMock(tx_hash="0xdef")
    phase3_tx = MagicMock(tx_hash="0x123")
    captured_phase2_raw = []
    captured_phase3_raw = []

    def capture_phase2(raw_amount: int):
        captured_phase2_raw.append(raw_amount)
        return phase2_tx, {"status": 1, "blockNumber": 101}

    def capture_phase3(raw_amount: int):
        captured_phase3_raw.append(raw_amount)
        return phase3_tx, {"status": 1, "blockNumber": 102}

    # 1. Simulate a withdrawal where vault equity drops by more than the intended 130 USDC sell value.
    # 2. Settle the phased withdrawal using the net amount for phases 2 and 3.
    # 3. Verify executed quantity does not exceed the planned partial sell quantity.
    with (
        patch.object(routing, "_fetch_safe_evm_usdc_balance", return_value=100_000_000),
        patch.object(routing, "_fetch_safe_perp_withdrawable_balance", return_value=Decimal("0")),
        patch.object(routing, "_fetch_safe_spot_free_usdc_balance", return_value=Decimal("0")),
        patch.object(routing, "_wait_for_perp_withdrawable_balance", return_value=Decimal("123.0")),
        patch.object(routing, "_broadcast_withdrawal_phase2", side_effect=capture_phase2),
        patch.object(routing, "_wait_for_spot_free_usdc_balance", return_value=Decimal("123.0")),
        patch.object(routing, "_broadcast_withdrawal_phase3", side_effect=capture_phase3),
        patch.object(routing, "_wait_for_usdc_arrival", return_value=122_990_000),
    ):
        routing._settle_withdrawal(
            routing.web3,
            state,
            trade,
            receipts,
            stop_on_execution_failure=False,
        )

    assert captured_phase2_raw == [123_000_000]
    assert captured_phase3_raw == [122_990_000]
    success_kwargs = state.mark_trade_success.call_args.kwargs
    assert success_kwargs["executed_amount"] == Decimal("-65")
    assert success_kwargs["executed_reserve"] == Decimal("122.99")
    assert success_kwargs["executed_price"] == pytest.approx(122.99 / 65)


@patch("tradeexecutor.ethereum.vault.hypercore_routing.get_block_timestamp")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.fetch_user_vault_equity")
def test_full_close_uses_observed_gross_decrease_when_phase1_under_redeems(
    mock_fetch_equity,
    mock_block_ts,
):
    """Avoid closing the full planned quantity when observed gross vault debit is short.

    1. Simulate a full-close withdrawal where phase 1 moves net USDC within slippage tolerance.
    2. Return a post-withdrawal equity snapshot showing only 100 USDC of gross vault debit for a 130 USDC close.
    3. Verify settlement uses the observed gross debit for quantity instead of marking the full close quantity sold.
    """
    routing = _make_routing()
    trade = _make_trade(planned_reserve=Decimal("130.0"))
    trade.planned_price = 2.0
    trade.planned_quantity = Decimal("-65")
    trade.closing = True
    state = MagicMock()
    state.portfolio.get_position_by_id.return_value.get_quantity.return_value = Decimal("65")
    state.portfolio.find_position_for_trade.return_value.get_quantity.return_value = Decimal("65")
    mock_block_ts.return_value = datetime.datetime(2025, 1, 1)
    mock_fetch_equity.side_effect = [
        _make_equity(Decimal("1000.0")),
        _make_equity(Decimal("900.0")),
    ]
    receipts = {HexBytes("0xabc"): {"status": 1, "blockNumber": 100}}

    phase2_tx = MagicMock(tx_hash="0xdef")
    phase3_tx = MagicMock(tx_hash="0x123")
    captured_phase2_raw = []
    captured_phase3_raw = []

    def capture_phase2(raw_amount: int):
        captured_phase2_raw.append(raw_amount)
        return phase2_tx, {"status": 1, "blockNumber": 101}

    def capture_phase3(raw_amount: int):
        captured_phase3_raw.append(raw_amount)
        return phase3_tx, {"status": 1, "blockNumber": 102}

    # 1. Simulate a full-close withdrawal where phase 1 moves net USDC within slippage tolerance.
    # 2. Return a post-withdrawal equity snapshot showing only 100 USDC of gross vault debit for a 130 USDC close.
    # 3. Verify settlement uses the observed gross debit for quantity instead of marking the full close quantity sold.
    with (
        patch.object(routing, "_fetch_safe_evm_usdc_balance", return_value=100_000_000),
        patch.object(routing, "_fetch_safe_perp_withdrawable_balance", return_value=Decimal("0")),
        patch.object(routing, "_fetch_safe_spot_free_usdc_balance", return_value=Decimal("0")),
        patch.object(routing, "_wait_for_perp_withdrawable_balance", return_value=Decimal("110.0")),
        patch.object(routing, "_broadcast_withdrawal_phase2", side_effect=capture_phase2),
        patch.object(routing, "_wait_for_spot_free_usdc_balance", return_value=Decimal("110.0")),
        patch.object(routing, "_broadcast_withdrawal_phase3", side_effect=capture_phase3),
        patch.object(routing, "_wait_for_usdc_arrival", return_value=109_990_000),
    ):
        routing._settle_withdrawal(
            routing.web3,
            state,
            trade,
            receipts,
            stop_on_execution_failure=False,
        )

    assert captured_phase2_raw == [110_000_000]
    assert captured_phase3_raw == [109_990_000]
    success_kwargs = state.mark_trade_success.call_args.kwargs
    assert success_kwargs["executed_amount"] == Decimal("-50")
    assert success_kwargs["executed_reserve"] == Decimal("109.99")
    assert success_kwargs["executed_price"] == pytest.approx(109.99 / 50)


@patch("tradeexecutor.ethereum.vault.hypercore_routing.report_failure")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.get_block_timestamp")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.fetch_user_vault_equity")
def test_withdrawal_aborts_if_perp_balance_does_not_appear(
    mock_fetch_equity,
    mock_block_ts,
    mock_report_failure,
):
    """Abort phased withdrawal if the vault withdrawal never reaches perp.

    1. Simulate a successful phase-1 EVM tx for a Hypercore withdrawal.
    2. Make the perp-balance wait fail before phase 2 can start.
    3. Verify the trade is failed with stranded-USDC recovery metadata.
    """
    from tradeexecutor.ethereum.vault.hypercore_routing import HypercoreWithdrawalVerificationError
    from hexbytes import HexBytes

    routing = _make_routing()
    trade = _make_trade(planned_reserve=Decimal("50.0"))
    state = MagicMock()
    mock_block_ts.return_value = datetime.datetime(2025, 1, 1)
    mock_fetch_equity.return_value = _make_equity(Decimal("500.0"))
    receipts = {HexBytes("0xabc"): {"status": 1, "blockNumber": 100}}

    # 1. Simulate a successful phase-1 EVM tx for a Hypercore withdrawal.
    # 2. Make the perp-balance wait fail before phase 2 can start.
    # 3. Verify the trade is failed with stranded-USDC recovery metadata.
    with patch.object(routing, "_fetch_safe_evm_usdc_balance", return_value=100_000_000):
        with patch.object(routing, "_fetch_safe_perp_withdrawable_balance", return_value=Decimal("0")):
            with patch.object(routing, "_fetch_safe_spot_free_usdc_balance", return_value=Decimal("0")):
                with patch.object(
                    routing,
                    "_wait_for_perp_withdrawable_balance",
                    side_effect=HypercoreWithdrawalVerificationError("perp did not increase"),
                ):
                    with patch.object(routing, "_broadcast_withdrawal_phase2") as phase2:
                        routing._settle_withdrawal(
                            routing.web3,
                            state,
                            trade,
                            receipts,
                            stop_on_execution_failure=False,
                        )

    phase2.assert_not_called()
    mock_report_failure.assert_called_once()
    assert trade.other_data["hypercore_stranded_usdc"]["location"] == "hypercore_perp"


@patch("tradeexecutor.ethereum.vault.hypercore_routing.report_failure")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.HyperliquidVault")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.get_block_timestamp")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.fetch_user_vault_equity")
def test_withdrawal_phase1_retry_handles_silent_noop_from_equity_drift(
    mock_fetch_equity,
    mock_block_ts,
    mock_vault_cls,
    mock_report_failure,
):
    """Retry withdrawal phase 1 when fresh vault equity explains a silent HyperCore no-op.

    1. Simulate the 2026-04-15 HyperAI DOEZOE crash where phase 1 asks for slightly more than fresh vault equity.
    2. Make the first perp-balance wait time out, matching a silent ``vaultTransfer`` no-op.
    3. Retry phase 1 once using fresh vault equity minus the safety margin.
    4. Continue phases 2 and 3 with the retry amount.
    5. Verify the trade succeeds, no stranded-USDC failure is reported, and the retry
       recomputes a smaller performance-fee tolerance for the smaller retry amount.
    """
    from hexbytes import HexBytes

    from eth_defi.hyperliquid.vault import estimate_max_withdrawal_commission
    from tradeexecutor.ethereum.vault.hypercore_routing import (
        HYPERCORE_WITHDRAWAL_SAFETY_MARGIN_RAW,
        HypercoreWithdrawalVerificationError,
        raw_to_usdc,
        usdc_to_raw,
    )

    # Vault reports a 10% leader performance fee, so the fee tolerance scales with the amount.
    mock_vault_cls.return_value.fetch_metadata.return_value = MagicMock(commission_rate=Decimal("0.10"))

    routing = _make_routing()
    trade = _make_trade(planned_reserve=Decimal("11.737146"))
    state = MagicMock()
    state.portfolio.get_position_by_id.return_value.get_quantity.return_value = Decimal("11.790353")
    mock_block_ts.return_value = datetime.datetime(2026, 4, 15, 13, 44, 29)
    current_equity = Decimal("11.725107")
    retry_raw = usdc_to_raw(current_equity) - HYPERCORE_WITHDRAWAL_SAFETY_MARGIN_RAW
    receipts = {HexBytes("0xabc"): {"status": 1, "blockNumber": 100}}

    phase1_retry_tx = MagicMock(tx_hash="0xretry")
    phase2_tx = MagicMock(tx_hash="0xdef")
    phase3_tx = MagicMock(tx_hash="0x123")
    captured_phase2_raw = []
    captured_phase3_raw = []

    def capture_phase2(raw_amount: int):
        captured_phase2_raw.append(raw_amount)
        return phase2_tx, {"status": 1, "blockNumber": 102}

    def capture_phase3(raw_amount: int):
        captured_phase3_raw.append(raw_amount)
        return phase3_tx, {"status": 1, "blockNumber": 103}

    # 1. Simulate the DOEZOE crash shape: post-phase-1 equity is below the requested amount.
    # 2. Make the first perp-balance wait time out before retrying.
    # 3. Retry phase 1 with fresh equity minus the safety margin.
    # 4. Continue phases 2 and 3 with the retry amount.
    # 5. Verify the trade succeeds and no stranded-USDC failure is reported.
    mock_fetch_equity.side_effect = [
        _make_equity(current_equity),
        _make_equity(current_equity),
        _make_equity(Decimal("1.500000")),
    ]
    with (
        patch.object(routing, "_fetch_safe_evm_usdc_balance", return_value=358_883_523),
        patch.object(routing, "_fetch_safe_perp_withdrawable_balance", return_value=Decimal("760.927156")),
        patch.object(routing, "_fetch_safe_spot_free_usdc_balance", return_value=Decimal("33.620982")),
        patch.object(
            routing,
            "_wait_for_perp_withdrawable_balance",
            side_effect=[
                HypercoreWithdrawalVerificationError("phase 1 timed out"),
                Decimal("771.152263"),
            ],
        ) as wait_perp,
        patch.object(
            routing,
            "_broadcast_withdrawal_phase1_retry",
            return_value=(phase1_retry_tx, {"status": 1, "blockNumber": 101}),
        ) as phase1_retry,
        patch.object(routing, "_broadcast_withdrawal_phase2", side_effect=capture_phase2),
        patch.object(routing, "_wait_for_spot_free_usdc_balance", return_value=Decimal("43.846089")),
        patch.object(routing, "_broadcast_withdrawal_phase3", side_effect=capture_phase3),
        patch.object(routing, "_wait_for_usdc_arrival", return_value=retry_raw),
    ):
        routing._settle_withdrawal(
            routing.web3,
            state,
            trade,
            receipts,
            stop_on_execution_failure=False,
        )

    phase1_retry.assert_called_once_with(
        vault_address=VAULT_ADDR,
        raw_amount=retry_raw,
    )
    assert wait_perp.call_count == 2
    assert captured_phase2_raw == [retry_raw]
    assert captured_phase3_raw == [retry_raw]
    assert trade.other_data["hypercore_capped_withdrawal_raw"] == retry_raw
    assert phase1_retry_tx in trade.blockchain_transactions
    state.mark_trade_success.assert_called_once()
    mock_report_failure.assert_not_called()

    # The retry wait must use a fee tolerance recomputed for the smaller retry
    # amount, not the original (larger) request — otherwise an over-large
    # tolerance could accept little or no post-retry perp increase.
    first_fee_tol = wait_perp.call_args_list[0].kwargs["performance_fee_tolerance"]
    retry_fee_tol = wait_perp.call_args_list[1].kwargs["performance_fee_tolerance"]
    assert first_fee_tol == estimate_max_withdrawal_commission(Decimal("11.737146"), Decimal("0.10"))
    assert retry_fee_tol == estimate_max_withdrawal_commission(raw_to_usdc(retry_raw), Decimal("0.10"))
    assert retry_fee_tol < first_fee_tol
