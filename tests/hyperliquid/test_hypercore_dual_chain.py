"""Test P6: Dual-chain confirmation after Hypercore withdrawals.

Tests that _settle_withdrawal() captures baseline vault equity before
the withdrawal and compares it against equity after USDC arrives on EVM.
"""

import datetime
import itertools
import logging
from decimal import Decimal
from unittest.mock import MagicMock, patch

from eth_defi.hyperliquid.api import UserVaultEquity


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


def test_withdrawal_already_reflected_in_vault_equity_is_detected():
    """Detect a phase 1 withdrawal that already reduced vault equity.

    1. Create a routing object with a state position quantity from before settlement.
    2. Feed the helper a vault equity snapshot that matches the expected post-withdrawal residual.
    3. Verify the helper recognises phase 1 as already applied.
    """
    routing = _make_routing()

    # 1. Create a routing object with a state position quantity from before settlement.
    position_quantity_before = Decimal("630.007301")
    current_vault_equity = Decimal("77.748241")

    # 2. Feed the helper a vault equity snapshot that matches the expected post-withdrawal residual.
    result = routing._is_withdrawal_already_reflected_in_vault_equity(
        position_quantity_before=position_quantity_before,
        current_vault_equity=current_vault_equity,
        expected_increase_raw=552_259_060,
    )

    # 3. Verify the helper recognises phase 1 as already applied.
    assert result is True


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
