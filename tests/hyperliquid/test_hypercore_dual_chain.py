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

    with patch.object(routing, "_fetch_safe_evm_usdc_balance", side_effect=[100_000_000, 150_000_000]):
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
    import pytest
    from tradeexecutor.ethereum.vault.hypercore_routing import HypercoreVaultRouting

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
