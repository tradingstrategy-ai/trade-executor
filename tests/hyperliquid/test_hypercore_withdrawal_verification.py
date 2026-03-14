"""Test Hypercore vault withdrawal verification (P2 fix).

Tests that the withdrawal poll loop correctly detects when CoreWriter
withdrawal actions fail silently on HyperCore, preventing phantom
withdrawals where the portfolio shows USDC arrived but it didn't.
"""

from unittest.mock import patch

import pytest

from tradeexecutor.ethereum.vault.hypercore_routing import (
    HypercoreWithdrawalVerificationError,
)


def test_withdrawal_verification_error_has_useful_message():
    """The exception message includes diagnostic details."""
    err = HypercoreWithdrawalVerificationError(
        "USDC did not arrive in Safe 0xABC within 30s. "
        "Expected increase: 50000000 raw, actual increase: 0 raw"
    )
    assert "did not arrive" in str(err)
    assert "50000000" in str(err)


@patch("tradeexecutor.ethereum.vault.hypercore_routing.time.sleep")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.time.time")
def test_wait_for_usdc_arrival_succeeds(mock_time, mock_sleep):
    """USDC arrives after 2 polls."""
    from unittest.mock import MagicMock
    from tradeexecutor.ethereum.vault.hypercore_routing import HypercoreVaultRouting

    routing = MagicMock(spec=HypercoreVaultRouting)
    routing.safe_address = "0xABC"

    # Balance: baseline, then +50 USDC (raw)
    balance_calls = [100_000_000, 100_000_000, 150_000_000]
    routing._fetch_safe_evm_usdc_balance = MagicMock(side_effect=balance_calls)

    mock_time.side_effect = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    # Call the unbound method with our mock as self
    result = HypercoreVaultRouting._wait_for_usdc_arrival(
        routing,
        baseline_balance_raw=100_000_000,
        expected_increase_raw=50_000_000,
        timeout=10.0,
        poll_interval=1.0,
    )

    assert result == 50_000_000


@patch("tradeexecutor.ethereum.vault.hypercore_routing.time.sleep")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.time.time")
def test_wait_for_usdc_arrival_timeout(mock_time, mock_sleep):
    """USDC never arrives: raises HypercoreWithdrawalVerificationError."""
    from unittest.mock import MagicMock
    from tradeexecutor.ethereum.vault.hypercore_routing import HypercoreVaultRouting

    routing = MagicMock(spec=HypercoreVaultRouting)
    routing.safe_address = "0xABC"
    routing._fetch_safe_evm_usdc_balance = MagicMock(return_value=100_000_000)

    mock_time.side_effect = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    with pytest.raises(HypercoreWithdrawalVerificationError) as exc_info:
        HypercoreVaultRouting._wait_for_usdc_arrival(
            routing,
            baseline_balance_raw=100_000_000,
            expected_increase_raw=50_000_000,
            timeout=3.0,
            poll_interval=1.0,
        )

    err_msg = str(exc_info.value)
    assert "did not arrive" in err_msg
    assert "50000000" in err_msg


@patch("tradeexecutor.ethereum.vault.hypercore_routing.time.sleep")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.time.time")
def test_wait_for_usdc_arrival_uses_actual_increase(mock_time, mock_sleep):
    """Returns the actual balance increase, even if slightly more than expected."""
    from unittest.mock import MagicMock
    from tradeexecutor.ethereum.vault.hypercore_routing import HypercoreVaultRouting

    routing = MagicMock(spec=HypercoreVaultRouting)
    routing.safe_address = "0xABC"

    # Balance increased by 50.5 USDC instead of expected 50
    routing._fetch_safe_evm_usdc_balance = MagicMock(return_value=150_500_000)
    mock_time.side_effect = [0.0, 1.0, 2.0]

    result = HypercoreVaultRouting._wait_for_usdc_arrival(
        routing,
        baseline_balance_raw=100_000_000,
        expected_increase_raw=50_000_000,
        timeout=10.0,
        poll_interval=1.0,
    )

    assert result == 50_500_000
