"""Test the strict HyperCore vault-equity observation helper.

Tests the strict equity-observation helper's threshold and deadline. Live
settlement now deliberately accepts its failures with a warning: equity
includes trading PnL and cannot prove rejection of an existing-position top-up.
"""

import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from eth_defi.hyperliquid.api import (
    HypercoreDepositVerificationError,
    UserVaultEquity,
    wait_for_vault_deposit_confirmation,
)


VAULT_ADDR = "0xdfc24b077bc1425ad1dea75bcb6f8158e10df303"
USER_ADDR = "0x1234567890abcdef1234567890abcdef12345678"
LOCKED_UNTIL = datetime.datetime(2030, 1, 1)


def _make_equity(equity: Decimal) -> UserVaultEquity:
    return UserVaultEquity(
        vault_address=VAULT_ADDR,
        equity=equity,
        locked_until=LOCKED_UNTIL,
    )


@patch("eth_defi.hyperliquid.api.time.sleep")
@patch("eth_defi.hyperliquid.api.fetch_user_vault_equity")
def test_deposit_verification_succeeds_new_position(mock_fetch, mock_sleep):
    """First deposit: equity appears on second poll."""
    mock_fetch.side_effect = [
        None,  # First poll: no position yet
        _make_equity(Decimal("50.0")),  # Second poll: equity appeared
    ]

    session = MagicMock()
    result = wait_for_vault_deposit_confirmation(
        session,
        user=USER_ADDR,
        vault_address=VAULT_ADDR,
        expected_deposit=Decimal("50.0"),
        existing_equity=None,
        timeout=10.0,
        poll_interval=1.0,
    )

    assert result.equity == Decimal("50.0")
    assert mock_fetch.call_count == 2
    # All calls should use bypass_cache=True
    for call in mock_fetch.call_args_list:
        assert call.kwargs.get("bypass_cache") is True or call[1].get("bypass_cache") is True


@patch("eth_defi.hyperliquid.api.time.sleep")
@patch("eth_defi.hyperliquid.api.fetch_user_vault_equity")
def test_deposit_verification_succeeds_existing_position(mock_fetch, mock_sleep):
    """Existing position: equity increases by expected amount."""
    mock_fetch.side_effect = [
        _make_equity(Decimal("100.0")),  # First poll: no increase yet
        _make_equity(Decimal("150.0")),  # Second poll: increased by 50
    ]

    session = MagicMock()
    result = wait_for_vault_deposit_confirmation(
        session,
        user=USER_ADDR,
        vault_address=VAULT_ADDR,
        expected_deposit=Decimal("50.0"),
        existing_equity=Decimal("100.0"),
        timeout=10.0,
        poll_interval=1.0,
    )

    assert result.equity == Decimal("150.0")
    assert mock_fetch.call_count == 2


@patch("eth_defi.hyperliquid.api.time.sleep")
@patch("eth_defi.hyperliquid.api.time.monotonic")
@patch("eth_defi.hyperliquid.api.fetch_user_vault_equity")
def test_deposit_verification_timeout_raises_error(mock_fetch, mock_time, mock_sleep):
    """Deposit never appears: raises HypercoreDepositVerificationError."""
    mock_fetch.return_value = None
    # Only waiting consumes the fake budget; reading the timer must not
    # accidentally bypass the polling branch this regression exercises.
    mock_time.return_value = 0.0

    def advance_clock(seconds: float) -> None:
        """Allow repeated missing-equity polls before the observation expires."""
        mock_time.return_value += seconds

    mock_sleep.side_effect = advance_clock

    session = MagicMock()
    with pytest.raises(HypercoreDepositVerificationError) as exc_info:
        wait_for_vault_deposit_confirmation(
            session,
            user=USER_ADDR,
            vault_address=VAULT_ADDR,
            expected_deposit=Decimal("50.0"),
            existing_equity=None,
            timeout=3.0,
            poll_interval=1.0,
        )

    err_msg = str(exc_info.value)
    assert "could not be verified" in err_msg
    assert "50.0" in err_msg
    assert VAULT_ADDR in err_msg


@patch("eth_defi.hyperliquid.api.time.sleep")
@patch("eth_defi.hyperliquid.api.fetch_user_vault_equity")
def test_deposit_verification_tolerates_small_difference(mock_fetch, mock_sleep):
    """Equity increase slightly below expected within tolerance still passes."""
    # Deposit 50 USDC, but equity only increased by 49.995 (vault PnL)
    mock_fetch.return_value = _make_equity(Decimal("149.995"))

    session = MagicMock()
    result = wait_for_vault_deposit_confirmation(
        session,
        user=USER_ADDR,
        vault_address=VAULT_ADDR,
        expected_deposit=Decimal("50.0"),
        existing_equity=Decimal("100.0"),
        timeout=10.0,
        poll_interval=1.0,
        tolerance=Decimal("0.01"),
    )

    assert result.equity == Decimal("149.995")


@patch("eth_defi.hyperliquid.api.time.sleep")
@patch("eth_defi.hyperliquid.api.time.monotonic")
@patch("eth_defi.hyperliquid.api.fetch_user_vault_equity")
def test_deposit_verification_existing_no_increase_times_out(mock_fetch, mock_time, mock_sleep):
    """Existing position with no equity increase times out."""
    # Flat equity can mean rejection or PnL masking a top-up. The helper only
    # reports that its threshold was not reached, without choosing a cause.
    mock_fetch.return_value = _make_equity(Decimal("100.0"))
    mock_time.return_value = 0.0

    def advance_clock(seconds: float) -> None:
        """Evaluate the flat-equity reading while the observation is still live."""
        mock_time.return_value += seconds

    mock_sleep.side_effect = advance_clock

    session = MagicMock()
    with pytest.raises(HypercoreDepositVerificationError) as exc_info:
        wait_for_vault_deposit_confirmation(
            session,
            user=USER_ADDR,
            vault_address=VAULT_ADDR,
            expected_deposit=Decimal("50.0"),
            existing_equity=Decimal("100.0"),
            timeout=3.0,
            poll_interval=1.0,
        )

    err_msg = str(exc_info.value)
    assert "does not prove whether the deposit was accepted" in err_msg
    assert mock_fetch.call_count == 2


@patch("eth_defi.hyperliquid.api.time.sleep")
@patch("eth_defi.hyperliquid.api.fetch_user_vault_equity")
def test_deposit_verification_tolerates_relative_existing_position_drift(
    mock_fetch, mock_sleep,
):
    """Large existing-vault deposit passes when the shortfall is within relative tolerance."""
    mock_fetch.return_value = _make_equity(Decimal("629.998483"))

    session = MagicMock()
    result = wait_for_vault_deposit_confirmation(
        session,
        user=USER_ADDR,
        vault_address=VAULT_ADDR,
        expected_deposit=Decimal("570.690753"),
        existing_equity=Decimal("59.559287"),
        timeout=10.0,
        poll_interval=1.0,
    )

    assert result.equity == Decimal("629.998483")


@patch("eth_defi.hyperliquid.api.time.sleep")
@patch("eth_defi.hyperliquid.api.time.monotonic")
@patch("eth_defi.hyperliquid.api.fetch_user_vault_equity")
def test_deposit_verification_rejects_large_existing_position_shortfall(
    mock_fetch, mock_time, mock_sleep,
):
    """Existing-vault deposit still fails when the shortfall exceeds relative tolerance."""
    mock_fetch.return_value = _make_equity(Decimal("590.0"))
    mock_time.return_value = 0.0

    def advance_clock(seconds: float) -> None:
        """Let the shortfall reach the comparison instead of skipping a late read."""
        mock_time.return_value += seconds

    mock_sleep.side_effect = advance_clock

    session = MagicMock()
    with pytest.raises(HypercoreDepositVerificationError) as exc_info:
        wait_for_vault_deposit_confirmation(
            session,
            user=USER_ADDR,
            vault_address=VAULT_ADDR,
            expected_deposit=Decimal("570.690753"),
            existing_equity=Decimal("59.559287"),
            timeout=3.0,
            poll_interval=1.0,
        )

    err_msg = str(exc_info.value)
    assert "could not be verified" in err_msg
    assert mock_fetch.call_count == 2
