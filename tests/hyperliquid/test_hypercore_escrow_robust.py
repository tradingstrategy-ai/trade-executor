"""Test P15: Robust escrow wait with spot balance verification."""

from decimal import Decimal
from unittest.mock import MagicMock, patch, call

from eth_defi.hyperliquid.api import SpotBalance, SpotClearinghouseState, EvmEscrow
from eth_defi.hyperliquid.evm_escrow import _get_usdc_spot_balance, wait_for_evm_escrow_clear


def _make_state(usdc_balance: Decimal, escrows: list[EvmEscrow] | None = None) -> SpotClearinghouseState:
    """Helper to build a SpotClearinghouseState with a USDC balance."""
    balances = [SpotBalance(coin="USDC", token=0, total=usdc_balance, hold=Decimal(0))]
    return SpotClearinghouseState(balances=balances, evm_escrows=escrows or [])


def test_get_usdc_spot_balance_found():
    """_get_usdc_spot_balance extracts USDC from balances."""
    state = _make_state(Decimal("100.5"))
    assert _get_usdc_spot_balance(state) == Decimal("100.5")


def test_get_usdc_spot_balance_missing():
    """_get_usdc_spot_balance returns 0 when no USDC balance."""
    state = SpotClearinghouseState(
        balances=[SpotBalance(coin="HYPE", token=1, total=Decimal("10"), hold=Decimal(0))],
        evm_escrows=[],
    )
    assert _get_usdc_spot_balance(state) == Decimal(0)


def test_escrow_wait_with_expected_usdc_succeeds(monkeypatch):
    """Escrow clears and spot USDC increases by expected amount."""
    session = MagicMock()

    escrow_entry = EvmEscrow(coin="USDC", token=0, total=Decimal("50"))

    # First call: baseline (has escrow, 100 USDC in spot)
    baseline = _make_state(Decimal("100"), [escrow_entry])
    # Second call (after initial sleep): still has escrow
    pending = _make_state(Decimal("100"), [escrow_entry])
    # Third call: escrow cleared, USDC increased
    cleared = _make_state(Decimal("150"))

    with patch(
        "eth_defi.hyperliquid.evm_escrow.fetch_spot_clearinghouse_state",
        side_effect=[baseline, pending, cleared],
    ), patch("eth_defi.hyperliquid.evm_escrow.time") as mock_time:
        mock_time.time.side_effect = [
            0,    # deadline = 0 + 60
            0.5,  # after initial sleep: baseline capture is before this
            2,    # first poll remaining check
            4,    # second poll remaining check
        ]
        mock_time.sleep = MagicMock()

        wait_for_evm_escrow_clear(
            session,
            user="0xABC",
            timeout=60.0,
            poll_interval=2.0,
            expected_usdc=Decimal("50"),
        )


def test_escrow_wait_with_expected_usdc_warns_on_shortfall(monkeypatch, caplog):
    """Escrow clears but spot USDC increase is less than expected — logs warning."""
    import logging
    session = MagicMock()

    escrow_entry = EvmEscrow(coin="USDC", token=0, total=Decimal("50"))

    # Baseline: 100 USDC
    baseline = _make_state(Decimal("100"), [escrow_entry])
    # Cleared but only +20 instead of +50
    cleared = _make_state(Decimal("120"))

    with patch(
        "eth_defi.hyperliquid.evm_escrow.fetch_spot_clearinghouse_state",
        side_effect=[baseline, cleared],
    ), patch("eth_defi.hyperliquid.evm_escrow.time") as mock_time:
        mock_time.time.side_effect = [0, 2]
        mock_time.sleep = MagicMock()

        with caplog.at_level(logging.WARNING):
            wait_for_evm_escrow_clear(
                session,
                user="0xABC",
                timeout=60.0,
                poll_interval=2.0,
                expected_usdc=Decimal("50"),
            )

        assert "Possible silent bridge failure" in caplog.text


def test_escrow_wait_without_expected_usdc_backward_compatible():
    """Without expected_usdc, the function behaves as before."""
    session = MagicMock()

    cleared = _make_state(Decimal("100"))

    with patch(
        "eth_defi.hyperliquid.evm_escrow.fetch_spot_clearinghouse_state",
        return_value=cleared,
    ), patch("eth_defi.hyperliquid.evm_escrow.time") as mock_time:
        mock_time.time.side_effect = [0, 2]
        mock_time.sleep = MagicMock()

        # Should work fine without expected_usdc
        wait_for_evm_escrow_clear(
            session,
            user="0xABC",
            timeout=60.0,
            poll_interval=2.0,
        )
