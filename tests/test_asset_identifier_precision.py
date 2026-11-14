"""Exact asset units across the executor's vault and native venue paths."""

from decimal import Decimal, localcontext

import pytest

from tradeexecutor.state.identifier import AssetIdentifier


@pytest.mark.parametrize("decimals", [0, 6, 8, 18])
@pytest.mark.parametrize("precision", [6, 28, 78])
def test_asset_conversion_preserves_raw_units(decimals: int, precision: int) -> None:
    """Keep asset balances exact and limit sub-unit rounding explicitly.

    1. Model small, signed, large and uint256-limit balances at each precision.
    2. Convert in both directions and bound human values to one raw unit.
    3. Verify fractional raw units truncate towards zero without context changes.
    """
    # 1. Model small, signed, large and uint256-limit balances at each precision.
    asset = AssetIdentifier(1, "0x0000000000000000000000000000000000000001", "SHARE", decimals)
    quantum = Decimal(f"1e-{decimals}")
    for raw in [0, 1, 1_001_000_000, 10**29 + 1, -(10**29 + 1), 2**256 - 1]:
        expected = Decimal(f"{raw}e-{decimals}")
        # 2. Convert in both directions and bound human values to one raw unit.
        with localcontext() as context:
            context.prec = precision
            actual = asset.convert_to_decimal(raw)
            assert actual == expected
            assert actual == pytest.approx(expected, rel=0, abs=quantum)
            assert asset.convert_to_raw_amount(actual) == raw
            assert context.prec == precision

    # 3. Verify fractional raw units truncate towards zero without context changes.
    for text in ["10000000000.1234567890123456789", "-10000000000.1234567890123456789"]:
        amount = Decimal(text)
        with localcontext() as oracle:
            oracle.prec = 100
            expected_raw = int(amount * 10**decimals)
        with localcontext() as context:
            context.prec = precision
            actual_raw = asset.convert_to_raw_amount(amount)
            assert actual_raw == expected_raw
            assert asset.convert_to_decimal(actual_raw) == pytest.approx(amount, rel=0, abs=quantum)
            assert context.prec == precision
