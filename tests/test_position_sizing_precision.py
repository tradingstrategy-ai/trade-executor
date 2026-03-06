"""Unit tests for position sizing precision in DeFi trades.

Tests decimal precision handling when calculating trade amounts across different
token decimals (6, 8, 18). Covers edge cases like USDC (6 decimals) → WETH (18 decimals)
conversions that commonly cause rounding errors in production.
"""

import pytest
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from typing import Tuple


def normalize_token_amount(
    amount: Decimal,
    source_decimals: int,
    target_decimals: int,
    round_mode: str = "down"
) -> Decimal:
    """Normalize token amount between different decimal precisions.
    
    Handles conversion from one token's decimal precision to another,
    ensuring no precision loss during cross-decimal calculations.
    
    Args:
        amount: The token amount to normalize
        source_decimals: Decimals of source token (e.g., 6 for USDC)
        target_decimals: Decimals of target token (e.g., 18 for WETH)
        round_mode: Rounding mode - "down" (floor) or "half_up" (standard)
        
    Returns:
        Normalized amount in target decimal precision
    """
    if source_decimals == target_decimals:
        return amount
    
    # Convert to base units (raw amount)
    decimal_diff = target_decimals - source_decimals
    
    rounding = ROUND_DOWN if round_mode == "down" else ROUND_HALF_UP
    
    if decimal_diff > 0:
        # Expanding precision (e.g., 6 → 18): multiply
        result = amount * Decimal(10 ** decimal_diff)
    else:
        # Reducing precision (e.g., 18 → 6): divide
        result = (amount / Decimal(10 ** abs(decimal_diff))).quantize(
            Decimal(10) ** -target_decimals,
            rounding=rounding
        )
    
    return result


def calculate_position_size(
    position_usd: Decimal,
    token_price: Decimal,
    token_decimals: int,
    min_position_usd: Decimal = Decimal("1.0")
) -> Tuple[Decimal, bool]:
    """Calculate position size in token amount from USD value.
    
    Handles decimal precision to avoid floating-point rounding errors.
    
    Args:
        position_usd: Position size in USD
        token_price: Token price in USD
        token_decimals: Token decimal precision
        min_position_usd: Minimum viable position in USD
        
    Returns:
        Tuple of (token_amount, is_valid) where is_valid indicates if
        position meets minimum requirements
    """
    if position_usd < min_position_usd or token_price <= Decimal("0"):
        return Decimal("0"), False
    
    # Calculate raw token amount with high precision
    raw_amount = position_usd / token_price
    
    # Quantize to token's decimal precision (using floor rounding for safety)
    quantized_amount = raw_amount.quantize(
        Decimal(10) ** -token_decimals,
        rounding=ROUND_DOWN
    )
    
    # Validate: verify position meets minimum after quantization
    final_position_value = quantized_amount * token_price
    is_valid = final_position_value >= min_position_usd
    
    return quantized_amount, is_valid


class TestPositionSizingPrecision:
    """Test suite for position sizing precision handling."""

    def test_usdc_to_weth_conversion(self):
        """Test USDC (6 decimals) → WETH (18 decimals) conversion."""
        # 1000 USDC with 6 decimals
        usdc_amount = Decimal("1000.000000")
        
        # Normalize to WETH's 18 decimals
        weth_amount = normalize_token_amount(
            usdc_amount,
            source_decimals=6,
            target_decimals=18
        )
        
        # Should multiply by 10^12
        expected = Decimal("1000000000000000000")
        assert weth_amount == expected
        assert len(str(weth_amount).split(".")[0]) == 16  # Correct magnitude

    def test_weth_to_usdc_conversion(self):
        """Test WETH (18 decimals) → USDC (6 decimals) conversion."""
        # 1 WETH with 18 decimals
        weth_amount = Decimal("1000000000000000000")
        
        # Normalize to USDC's 6 decimals
        usdc_amount = normalize_token_amount(
            weth_amount,
            source_decimals=18,
            target_decimals=6,
            round_mode="down"
        )
        
        # Should divide by 10^12 with floor rounding
        expected = Decimal("1.000000")
        assert usdc_amount == expected

    def test_fractional_weth_to_usdc(self):
        """Test fractional WETH conversion handling rounding correctly."""
        # 0.123456789123456789 WETH (18 decimals)
        weth_amount = Decimal("0.123456789123456789")
        
        usdc_amount = normalize_token_amount(
            weth_amount,
            source_decimals=18,
            target_decimals=6,
            round_mode="down"
        )
        
        # Should floor to 0.123456 USDC
        assert usdc_amount == Decimal("0.123456")
        assert usdc_amount < weth_amount  # Verify floor behavior

    def test_position_size_calculation_usdc(self):
        """Test position size calculation for USDC (6 decimals)."""
        position_usd = Decimal("5000.00")
        usdc_price = Decimal("1.0001")  # Slightly above peg
        usdc_decimals = 6
        
        amount, is_valid = calculate_position_size(
            position_usd=position_usd,
            token_price=usdc_price,
            token_decimals=usdc_decimals
        )
        
        assert is_valid is True
        # 5000 / 1.0001 ≈ 4999.5 USDC (truncated to 6 decimals)
        assert amount > Decimal("4999")
        assert amount < Decimal("5000")
        assert str(amount).count(".") <= 1  # Valid decimal notation

    def test_position_size_calculation_weth(self):
        """Test position size calculation for WETH (18 decimals)."""
        position_usd = Decimal("10000.00")
        weth_price = Decimal("2345.67")  # WETH price in USD
        weth_decimals = 18
        
        amount, is_valid = calculate_position_size(
            position_usd=position_usd,
            token_price=weth_price,
            token_decimals=weth_decimals
        )
        
        assert is_valid is True
        # 10000 / 2345.67 ≈ 4.264 WETH
        assert amount > Decimal("4")
        assert amount < Decimal("5")
        # Verify precision to 18 decimals
        decimal_part = str(amount).split(".")[1] if "." in str(amount) else ""
        assert len(decimal_part) <= 18

    def test_position_size_below_minimum(self):
        """Test position size validation when below minimum threshold."""
        position_usd = Decimal("0.50")  # Below typical $1.00 minimum
        btc_price = Decimal("45000.00")
        btc_decimals = 8
        min_usd = Decimal("1.00")
        
        amount, is_valid = calculate_position_size(
            position_usd=position_usd,
            token_price=btc_price,
            token_decimals=btc_decimals,
            min_position_usd=min_usd
        )
        
        assert is_valid is False
        assert amount == Decimal("0")

    def test_same_decimal_precision_passthrough(self):
        """Test normalization when source and target decimals match."""
        amount = Decimal("100.123456")
        
        result = normalize_token_amount(
            amount,
            source_decimals=18,
            target_decimals=18
        )
        
        assert result == amount

    def test_btc_usdc_conversion(self):
        """Test BTC (8 decimals) → USDC (6 decimals) conversion."""
        # 0.5 BTC with 8 decimals
        btc_amount = Decimal("0.50000000")
        
        usdc_amount = normalize_token_amount(
            btc_amount,
            source_decimals=8,
            target_decimals=6,
            round_mode="down"
        )
        
        # Should divide by 100 with floor rounding
        expected = Decimal("0.005000")
        assert usdc_amount == expected

    def test_precision_accumulation_multiple_conversions(self):
        """Test that precision is preserved through multiple conversions."""
        original = Decimal("123.456789")
        
        # Convert through chain: 6 → 18 → 8 → 6
        step1 = normalize_token_amount(original, 6, 18)
        step2 = normalize_token_amount(step1, 18, 8, round_mode="down")
        step3 = normalize_token_amount(step2, 8, 6, round_mode="down")
        
        # Due to rounding, final amount should be <= original
        assert step3 <= original
        # But should be very close (within token precision)
        assert original - step3 < Decimal("0.000001")
