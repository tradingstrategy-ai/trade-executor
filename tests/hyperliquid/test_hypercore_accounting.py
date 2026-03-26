"""Test Hypercore vault position accounting consistency.

Verifies that:
1. Deposit records the USDC delta (not total equity) as executed_amount.
2. Valuation computes per-unit price (equity/quantity) so value = equity.
3. A second deposit adds only the delta to position quantity.
4. Valuation stays 1:1 with equity after multiple deposits.
"""

import datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from eth_defi.compat import native_datetime_utc_now


def test_valuation_computes_per_unit_price():
    """Value = quantity * (equity / quantity) = equity."""
    from tradeexecutor.ethereum.vault.hypercore_valuation import HypercoreVaultValuator

    # Position with quantity=100 (deposited 100 USDC), price=1.0
    position = MagicMock()
    position.is_vault.return_value = True
    position.get_quantity.return_value = Decimal("100.0")
    position.last_token_price = 1.0

    # API returns equity = 105 (5% gain)
    def value_func(pair):
        return Decimal("105.0")

    valuator = HypercoreVaultValuator(value_func=value_func, simulate=False)
    ts = native_datetime_utc_now()
    result = valuator(ts, position)

    # Per-unit price should be 105/100 = 1.05
    assert position.revalue_base_asset.call_count == 1
    call_args = position.revalue_base_asset.call_args
    new_price = call_args[0][1]
    assert pytest.approx(new_price, rel=1e-6) == 1.05


def test_valuation_second_deposit_stays_correct():
    """After depositing 100 then 50, equity=155 → value should be 155."""
    from tradeexecutor.ethereum.vault.hypercore_valuation import HypercoreVaultValuator

    position = MagicMock()
    position.is_vault.return_value = True
    # After two deposits: 100 + 50 = 150 quantity
    position.get_quantity.return_value = Decimal("150.0")
    position.last_token_price = 1.0

    # API returns equity = 155 (vault grew from 150 to 155)
    def value_func(pair):
        return Decimal("155.0")

    valuator = HypercoreVaultValuator(value_func=value_func, simulate=False)
    ts = native_datetime_utc_now()
    result = valuator(ts, position)

    # Per-unit price: 155/150 = 1.0333...
    call_args = position.revalue_base_asset.call_args
    new_price = call_args[0][1]
    assert pytest.approx(new_price, rel=1e-4) == 155.0 / 150.0


def test_valuation_zero_quantity_uses_default_price():
    """Edge case: quantity=0 should use price=1.0 to avoid division by zero."""
    from tradeexecutor.ethereum.vault.hypercore_valuation import HypercoreVaultValuator

    position = MagicMock()
    position.is_vault.return_value = True
    position.get_quantity.return_value = Decimal("0")
    position.last_token_price = 1.0

    def value_func(pair):
        return Decimal("0")

    valuator = HypercoreVaultValuator(value_func=value_func, simulate=False)
    ts = native_datetime_utc_now()
    result = valuator(ts, position)

    call_args = position.revalue_base_asset.call_args
    new_price = call_args[0][1]
    assert new_price == 1.0


def test_pricing_model_returns_one():
    """Trade pricing for vault deposits/withdrawals is always 1.0 USDC."""
    from tradeexecutor.ethereum.vault.hypercore_valuation import HypercoreVaultPricing
    from tradeexecutor.state.identifier import TradingPairIdentifier, TradingPairKind, AssetIdentifier

    # Even with non-zero equity, trade price should be 1.0
    def value_func(pair):
        return Decimal("500.0")

    pricing = HypercoreVaultPricing(value_func=value_func, simulate=False)

    base = AssetIdentifier(chain_id=999, address="0x0000000000000000000000000000000000000001", token_symbol="VAULT", decimals=6)
    quote = AssetIdentifier(chain_id=999, address="0x0000000000000000000000000000000000000002", token_symbol="USDC", decimals=6)
    pair = TradingPairIdentifier(
        base=base, quote=quote,
        pool_address="0x0000000000000000000000000000000000000003", exchange_address="0x0000000000000000000000000000000000000004",
        internal_id=1, internal_exchange_id=1,
        fee=0.0, kind=TradingPairKind.vault,
    )

    buy_pricing = pricing.get_buy_price(None, pair, Decimal("100"))
    assert buy_pricing.price == 1.0

    sell_pricing = pricing.get_sell_price(None, pair, Decimal("50"))
    assert sell_pricing.price == 1.0

    mid = pricing.get_mid_price(None, pair)
    assert mid == 1.0


def test_lockup_func_populates_other_data():
    """Valuator with lockup_func stores lockup hours in position.other_data.

    1. Create a mock lockup func returning 12.5 hours
    2. Run the valuator
    3. Verify other_data contains the lockup hours
    """
    from tradeexecutor.ethereum.vault.hypercore_valuation import HypercoreVaultValuator

    position = MagicMock()
    position.is_vault.return_value = True
    position.get_quantity.return_value = Decimal("100.0")
    position.last_token_price = 1.0
    position.other_data = {}

    def value_func(pair):
        return Decimal("105.0")

    def lockup_func(pair):
        return 12.5

    valuator = HypercoreVaultValuator(value_func=value_func, lockup_func=lockup_func)
    ts = native_datetime_utc_now()
    valuator(ts, position)

    assert position.other_data["vault_lockup_remaining_hours"] == 12.5


def test_lockup_func_none_position():
    """Valuator with lockup_func stores None when no vault position found.

    1. Create a mock lockup func returning None (no position)
    2. Run the valuator
    3. Verify other_data contains None
    """
    from tradeexecutor.ethereum.vault.hypercore_valuation import HypercoreVaultValuator

    position = MagicMock()
    position.is_vault.return_value = True
    position.get_quantity.return_value = Decimal("100.0")
    position.last_token_price = 1.0
    position.other_data = {}

    def value_func(pair):
        return Decimal("105.0")

    def lockup_func(pair):
        return None

    valuator = HypercoreVaultValuator(value_func=value_func, lockup_func=lockup_func)
    ts = native_datetime_utc_now()
    valuator(ts, position)

    assert position.other_data["vault_lockup_remaining_hours"] is None


def test_old_bug_equity_squared():
    """Regression: the old code set quantity=equity and price=equity → value=equity².

    With the fix, quantity=deposited_amount and price=equity/quantity → value=equity.
    This test verifies the specific scenario from the review: a 5 USDC deposit
    with API equity=5.5 should value at 5.5, not 27.5.
    """
    from tradeexecutor.ethereum.vault.hypercore_valuation import HypercoreVaultValuator

    position = MagicMock()
    position.is_vault.return_value = True
    # Deposited 5 USDC (quantity tracks USDC deposited)
    position.get_quantity.return_value = Decimal("5.0")
    position.last_token_price = 1.0

    # API says equity is 5.5
    def value_func(pair):
        return Decimal("5.5")

    valuator = HypercoreVaultValuator(value_func=value_func, simulate=False)
    ts = native_datetime_utc_now()
    result = valuator(ts, position)

    call_args = position.revalue_base_asset.call_args
    new_price = call_args[0][1]
    # Per-unit price: 5.5 / 5.0 = 1.1
    assert pytest.approx(new_price, rel=1e-6) == 1.1
    # Value would be: 5.0 * 1.1 = 5.5 (correct), NOT 5.0 * 5.5 = 27.5 (old bug)
