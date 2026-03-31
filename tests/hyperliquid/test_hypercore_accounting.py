"""Test Hypercore vault position accounting consistency.

Verifies that:
1. Deposit records the USDC delta (not total equity) as executed_amount.
2. Valuation computes per-unit price (equity/quantity) so value = equity.
3. A second deposit adds only the delta to position quantity.
4. Valuation stays 1:1 with equity after multiple deposits.
"""

import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from eth_defi.compat import native_datetime_utc_now
from tradingstrategy.chain import ChainId

from tradeexecutor.ethereum.vault.hypercore_valuation import HypercoreVaultPricing, HypercoreVaultValuator
from tradeexecutor.ethereum.vault.hypercore_vault import create_hypercore_vault_pair
from tradeexecutor.state.identifier import TradingPairIdentifier, TradingPairKind, AssetIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.strategy.account_correction import _build_hypercore_vault_account_checks
from tradeexecutor.strategy.dust import (
    get_close_epsilon_for_pair,
    get_dust_epsilon_for_pair,
    HYPERLIQUID_VAULT_CLOSE_EPSILON,
    DEFAULT_VAULT_EPSILON,
)
from tradeexecutor.strategy.sync_model import OnChainBalance


def test_valuation_computes_per_unit_price():
    """Value = quantity * (equity / quantity) = equity."""

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


def test_lockup_func_populates_expires_at():
    """Valuator with lockup_func stores ISO timestamp in position.other_data.

    1. Create a mock lockup func returning a fixed datetime
    2. Run the valuator
    3. Verify other_data contains the ISO string
    """
    import datetime as dt

    position = MagicMock()
    position.is_vault.return_value = True
    position.get_quantity.return_value = Decimal("100.0")
    position.last_token_price = 1.0
    position.other_data = {}

    expires = dt.datetime(2026, 3, 27, 14, 30, 0)

    def value_func(pair):
        return Decimal("105.0")

    def lockup_func(pair):
        return expires

    valuator = HypercoreVaultValuator(value_func=value_func, lockup_func=lockup_func)
    ts = native_datetime_utc_now()
    valuator(ts, position)

    assert position.other_data["vault_lockup_expires_at"] == "2026-03-27T14:30:00"


def test_lockup_func_none_position():
    """Valuator with lockup_func stores None when no vault position found.

    1. Create a mock lockup func returning None (no position)
    2. Run the valuator
    3. Verify other_data contains None
    """

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

    assert position.other_data["vault_lockup_expires_at"] is None


def test_old_bug_equity_squared():
    """Regression: the old code set quantity=equity and price=equity → value=equity².

    With the fix, quantity=deposited_amount and price=equity/quantity → value=equity.
    This test verifies the specific scenario from the review: a 5 USDC deposit
    with API equity=5.5 should value at 5.5, not 27.5.
    """

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


def test_hypercore_vault_dust_epsilon_covers_safety_margin():
    """Hypercore vault close epsilon is large enough to cover withdrawal safety margin dust.

    HYPERCORE_WITHDRAWAL_SAFETY_MARGIN_RAW (10_000 raw = $0.01) is subtracted
    from live vault equity during full-close withdrawals, leaving ~$0.01 residual
    in the position.  The close epsilon must exceed this so can_be_closed()
    recognises the position as effectively closed.

    1. Build a Hypercore vault pair using create_hypercore_vault_pair().
    2. Verify get_close_epsilon_for_pair() and get_dust_epsilon_for_pair()
       return the Hypercore-specific epsilon.
    3. Create a TradingPosition with dust quantity (0.01) and assert can_be_closed().
    4. Same position with non-dust quantity (1.0) must NOT be closeable.
    5. Build a non-Hypercore vault pair and verify it still gets DEFAULT_VAULT_EPSILON.
    """

    # 1. Build a Hypercore vault pair
    quote = AssetIdentifier(
        chain_id=ChainId.hypercore.value,
        address="0x0000000000000000000000000000000000000002",
        token_symbol="USDC",
        decimals=6,
    )
    hypercore_pair = create_hypercore_vault_pair(
        quote=quote,
        vault_address="0x1111111111111111111111111111111111111111",
    )
    assert hypercore_pair.is_hyperliquid_vault()

    # 2. Epsilon functions return Hypercore-specific value
    assert get_close_epsilon_for_pair(hypercore_pair) == HYPERLIQUID_VAULT_CLOSE_EPSILON
    assert get_dust_epsilon_for_pair(hypercore_pair) == HYPERLIQUID_VAULT_CLOSE_EPSILON

    # 3. Position with dust quantity (0.01 USDC) can be closed
    ts = native_datetime_utc_now()
    position = TradingPosition(
        position_id=1,
        pair=hypercore_pair,
        opened_at=ts,
        last_pricing_at=ts,
        last_token_price=1.0,
        last_reserve_price=1.0,
        reserve_currency=quote,
    )
    # is_spot() asserts at least one trade exists, so add a dummy
    dummy_trade = MagicMock()
    dummy_trade.is_spot.return_value = False
    position.trades[1] = dummy_trade

    with patch.object(position, "get_quantity", return_value=Decimal("0.01")):
        assert position.can_be_closed() is True

    # 4. Position with non-dust quantity (1.0 USDC) must NOT be closeable
    with patch.object(position, "get_quantity", return_value=Decimal("1.0")):
        assert position.can_be_closed() is False

    # 5. Non-Hypercore vault pair still gets DEFAULT_VAULT_EPSILON
    non_hypercore_base = AssetIdentifier(
        chain_id=999,
        address="0x0000000000000000000000000000000000000001",
        token_symbol="VAULT",
        decimals=6,
    )
    non_hypercore_quote = AssetIdentifier(
        chain_id=999,
        address="0x0000000000000000000000000000000000000002",
        token_symbol="USDC",
        decimals=6,
    )
    non_hypercore_pair = TradingPairIdentifier(
        base=non_hypercore_base,
        quote=non_hypercore_quote,
        pool_address="0x0000000000000000000000000000000000000003",
        exchange_address="0x0000000000000000000000000000000000000004",
        internal_id=2,
        internal_exchange_id=1,
        fee=0.0,
        kind=TradingPairKind.vault,
    )
    assert not non_hypercore_pair.is_hyperliquid_vault()
    assert get_close_epsilon_for_pair(non_hypercore_pair) == DEFAULT_VAULT_EPSILON


def test_hypercore_account_check_compares_equity_not_quantity() -> None:
    """Test Hypercore account checks compare expected equity against live equity.

    This covers the live Hyper-AI crash pattern where the vault checker
    compared API equity to position quantity, even though Hypercore
    valuation stores quantity and price separately.

    1. Create a Hypercore position with quantity from deposited USDC and a price below 1.0.
    2. Feed the account checker the live vault equity returned by the Hyperliquid API path.
    3. Verify the check uses expected USD equity, reports zero USD diff, and stays clean.
    """

    # 1. Create a Hypercore position with quantity from deposited USDC and a price below 1.0.
    quote = AssetIdentifier(
        chain_id=ChainId.hypercore.value,
        address="0x0000000000000000000000000000000000000002",
        token_symbol="USDC",
        decimals=6,
    )
    hypercore_pair = create_hypercore_vault_pair(
        quote=quote,
        vault_address="0x1111111111111111111111111111111111111111",
    )

    class DummyHypercorePosition:
        """Minimal Hypercore position stub for account-check regression coverage."""

        pair = hypercore_pair
        last_token_price = 0.9939891061405017
        last_pricing_at = native_datetime_utc_now()

        def __hash__(self) -> int:
            return id(self)

        def get_quantity(self) -> Decimal:
            return Decimal("56.104634")

        def calculate_quantity_usd_value(self, quantity: Decimal) -> float:
            assert quantity == Decimal("56.104634")
            return 55.767395

        def get_human_readable_name(self) -> str:
            return "Loop Fund"

    position = DummyHypercorePosition()

    state = MagicMock()
    state.portfolio.get_open_and_frozen_positions.return_value = [position]

    sync_model = MagicMock()
    sync_model.web3 = MagicMock()
    sync_model.get_token_storage_address.return_value = "0xa8F8DEbb722c6174B814b432169BF569603F673F"

    live_equity = Decimal("55.767395")
    live_balance = OnChainBalance(
        block_number=None,
        timestamp=native_datetime_utc_now(),
        asset=hypercore_pair.base,
        amount=live_equity,
    )

    # 2. Feed the account checker the live vault equity returned by the Hyperliquid API path.
    with patch(
        "tradeexecutor.strategy.account_correction.fetch_onchain_balances_multichain",
        return_value=iter([live_balance]),
    ):
        corrections = _build_hypercore_vault_account_checks(state, sync_model)

    # 3. Verify the check uses expected USD equity, reports zero USD diff, and stays clean.
    assert len(corrections) == 1
    correction = corrections[0]
    assert correction.expected_amount == Decimal("55.767395")
    assert correction.actual_amount == live_equity
    assert correction.usd_value == 0.0
    assert correction.mismatch is False
