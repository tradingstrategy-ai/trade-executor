"""Test ERC-4626 vault price reading and estimation."""
from decimal import Decimal

import pytest
from tradeexecutor.ethereum.vault.vault_live_pricing import VaultPricing
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing
from tradeexecutor.state.identifier import TradingPairIdentifier


@pytest.fixture()
def vault_pricing(
    pricing_model: GenericPricing,
    ipor_usdc: TradingPairIdentifier
) -> VaultPricing:
    """Create a vault pricing model fixture."""

    vault_pricing = pricing_model.pair_configurator.get_pricing(ipor_usdc)
    assert isinstance(vault_pricing, VaultPricing)
    return vault_pricing


def test_vault_estimate_buy(
    vault_pricing,
    ipor_usdc: TradingPairIdentifier
):
    """Estimate buy price / deosit estimate for a vault."""

    estimate = vault_pricing.get_buy_price(
        ts=None,
        pair=ipor_usdc,
        reserve=Decimal("100.00")
    )
    assert estimate.block_number > 0
    # Price of one share
    assert estimate.mid_price == pytest.approx(1.0335669634763602)  # We use forked by block mainnet


def test_vault_estimate_sell(
    vault_pricing,
    ipor_usdc: TradingPairIdentifier
):
    """Estimate sell price / redeem estimate for a vault."""

    estimate = vault_pricing.get_sell_price(
        ts=None,
        pair=ipor_usdc,
        quantity=Decimal("100.00")
    )
    assert estimate.block_number > 0
    # Price of one share
    assert estimate.mid_price == pytest.approx(1.0335669634763602)  # We use forked by block mainnet


def test_vault_mid_price(
    vault_pricing,
    ipor_usdc: TradingPairIdentifier
):
    """Because there is no trading fees, vaults mid price is same as share price."""

    mid_price = vault_pricing.get_mid_price(
        ts=None,
        pair=ipor_usdc,
    )
    assert mid_price == pytest.approx(1.0335669634763602)  # We use forked by block mainnet


def test_vault_tvl(
    vault_pricing,
    ipor_usdc: TradingPairIdentifier
):
    """Get the real-time TVL of ERC-4626 vault."""
    tvl = vault_pricing.get_usd_tvl(
        timestamp=None,
        pair=ipor_usdc,
    )
    assert tvl == pytest.approx(1437072.77357)  # We use forked by block mainnet
