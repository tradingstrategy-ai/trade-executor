"""Test HypercoreVaultPricing.get_usd_tvl() against the live Hyperliquid API.

Validates that the pricing model can fetch TVL from the Hyperliquid
``vaultDetails`` API when no replay ``market_data_source`` is configured,
which is the code path used in live trading.

This test was added because the live trading path crashed with
``NotImplementedError`` — ``get_usd_tvl()`` only supported the replay
data source used in backtesting, but not the live API fallback.

1. Create a HypercoreVaultPricing without market_data_source (live mode)
2. Call get_usd_tvl() on the HLP vault via the Hyperliquid API
3. Verify TVL is positive and reasonable (HLP is the largest vault)
4. Verify USDTVLSizeRiskModel works end-to-end with the live pricing model
5. Verify size risk capping produces sensible results
"""

import os
from decimal import Decimal

import pytest
from eth_defi.compat import native_datetime_utc_now
from eth_defi.token import USDC_NATIVE_TOKEN

from tradeexecutor.ethereum.vault.hypercore_valuation import HypercoreVaultPricing
from tradeexecutor.ethereum.vault.hypercore_vault import (
    HLP_VAULT_ADDRESS,
    create_hypercore_vault_pair,
)
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.strategy.tvl_size_risk import USDTVLSizeRiskModel

pytestmark = pytest.mark.skipif(
    not os.environ.get("HYPERCORE_WRITER_TEST_PRIVATE_KEY"),
    reason="HYPERCORE_WRITER_TEST_PRIVATE_KEY not set",
)

CHAIN_ID = 999


@pytest.fixture()
def usdc_asset() -> AssetIdentifier:
    """USDC AssetIdentifier for HyperEVM mainnet."""
    return AssetIdentifier(
        chain_id=CHAIN_ID,
        address=USDC_NATIVE_TOKEN[CHAIN_ID],
        token_symbol="USDC",
        decimals=6,
    )


@pytest.fixture()
def hlp_pair(usdc_asset: AssetIdentifier):
    """HLP vault TradingPairIdentifier for mainnet."""
    return create_hypercore_vault_pair(
        quote=usdc_asset,
        vault_address=HLP_VAULT_ADDRESS["mainnet"],
    )


@pytest.fixture()
def live_pricing() -> HypercoreVaultPricing:
    """Pricing model without market_data_source, simulating live trading."""
    return HypercoreVaultPricing(
        value_func=None,
        market_data_source=None,
    )


def test_get_usd_tvl_live(
    live_pricing: HypercoreVaultPricing,
    hlp_pair,
):
    """Verify get_usd_tvl fetches TVL from the Hyperliquid API in live mode.

    1. Create a pricing model without market_data_source (live trading path)
    2. Call get_usd_tvl() on the HLP vault
    3. Assert TVL is positive and above 1M USD (HLP is the largest Hyperliquid vault)
    """
    ts = native_datetime_utc_now()
    tvl = live_pricing.get_usd_tvl(ts, hlp_pair)
    assert tvl > 1_000_000, f"HLP TVL should be above 1M USD, got {tvl:,.2f}"


def test_size_risk_model_with_live_tvl(
    live_pricing: HypercoreVaultPricing,
    hlp_pair,
):
    """Verify USDTVLSizeRiskModel works end-to-end with live Hypercore TVL.

    1. Create a USDTVLSizeRiskModel with 20% per-position cap
    2. Ask for a position size of 100 USD (well within cap)
    3. Assert full amount is accepted
    4. Ask for a position size larger than 20% of TVL
    5. Assert the position is capped to 20% of TVL
    """
    ts = native_datetime_utc_now()
    size_risk_model = USDTVLSizeRiskModel(
        pricing_model=live_pricing,
        per_position_cap=0.20,
    )

    # 2-3. Small position — should be fully accepted
    small_result = size_risk_model.get_acceptable_size_for_position(
        ts, hlp_pair, asked_value=100.0,
    )
    assert small_result.accepted_size == pytest.approx(100.0)
    assert not small_result.capped
    assert small_result.tvl > 1_000_000

    # 4-5. Absurdly large position — should be capped at 20% of TVL
    huge_ask = small_result.tvl * 10
    capped_result = size_risk_model.get_acceptable_size_for_position(
        ts, hlp_pair, asked_value=huge_ask,
    )
    assert capped_result.capped
    assert capped_result.accepted_size == pytest.approx(small_result.tvl * 0.20, rel=0.01)
