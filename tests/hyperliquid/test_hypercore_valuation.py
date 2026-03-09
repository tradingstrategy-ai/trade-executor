"""Test Hypercore vault valuation and pricing models against HyperEVM mainnet.

Uses the Trial 3 deployment (Safe-Hypercore-Writer-trials.md) which has
a real HLP deposit on mainnet. Validates that the valuation model returns
positive equity from the Hyperliquid info API.

Environment variables:
    HYPERCORE_WRITER_TEST_PRIVATE_KEY: deployer key for Trial 3 Lagoon vault
"""

import datetime
import os
from decimal import Decimal

import pytest
from eth_defi.compat import native_datetime_utc_now
from eth_defi.hyperliquid.api import fetch_user_vault_equity
from eth_defi.hyperliquid.session import (
    HYPERLIQUID_API_URL,
    create_hyperliquid_session,
)
from eth_defi.token import USDC_NATIVE_TOKEN

from tradeexecutor.ethereum.vault.hypercore_valuation import (
    HypercoreVaultPricing,
    HypercoreVaultValuator,
)
from tradeexecutor.ethereum.vault.hypercore_vault import (
    HLP_VAULT_ADDRESS,
    create_hypercore_vault_pair,
    create_hypercore_vault_value_func,
)
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeExecution, TradeType

pytestmark = pytest.mark.skipif(
    not os.environ.get("HYPERCORE_WRITER_TEST_PRIVATE_KEY"),
    reason="HYPERCORE_WRITER_TEST_PRIVATE_KEY not set",
)

#: Trial 3 Safe address on HyperEVM mainnet
TRIAL3_SAFE = "0x7bEfA4a93A5c19b578b14A1BC5c4CaBb0B8D7991"
CHAIN_ID = 999


@pytest.fixture()
def session():
    """Hyperliquid mainnet API session."""
    return create_hyperliquid_session(api_url=HYPERLIQUID_API_URL)


@pytest.fixture()
def usdc_asset():
    """USDC AssetIdentifier for HyperEVM mainnet."""
    return AssetIdentifier(
        chain_id=CHAIN_ID,
        address=USDC_NATIVE_TOKEN[CHAIN_ID],
        token_symbol="USDC",
        decimals=6,
    )


@pytest.fixture()
def vault_pair(usdc_asset):
    """HLP vault TradingPairIdentifier for mainnet."""
    return create_hypercore_vault_pair(
        quote=usdc_asset,
        vault_address=HLP_VAULT_ADDRESS["mainnet"],
    )


@pytest.fixture()
def value_func(session):
    """Hypercore vault value function for Trial 3 Safe on mainnet."""
    return create_hypercore_vault_value_func(
        session=session,
        safe_address=TRIAL3_SAFE,
        is_testnet=False,
    )


def test_fetch_vault_equity_directly(session):
    """Test that the Hyperliquid API returns equity for the Trial 3 Safe."""
    eq = fetch_user_vault_equity(
        session,
        user=TRIAL3_SAFE,
        vault_address=HLP_VAULT_ADDRESS["mainnet"],
    )
    assert eq is not None, \
        f"No vault equity found for Safe {TRIAL3_SAFE} in HLP"
    assert eq.equity > 0, \
        f"Expected positive equity, got {eq.equity}"


def test_value_func_returns_positive_equity(value_func, vault_pair):
    """Test that create_hypercore_vault_value_func returns positive equity."""
    equity = value_func(vault_pair)
    assert isinstance(equity, Decimal)
    assert equity > 0, f"Expected positive equity, got {equity}"


def test_pricing_model(value_func, vault_pair):
    """Test HypercoreVaultPricing returns valid prices from the API."""
    pricing = HypercoreVaultPricing(value_func)
    ts = native_datetime_utc_now()

    buy_price = pricing.get_buy_price(ts, vault_pair, reserve=Decimal(5))
    assert buy_price.price > 0

    sell_price = pricing.get_sell_price(ts, vault_pair, quantity=Decimal(5))
    assert sell_price.price > 0

    mid_price = pricing.get_mid_price(ts, vault_pair)
    assert mid_price > 0

    assert pricing.get_pair_fee(ts, vault_pair) == 0.0


def test_valuation_model(value_func, vault_pair):
    """Test HypercoreVaultValuator revalues a position with real API data."""
    ts = native_datetime_utc_now()

    # Create a mock position with a known initial value
    position = TradingPosition(
        position_id=1,
        pair=vault_pair,
        opened_at=ts,
        last_pricing_at=ts,
        last_token_price=1.0,
        last_reserve_price=1.0,
        reserve_currency=vault_pair.quote,
    )
    trade = TradeExecution(
        trade_id=1,
        position_id=1,
        trade_type=TradeType.rebalance,
        pair=vault_pair,
        opened_at=ts,
        planned_quantity=Decimal(5),
        planned_price=1.0,
        planned_reserve=Decimal(5),
        reserve_currency=vault_pair.quote,
    )
    trade.started_at = ts
    trade.mark_broadcasted(ts)
    trade.mark_success(
        executed_at=ts,
        executed_price=1.0,
        executed_quantity=Decimal(5),
        executed_reserve=Decimal(5),
        lp_fees=0.0,
        native_token_price=0.0,
    )
    position.trades[1] = trade

    valuator = HypercoreVaultValuator(value_func)
    evt = valuator(ts, position)

    assert evt.new_price > 0, f"Expected positive price, got {evt.new_price}"
    assert evt.new_value > 0, f"Expected positive value, got {evt.new_value}"
    assert position.last_token_price > 0
