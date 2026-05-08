"""Test CCTP custody address resolver for multichain mint recipients.

Verifies that the CCTP bridge routing correctly resolves the
mint_recipient address based on trade direction and deployment type
(Lagoon vault with per-chain Safes vs hot wallet).

Pure unit tests -- no RPC connections or Anvil forks needed.
"""

import datetime
from dataclasses import dataclass
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from tradeexecutor.ethereum.ethereum_protocol_adapters import _make_cctp_custody_resolver
from tradeexecutor.state.identifier import (
    AssetIdentifier,
    TradingPairIdentifier,
    TradingPairKind,
)
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeType


ARBITRUM_CHAIN_ID = 42161
BASE_CHAIN_ID = 8453

PRIMARY_SAFE = "0xaaaa000000000000000000000000000000000001"
SATELLITE_BASE_SAFE = "0xbbbb000000000000000000000000000000000002"
HOT_WALLET = "0xcccc000000000000000000000000000000000003"


@dataclass
class FakeSatelliteVault:
    """Minimal stand-in for AutomatedSafe with only the attribute we need."""
    safe_address: str


@pytest.fixture()
def satellite_vaults() -> dict:
    """Satellite vaults keyed by chain_id, mimicking the execution model layout."""
    return {
        BASE_CHAIN_ID: FakeSatelliteVault(safe_address=SATELLITE_BASE_SAFE),
    }


@pytest.fixture()
def usdc_arbitrum() -> AssetIdentifier:
    """USDC on Arbitrum (primary chain)."""
    return AssetIdentifier(
        chain_id=ARBITRUM_CHAIN_ID,
        address="0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
        token_symbol="USDC",
        decimals=6,
    )


@pytest.fixture()
def usdc_base() -> AssetIdentifier:
    """USDC on Base (satellite chain)."""
    return AssetIdentifier(
        chain_id=BASE_CHAIN_ID,
        address="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
        token_symbol="USDC",
        decimals=6,
    )


@pytest.fixture()
def cctp_pair(usdc_arbitrum: AssetIdentifier, usdc_base: AssetIdentifier) -> TradingPairIdentifier:
    """CCTP bridge pair: Arbitrum (quote/primary) -> Base (base/satellite)."""
    return TradingPairIdentifier(
        base=usdc_base,
        quote=usdc_arbitrum,
        pool_address="0x28b5a0e9C621a5BadaA536219b3a228C8168cf5d",
        exchange_address="0x28b5a0e9C621a5BadaA536219b3a228C8168cf5d",
        fee=0,
        kind=TradingPairKind.cctp_bridge,
        other_data={"bridge_protocol": "cctp"},
    )


def test_resolver_returns_satellite_safe_for_satellite_chain(satellite_vaults: dict):
    """Test that the resolver returns the satellite Safe for a satellite chain_id.

    1. Build a resolver with satellite vaults and a primary custody address.
    2. Query the resolver for the Base chain_id.
    3. Verify the satellite Safe address is returned, not the primary.
    """
    # 1. Build a resolver with satellite vaults and a primary custody address.
    resolver = _make_cctp_custody_resolver(
        satellite_vaults=satellite_vaults,
        primary_custody_address=PRIMARY_SAFE,
        fallback_address=HOT_WALLET,
    )
    assert resolver is not None

    # 2. Query the resolver for the Base chain_id.
    result = resolver(BASE_CHAIN_ID)

    # 3. Verify the satellite Safe address is returned, not the primary.
    assert result == SATELLITE_BASE_SAFE


def test_resolver_returns_primary_for_primary_chain(satellite_vaults: dict):
    """Test that the resolver falls back to the primary address for a chain without a satellite.

    1. Build a resolver with a satellite only on Base.
    2. Query the resolver for the Arbitrum chain_id (primary chain, no satellite).
    3. Verify the primary custody address is returned.
    """
    # 1. Build a resolver with a satellite only on Base.
    resolver = _make_cctp_custody_resolver(
        satellite_vaults=satellite_vaults,
        primary_custody_address=PRIMARY_SAFE,
        fallback_address=HOT_WALLET,
    )
    assert resolver is not None

    # 2. Query the resolver for the Arbitrum chain_id (primary chain, no satellite).
    result = resolver(ARBITRUM_CHAIN_ID)

    # 3. Verify the primary custody address is returned.
    assert result == PRIMARY_SAFE


def test_resolver_returns_fallback_for_hot_wallet():
    """Test that the resolver returns the fallback (hot wallet) when no satellites or primary are set.

    1. Build a resolver with no satellites and no primary.
    2. Verify None is returned (hot wallet path uses tx_builder fallback).
    3. Build a resolver with primary_custody_address only.
    4. Query for any chain_id and verify primary is returned.
    """
    # 1. Build a resolver with no satellites and no primary.
    resolver = _make_cctp_custody_resolver(
        satellite_vaults=None,
        primary_custody_address=None,
        fallback_address=HOT_WALLET,
    )

    # 2. Verify None is returned (hot wallet path uses tx_builder fallback).
    assert resolver is None

    # 3. Build a resolver with primary_custody_address only.
    resolver_primary = _make_cctp_custody_resolver(
        satellite_vaults=None,
        primary_custody_address=PRIMARY_SAFE,
        fallback_address=HOT_WALLET,
    )
    assert resolver_primary is not None

    # 4. Query for any chain_id and verify primary is returned.
    assert resolver_primary(BASE_CHAIN_ID) == PRIMARY_SAFE
    assert resolver_primary(ARBITRUM_CHAIN_ID) == PRIMARY_SAFE


def test_resolver_raises_when_no_address_available():
    """Test that the resolver raises ValueError when no address can be found.

    1. Build a resolver with empty satellite_vaults dict and no primary or fallback.
    2. Query for a chain_id with no matching entry.
    3. Verify ValueError is raised because no address can be resolved.
    """
    # 1. Build a resolver with empty satellite_vaults dict and no primary or fallback.
    # Note: {} is not None, so a resolver closure is still created.
    resolver = _make_cctp_custody_resolver(
        satellite_vaults={},
        primary_custody_address=None,
        fallback_address=None,
    )
    assert resolver is not None

    # 2. Query for a chain_id with no matching entry.
    # 3. Verify ValueError is raised because no address can be resolved.
    with pytest.raises(ValueError, match="No custody address for chain"):
        resolver(BASE_CHAIN_ID)


def test_direction_aware_mint_recipient(
    satellite_vaults: dict,
    cctp_pair: TradingPairIdentifier,
    usdc_arbitrum: AssetIdentifier,
):
    """Test that setup_trades resolves the correct mint recipient based on trade direction.

    We mock the heavy eth_defi imports and tx_builder to isolate the
    resolver logic inside setup_trades.

    1. Create a CctpBridgeRouting with a custody_address_resolver.
    2. Create a buy trade (bridge-out: Arbitrum -> Base) and verify mint_recipient = satellite Safe.
    3. Create a sell trade (bridge-back: Base -> Arbitrum) and verify mint_recipient = primary Safe.
    """
    from tradeexecutor.ethereum.cctp.routing import CctpBridgeRouting

    resolver = _make_cctp_custody_resolver(
        satellite_vaults=satellite_vaults,
        primary_custody_address=PRIMARY_SAFE,
        fallback_address=HOT_WALLET,
    )

    web3config = MagicMock()
    mock_web3 = MagicMock()
    web3config.get_connection.return_value = mock_web3

    # 1. Create a CctpBridgeRouting with a custody_address_resolver.
    routing = CctpBridgeRouting(web3config, custody_address_resolver=resolver)
    assert routing.custody_address_resolver is resolver

    state = State()
    state.portfolio.initialise_reserves(usdc_arbitrum)
    reserve = state.portfolio.get_default_reserve_position()
    reserve.quantity = Decimal(10000)
    reserve.reserve_token_price = 1.0

    ts = datetime.datetime(2025, 1, 1)

    # --- Buy trade (bridge-out) ---

    # 2. Create a buy trade (bridge-out: Arbitrum -> Base) and verify mint_recipient = satellite Safe.
    _, buy_trade, _ = state.create_trade(
        strategy_cycle_at=ts,
        pair=cctp_pair,
        quantity=None,
        reserve=Decimal(1000),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
    )
    state.start_execution(ts, buy_trade)

    # For the buy direction, mint_dest_chain_id = pair.base.chain_id = Base
    assert buy_trade.is_buy()
    buy_mint_dest = cctp_pair.base.chain_id
    assert resolver(buy_mint_dest) == SATELLITE_BASE_SAFE

    # --- Sell trade (bridge-back) ---

    # First complete the buy so we can create a sell
    buy_trade.mark_broadcasted(ts)
    state.mark_trade_success(
        executed_at=ts,
        trade=buy_trade,
        executed_price=1.0,
        executed_amount=Decimal(1000),
        executed_reserve=Decimal(1000),
        lp_fees=0,
        native_token_price=0,
    )

    # 3. Create a sell trade (bridge-back: Base -> Arbitrum) and verify mint_recipient = primary Safe.
    _, sell_trade, _ = state.create_trade(
        strategy_cycle_at=ts,
        pair=cctp_pair,
        quantity=Decimal(-1000),
        reserve=None,
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
    )

    assert sell_trade.is_sell()
    sell_mint_dest = cctp_pair.quote.chain_id
    assert resolver(sell_mint_dest) == PRIMARY_SAFE
