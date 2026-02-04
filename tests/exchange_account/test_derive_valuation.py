"""Test exchange account valuation model with Derive testnet.

This test uses the Derive.xyz API with credentials to verify
that the valuation model correctly reads account values.

Required environment variables:

- ``DERIVE_OWNER_PRIVATE_KEY``: Owner wallet private key (from web UI wallet)
- ``DERIVE_SESSION_PRIVATE_KEY``: Session key private key (from testnet developer page)
- ``DERIVE_WALLET_ADDRESS``: Derive wallet address (optional, auto-derived from owner key)

See tests/derive/derive-test-key-setup.md for detailed instructions.
"""

import datetime
import os
from decimal import Decimal

import pytest
from eth_account import Account

from eth_defi.derive.account import fetch_account_summary, fetch_subaccount_ids
from eth_defi.derive.authentication import DeriveApiClient
from eth_defi.derive.onboarding import fetch_derive_wallet_address

from tradeexecutor.state.identifier import (
    TradingPairIdentifier,
    AssetIdentifier,
    TradingPairKind,
)
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeExecution, TradeType
from tradeexecutor.exchange_account.pricing import ExchangeAccountPricingModel
from tradeexecutor.exchange_account.valuation import ExchangeAccountValuator
from tradeexecutor.exchange_account.derive import create_derive_account_value_func


# Skip tests if credentials not configured
pytestmark = pytest.mark.skipif(
    not os.environ.get("DERIVE_OWNER_PRIVATE_KEY") or not os.environ.get("DERIVE_SESSION_PRIVATE_KEY"),
    reason="Set DERIVE_OWNER_PRIVATE_KEY and DERIVE_SESSION_PRIVATE_KEY for Derive tests. See tests/derive/derive-test-key-setup.md",
)


@pytest.fixture(scope="module")
def owner_account():
    """Owner wallet from environment."""
    return Account.from_key(os.environ["DERIVE_OWNER_PRIVATE_KEY"])


@pytest.fixture(scope="module")
def derive_wallet_address(owner_account):
    """Derive wallet address from env, or auto-derived from owner key."""
    env_address = os.environ.get("DERIVE_WALLET_ADDRESS")
    if env_address:
        return env_address
    return fetch_derive_wallet_address(owner_account.address, is_testnet=True)


@pytest.fixture(scope="module")
def authenticated_client(owner_account, derive_wallet_address):
    """Derive API client authenticated with session key.

    Automatically resolves the first subaccount ID from the API.
    """
    client = DeriveApiClient(
        owner_account=owner_account,
        derive_wallet_address=derive_wallet_address,
        is_testnet=True,
        session_key_private=os.environ["DERIVE_SESSION_PRIVATE_KEY"],
    )
    # Resolve subaccount ID from the API
    ids = fetch_subaccount_ids(client)
    if ids:
        client.subaccount_id = ids[0]
    return client


@pytest.fixture
def exchange_account_pair(authenticated_client):
    """Create an exchange account pair for Derive testnet."""
    # Use a generic chain ID for the synthetic assets
    chain_id = 901  # Derive testnet chain ID

    usdc = AssetIdentifier(
        chain_id=chain_id,
        address="0x0000000000000000000000000000000000000001",
        token_symbol="USDC",
        decimals=6,
    )
    derive_account = AssetIdentifier(
        chain_id=chain_id,
        address="0x0000000000000000000000000000000000000002",
        token_symbol="DERIVE-ACCOUNT",
        decimals=6,
    )
    return TradingPairIdentifier(
        base=derive_account,
        quote=usdc,
        pool_address="0x0000000000000000000000000000000000000003",
        exchange_address="0x0000000000000000000000000000000000000004",
        internal_id=1,
        internal_exchange_id=1,
        fee=0.0,
        kind=TradingPairKind.exchange_account,
        exchange_name="derive",
        other_data={
            "exchange_protocol": "derive",
            "exchange_subaccount_id": authenticated_client.subaccount_id,
            "exchange_is_testnet": True,
        },
    )


@pytest.fixture
def position_with_deposit(exchange_account_pair):
    """Create a position representing capital deposited to Derive.

    The testnet account has been funded with 100k USDC.
    """
    opened_at = datetime.datetime(2024, 1, 1)
    position = TradingPosition(
        position_id=1,
        pair=exchange_account_pair,
        opened_at=opened_at,
        last_pricing_at=opened_at,
        last_token_price=1.0,
        last_reserve_price=1.0,
        reserve_currency=exchange_account_pair.quote,
    )

    # Add initial deposit trade (100k USDC matches testnet funding)
    trade = TradeExecution(
        trade_id=1,
        position_id=1,
        trade_type=TradeType.rebalance,
        pair=exchange_account_pair,
        opened_at=opened_at,
        planned_quantity=Decimal("100000.0"),
        planned_price=1.0,
        planned_reserve=Decimal("100000.0"),
        reserve_currency=exchange_account_pair.quote,
    )
    trade.started_at = opened_at
    trade.mark_broadcasted(opened_at)
    trade.mark_success(
        executed_at=opened_at,
        executed_price=1.0,
        executed_quantity=Decimal("100000.0"),
        executed_reserve=Decimal("100000.0"),
        lp_fees=0.0,
        native_token_price=1.0,
    )
    position.trades[1] = trade
    return position


def test_derive_valuation_reads_account_value(
    authenticated_client,
    exchange_account_pair,
    position_with_deposit,
):
    """Test that valuation model correctly reads Derive account value.

    This test verifies the full valuation flow:
    1. Create account value function for Derive
    2. Create pricing and valuation models
    3. Revalue position using Derive API
    4. Verify the returned value matches expected testnet funding
    """
    # Skip if no subaccounts
    ids = fetch_subaccount_ids(authenticated_client)
    if not ids:
        pytest.skip("Account has no subaccounts yet")

    # Create account value function for Derive
    clients = {authenticated_client.subaccount_id: authenticated_client}
    account_value_func = create_derive_account_value_func(clients)

    # Create pricing and valuation models
    pricing_model = ExchangeAccountPricingModel(account_value_func)
    valuator = ExchangeAccountValuator(pricing_model)

    # Revalue position
    ts = datetime.datetime.utcnow()
    update = valuator(ts, position_with_deposit)

    # Verify valuation came from API
    # Testnet account has been funded with 100k USDC
    assert update.new_value >= 100_000, f"Expected at least 100k USD, got {update.new_value}"
    assert update.new_price == 1.0, "Price should be 1.0 for USD denominated"
    assert isinstance(update.quantity, Decimal), "Quantity should be Decimal"
    assert update.quantity >= 100_000, f"Expected at least 100k quantity, got {update.quantity}"


def test_derive_valuation_updates_position_timestamp(
    authenticated_client,
    exchange_account_pair,
    position_with_deposit,
):
    """Test that valuation updates position.last_pricing_at."""
    ids = fetch_subaccount_ids(authenticated_client)
    if not ids:
        pytest.skip("Account has no subaccounts yet")

    clients = {authenticated_client.subaccount_id: authenticated_client}
    account_value_func = create_derive_account_value_func(clients)
    pricing_model = ExchangeAccountPricingModel(account_value_func)
    valuator = ExchangeAccountValuator(pricing_model)

    old_pricing_at = position_with_deposit.last_pricing_at
    ts = datetime.datetime.utcnow()
    valuator(ts, position_with_deposit)

    assert position_with_deposit.last_pricing_at == ts
    assert position_with_deposit.last_pricing_at != old_pricing_at


def test_derive_account_value_func_direct(authenticated_client, exchange_account_pair):
    """Test the account value function directly without valuation model."""
    ids = fetch_subaccount_ids(authenticated_client)
    if not ids:
        pytest.skip("Account has no subaccounts yet")

    clients = {authenticated_client.subaccount_id: authenticated_client}
    account_value_func = create_derive_account_value_func(clients)

    value = account_value_func(exchange_account_pair)

    assert isinstance(value, Decimal), f"Expected Decimal, got {type(value)}"
    assert value >= 100_000, f"Expected at least 100k USD, got {value}"
