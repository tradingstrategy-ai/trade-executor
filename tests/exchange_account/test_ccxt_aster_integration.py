"""Integration test for CCXT Aster exchange account.

Requires real Aster API credentials. Skipped in CI.

Set ASTER_API_KEY and ASTER_API_SECRET environment variables to run.
"""

import os
from decimal import Decimal

import pytest

from tradeexecutor.exchange_account.ccxt_exchange import (
    create_ccxt_exchange,
    aster_total_equity,
    create_ccxt_account_value_func,
)
from tradeexecutor.state.identifier import (
    TradingPairIdentifier,
    AssetIdentifier,
    TradingPairKind,
)


pytestmark = pytest.mark.skipif(
    not os.environ.get("ASTER_API_KEY") or not os.environ.get("ASTER_API_SECRET"),
    reason="Set ASTER_API_KEY and ASTER_API_SECRET for Aster integration tests",
)


@pytest.fixture
def aster_exchange():
    """Create authenticated Aster exchange via CCXT."""
    return create_ccxt_exchange("aster", {
        "apiKey": os.environ["ASTER_API_KEY"],
        "secret": os.environ["ASTER_API_SECRET"],
    })


@pytest.fixture
def aster_exchange_account_pair():
    """Create Aster exchange account pair for testing."""
    usdc = AssetIdentifier(
        chain_id=1,
        address="0x0000000000000000000000000000000000000001",
        token_symbol="USDC",
        decimals=6,
    )
    aster_account = AssetIdentifier(
        chain_id=1,
        address="0x0000000000000000000000000000000000000002",
        token_symbol="ASTER-ACCOUNT",
        decimals=6,
    )
    return TradingPairIdentifier(
        base=aster_account,
        quote=usdc,
        pool_address="0x0000000000000000000000000000000000000003",
        exchange_address="0x0000000000000000000000000000000000000004",
        internal_id=1,
        internal_exchange_id=1,
        fee=0.0,
        kind=TradingPairKind.exchange_account,
        exchange_name="ccxt_aster",
        other_data={
            "exchange_protocol": "ccxt",
            "ccxt_account_id": "aster_main",
            "ccxt_exchange_id": "aster",
            "exchange_is_testnet": False,
        },
    )


def test_aster_total_equity_returns_value(aster_exchange):
    """Test that aster_total_equity returns a valid Decimal from real API."""
    value = aster_total_equity(aster_exchange)
    assert isinstance(value, Decimal)
    assert value >= 0


def test_ccxt_account_value_func_with_real_aster(
    aster_exchange,
    aster_exchange_account_pair,
):
    """Test full account value function with real Aster API."""
    exchanges = {"aster_main": aster_exchange}
    account_value_func = create_ccxt_account_value_func(exchanges)

    value = account_value_func(aster_exchange_account_pair)
    assert isinstance(value, Decimal)
    assert value >= 0
