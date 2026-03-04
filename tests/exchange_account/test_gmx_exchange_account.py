"""Unit tests for GMX exchange account pair creation and metadata."""

import pytest

from eth_defi.gmx.contracts import get_contract_addresses
from eth_defi.token import USDC_NATIVE_TOKEN

from tradeexecutor.exchange_account.gmx import create_gmx_exchange_account_pair
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairKind

#: Arbitrum mainnet chain ID
ARBITRUM_CHAIN_ID = 42161


@pytest.fixture()
def usdc() -> AssetIdentifier:
    return AssetIdentifier(
        chain_id=ARBITRUM_CHAIN_ID,
        address=USDC_NATIVE_TOKEN[ARBITRUM_CHAIN_ID],
        token_symbol="USDC",
        decimals=6,
    )


@pytest.fixture()
def safe_address() -> str:
    return "0x1234567890abcdef1234567890abcdef12345678"


def test_create_gmx_exchange_account_pair(usdc, safe_address):
    """Verify pair fields, kind, and is_exchange_account()."""
    pair = create_gmx_exchange_account_pair(
        quote=usdc,
        safe_address=safe_address,
    )

    assert pair.kind == TradingPairKind.exchange_account
    assert pair.is_exchange_account()
    assert pair.base.token_symbol == "GMX-ACCOUNT"
    assert pair.base.decimals == 6
    assert pair.base.chain_id == ARBITRUM_CHAIN_ID
    assert pair.quote.token_symbol == "USDC"
    assert pair.pool_address == safe_address
    assert pair.exchange_address == get_contract_addresses("arbitrum").exchangerouter
    assert pair.exchange_name == "GMX"
    assert pair.fee == 0.0


def test_gmx_pair_protocol_detection(usdc, safe_address):
    """Verify get_exchange_account_protocol() and other_data."""
    pair = create_gmx_exchange_account_pair(
        quote=usdc,
        safe_address=safe_address,
        is_testnet=True,
    )

    assert pair.get_exchange_account_protocol() == "gmx"

    config = pair.get_exchange_account_config()
    assert config["exchange_protocol"] == "gmx"
    assert config["exchange_is_testnet"] is True

    # Mainnet variant
    pair_mainnet = create_gmx_exchange_account_pair(
        quote=usdc,
        safe_address=safe_address,
        is_testnet=False,
    )
    assert pair_mainnet.get_exchange_account_config()["exchange_is_testnet"] is False
