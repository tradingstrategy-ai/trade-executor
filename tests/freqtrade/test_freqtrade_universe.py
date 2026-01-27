"""Test Freqtrade universe helper functions.

Tests for create_freqtrade_pair() and load_freqtrade_bots().
"""

import os

import pytest

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairKind
from tradeexecutor.strategy.freqtrade.universe import create_freqtrade_pair, load_freqtrade_bots
from tradingstrategy.chain import ChainId


@pytest.fixture
def usdc_asset():
    """USDC asset for testing."""
    return AssetIdentifier(
        chain_id=ChainId.polygon.value,
        address="0x2791bca1f2de4661ed88a30c99a7a9449aa84174",
        token_symbol="USDC",
        decimals=6,
    )


def test_create_freqtrade_pair_on_chain_transfer(usdc_asset):
    """Test creating a Freqtrade pair with on-chain transfer deposit method."""
    pair = create_freqtrade_pair(
        freqtrade_id="momentum-bot",
        api_url="http://localhost:8080",
        exchange_name="binance",
        reserve_currency=usdc_asset,
        deposit_method="on_chain_transfer",
        recipient_address="0xabcdef0123456789abcdef0123456789abcdef01",
    )

    # Verify basic pair structure
    assert pair.kind == TradingPairKind.freqtrade
    assert pair.is_freqtrade()
    assert pair.base.token_symbol == "MOMENTUM-BOT"
    assert pair.quote == usdc_asset
    assert pair.fee == 0.0

    # Verify other_data contains Freqtrade config
    assert pair.other_data["freqtrade_id"] == "momentum-bot"
    assert pair.other_data["freqtrade_api_url"] == "http://localhost:8080"
    assert pair.other_data["freqtrade_exchange"] == "binance"
    assert pair.other_data["freqtrade_deposit_method"] == "on_chain_transfer"
    assert pair.other_data["freqtrade_recipient_address"] == "0xabcdef0123456789abcdef0123456789abcdef01"

    # Verify helper methods work
    assert pair.get_freqtrade_id() == "momentum-bot"
    assert pair.get_freqtrade_api_url() == "http://localhost:8080"
    assert pair.get_freqtrade_deposit_method() == "on_chain_transfer"


def test_create_freqtrade_pair_aster_vault(usdc_asset):
    """Test creating a Freqtrade pair with Aster vault deposit method."""
    pair = create_freqtrade_pair(
        freqtrade_id="aster-bot",
        api_url="http://localhost:8081",
        exchange_name="aster",
        reserve_currency=usdc_asset,
        deposit_method="aster_vault",
        vault_address="0x1234567890abcdef1234567890abcdef12345678",
        broker_id=5,
    )

    assert pair.kind == TradingPairKind.freqtrade
    assert pair.other_data["freqtrade_deposit_method"] == "aster_vault"
    assert pair.other_data["freqtrade_vault_address"] == "0x1234567890abcdef1234567890abcdef12345678"
    assert pair.other_data["freqtrade_broker_id"] == 5


def test_create_freqtrade_pair_hyperliquid(usdc_asset):
    """Test creating a Freqtrade pair with Hyperliquid deposit method."""
    pair = create_freqtrade_pair(
        freqtrade_id="hyperliquid-bot",
        api_url="http://localhost:8082",
        exchange_name="hyperliquid",
        reserve_currency=usdc_asset,
        deposit_method="hyperliquid",
        vault_address="0xabcdef1234567890abcdef1234567890abcdef12",
        is_mainnet=True,
    )

    assert pair.kind == TradingPairKind.freqtrade
    assert pair.other_data["freqtrade_deposit_method"] == "hyperliquid"
    assert pair.other_data["freqtrade_vault_address"] == "0xabcdef1234567890abcdef1234567890abcdef12"
    assert pair.other_data["freqtrade_is_mainnet"] is True


def test_create_freqtrade_pair_orderly(usdc_asset):
    """Test creating a Freqtrade pair with Orderly vault deposit method."""
    pair = create_freqtrade_pair(
        freqtrade_id="orderly-bot",
        api_url="http://localhost:8083",
        exchange_name="orderly",
        reserve_currency=usdc_asset,
        deposit_method="orderly_vault",
        vault_address="0x9876543210fedcba9876543210fedcba98765432",
        orderly_account_id="0x1111111111111111111111111111111111111111111111111111111111111111",
        broker_id="woofi_pro",
        token_id="USDC",
    )

    assert pair.kind == TradingPairKind.freqtrade
    assert pair.other_data["freqtrade_deposit_method"] == "orderly_vault"
    assert pair.other_data["freqtrade_vault_address"] == "0x9876543210fedcba9876543210fedcba98765432"
    assert pair.other_data["freqtrade_orderly_account_id"] == "0x1111111111111111111111111111111111111111111111111111111111111111"
    assert pair.other_data["freqtrade_broker_id"] == "woofi_pro"
    assert pair.other_data["freqtrade_token_id"] == "USDC"


def test_get_freqtrade_config_with_env_vars(usdc_asset):
    """Test get_freqtrade_config() retrieves credentials from environment variables."""
    # Set environment variables
    os.environ["FREQTRADE_MOMENTUM_BOT_USERNAME"] = "testuser"
    os.environ["FREQTRADE_MOMENTUM_BOT_PASSWORD"] = "testpass"

    try:
        pair = create_freqtrade_pair(
            freqtrade_id="momentum-bot",
            api_url="http://localhost:8080",
            exchange_name="binance",
            reserve_currency=usdc_asset,
            deposit_method="on_chain_transfer",
            recipient_address="0xabcdef0123456789abcdef0123456789abcdef01",
        )

        config = pair.get_freqtrade_config()

        assert config is not None
        assert config["freqtrade_id"] == "momentum-bot"
        assert config["api_url"] == "http://localhost:8080"
        assert config["api_username"] == "testuser"
        assert config["api_password"] == "testpass"
        assert config["exchange_name"] == "binance"
        assert config["deposit_method"] == "on_chain_transfer"
        assert config["recipient_address"] == "0xabcdef0123456789abcdef0123456789abcdef01"

    finally:
        # Clean up environment variables
        del os.environ["FREQTRADE_MOMENTUM_BOT_USERNAME"]
        del os.environ["FREQTRADE_MOMENTUM_BOT_PASSWORD"]


def test_get_freqtrade_config_missing_env_vars(usdc_asset):
    """Test get_freqtrade_config() returns None credentials when env vars are missing."""
    # Ensure env vars are not set
    for key in ["FREQTRADE_TEST_BOT_USERNAME", "FREQTRADE_TEST_BOT_PASSWORD"]:
        if key in os.environ:
            del os.environ[key]

    pair = create_freqtrade_pair(
        freqtrade_id="test-bot",
        api_url="http://localhost:8080",
        exchange_name="binance",
        reserve_currency=usdc_asset,
        deposit_method="on_chain_transfer",
        recipient_address="0xabcdef0123456789abcdef0123456789abcdef01",
    )

    config = pair.get_freqtrade_config()

    assert config is not None
    assert config["api_username"] is None
    assert config["api_password"] is None


def test_load_freqtrade_bots(usdc_asset):
    """Test load_freqtrade_bots() creates multiple Freqtrade pairs from config."""
    freqtrade_bots = [
        {
            "freqtrade_id": "momentum-bot",
            "api_url": "http://localhost:8080",
            "exchange_name": "binance",
            "deposit_method": "on_chain_transfer",
            "recipient_address": "0xabcdef0123456789abcdef0123456789abcdef01",
        },
        {
            "freqtrade_id": "aster-bot",
            "api_url": "http://localhost:8081",
            "exchange_name": "aster",
            "deposit_method": "aster_vault",
            "vault_address": "0x1234567890abcdef1234567890abcdef12345678",
            "broker_id": 0,
        },
    ]

    pairs = load_freqtrade_bots(freqtrade_bots, usdc_asset)

    assert len(pairs) == 2

    # Verify first pair (on-chain transfer)
    assert pairs[0].get_freqtrade_id() == "momentum-bot"
    assert pairs[0].get_freqtrade_deposit_method() == "on_chain_transfer"
    assert pairs[0].other_data["freqtrade_recipient_address"] == "0xabcdef0123456789abcdef0123456789abcdef01"

    # Verify second pair (Aster vault)
    assert pairs[1].get_freqtrade_id() == "aster-bot"
    assert pairs[1].get_freqtrade_deposit_method() == "aster_vault"
    assert pairs[1].other_data["freqtrade_vault_address"] == "0x1234567890abcdef1234567890abcdef12345678"
    assert pairs[1].other_data["freqtrade_broker_id"] == 0


def test_load_freqtrade_bots_with_custom_timeouts(usdc_asset):
    """Test load_freqtrade_bots() with custom timeout and polling settings."""
    freqtrade_bots = [
        {
            "freqtrade_id": "custom-bot",
            "api_url": "http://localhost:8080",
            "exchange_name": "binance",
            "deposit_method": "on_chain_transfer",
            "recipient_address": "0xabcdef0123456789abcdef0123456789abcdef01",
            "fee_tolerance": "2.0",
            "confirmation_timeout": 300,
            "poll_interval": 5,
        },
    ]

    pairs = load_freqtrade_bots(freqtrade_bots, usdc_asset)

    assert len(pairs) == 1
    assert pairs[0].other_data["freqtrade_fee_tolerance"] == "2.0"
    assert pairs[0].other_data["freqtrade_confirmation_timeout"] == 300
    assert pairs[0].other_data["freqtrade_poll_interval"] == 5
