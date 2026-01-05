"""Freqtrade pricing model tests."""

import datetime
from decimal import Decimal
from unittest.mock import Mock, MagicMock

import pytest

from tradeexecutor.state.identifier import (
    TradingPairIdentifier,
    AssetIdentifier,
    TradingPairKind,
)
from tradeexecutor.strategy.freqtrade.freqtrade_pricing import FreqtradePricingModel
from tradeexecutor.strategy.freqtrade.freqtrade_client import FreqtradeClient
from tradingstrategy.chain import ChainId


@pytest.fixture
def freqtrade_pair():
    """Create a Freqtrade trading pair."""
    usdt = AssetIdentifier(
        ChainId.centralised_exchange.value,
        "0x1",
        "USDT",
        6,
        1,
    )
    ft_strategy = AssetIdentifier(
        ChainId.centralised_exchange.value,
        "0x2",
        "FT-MOMENTUM",
        6,
        2,
    )
    return TradingPairIdentifier(
        ft_strategy,
        usdt,
        "freqtrade-momentum/usdt",
        "binance",
        internal_id=1,
        internal_exchange_id=1,
        fee=0.0,
        kind=TradingPairKind.freqtrade,
        other_data={
            "freqtrade_id": "momentum-bot",
        },
    )


@pytest.fixture
def mock_clients(freqtrade_pair):
    """Create mock Freqtrade clients."""
    mock_client = Mock(spec=FreqtradeClient)
    mock_client.get_balance.return_value = {
        "total": 1000.0,
        "free": 900.0,
        "used": 100.0,
    }
    return {"momentum-bot": mock_client}


def test_freqtrade_pricing_model_creation(mock_clients):
    """Test creating a FreqtradePricingModel."""
    pricing_model = FreqtradePricingModel(mock_clients)
    assert pricing_model is not None
    assert "momentum-bot" in pricing_model.clients


def test_get_buy_price(freqtrade_pair, mock_clients):
    """Test get_buy_price returns 1.0."""
    pricing_model = FreqtradePricingModel(mock_clients)
    ts = datetime.datetime.utcnow()
    reserve = Decimal("1000.0")

    pricing = pricing_model.get_buy_price(ts, freqtrade_pair, reserve)

    assert pricing.price == 1.0
    assert pricing.mid_price == 1.0
    assert pricing.lp_fee == [0.0]
    assert pricing.pair_fee == [0.0]
    assert pricing.token_in == reserve
    assert pricing.token_out == reserve


def test_get_sell_price(freqtrade_pair, mock_clients):
    """Test get_sell_price returns 1.0."""
    pricing_model = FreqtradePricingModel(mock_clients)
    ts = datetime.datetime.utcnow()
    quantity = Decimal("1000.0")

    pricing = pricing_model.get_sell_price(ts, freqtrade_pair, quantity)

    assert pricing.price == 1.0
    assert pricing.mid_price == 1.0
    assert pricing.lp_fee == [0.0]
    assert pricing.pair_fee == [0.0]
    assert pricing.token_in == quantity
    assert pricing.token_out == quantity


def test_get_sell_price_with_none_quantity(freqtrade_pair, mock_clients):
    """Test get_sell_price with None quantity."""
    pricing_model = FreqtradePricingModel(mock_clients)
    ts = datetime.datetime.utcnow()

    pricing = pricing_model.get_sell_price(ts, freqtrade_pair, None)

    assert pricing.price == 1.0
    assert pricing.mid_price == 1.0


def test_get_mid_price(freqtrade_pair, mock_clients):
    """Test get_mid_price returns 1.0."""
    pricing_model = FreqtradePricingModel(mock_clients)
    ts = datetime.datetime.utcnow()

    price = pricing_model.get_mid_price(ts, freqtrade_pair)

    assert price == 1.0


def test_get_pair_fee(freqtrade_pair, mock_clients):
    """Test get_pair_fee returns 0.0."""
    pricing_model = FreqtradePricingModel(mock_clients)
    ts = datetime.datetime.utcnow()

    fee = pricing_model.get_pair_fee(ts, freqtrade_pair)

    assert fee == 0.0


def test_get_freqtrade_balance(freqtrade_pair, mock_clients):
    """Test querying Freqtrade balance."""
    pricing_model = FreqtradePricingModel(mock_clients)

    balance = pricing_model._get_freqtrade_balance(freqtrade_pair)

    assert balance == Decimal("1000.0")
    mock_clients["momentum-bot"].get_balance.assert_called_once()


def test_get_freqtrade_balance_api_failure(freqtrade_pair, mock_clients):
    """Test handling of API failures."""
    mock_clients["momentum-bot"].get_balance.side_effect = Exception("API error")
    pricing_model = FreqtradePricingModel(mock_clients)

    with pytest.raises(Exception):
        pricing_model._get_freqtrade_balance(freqtrade_pair)


def test_get_freqtrade_balance_missing_total(freqtrade_pair, mock_clients):
    """Test handling of missing 'total' field in response."""
    mock_clients["momentum-bot"].get_balance.return_value = {
        "free": 900.0,
        "used": 100.0,
    }
    pricing_model = FreqtradePricingModel(mock_clients)

    balance = pricing_model._get_freqtrade_balance(freqtrade_pair)

    assert balance == Decimal("0")


def test_pricing_consistent_across_buy_sell(freqtrade_pair, mock_clients):
    """Test that buy and sell prices are consistent (always 1.0)."""
    pricing_model = FreqtradePricingModel(mock_clients)
    ts = datetime.datetime.utcnow()
    reserve = Decimal("1000.0")

    buy_pricing = pricing_model.get_buy_price(ts, freqtrade_pair, reserve)
    sell_pricing = pricing_model.get_sell_price(ts, freqtrade_pair, reserve)

    assert buy_pricing.price == sell_pricing.price
    assert buy_pricing.mid_price == sell_pricing.mid_price


def test_get_buy_price_assertion_on_non_freqtrade_pair(mock_clients):
    """Test that get_buy_price asserts on non-Freqtrade pairs."""
    # Create a spot market pair
    usdt = AssetIdentifier(ChainId.ethereum.value, "0x1", "USDT", 6, 1)
    btc = AssetIdentifier(ChainId.ethereum.value, "0x2", "BTC", 8, 2)
    spot_pair = TradingPairIdentifier(
        btc, usdt, "BTC/USDT", "uniswap", kind=TradingPairKind.spot_market_hold
    )

    pricing_model = FreqtradePricingModel(mock_clients)
    ts = datetime.datetime.utcnow()

    with pytest.raises(AssertionError):
        pricing_model.get_buy_price(ts, spot_pair, Decimal("1000"))


def test_get_sell_price_assertion_on_non_freqtrade_pair(mock_clients):
    """Test that get_sell_price asserts on non-Freqtrade pairs."""
    # Create a vault pair
    usdt = AssetIdentifier(ChainId.ethereum.value, "0x1", "USDT", 6, 1)
    vault_share = AssetIdentifier(ChainId.ethereum.value, "0x3", "vUSDT", 6, 3)
    vault_pair = TradingPairIdentifier(
        vault_share, usdt, "vUSDT/USDT", "vault", kind=TradingPairKind.vault
    )

    pricing_model = FreqtradePricingModel(mock_clients)
    ts = datetime.datetime.utcnow()

    with pytest.raises(AssertionError):
        pricing_model.get_sell_price(ts, vault_pair, Decimal("1000"))
