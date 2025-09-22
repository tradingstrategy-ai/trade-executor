"""Test Orderly pricing and valuation functionality."""

import os
import pytest
import datetime
from decimal import Decimal

from eth_defi.orderly.vault import OrderlyVault
from tradeexecutor.ethereum.orderly.orderly_pricing import OrderlyPricing
from tradeexecutor.ethereum.orderly.orderly_valuation import OrderlyValuator
from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.valuation import ValuationUpdate
from tradeexecutor.strategy.trade_pricing import TradePricing


JSON_RPC_ARBITRUM_SEPOLIA = os.environ.get("JSON_RPC_ARBITRUM_SEPOLIA")
pytestmark = pytest.mark.skipif(not JSON_RPC_ARBITRUM_SEPOLIA, reason="No JSON_RPC_ARBITRUM_SEPOLIA environment variable")


def test_orderly_prices(
    web3,
    orderly_vault: OrderlyVault,
    vault_pair: TradingPairIdentifier,
):
    """Test getting buy/sell/mid price for Orderly vault deposits."""

    pricing = OrderlyPricing(
        web3=web3,
        vault=orderly_vault,
    )

    reserve_amount = Decimal("100")  # 100 USDC
    ts = datetime.datetime.utcnow()

    buy_price_info = pricing.get_buy_price(ts, vault_pair, reserve_amount)

    assert buy_price_info.price == 1.0  # 1:1 ratio for now
    assert buy_price_info.mid_price == 1.0
    assert buy_price_info.token_in == reserve_amount
    assert buy_price_info.token_out == reserve_amount  # 1:1 for Orderly
    assert buy_price_info.lp_fee == [0.0]
    assert buy_price_info.pair_fee == [0.0]
    assert isinstance(buy_price_info.read_at, datetime.datetime)

    # sell price
    shares_amount = Decimal("100")  # 100 vault shares
    price_info = pricing.get_sell_price(ts, vault_pair, shares_amount)

    assert price_info.price == 1.0  # 1:1 ratio for now
    assert price_info.mid_price == 1.0
    assert price_info.token_in == shares_amount
    assert price_info.token_out == shares_amount  # 1:1 for Orderly
    assert price_info.lp_fee == [0.0]
    assert price_info.pair_fee == [0.0]
    assert isinstance(price_info.read_at, datetime.datetime)

    # mid
    mid_price = pricing.get_mid_price(ts, vault_pair)

    assert mid_price == 1.0  # Should match the 1:1 ratio



def test_orderly_pricing_pair_fee(
    web3,
    orderly_vault: OrderlyVault,
    usdc: AssetIdentifier,
    weth: AssetIdentifier,
):
    """Test getting pair fee for Orderly vault."""

    pricing = OrderlyPricing(
        web3=web3,
        vault=orderly_vault,
    )

    # Create a mock vault trading pair
    vault_pair = TradingPairIdentifier(
        base=weth,  # Vault shares
        quote=usdc,  # USDC denomination
        pool_address="0x0EaC556c0C2321BA25b9DC01e4e3c95aD5CDCd2f",  # Vault address
        exchange_address="0x0000000000000000000000000000000000000000",
        fee=0.0,
    )

    ts = datetime.datetime.utcnow()
    fee = pricing.get_pair_fee(ts, vault_pair)

    assert fee == 0.0  # No trading fees for vault operations


def test_orderly_valuator_zero_shares(
    web3,
    orderly_vault: OrderlyVault,
    vault_pair: TradingPairIdentifier,
    usdc: AssetIdentifier,
):
    """Test valuator with zero shares (frozen position)."""

    pricing = OrderlyPricing(
        web3=web3,
        vault=orderly_vault,
    )
    valuator = OrderlyValuator(pricing)

    # Create position with zero shares
    position = TradingPosition(
        position_id=1,
        pair=vault_pair,
        opened_at=datetime.datetime.utcnow(),
        last_token_price=1.0,
        last_reserve_price=1.0,
        reserve_currency=usdc,
        last_pricing_at=datetime.datetime.utcnow(),
    )

    # Set quantity to 0
    position.trades = {}

    ts = datetime.datetime.utcnow()
    valuation_update = valuator(ts, position)

    assert isinstance(valuation_update, ValuationUpdate)
    assert valuation_update.position_id == 1
    assert valuation_update.old_value == 0
    assert valuation_update.new_value == 0
    assert valuation_update.old_price == 0
    assert valuation_update.new_price == 0
    assert valuation_update.quantity == 0
    assert position.last_token_price == 0


def test_orderly_valuator_with_shares(
    vault_pair: TradingPairIdentifier,
    mocker,
):
    """Test valuator with actual shares."""

    mock_pricing = mocker.Mock(spec=OrderlyPricing)

    # Mock the pricing response
    mock_price_structure = TradePricing(
        price=1.1,  # Slightly higher than 1:1
        mid_price=1.1,
        lp_fee=[0.0],
        pair_fee=[0.0],
        side=False,
        path=[],
        read_at=datetime.datetime.utcnow(),
        block_number=12345,
        token_in=Decimal("100"),
        token_out=Decimal("110"),
    )

    mock_pricing.get_sell_price.return_value = mock_price_structure

    valuator = OrderlyValuator(mock_pricing)

    # Create position with shares
    position = mocker.Mock(spec=TradingPosition)
    position.is_vault.return_value = True
    position.get_quantity.return_value = Decimal("100")  # 100 shares
    position.pair = vault_pair
    position.last_token_price = 1.0  # Old price
    position.get_value.return_value = 100.0  # Old value
    position.revalue_base_asset.return_value = 110.0  # New value
    position.position_id = 1

    ts = datetime.datetime.utcnow()
    valuation_update = valuator(ts, position)

    assert isinstance(valuation_update, ValuationUpdate)
    assert valuation_update.position_id == 1
    assert valuation_update.old_value == 100.0
    assert valuation_update.new_value == 110.0
    assert valuation_update.old_price == 1.0
    assert valuation_update.new_price == 1.1
    assert valuation_update.quantity == 100

    # Verify that pricing was called correctly
    mock_pricing.get_sell_price.assert_called_once_with(ts, vault_pair, Decimal("100"))

    # Verify position was updated
    position.revalue_base_asset.assert_called_once_with(ts, 1.1)
    assert position.last_token_price == 1.1
