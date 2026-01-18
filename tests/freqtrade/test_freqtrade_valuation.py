"""Freqtrade valuation model tests."""

import datetime
from decimal import Decimal
from unittest.mock import Mock

import pytest

from tradeexecutor.state.identifier import (
    TradingPairIdentifier,
    AssetIdentifier,
    TradingPairKind,
)
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeExecution, TradeType
from tradeexecutor.strategy.freqtrade.freqtrade_pricing import FreqtradePricingModel
from tradeexecutor.strategy.freqtrade.freqtrade_valuation import FreqtradeValuator
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
def mock_pricing_model(freqtrade_pair):
    """Create a mock pricing model."""
    mock_client = Mock(spec=FreqtradeClient)
    mock_client.get_balance.return_value = {
        "total": 1000.0,
        "free": 900.0,
        "used": 100.0,
    }
    clients = {"momentum-bot": mock_client}
    return FreqtradePricingModel(clients)


@pytest.fixture
def position_with_deposit(freqtrade_pair):
    """Create a position with a completed deposit trade."""
    opened_at = datetime.datetime(2024, 1, 1)
    position = TradingPosition(
        1,
        freqtrade_pair,
        opened_at=opened_at,
        last_pricing_at=opened_at,
        last_token_price=1.0,
        last_reserve_price=1.0,
        reserve_currency=freqtrade_pair.quote,
    )

    # Add deposit trade
    trade = TradeExecution(
        trade_id=1,
        position_id=1,
        trade_type=TradeType.rebalance,
        pair=freqtrade_pair,
        opened_at=opened_at,
        planned_quantity=Decimal("1000.0"),
        planned_price=1.0,
        planned_reserve=Decimal("1000.0"),
        reserve_currency=freqtrade_pair.quote,
    )
    trade.started_at = opened_at
    trade.mark_broadcasted(opened_at)
    trade.mark_success(
        executed_at=opened_at,
        executed_price=1.0,
        executed_quantity=Decimal("1000.0"),
        executed_reserve=Decimal("1000.0"),
        lp_fees=0.0,
        native_token_price=1.0,
    )
    position.trades[1] = trade
    return position


def test_valuation_balance_drift_warning(
    freqtrade_pair, mock_pricing_model, position_with_deposit, caplog
):
    """Test that balance drift >1% triggers warning."""
    # Set API to return much higher balance (>1% drift)
    mock_pricing_model.clients["momentum-bot"].get_balance.return_value = {
        "total": 1200.0,  # 20% drift from tracked 1000
        "free": 1150.0,
        "used": 50.0,
    }

    valuator = FreqtradeValuator(mock_pricing_model)
    ts = datetime.datetime(2024, 1, 2)

    with caplog.at_level("WARNING"):
        valuator(ts, position_with_deposit)

    # Check that warning was logged
    assert "Balance drift detected" in caplog.text
    assert "position 1" in caplog.text


def test_valuation_api_failure_graceful_handling(
    freqtrade_pair, mock_pricing_model, position_with_deposit
):
    """Test graceful handling when Freqtrade API is unreachable."""
    # Make API call fail
    mock_pricing_model.clients["momentum-bot"].get_balance.side_effect = Exception(
        "API unreachable"
    )

    valuator = FreqtradeValuator(mock_pricing_model)
    ts = datetime.datetime(2024, 1, 2)
    update = valuator(ts, position_with_deposit)

    # Should return no-change valuation
    assert update.old_value == update.new_value
    assert update.quantity == Decimal("1000.0")
