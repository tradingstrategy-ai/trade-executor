"""Freqtrade position state unit tests."""

import datetime
from decimal import Decimal

import pytest

from tradeexecutor.state.identifier import (
    TradingPairIdentifier,
    AssetIdentifier,
    TradingPairKind,
)
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeExecution, TradeType
from tradeexecutor.strategy.pnl import calculate_pnl_generic
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
            "freqtrade_url": "http://localhost:8080",
            "exchange": "binance",
        },
    )


def test_freqtrade_kind_is_recognized(freqtrade_pair):
    """Verify that freqtrade_spot is recognized as Freqtrade position."""
    assert freqtrade_pair.kind.is_freqtrade()
    assert not freqtrade_pair.kind.is_spot()  # Not a spot position
    assert not freqtrade_pair.kind.is_vault()


def test_freqtrade_position_is_freqtrade(freqtrade_pair):
    """Verify that TradingPosition recognizes Freqtrade type."""
    opened_at = datetime.datetime.utcnow()
    position = TradingPosition(
        1,
        freqtrade_pair,
        opened_at=opened_at,
        last_pricing_at=opened_at,
        last_token_price=1.0,
        last_reserve_price=1.0,
        reserve_currency=freqtrade_pair.quote,
    )

    # Add a trade so we can call is_spot()
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
    position.trades[1] = trade

    assert position.is_freqtrade()
    assert not position.is_vault()
    # Note: is_spot() would return False because Freqtrade is not in is_spot()'s pair check


def test_freqtrade_position_creation(freqtrade_pair):
    """Test creating a basic Freqtrade position."""
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

    assert position.position_id == 1
    assert position.pair.kind == TradingPairKind.freqtrade
    assert position.opened_at == opened_at
    assert position.is_freqtrade()


def test_freqtrade_position_valuation(freqtrade_pair):
    """Test valuation of a Freqtrade position."""
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
        planned_quantity=Decimal("1000.0"),  # Deposit 1000 USDT
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

    # Position value should be 1000 USDT
    value = position.get_value()
    assert value == 1000.0


def test_freqtrade_pnl_calculation(freqtrade_pair):
    """Test PnL calculation for Freqtrade positions."""
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

    # Deposit 1000 USDT
    trade1 = TradeExecution(
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
    trade1.started_at = opened_at
    trade1.mark_broadcasted(opened_at)
    trade1.mark_success(
        executed_at=opened_at,
        executed_price=1.0,
        executed_quantity=Decimal("1000.0"),
        executed_reserve=Decimal("1000.0"),
        lp_fees=0.0,
        native_token_price=1.0,
    )
    position.trades[1] = trade1

    # Update position price to reflect 10% profit
    now = opened_at + datetime.timedelta(days=1)
    position.last_token_price = 1.1  # 10% profit
    position.last_pricing_at = now

    pnl = calculate_pnl_generic(position, end_at=now)

    assert pnl.profit_usd == 100.0  # 1000 * 0.1
    assert pnl.profit_pct == 0.1  # 10%


def test_freqtrade_multiple_deposits(freqtrade_pair):
    """Test multiple deposits to Freqtrade position."""
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

    # First deposit: 1000 USDT
    trade1 = TradeExecution(
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
    trade1.started_at = opened_at
    trade1.mark_broadcasted(opened_at)
    trade1.mark_success(
        executed_at=opened_at,
        executed_price=1.0,
        executed_quantity=Decimal("1000.0"),
        executed_reserve=Decimal("1000.0"),
        lp_fees=0.0,
        native_token_price=1.0,
    )
    position.trades[1] = trade1

    # Second deposit: 500 USDT
    later = opened_at + datetime.timedelta(hours=1)
    trade2 = TradeExecution(
        trade_id=2,
        position_id=1,
        trade_type=TradeType.rebalance,
        pair=freqtrade_pair,
        opened_at=later,
        planned_quantity=Decimal("500.0"),
        planned_price=1.0,
        planned_reserve=Decimal("500.0"),
        reserve_currency=freqtrade_pair.quote,
    )
    trade2.started_at = later
    trade2.mark_broadcasted(later)
    trade2.mark_success(
        executed_at=later,
        executed_price=1.0,
        executed_quantity=Decimal("500.0"),
        executed_reserve=Decimal("500.0"),
        lp_fees=0.0,
        native_token_price=1.0,
    )
    position.trades[2] = trade2

    # Total quantity should be 1500
    quantity = position.get_quantity()
    assert quantity == Decimal("1500.0")

    # Position value should be 1500 USDT
    value = position.get_value()
    assert value == 1500.0


def test_freqtrade_position_with_profit(freqtrade_pair):
    """Test Freqtrade position PnL with trading profit."""
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

    # Deposit 1000 USDT
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

    # After 1 week, balance increased to 1150 (15% profit)
    week_later = opened_at + datetime.timedelta(days=7)
    position.last_token_price = 1.15  # Freqtrade bot made 15% profit
    position.last_pricing_at = week_later

    pnl = calculate_pnl_generic(position, end_at=week_later)

    assert pnl.profit_usd == 150.0  # 1000 * 0.15
    assert pnl.profit_pct == 0.15  # 15%
