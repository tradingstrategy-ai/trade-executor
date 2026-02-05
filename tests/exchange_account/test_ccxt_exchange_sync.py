"""Test CCXT exchange account sync with mocked balance data.

Tests verify that:
1. create_ccxt_exchange creates exchange instances correctly
2. aster_total_equity extracts totalMarginBalance from API response
3. create_ccxt_account_value_func correctly wires exchange to value extractor
4. Integration with ExchangeAccountSyncModel detects PnL changes
"""

import datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from tradeexecutor.state.state import State
from tradeexecutor.state.identifier import (
    TradingPairIdentifier,
    AssetIdentifier,
    TradingPairKind,
)
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeExecution, TradeType
from tradeexecutor.exchange_account.ccxt_exchange import (
    create_ccxt_exchange,
    aster_total_equity,
    create_ccxt_account_value_func,
)
from tradeexecutor.exchange_account.sync_model import ExchangeAccountSyncModel


@pytest.fixture
def ccxt_exchange_account_pair():
    """Create CCXT exchange account pair for testing."""
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


@pytest.fixture
def state_with_ccxt_position(ccxt_exchange_account_pair):
    """Create state with an open CCXT exchange account position (100k deposit)."""
    state = State()
    opened_at = datetime.datetime(2024, 1, 1)

    position = TradingPosition(
        position_id=1,
        pair=ccxt_exchange_account_pair,
        opened_at=opened_at,
        last_pricing_at=opened_at,
        last_token_price=1.0,
        last_reserve_price=1.0,
        reserve_currency=ccxt_exchange_account_pair.quote,
    )

    trade = TradeExecution(
        trade_id=1,
        position_id=1,
        trade_type=TradeType.rebalance,
        pair=ccxt_exchange_account_pair,
        opened_at=opened_at,
        planned_quantity=Decimal("100000.0"),
        planned_price=1.0,
        planned_reserve=Decimal("100000.0"),
        reserve_currency=ccxt_exchange_account_pair.quote,
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

    state.portfolio.open_positions[1] = position
    return state


def test_create_ccxt_exchange_valid():
    """Test creating a CCXT exchange instance with valid exchange_id."""
    pytest.importorskip("ccxt")
    exchange = create_ccxt_exchange("aster", {
        "apiKey": "test-key",
        "secret": "test-secret",
    })
    assert exchange is not None


def test_create_ccxt_exchange_invalid():
    """Test that invalid exchange_id raises ValueError."""
    pytest.importorskip("ccxt")
    with pytest.raises(ValueError, match="Unknown CCXT exchange"):
        create_ccxt_exchange("nonexistent_exchange_xyz")


def test_aster_total_equity_extracts_field():
    """Test that aster_total_equity extracts totalMarginBalance correctly."""
    mock_exchange = MagicMock()
    mock_exchange.fapiPrivateGetV4Account.return_value = {
        "totalWalletBalance": "50000.00",
        "totalUnrealizedProfit": "2500.00",
        "totalMarginBalance": "52500.00",
        "totalInitialMargin": "10000.00",
        "availableBalance": "42500.00",
        "assets": [],
    }

    result = aster_total_equity(mock_exchange)

    assert result == Decimal("52500.00")
    mock_exchange.fapiPrivateGetV4Account.assert_called_once()


def test_aster_total_equity_missing_field():
    """Test that aster_total_equity raises KeyError when field is missing."""
    mock_exchange = MagicMock()
    mock_exchange.fapiPrivateGetV4Account.return_value = {
        "totalWalletBalance": "50000.00",
        "assets": [],
    }

    with pytest.raises(KeyError, match="totalMarginBalance"):
        aster_total_equity(mock_exchange)


def test_aster_total_equity_api_error():
    """Test that API errors propagate from aster_total_equity."""
    mock_exchange = MagicMock()
    mock_exchange.fapiPrivateGetV4Account.side_effect = Exception("API timeout")

    with pytest.raises(Exception, match="API timeout"):
        aster_total_equity(mock_exchange)


def test_create_ccxt_account_value_func_returns_value(ccxt_exchange_account_pair):
    """Test that create_ccxt_account_value_func returns correct account value."""
    mock_exchange = MagicMock()
    mock_exchange.id = "aster"
    mock_exchange.fapiPrivateGetV4Account.return_value = {
        "totalMarginBalance": "52500.00",
    }

    exchanges = {"aster_main": mock_exchange}
    account_value_func = create_ccxt_account_value_func(exchanges)

    result = account_value_func(ccxt_exchange_account_pair)
    assert result == Decimal("52500.00")


def test_create_ccxt_account_value_func_unknown_account_id(ccxt_exchange_account_pair):
    """Test that unknown ccxt_account_id raises KeyError."""
    exchanges = {}
    account_value_func = create_ccxt_account_value_func(exchanges)

    with pytest.raises(KeyError, match="aster_main"):
        account_value_func(ccxt_exchange_account_pair)


def test_create_ccxt_account_value_func_custom_extractor(ccxt_exchange_account_pair):
    """Test that custom value_extractor is used when provided."""
    mock_exchange = MagicMock()
    mock_exchange.id = "custom"

    def custom_extractor(exchange):
        return Decimal("99999.99")

    exchanges = {"aster_main": mock_exchange}
    account_value_func = create_ccxt_account_value_func(
        exchanges, value_extractor=custom_extractor,
    )

    result = account_value_func(ccxt_exchange_account_pair)
    assert result == Decimal("99999.99")


def test_create_ccxt_account_value_func_api_error(ccxt_exchange_account_pair):
    """Test that API errors propagate from the account value function."""
    mock_exchange = MagicMock()
    mock_exchange.id = "aster"
    mock_exchange.fapiPrivateGetV4Account.side_effect = Exception("Connection refused")

    exchanges = {"aster_main": mock_exchange}
    account_value_func = create_ccxt_account_value_func(exchanges)

    with pytest.raises(Exception, match="Connection refused"):
        account_value_func(ccxt_exchange_account_pair)


def test_ccxt_with_sync_model_detects_profit(
    state_with_ccxt_position,
    ccxt_exchange_account_pair,
):
    """Test full integration: CCXT account value func + ExchangeAccountSyncModel detects profit."""
    mock_exchange = MagicMock()
    mock_exchange.id = "aster"
    mock_exchange.fapiPrivateGetV4Account.return_value = {
        "totalMarginBalance": "105000.00",
    }

    exchanges = {"aster_main": mock_exchange}
    account_value_func = create_ccxt_account_value_func(exchanges)

    sync_model = ExchangeAccountSyncModel(account_value_func)
    events = sync_model.sync_positions(
        timestamp=datetime.datetime.utcnow(),
        state=state_with_ccxt_position,
        strategy_universe=None,
        pricing_model=None,
    )

    assert len(events) == 1
    evt = events[0]
    assert evt.quantity == Decimal("5000.00")
    assert evt.old_balance == Decimal("100000.0")
    assert "ccxt" in evt.notes.lower()


def test_ccxt_with_sync_model_detects_loss(
    state_with_ccxt_position,
    ccxt_exchange_account_pair,
):
    """Test full integration: CCXT account value func + ExchangeAccountSyncModel detects loss."""
    mock_exchange = MagicMock()
    mock_exchange.id = "aster"
    mock_exchange.fapiPrivateGetV4Account.return_value = {
        "totalMarginBalance": "93000.00",
    }

    exchanges = {"aster_main": mock_exchange}
    account_value_func = create_ccxt_account_value_func(exchanges)

    sync_model = ExchangeAccountSyncModel(account_value_func)
    events = sync_model.sync_positions(
        timestamp=datetime.datetime.utcnow(),
        state=state_with_ccxt_position,
        strategy_universe=None,
        pricing_model=None,
    )

    assert len(events) == 1
    evt = events[0]
    assert evt.quantity == Decimal("-7000.00")


def test_ccxt_with_sync_model_pnl_tracking(
    state_with_ccxt_position,
    ccxt_exchange_account_pair,
):
    """Test that consecutive syncs correctly track cumulative PnL."""
    mock_exchange = MagicMock()
    mock_exchange.id = "aster"

    exchanges = {"aster_main": mock_exchange}
    account_value_func = create_ccxt_account_value_func(exchanges)
    sync_model = ExchangeAccountSyncModel(account_value_func)

    # First sync: profit of 5k
    mock_exchange.fapiPrivateGetV4Account.return_value = {
        "totalMarginBalance": "105000.00",
    }
    events1 = sync_model.sync_positions(
        timestamp=datetime.datetime.utcnow(),
        state=state_with_ccxt_position,
        strategy_universe=None,
        pricing_model=None,
    )
    assert len(events1) == 1
    assert events1[0].quantity == Decimal("5000.00")

    # Second sync: additional profit of 3k (total 108k)
    mock_exchange.fapiPrivateGetV4Account.return_value = {
        "totalMarginBalance": "108000.00",
    }
    events2 = sync_model.sync_positions(
        timestamp=datetime.datetime.utcnow(),
        state=state_with_ccxt_position,
        strategy_universe=None,
        pricing_model=None,
    )
    assert len(events2) == 1
    assert events2[0].quantity == Decimal("3000.00")

    # Position quantity should now be 108k
    position = state_with_ccxt_position.portfolio.open_positions[1]
    assert position.get_quantity() == Decimal("108000.00")
