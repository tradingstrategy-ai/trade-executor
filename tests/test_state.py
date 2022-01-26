"""Test trade execution state management."""
import datetime
from decimal import Decimal

import pytest

from tradeexecutor.state.state import State, AssetIdentifier, TradingPairIdentifier, ReservePosition, TradeType, \
    TradeStatus
from tradingstrategy.chain import ChainId


@pytest.fixture
def usdc() -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(ChainId.ethereum, "0x0", "USDC", 6)


@pytest.fixture
def weth() -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(ChainId.ethereum, "0x1", "WETH", 18)


@pytest.fixture
def weth_usdc(usdc, weth) -> TradingPairIdentifier:
    """Mock some assets"""
    return TradingPairIdentifier(weth, usdc, 1, "0x2")


@pytest.fixture
def start_ts(usdc, weth) -> datetime.datetime:
    """Timestamp of action started"""
    return datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc)


def test_empty_state():
    """"Create new empty trade executor state."""
    state = State()
    assert state.is_empty()


def test_update_reserves(usdc, weth, weth_usdc, start_ts):
    """Set currency reserves for a portfolio."""
    state = State()
    state.update_reserves([ReservePosition(usdc, Decimal(1000), 1.0, start_ts)])
    assert state.portfolio.get_current_cash() == 1_000
    assert state.portfolio.get_total_equity() == 1_000


def test_single_buy(usdc, weth, weth_usdc, start_ts):
    """Do a single token purchase."""
    state = State()
    state.update_reserves([ReservePosition(usdc, Decimal(1000), 1.0, start_ts)])

    # #1 Planning stage
    # Buy 0.1 ETH at 1700 USD/ETH
    position, trade = state.create_trade(
        ts=start_ts,
        pair=weth_usdc,
        quantity=Decimal("0.1"),
        assumed_price=1700,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc)
    assert trade.get_status() == TradeStatus.planned
    assert position.has_planned_trades()
    assert state.portfolio.get_current_cash() == 1_000
    assert state.portfolio.get_total_equity() == 1_000

    # #2 Capital allocation
    txid = "0xffffff"
    nonce = 1
    ts = start_ts + datetime.timedelta(minutes=1)
    state.start_execution(ts, trade, txid, nonce)

    assert trade.get_status() == TradeStatus.started
    assert trade.reserve_currency_allocated == 170
    assert position.has_unexecuted_trades()
    assert not position.has_planned_trades()
    assert state.portfolio.get_current_cash() == 830  # Trades being executed do not show in the portfolio value
    assert state.portfolio.get_total_equity() == 1000  # Trades being executed do not show in the portfolio value

    # #3 broadcast
    ts = ts + datetime.timedelta(minutes=1)
    state.mark_broadcasted(ts, trade)
    assert state.portfolio.get_current_cash() == 830  # Trades being executed do not show in the portfolio value
    assert state.portfolio.get_total_equity() == 1000  # Trades being executed do not show in the portfolio value

    # #4 success
    ts = ts + datetime.timedelta(minutes=1)
    executed_price = 1690
    executed_quantity = Decimal("0.09")
    lp_fees = 2.50  # $2.5

    gas_units_consumed = 150_000  # 150k gas units per swap
    gas_price = 15 * 10**9  # 15 Gwei/gas unit
    native_token_price = 1.9  # 1.9 USD/ETH
    state.mark_trade_success(ts, trade, executed_price, executed_quantity, lp_fees, gas_price, gas_units_consumed, native_token_price)

    value_after_trade = 1690 * 0.09
    assert trade.get_status() == TradeStatus.success
    assert trade.get_value() == pytest.approx(value_after_trade)
    assert trade.reserve_currency_allocated == 0
    assert trade.get_gas_fees_paid() == pytest.approx(0.004274999999999999)
    assert trade.get_fees_paid() == pytest.approx(0.004274999999999999 + 2.50)

    assert not position.has_unexecuted_trades()
    assert not position.has_planned_trades()
    assert state.portfolio.get_current_cash() == 830  # Trades being executed do not show in the portfolio value
    assert state.portfolio.get_total_equity() == 982.1  # Trades being executed do not show in the portfolio value

    assert len(state.portfolio.open_positions) == 1