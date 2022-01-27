"""Test trade execution state management."""
import datetime
from decimal import Decimal

import pytest

from tradeexecutor.state.state import State, AssetIdentifier, TradingPairIdentifier, ReservePosition, TradeType, \
    TradeStatus, Portfolio, TradeExecution, TradingPosition
from tradeexecutor.utils.testtrader import TestTrader
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
    return TradingPairIdentifier(weth, usdc, "0x2")


@pytest.fixture
def start_ts(usdc, weth) -> datetime.datetime:
    """Timestamp of action started"""
    return datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc)


@pytest.fixture
def single_asset_portfolio(start_ts, weth_usdc, weth, usdc) -> Portfolio:
    """Creates a mock portfolio that holds some reserve currency and WETH"""
    p = Portfolio()

    reserve = ReservePosition(usdc, Decimal(500), 1.0, start_ts)
    p.reserves[reserve.get_identifier()] = reserve

    trade = TradeExecution(
        trade_id = 1,
        position_id =1,
        trade_type = TradeType.rebalance,
        pair=weth_usdc,
        opened_at = start_ts,
        planned_quantity = Decimal(0.1),
        planned_price=1670,
        planned_reserve=Decimal(167),
        reserve_currency = usdc,
        started_at = start_ts,
        reserve_currency_allocated = 167,
        broadcasted_at =start_ts,
        executed_at = start_ts,
        executed_price=1660,
        executed_quantity=Decimal(0.095),
        lp_fees_paid =2.5,
        gas_units_consumed=150_000,
        gas_price=15,
        native_token_price=1.9,
        # Blockchain bookkeeping
        txid="0x01",
        nonce=1,
    )

    assert trade.is_buy()
    assert trade.is_success()

    position = TradingPosition(
        position_id=1,
        pair=weth_usdc,
        opened_at=start_ts,
        last_token_price=1660,
        last_reserve_price=1,
        last_pricing_at=start_ts,
        reserve_currency=usdc,
        trades={1: trade},
        next_trade_id=2,
    )

    p.open_positions = {1: position}
    return p



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
        reserve_currency=usdc,
        reserve_currency_price=1.0)

    assert trade.get_status() == TradeStatus.planned
    assert trade.planned_reserve == Decimal("170")
    assert trade.planned_quantity == Decimal("0.1")
    assert trade.planned_price == 1700
    assert trade.get_planned_value() == 170

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
    assert trade.get_value() == 170
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
    assert state.portfolio.get_total_equity() == 983  # Trades being executed do not show in the portfolio value7

    assert len(state.portfolio.open_positions) == 1


def test_single_sell_all(usdc, weth, weth_usdc, start_ts, single_asset_portfolio):
    """Sell the single open ETH position in the portfolio."""
    state = State(portfolio=single_asset_portfolio)

    # 0: Check the starting state
    assert state.portfolio.get_current_cash() == 500
    assert state.portfolio.get_total_equity() == 657.7
    assert state.portfolio.get_open_position_equity() == 157.7
    eth_quantity = state.portfolio.open_positions[1].get_equity_for_position()
    assert eth_quantity == Decimal("0.09500000000000000111022302463")
    assert state.portfolio.open_positions[1].next_trade_id == 2

    # #1 Planning stage
    # Sell all ETH at 1700 USD/ETH
    position, trade = state.create_trade(
        ts=start_ts,
        pair=weth_usdc,
        quantity=-eth_quantity,
        assumed_price=1700,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        reserve_currency_price=1.0)

    # State and position is currently updated
    assert position == state.portfolio.open_positions[1]
    assert position.position_id == 1
    assert trade.trade_id == 2
    assert position.next_trade_id == 3
    assert len(state.portfolio.open_positions) == 1
    assert len(state.portfolio.open_positions[1].trades) == 2
    assert trade.is_sell()
    assert trade.get_status() == TradeStatus.planned
    assert position.has_planned_trades()
    assert state.portfolio.get_current_cash() == 500
    assert state.portfolio.get_total_equity() == 657.7

    # #2 preparation
    txid = "0xffffff"
    nonce = 2
    ts = start_ts + datetime.timedelta(minutes=1)
    state.start_execution(ts, trade, txid, nonce)
    assert state.portfolio.get_current_cash() == 500
    assert state.portfolio.get_total_equity() == 657.7
    assert state.portfolio.open_positions[1].trades[2].txid == txid
    assert state.portfolio.open_positions[1].trades[2].nonce == nonce
    assert state.portfolio.open_positions[1].trades[2].started_at == ts
    assert state.portfolio.open_positions[1].trades[2].is_started()
    assert trade.planned_quantity == -eth_quantity
    assert trade.planned_reserve == 0
    assert trade.reserve_currency_allocated is None

    # #3 broadcast
    ts = ts + datetime.timedelta(minutes=1)
    state.mark_broadcasted(ts, trade)
    assert state.portfolio.get_current_cash() == 500
    assert state.portfolio.get_total_equity() == 657.7

    # #4 success
    ts = ts + datetime.timedelta(minutes=1)
    executed_price = 1640.0
    executed_quantity = -eth_quantity
    executed_reserve = eth_quantity * Decimal(executed_price)
    lp_fees = 2.50  # $2.5

    gas_units_consumed = 150_000  # 150k gas units per swap
    gas_price = 15 * 10**9  # 15 Gwei/gas unit
    native_token_price = 1.9  # 1.9 USD/ETH
    state.mark_trade_success(ts, trade, executed_price, executed_quantity, executed_reserve, lp_fees, gas_price, gas_units_consumed, native_token_price)

    value_after_trade = 1640 * float(eth_quantity)
    assert trade.get_status() == TradeStatus.success
    assert trade.executed_price == executed_price
    assert trade.executed_quantity == -eth_quantity
    assert trade.executed_reserve == executed_reserve
    assert trade.get_equity_for_position() == -eth_quantity
    assert trade.executed_quantity == -eth_quantity
    assert trade.executed_reserve == executed_reserve
    assert trade.get_executed_value() == pytest.approx(value_after_trade)
    assert trade.get_executed_value() == pytest.approx(float(executed_reserve))
    assert trade.get_value() == pytest.approx(value_after_trade)
    assert trade.reserve_currency_allocated == 0
    assert trade.get_gas_fees_paid() == pytest.approx(0.004274999999999999)
    assert trade.get_fees_paid() == pytest.approx(0.004274999999999999 + 2.50)

    # Position is properly finished
    assert not position.has_unexecuted_trades()
    assert not position.has_planned_trades()
    assert position.get_value() == 0

    # We lost some money in the trade, so equity is now lower
    # We originally bought ETH at 1690
    # now sold at 1640
    assert state.portfolio.get_current_cash() == 655.8
    assert state.portfolio.get_total_equity() == 655.8
    assert state.portfolio.get_open_position_equity() == 0

    assert len(state.portfolio.open_positions) == 0
    assert len(state.portfolio.closed_positions) == 1


def test_buy_buy_sell_sell(usdc, weth, weth_usdc, start_ts):
    """Execute four trades on a position."""

    state = State()
    state.update_reserves([ReservePosition(usdc, Decimal(1000), 1.0, start_ts)])
    trader = TestTrader(state)

    # 0: start
    assert state.portfolio.get_total_equity() == 1000.0

    # 1: buy 1
    position, trade = trader.buy(weth_usdc, Decimal(0.1), 1700)
    assert state.portfolio.get_total_equity() == 998.3
    assert position.get_equity_for_position() == pytest.approx(Decimal(0.099))

    # 2: buy 2
    position, trade = trader.buy(weth_usdc, Decimal(0.1), 1700)
    assert position.get_equity_for_position() == pytest.approx(Decimal(0.099 * 2))
    assert len(position.trades) == 2
    assert position.trades[1].get_position_quantity() == pytest.approx(Decimal(0.1 * 0.99))
    assert position.trades[2].get_position_quantity() == pytest.approx(Decimal(0.1 * 0.99))
    assert position.trades[1].get_executed_value() == pytest.approx(166.617)
    assert position.trades[2].get_executed_value() == pytest.approx(166.617)

    # Portfolio valuation still uses 1700 USD/ETH because
    # it is not automatically revalued on loss on trade execution and this is why
    # it seems the portfolio value is growing when we buy more ETH
    assert position.last_token_price == 1700
    assert state.portfolio.get_total_equity() == 996.6

    assert len(state.portfolio.open_positions) == 1

    # Sell all accrued tokens in two trades
    half_1 = position.get_equity_for_position() / 2
    half_2 = position.get_equity_for_position() - half_1

    position, trade = trader.sell(weth_usdc, half_1, 1700)
    assert trade.executed_quantity == -half_1
    assert trade.get_equity_for_position() == -half_1
    assert position.get_equity_for_position() == half_2

    trader.sell(weth_usdc, half_2, 1700)

    assert position.get_equity_for_position() == 0
    assert len(state.portfolio.open_positions) == 0
    assert state.portfolio.get_total_equity() == pytest.approx(993.234)

    # All done
    assert len(position.trades) == 4
    assert len(state.portfolio.open_positions) == 0



