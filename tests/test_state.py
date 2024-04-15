"""Test trade execution state management.

TODO: Clean txid and nonce references properly.
"""
import datetime
from decimal import Decimal
from typing import Tuple
import numpy as np
import pandas as pd

import pytest
from hexbytes import HexBytes

from tradeexecutor.state.valuation import ValuationUpdate
from tradeexecutor.strategy.trade_pricing import TradePricing
from tradeexecutor.monkeypatch.dataclasses_json import patch_dataclasses_json
from tradeexecutor.state.state import State, TradeType
from tradeexecutor.state.portfolio import NotEnoughMoney, TooSmallTrade
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeExecution, TradeStatus
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction, solidity_arg_encoder
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.validator import validate_nested_state_dict, BadStateData
from tradeexecutor.statistics.core import update_statistics
from tradeexecutor.statistics.statistics_table import serialise_long_short_stats_as_json_table
from tradeexecutor.strategy.valuation import revalue_state
from tradeexecutor.testing.unit_test_trader import UnitTestTrader
from tradingstrategy.chain import ChainId
from tradingstrategy.types import USDollarAmount
from tradeexecutor.strategy.execution_context import ExecutionMode



@pytest.fixture
def mock_exchange_address() -> str:
    """Mock some assets"""
    return "0x1"


@pytest.fixture
def usdc() -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(ChainId.ethereum.value, "0x0", "USDC", 6)


@pytest.fixture
def weth() -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(ChainId.ethereum.value, "0x1", "WETH", 18)


@pytest.fixture
def aave() -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(ChainId.ethereum.value, "0x3", "AAVE", 18)


@pytest.fixture
def weth_usdc(mock_exchange_address, usdc, weth) -> TradingPairIdentifier:
    """Mock some assets"""
    return TradingPairIdentifier(weth, usdc, "0x4", mock_exchange_address, internal_id=1)


@pytest.fixture
def aave_usdc(mock_exchange_address, usdc, aave) -> TradingPairIdentifier:
    """Mock some assets"""
    return TradingPairIdentifier(aave, usdc, "0x5", mock_exchange_address, internal_id=2)


@pytest.fixture
def start_ts(usdc, weth) -> datetime.datetime:
    """Timestamp of action started"""
    return datetime.datetime(2022, 1, 1, tzinfo=None)


@pytest.fixture
def single_asset_portfolio(start_ts, weth_usdc, weth, usdc) -> Portfolio:
    """Creates a mock portfolio that holds some reserve currency and WETH"""
    p = Portfolio()

    reserve = ReservePosition(usdc, Decimal(500), start_ts, 1.0, start_ts)
    p.reserves[reserve.get_identifier()] = reserve

    trade = TradeExecution(
        trade_id = 1,
        position_id =1,
        trade_type = TradeType.rebalance,
        pair=weth_usdc,
        opened_at = start_ts,
        planned_quantity = Decimal(0.1),
        planned_price=1670.0,
        planned_reserve=Decimal(167),
        reserve_currency = usdc,
        started_at = start_ts,
        reserve_currency_allocated = 167,
        broadcasted_at =start_ts,
        executed_at = start_ts,
        executed_price=1660,
        executed_quantity=Decimal(0.095),
        lp_fees_paid =2.5,
        native_token_price=1.9,
    )

    tx = BlockchainTransaction(
        tx_hash=HexBytes("0x01"),
        nonce=1,
        realised_gas_units_consumed=150_000,
        realised_gas_price=15,
    )
    trade.blockchain_transactions = [tx]

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
    )

    p.open_positions = {1: position}

    p.next_trade_id = 2

    return p


def test_empty_state():
    """"Create new empty trade executor state."""
    state = State()
    assert state.is_empty()
    state.perform_integrity_check()


def test_update_reserves(usdc, weth, weth_usdc, start_ts):
    """Set currency reserves for a portfolio."""
    state = State()
    state.update_reserves([ReservePosition(usdc, Decimal(1000), start_ts, 1.0, start_ts)])
    assert state.portfolio.get_cash() == 1_000
    assert state.portfolio.get_total_equity() == 1_000


def test_single_buy(usdc, weth, weth_usdc, start_ts):
    """Do a single token purchase."""
    state = State()
    state.update_reserves([ReservePosition(usdc, Decimal(1000), start_ts, 1.0, start_ts)])

    # #1 Planning stage
    # Buy 0.1 ETH at 1700 USD/ETH
    position, trade, created = state.create_trade(
        strategy_cycle_at=start_ts,
        pair=weth_usdc,
        quantity=Decimal("0.1"),
        reserve=None,
        assumed_price=1700,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        reserve_currency_price=1.0,
        slippage_tolerance=0.01,
    )

    assert trade.get_status() == TradeStatus.planned
    assert trade.planned_reserve == Decimal("170")
    assert trade.planned_quantity == Decimal("0.1")
    assert trade.planned_price == 1700
    assert trade.get_planned_value() == 170
    assert trade.slippage_tolerance == 0.01

    asset_deltas = trade.calculate_asset_deltas()
    assert asset_deltas[0].is_spending()
    assert asset_deltas[0].asset == usdc.address
    assert asset_deltas[0].raw_amount == -(170 * 10**6)
    assert asset_deltas[1].asset == weth.address
    assert asset_deltas[1].is_incoming()
    assert asset_deltas[1].raw_amount == pytest.approx(98999999999999999)

    assert position.has_planned_trades()
    assert state.portfolio.get_cash() == 1_000
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
    assert state.portfolio.get_cash() == 830  # Trades being executed do not show in the portfolio value
    assert state.portfolio.get_total_equity() == 1000  # Trades being executed do not show in the portfolio value

    # #3 broadcast
    ts = ts + datetime.timedelta(minutes=1)
    state.mark_broadcasted(ts, trade)
    assert state.portfolio.get_cash() == 830  # Trades being executed do not show in the portfolio value
    assert state.portfolio.get_total_equity() == 1000  # Trades being executed do not show in the portfolio value

    # #4 success
    ts = ts + datetime.timedelta(minutes=1)
    executed_price = 1690.0
    executed_quantity = Decimal("0.09")
    lp_fees = 2.50  # $2.5

    gas_units_consumed = 150_000  # 150k gas units per swap
    gas_price = 15 * 10**9  # 15 Gwei/gas unit
    native_token_price = 1.9  # 1.9 USD/ETH
    state.mark_trade_success(ts, trade, executed_price, executed_quantity, 0, lp_fees, native_token_price)

    value_after_trade = 1690 * 0.09
    assert trade.get_status() == TradeStatus.success
    assert trade.get_value() == pytest.approx(value_after_trade)
    assert trade.reserve_currency_allocated == 0
    assert trade.get_fees_paid() == pytest.approx(2.50)

    assert not position.has_unexecuted_trades()
    assert not position.has_planned_trades()
    assert state.portfolio.get_cash() == 830  # Trades being executed do not show in the portfolio value
    assert state.portfolio.get_total_equity() == 983  # Trades being executed do not show in the portfolio value7

    assert len(state.portfolio.open_positions) == 1


def test_single_sell_all(usdc, weth, weth_usdc, start_ts, single_asset_portfolio):
    """Sell the single open ETH position in the portfolio."""
    state = State(portfolio=single_asset_portfolio)

    # 0: Check the starting state
    assert state.portfolio.get_cash() == 500
    assert state.portfolio.get_total_equity() == 657.7
    assert state.portfolio.get_position_equity_and_loan_nav() == 157.7
    eth_quantity = state.portfolio.open_positions[1].get_quantity_old()
    assert eth_quantity == Decimal("0.09500000000000000111022302463")

    # #1 Planning stage
    # Sell all ETH at 1700 USD/ETH
    position, trade, created = state.create_trade(
        strategy_cycle_at=start_ts,
        pair=weth_usdc,
        quantity=-eth_quantity,
        reserve=None,
        assumed_price=1700,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        reserve_currency_price=1.0,
        slippage_tolerance=0.01,
    )

    # State and position is currently updated
    assert position == state.portfolio.open_positions[1]
    assert position.position_id == 1
    assert trade.trade_id == 2
    assert len(state.portfolio.open_positions) == 1
    assert len(state.portfolio.open_positions[1].trades) == 2
    assert trade.is_sell()
    assert trade.get_status() == TradeStatus.planned
    assert position.has_planned_trades()
    assert state.portfolio.get_cash() == 500
    assert state.portfolio.get_total_equity() == 657.7

    asset_deltas = trade.calculate_asset_deltas()
    assert asset_deltas[0].is_spending()
    assert asset_deltas[0].asset == weth.address
    assert asset_deltas[0].raw_amount == pytest.approx(-0.09500000000000000111022302463 * 10**18)
    assert asset_deltas[1].asset == usdc.address
    assert asset_deltas[1].is_incoming()
    assert asset_deltas[1].raw_amount == pytest.approx(159885000)  # USD 159

    # #2 preparation
    txid = "0xffffff"
    nonce = 2
    ts = start_ts + datetime.timedelta(minutes=1)
    state.start_execution(ts, trade, txid, nonce)
    assert state.portfolio.get_cash() == 500
    assert state.portfolio.get_total_equity() == 657.7
    assert state.portfolio.open_positions[1].trades[2].txid == txid
    assert state.portfolio.open_positions[1].trades[2].nonce == nonce
    assert state.portfolio.open_positions[1].trades[2].started_at == ts
    assert state.portfolio.open_positions[1].trades[2].is_started()
    assert trade.planned_quantity == -eth_quantity
    assert trade.reserve_currency_allocated is None

    # #3 broadcast
    ts = ts + datetime.timedelta(minutes=1)
    state.mark_broadcasted(ts, trade)
    assert state.portfolio.get_cash() == 500
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
    state.mark_trade_success(ts, trade, executed_price, executed_quantity, executed_reserve, lp_fees, native_token_price)

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
    assert trade.get_fees_paid() == pytest.approx(2.50)

    # Position is properly finished
    assert not position.has_unexecuted_trades()
    assert not position.has_planned_trades()
    assert position.get_value() == 0

    # We lost some money in the trade, so equity is now lower
    # We originally bought ETH at 1690
    # now sold at 1640
    assert state.portfolio.get_cash() == 655.8
    assert state.portfolio.get_total_equity() == 655.8
    assert state.portfolio.get_position_equity_and_loan_nav() == 0

    assert len(state.portfolio.open_positions) == 0
    assert len(state.portfolio.closed_positions) == 1


def test_buy_buy_sell_sell(usdc, weth, weth_usdc, start_ts):
    """Execute four trades on a position."""

    state = State()
    state.update_reserves([ReservePosition(usdc, Decimal(1000), start_ts, 1.0, start_ts)])
    trader = UnitTestTrader(state)

    # 0: start
    assert state.portfolio.get_total_equity() == 1000.0

    # 1: buy 1
    position, trade = trader.buy(weth_usdc, Decimal(0.1), 1700)
    assert state.portfolio.get_total_equity() == 998.3
    assert position.get_quantity_old() == pytest.approx(Decimal(0.099))

    # 2: buy 2
    position, trade = trader.buy(weth_usdc, Decimal(0.1), 1700)
    assert position.get_quantity_old() == pytest.approx(Decimal(0.099 * 2))
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
    half_1 = position.get_quantity_old() / 2
    half_2 = position.get_quantity_old() - half_1

    position, trade = trader.sell(weth_usdc, half_1, 1700)
    assert trade.executed_quantity == -half_1
    assert trade.get_equity_for_position() == -half_1
    assert position.get_quantity_old() == half_2

    trader.sell(weth_usdc, half_2, 1700)

    assert position.get_quantity_old() == 0
    assert len(state.portfolio.open_positions) == 0
    assert state.portfolio.get_total_equity() == pytest.approx(993.234)

    # All done
    assert len(position.trades) == 4
    assert len(state.portfolio.open_positions) == 0


def test_buy_sell_two_positions(usdc, weth_usdc, aave_usdc, start_ts):
    """Open two parallel positions."""

    state = State()
    state.update_reserves([ReservePosition(usdc, Decimal(1000), start_ts, 1.0, start_ts)])
    trader = UnitTestTrader(state)

    # 0: start
    assert state.portfolio.get_total_equity() == 1000.0

    # 1: buy token 1
    position, trade = trader.buy(weth_usdc, Decimal(0.1), 1700)
    assert state.portfolio.get_total_equity() == 998.3
    assert position.get_quantity_old() == pytest.approx(Decimal(0.099))

    # 2: buy  token 3
    position, trade = trader.buy(aave_usdc, Decimal(0.5), 200)
    assert position.get_quantity_old() == pytest.approx(Decimal(0.5 * 0.99))

    assert len(state.portfolio.open_positions) == 2
    assert state.portfolio.get_total_equity() == 997.3

    # 3: sell both
    portfolio = state.portfolio
    trader.sell(weth_usdc, portfolio.get_equity_for_pair(weth_usdc), 1700)
    trader.sell(aave_usdc, portfolio.get_equity_for_pair(aave_usdc), 200)

    assert len(state.portfolio.open_positions) == 0
    assert state.portfolio.get_total_equity() == pytest.approx(994.627)


def test_statistics(usdc, weth_usdc, aave_usdc, start_ts):
    """Open and close two parallel positions and calculate statistics for them."""

    state = State()
    state.update_reserves([ReservePosition(
        usdc, 
        Decimal(1000), 
        start_ts, 
        1.0, 
        start_ts,
        initial_deposit=Decimal(1000),
        initial_deposit_reserve_token_price=1.0,
    )])
    trader = UnitTestTrader(state)

    # 0: start
    assert state.portfolio.get_total_equity() == 1000.0
    portfolio = state.portfolio

    # 1: buy token 1
    trader.buy(weth_usdc, Decimal(0.1), 1700)
    trader.buy(aave_usdc, Decimal(0.5), 200)

    long_short_metrics_latest = serialise_long_short_stats_as_json_table(
        state, None
    )
    
    update_statistics(datetime.datetime.utcnow(), state.stats, portfolio, ExecutionMode.real_trading, long_short_metrics_latest=long_short_metrics_latest)
    
    stats = state.stats
    portfolio_stats = stats.get_latest_portfolio_stats()
    summary = portfolio_stats.summary
    
    # no closed positions
    assert stats.long_short_metrics_latest["live_stats"].rows['total_positions'].value['All'] == '0' 
    assert stats.long_short_metrics_latest["backtested_stats"].rows == {}
    
    for key, value in stats.long_short_metrics_latest["live_stats"].rows.items():
        all_value = value.value['All']
        short_value = value.value['Short']
        long_value = value.value['Long']
        assert isinstance(all_value, str | None), f"Should be string. Got {type(all_value)} for key: {key}"
        assert isinstance(short_value, str | None), f"Should be string. Got {type(short_value)} for key: {key}"
        assert isinstance(long_value, str | None), f"Should be string. Got {type(long_value)} for key: {key}"
    
    
    assert len(stats.positions) == 2
    assert len(stats.closed_positions) == 0
    assert stats.get_latest_position_stats(1).quantity == pytest.approx(0.099)
    assert stats.get_latest_position_stats(2).quantity == pytest.approx(0.495)

    assert stats.get_latest_position_stats(1).value == pytest.approx(168.3)
    assert stats.get_latest_position_stats(2).value == pytest.approx(99.0)

    assert portfolio_stats.unrealised_profit_usd == pytest.approx(2.673)
    assert portfolio_stats.realised_profit_usd == 0

    assert summary.undecided == portfolio_stats.open_position_count
    assert summary.uninvested_cash == portfolio_stats.free_cash
    assert summary.realised_profit == portfolio_stats.realised_profit_usd

    long_short_metrics_latest = serialise_long_short_stats_as_json_table(
        state, None
    )
    
    update_statistics(datetime.datetime.utcnow(), state.stats, portfolio, ExecutionMode.real_trading, long_short_metrics_latest=long_short_metrics_latest)

    assert stats.get_latest_position_stats(1).value == pytest.approx(168.3)
    assert stats.get_latest_position_stats(2).value == pytest.approx(99.0)

    trader.sell(weth_usdc, portfolio.get_equity_for_pair(weth_usdc), 1700)
    trader.sell(aave_usdc, portfolio.get_equity_for_pair(aave_usdc), 300)

    assert len(state.portfolio.open_positions) == 0
    assert state.portfolio.get_total_equity() == pytest.approx(1043.632)

    long_short_metrics_latest = serialise_long_short_stats_as_json_table(
        state, None
    )
    
    update_statistics(datetime.datetime.utcnow(), state.stats, portfolio, ExecutionMode.real_trading, long_short_metrics_latest=long_short_metrics_latest)

    stats = state.stats
    portfolio_stats = stats.get_latest_portfolio_stats()
    summary = portfolio_stats.summary
    
    # 1 zero loss position
    assert stats.long_short_metrics_latest["live_stats"].rows['total_positions'].value['All'] == '2' 
    assert stats.long_short_metrics_latest["live_stats"].rows['won_positions'].value['Long'] == '1'
    assert stats.long_short_metrics_latest["live_stats"].rows['won_positions'].value['Short'] == '0'
    assert stats.long_short_metrics_latest["live_stats"].rows['average_position'].value['Long'] == '25.00%'
    assert stats.long_short_metrics_latest["live_stats"].rows['return_percent'].value['All'] == '4.91%'
    assert stats.long_short_metrics_latest["backtested_stats"].rows == {}

    assert len(stats.positions) == 2
    assert len(stats.closed_positions) == 2
    assert stats.get_latest_portfolio_stats().total_equity == pytest.approx(1043.632)
    assert stats.get_latest_portfolio_stats().free_cash == pytest.approx(1043.632)
    assert stats.get_latest_portfolio_stats().open_position_count == 0
    assert stats.get_latest_portfolio_stats().closed_position_count == 2
    assert stats.get_latest_portfolio_stats().frozen_position_count == 0
    assert stats.get_latest_portfolio_stats().unrealised_profit_usd == 0
    assert stats.get_latest_portfolio_stats().realised_profit_usd == 49.005

    # Both positions have three stats samples
    # - One before valuations
    # - One after revaluations
    # - One after close
    assert len(stats.positions[1]) == 3
    assert len(stats.positions[2]) == 3

    assert stats.get_latest_position_stats(1).profitability == 0
    assert stats.get_latest_position_stats(1).profit_usd == 0
    assert stats.get_latest_position_stats(1).quantity == 0
    assert stats.get_latest_position_stats(2).profitability == 0.5

    assert stats.closed_positions[1].value_at_open == pytest.approx(168.3)
    assert stats.closed_positions[1].value_at_max == pytest.approx(168.3)

    assert stats.closed_positions[2].value_at_open == pytest.approx(99)
    assert stats.closed_positions[2].value_at_max == pytest.approx(99)

    assert summary.won == 1
    assert summary.lost == 0
    assert summary.zero_loss == 1
    assert summary.total_positions == 2
    assert summary.win_percent == 0.5
    assert summary.return_percent == pytest.approx(0.04363200000000006)
    assert summary.annualised_return_percent == pytest.approx(152886.528)
    assert summary.realised_profit == pytest.approx(49.0049999)
    assert summary.uninvested_cash == portfolio_stats.free_cash
    assert summary.average_net_profit == pytest.approx(24.50249)

    long_short_metrics_latest = serialise_long_short_stats_as_json_table(
        state, None
    )
    
    update_statistics(datetime.datetime.utcnow(), state.stats, portfolio, ExecutionMode.real_trading, long_short_metrics_latest=long_short_metrics_latest)
    

def test_not_enough_cash(usdc, weth_usdc, start_ts):
    """Try to buy too much at once."""

    state = State()
    state.update_reserves([ReservePosition(usdc, Decimal(1000), start_ts, 1.0, start_ts)])
    trader = UnitTestTrader(state)

    # 0: start
    assert state.portfolio.get_total_equity() == 1000.0

    # Has $1000, needs $1700
    with pytest.raises(NotEnoughMoney):
        trader.buy(weth_usdc, Decimal(1), 1700)


def test_buy_sell_buy(usdc, weth, weth_usdc, start_ts):
    """Execute three trades on a position."""

    state = State()
    state.update_reserves([ReservePosition(usdc, Decimal(1000), start_ts, 1.0, start_ts)])
    trader = UnitTestTrader(state)

    # 0: start
    assert state.portfolio.get_total_equity() == 1000.0

    # 1: buy 1
    position, trade = trader.buy(weth_usdc, Decimal(0.1), 1700)
    assert state.portfolio.get_total_equity() == 998.3
    assert position.get_quantity_old() == pytest.approx(Decimal(0.099))

    # 2: Sell half of the tokens
    half_1 = position.get_quantity_old() / 2
    position, trade = trader.sell(weth_usdc, half_1, 1700)
    assert position.get_quantity_old() == pytest.approx(Decimal(0.0495))
    assert len(position.trades) == 2

    # 3: buy more
    position, trade = trader.buy(weth_usdc, Decimal(0.1), 1700)
    assert position.get_quantity_old() == pytest.approx(Decimal(0.1485))

    # All done
    assert len(position.trades) == 3
    assert len(state.portfolio.open_positions) == 1


def test_realised_profit_calculation(usdc, weth_usdc, start_ts: datetime.datetime):
    """Calculate realised profits correctly."""

    state = State()
    state.update_reserves([ReservePosition(usdc, Decimal(10000), start_ts, 1.0, start_ts)])
    trader = UnitTestTrader(state, lp_fees=0, price_impact=1)

    assert state.portfolio.get_total_equity() == 10000.0

    # Do 2 buys w/ different prices
    position, trade = trader.buy(weth_usdc, Decimal("1.0"), 1700.0)
    position, trade = trader.buy(weth_usdc, Decimal("0.5"), 1900.0)
    spent = 1700 + 1900/2
    estimated_avg_buy = (1*1700 + 0.5*1900) / 1.5

    assert position.is_long()
    assert position.get_net_quantity() == pytest.approx(Decimal("1.5"))
    assert position.get_total_bought_usd() == spent
    assert position.get_average_buy() == pytest.approx(estimated_avg_buy)
    assert position.get_buy_quantity() == Decimal("1.5")

    # No sells yet, no realised profit
    assert position.get_average_sell() is None

    assert position.get_realised_profit_usd() is None

    position, trade = trader.sell(weth_usdc, Decimal("1.5"), 1850.0)
    received = 1850 * 1.5
    assert position.is_closed()
    assert position.is_long()  # No change here
    assert position.get_average_sell() == 1850
    assert position.get_buy_quantity() == Decimal("1.5")
    assert position.get_sell_quantity() == Decimal("1.5")
    assert position.get_net_quantity() == Decimal("0")
    assert position.get_total_bought_usd() == spent
    assert position.get_total_sold_usd() == received

    profit = received - spent
    assert profit == 125
    assert position.get_realised_profit_usd() == pytest.approx(profit)


def test_realised_partial_profit_calculation(usdc, weth_usdc, start_ts: datetime.datetime):
    """Calculate realised profits correctly when some of the position is still open."""

    state = State()
    state.update_reserves([ReservePosition(usdc, Decimal(10000), start_ts, 1.0, start_ts)])
    trader = UnitTestTrader(state, lp_fees=0, price_impact=1)

    assert state.portfolio.get_total_equity() == 10000.0

    # Do 2 buys w/ different prices
    trader.buy(weth_usdc, Decimal("1.0"), 1700.0)
    trader.buy(weth_usdc, Decimal("0.5"), 1900.0)
    spent = 1700 + 1900/2

    # Sell half, gain 50% of the profit of the test_realised_profit_calculation
    position, trade = trader.sell(weth_usdc, Decimal("0.75"), 1850.0)
    received = 1850 * 0.75
    assert not position.is_closed()
    assert position.is_long()  # No change here
    assert position.get_average_sell() == 1850
    assert position.get_buy_quantity() == Decimal("1.5")
    assert position.get_sell_quantity() == Decimal("0.75")
    assert position.get_net_quantity() == Decimal("0.75")
    assert position.get_total_bought_usd() == spent
    assert position.get_total_sold_usd() == received

    # TODO: Should we express this differently?
    assert position.get_realised_profit_usd() == pytest.approx(62.5)
    assert position.get_realised_profit_percent() == pytest.approx(0.023584905660377336)
    assert position.get_unrealised_profit_usd() == pytest.approx(-50)
    assert position.get_unrealised_and_realised_profit_percent() == pytest.approx(0.004716981132075429)

    # Realise the remaining 50% of position profit
    position, trade = trader.sell(weth_usdc, Decimal("0.75"), 1850.0)
    assert position.get_realised_profit_usd() == pytest.approx(124.99999999999989)
    assert position.get_realised_profit_percent() == pytest.approx(0.04716981132075467)
    assert position.get_unrealised_profit_usd() == pytest.approx(0)
    assert position.get_unrealised_and_realised_profit_percent() == pytest.approx(0.04716981132075467)


def test_unrealised_profit_calculation(usdc, weth_usdc, start_ts: datetime.datetime):
    """Calculate unrealised profits correctly."""

    state = State()
    state.update_reserves([ReservePosition(usdc, Decimal(10000), start_ts, 1.0, start_ts)])
    trader = UnitTestTrader(state, lp_fees=0, price_impact=1)

    assert state.portfolio.get_total_equity() == 10000.0

    # Do 2 buys w/ different prices
    trader.buy(weth_usdc, Decimal("1.0"), 1700.0)
    position, _ = trader.buy(weth_usdc, Decimal("0.5"), 1900.0)
    spent = 1700 + 1900/2

    # Helper class to set ETH price
    class EthValuator:

        def __init__(self, price):
            self.price = price

        def __call__(self, ts, position: TradingPosition):
            position.revalue_base_asset(ts, float(self.price))
            return ValuationUpdate(
                position_id=position.position_id,
                created_at=ts,
                valued_at=ts,
                new_price=self.price,
                new_value=position.get_value()
            )

    # Revalue ETH to 1500 USD
    revalue_state(state, start_ts, EthValuator(1500))
    assert position.is_open()
    assert position.is_long()  # No change here
    assert position.get_realised_profit_usd() is None
    assert position.get_unrealised_profit_usd() == pytest.approx(-400)
    assert position.get_total_profit_percent() == pytest.approx(-0.15094339622641514)
    assert position.get_unrealised_and_realised_profit_percent() == pytest.approx(-0.15094339622641514)

    # Revalue ETH to 2000 USD, we are on green
    revalue_state(state, start_ts, EthValuator(2000))
    assert position.get_unrealised_profit_usd() == pytest.approx(350)
    assert position.get_total_profit_percent() == pytest.approx(0.13207547169811318)

    # Sell half, gain 50% of the profit of the test_realised_profit_calculation
    revalue_state(state, start_ts, EthValuator(1850))
    position, trade = trader.sell(weth_usdc, Decimal("0.75"), 1850.0)
    received = 1850 * 0.75
    assert not position.is_closed()
    assert position.is_long()  # No change here
    assert position.get_average_sell() == 1850
    assert position.get_buy_quantity() == Decimal("1.5")
    assert position.get_sell_quantity() == Decimal("0.75")
    assert position.get_net_quantity() == Decimal("0.75")
    assert position.get_total_bought_usd() == spent
    assert position.get_total_sold_usd() == received

    assert position.get_realised_profit_usd() == pytest.approx(62.5)
    assert position.get_unrealised_profit_usd() == pytest.approx(62.5)
    assert position.get_total_profit_percent() == pytest.approx(0.04716981132075467)

    # Close the remaining position
    position, trade = trader.sell(weth_usdc, Decimal("0.75"), 1850.0)
    assert position.is_closed()
    assert position.get_realised_profit_usd() == pytest.approx(125)
    assert position.get_unrealised_profit_usd() == 0
    assert position.get_total_profit_percent() == pytest.approx(0.04716981132075467)


def test_position_risk_calculation(usdc, weth_usdc, start_ts: datetime.datetime):
    """Calculate risk for an individual position."""

    state = State()
    state.update_reserves([ReservePosition(usdc, Decimal(10000), start_ts, 1.0, start_ts)])
    trader = UnitTestTrader(state, lp_fees=0, price_impact=1)

    assert state.portfolio.get_total_equity() == 10000.0

    # Do 2 buys, see open value does not change
    position, trade = trader.buy(weth_usdc, Decimal("1.0"), 1700.0)
    assert position.get_value_at_open() == 1700
    position.stop_loss = 1400  # Manually set stop loss for testing

    # Close the position, see open value does not change
    position, trade = trader.sell(weth_usdc, Decimal("1.0"), 1850.0)
    assert position.is_closed()
    assert position.get_value_at_open() == 1700

    assert position.portfolio_value_at_open == 10_000
    assert position.get_value_at_open() == 1700

    # Calculate the risk we took with the position
    assert position.get_capital_tied_at_open_pct() == 0.17

    risked_capital = position.get_loss_risk_at_open()
    assert risked_capital == 300

    # With stop loss, the risked capital is much lower
    assert position.get_loss_risk_at_open_pct() == 0.03


def test_single_buy_failed(usdc, weth, weth_usdc, start_ts):
    """A single token purchase tx fails."""
    state = State()
    state.update_reserves([ReservePosition(usdc, Decimal(1000), start_ts, 1.0, start_ts)])

    # #1 Planning stage
    # Buy 0.1 ETH at 1700 USD/ETH
    position, trade, created = state.create_trade(
        strategy_cycle_at=start_ts,
        pair=weth_usdc,
        quantity=Decimal("0.1"),
        reserve=None,
        assumed_price=1700,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        reserve_currency_price=1.0)

    # #2 Capital allocation
    txid = "0xffffff"
    nonce = 1
    ts = start_ts + datetime.timedelta(minutes=1)
    state.start_execution(ts, trade, txid, nonce)

    # #3 broadcast
    ts = ts + datetime.timedelta(minutes=1)
    state.mark_broadcasted(ts, trade)
    assert state.portfolio.get_cash() == 830  # Trades being executed do not show in the portfolio value
    assert state.portfolio.get_total_equity() == 1000  # Trades being executed do not show in the portfolio value

    # #4 fail
    ts = ts + datetime.timedelta(minutes=1)
    executed_price = 1690.0
    executed_quantity = Decimal("0.09")
    lp_fees = 2.50  # $2.5

    gas_units_consumed = 150_000  # 150k gas units per swap
    gas_price = 15 * 10**9  # 15 Gwei/gas unit
    native_token_price = 1.9  # 1.9 USD/ETH
    state.mark_trade_failed(ts, trade)

    assert trade.get_status() == TradeStatus.failed

    assert not position.has_unexecuted_trades()
    assert not position.has_planned_trades()
    assert state.portfolio.get_cash() == 1000.0  # Trades being executed do not show in the portfolio value
    assert state.portfolio.get_total_equity() == 1000.0  # Trades being executed do not show in the portfolio value7

    assert len(state.portfolio.open_positions) == 1

    # Check portfolio value recording
    assert position.portfolio_value_at_open == 1000.0
    assert trade.portfolio_value_at_creation == 1000.0


def test_serialize_state(usdc, weth_usdc, start_ts: datetime.datetime):
    """Dump and reload the internal state."""

    state = State()
    state.update_reserves([ReservePosition(
        usdc, 
        Decimal(1000), 
        start_ts, 
        1.0, 
        start_ts,
        initial_deposit=Decimal(1000),
        initial_deposit_reserve_token_price=1.0,
    )])
    trader = UnitTestTrader(state)

    # 1: buy 1
    position, trade = trader.buy(weth_usdc, Decimal(0.1), 1700)
    assert state.portfolio.get_total_equity() == 998.3
    assert position.get_value() == pytest.approx(168.3)
    assert position.last_pricing_at == start_ts

    update_statistics(datetime.datetime.utcnow(), state.stats, state.portfolio, ExecutionMode.real_trading)

    state.perform_integrity_check()

    # test restore from dump
    dump = state.to_json()
    state2 = State.from_json(dump)
    state2.perform_integrity_check()

    # Check we decoded correctly
    portfolio2 = state2.portfolio
    position2 = portfolio2.open_positions[1]
    summary = state2.stats.get_latest_portfolio_stats().summary
    assert position2.get_value() == pytest.approx(168.3)
    assert position2.last_pricing_at == start_ts
    assert position2.last_pricing_at.tzinfo == None  # Be especially careful with timestamps
    assert isinstance(summary.duration, datetime.timedelta)
    assert isinstance(summary.average_duration_of_winning_trades, datetime.timedelta)
    assert isinstance(summary.average_duration_of_losing_trades, datetime.timedelta)

    # test restore from dump using different method
    dump = state.to_json_safe()
    state3 = State.from_json(dump)
    state3.perform_integrity_check()


def test_state_summary_without_initial_cash(usdc, weth_usdc, start_ts: datetime.datetime):
    """Backward compat test for reverse without init cash info."""
    state = State()
    state.update_reserves([ReservePosition(
        usdc, 
        Decimal(1000), 
        start_ts, 
        1.0, 
        start_ts
    )])
    trader = UnitTestTrader(state)

    position, trade = trader.buy(weth_usdc, Decimal(0.1), 1700)
    assert state.portfolio.get_total_equity() == 998.3
    assert position.get_value() == pytest.approx(168.3)
    assert position.last_pricing_at == start_ts

    assert position.is_open()
    assert trade.is_success()
    update_statistics(datetime.datetime.utcnow(), state.stats, state.portfolio, ExecutionMode.unit_testing_trading)

    trader.sell(weth_usdc, state.portfolio.get_equity_for_pair(weth_usdc), 1800)

    update_statistics(datetime.datetime.utcnow(), state.stats, state.portfolio, ExecutionMode.unit_testing_trading)

    state.perform_integrity_check()
    summary = state.stats.get_latest_portfolio_stats().summary

    assert summary.initial_cash == 0
    assert summary.return_percent is None
    assert summary.annualised_return_percent is None
    assert summary.total_positions == 1
    assert summary.end_value == pytest.approx(1006.418)
    assert summary.average_net_profit == pytest.approx(9.800999)


def test_validate_state():
    """Catch common errors in setting bad state data."""

    ok = {"foo": datetime.datetime(1970, 1, 1)}
    validate_nested_state_dict(ok)

    ok = {"foo": Decimal(1)}
    validate_nested_state_dict(ok)

    ok = {"foo": 1}
    validate_nested_state_dict(ok)

    ok = {"foo": None}
    validate_nested_state_dict(ok)

    ok = {"foo": "1"}
    validate_nested_state_dict(ok)

    with pytest.raises(BadStateData):
        bad = {"foo": np.float32(1)}
        validate_nested_state_dict(bad)

    with pytest.raises(BadStateData):
        bad = {"foo": {"bar": np.float32(1)}}
        validate_nested_state_dict(bad)


def test_serialise_timedelta():
    """Serialise timedelta, that is part of a TradePricing object."""

    p = TradePricing(
        price=1000.0,
        mid_price=1000.0,
        lp_fee=1000.0 * 0.0030,
        pair_fee=0.0030,
        market_feed_delay=datetime.timedelta(minutes=2),
        side=True,
    )

    buf = p.to_json()
    p2 = TradePricing.from_json(buf)

    assert p2.market_feed_delay == datetime.timedelta(minutes=2)


def test_validate_state_with_too_large_int():
    """Integer is too big."""
    int_too_big = {"foo": 2**80}
    with pytest.raises(BadStateData):
        validate_nested_state_dict(int_too_big)


def test_validate_state_with_nan():
    """We have NaN."""
    nan = {"foo": float('inf')}
    with pytest.raises(BadStateData):
        validate_nested_state_dict(nan)


def test_blockchain_transaction_params():
    """Blockchain transactions must be able encode very large numbers."""
    args = (2**80,)
    bt = BlockchainTransaction(transaction_args=args)
    data = bt.to_dict()
    assert data["transaction_args"] == "80049510000000000000008a0b000000000000000000000185942e"
    validate_nested_state_dict(data)
    bt.to_json()


def test_serialize_state(usdc, weth_usdc, start_ts: datetime.datetime):
    """Dump and reload the internal state."""

    state = State()
    state.update_reserves([ReservePosition(
        usdc,
        Decimal(1000),
        start_ts,
        1.0,
        start_ts,
        initial_deposit=Decimal(1000),
        initial_deposit_reserve_token_price=1.0,
    )])
    trader = UnitTestTrader(state)

    # 1: buy 1
    position, trade = trader.buy(weth_usdc, Decimal(0.1), 1700)
    assert state.portfolio.get_total_equity() == 998.3
    assert position.get_value() == pytest.approx(168.3)
    assert position.last_pricing_at == start_ts

    update_statistics(datetime.datetime.utcnow(), state.stats, state.portfolio, ExecutionMode.unit_testing_trading)

    state.perform_integrity_check()

    # test restore from dump
    dump = state.to_json()
    state2 = State.from_json(dump)
    state2.perform_integrity_check()

    # Check we decoded correctly
    portfolio2 = state2.portfolio
    position2 = portfolio2.open_positions[1]
    summary = state2.stats.get_latest_portfolio_stats().summary
    assert position2.get_value() == pytest.approx(168.3)
    assert position2.last_pricing_at == start_ts
    assert position2.last_pricing_at.tzinfo == None  # Be especially careful with timestamps
    assert isinstance(summary.duration, datetime.timedelta)
    assert isinstance(summary.average_duration_of_winning_trades, datetime.timedelta)
    assert isinstance(summary.average_duration_of_losing_trades, datetime.timedelta)

    # test restore from dump using different method
    dump = state.to_json_safe()
    state3 = State.from_json(dump)
    state3.perform_integrity_check()


def test_serialize_panda_timestamp():
    """Handle pd.Timestamp conversion in serialisation.."""

    state: State = State()
    state.created_at = pd.Timestamp(2020, 1, 1)

    # test restore from dump using different method
    dump = state.to_json_safe()
    state2 = State.from_json(dump)
    state2.perform_integrity_check()

    assert state.created_at == datetime.datetime(2020, 1, 1)


def test_encode_solidty_args():
    """See we can make Solidity ABI values JSON serializable."""

    # Nested lists,
    # 256bit uint
    args = [[["0x8dccfFD0FB877890Bd7c5B21941932a5D3EaDea3",9223372036854776000,513172,682184]]]
    solidity_madness = {"foo": solidity_arg_encoder(args)}
    validate_nested_state_dict(solidity_madness)


def test_trade_too_small(usdc, weth_usdc, start_ts: datetime.datetime):
    """We cannot open very small trades."""
    state = State()
    state.update_reserves([ReservePosition(
        usdc,
        Decimal(1000),
        start_ts,
        1.0,
        start_ts
    )])
    trader = UnitTestTrader(state)

    with pytest.raises(TooSmallTrade):
         trader.buy(weth_usdc, Decimal(0.000000001), 1700)
