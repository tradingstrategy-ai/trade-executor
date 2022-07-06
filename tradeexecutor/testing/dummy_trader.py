"""Simple trade generator without execution."""
import datetime
from decimal import Decimal
from typing import Tuple

import pandas as pd

from tradeexecutor.state.state import State, TradeType
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradingstrategy.candle import GroupedCandleUniverse


class DummyTestTrader:
    """Helper class to generate trades for tests.

    This trade helper is not connected to any blockchain - it just simulates txid and nonce values.
    """

    def __init__(self, state: State, lp_fees=2.50, price_impact=0.99):
        self.state = state
        self.nonce = 1
        self.ts = datetime.datetime(2022, 1, 1, tzinfo=None)

        self.lp_fees = lp_fees
        self.price_impact = price_impact
        self.native_token_price = 1

    def time_travel(self, timestamp: datetime.datetime):
        self.ts = timestamp

    def create(self, pair: TradingPairIdentifier, quantity: Decimal, price: float) -> Tuple[TradingPosition, TradeExecution]:
        """Open a new trade."""
        # 1. Plan
        position, trade, created = self.state.create_trade(
            ts=self.ts,
            pair=pair,
            quantity=quantity,
            reserve=None,
            assumed_price=price,
            trade_type=TradeType.rebalance,
            reserve_currency=pair.quote,
            reserve_currency_price=1.0)

        self.ts += datetime.timedelta(seconds=1)
        return position, trade

    def create_and_execute(self, pair: TradingPairIdentifier, quantity: Decimal, price: float) -> Tuple[TradingPosition, TradeExecution]:

        assert price > 0
        assert quantity != 0

        price_impact = self.price_impact

        # 1. Plan
        position, trade = self.create(
            pair=pair,
            quantity=quantity,
            price=price)

        # 2. Capital allocation
        txid = hex(self.nonce)
        nonce = self.nonce
        self.state.start_execution(self.ts, trade, txid, nonce)

        # 3. broadcast
        self.nonce += 1
        self.ts += datetime.timedelta(seconds=1)

        self.state.mark_broadcasted(self.ts, trade)
        self.ts += datetime.timedelta(seconds=1)

        # 4. executed
        executed_price = price * price_impact
        if trade.is_buy():
            executed_quantity = quantity * Decimal(price_impact)
            executed_reserve = Decimal(0)
        else:
            executed_quantity = quantity
            executed_reserve = abs(quantity * Decimal(executed_price))

        self.state.mark_trade_success(self.ts, trade, executed_price, executed_quantity, executed_reserve, self.lp_fees, self.native_token_price)
        return position, trade

    def buy(self, pair, quantity, price) -> Tuple[TradingPosition, TradeExecution]:
        return self.create_and_execute(pair, quantity, price)

    def prepare_buy(self, pair, quantity, price) -> Tuple[TradingPosition, TradeExecution]:
        return self.create(pair, quantity, price)

    def sell(self, pair, quantity, price) -> Tuple[TradingPosition, TradeExecution]:
        return self.create_and_execute(pair, -quantity, price)

    def buy_with_price_data(self, pair, quantity, candle_universe: GroupedCandleUniverse) -> Tuple[TradingPosition, TradeExecution]:
        price = candle_universe.get_closest_price(pair.internal_id, pd.Timestamp(self.ts))
        return self.create_and_execute(pair, quantity, float(price))

    def sell_with_price_data(self, pair, quantity, candle_universe: GroupedCandleUniverse) -> Tuple[TradingPosition, TradeExecution]:
        price = candle_universe.get_closest_price(pair.internal_id, pd.Timestamp(self.ts))
        return self.create_and_execute(pair, -quantity, float(price))
