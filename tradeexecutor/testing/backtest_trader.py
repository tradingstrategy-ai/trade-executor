import datetime
from decimal import Decimal
from typing import Tuple, Optional

import pandas as pd

from tradeexecutor.backtest.backtest_execution import BacktestExecutionModel
from tradeexecutor.backtest.backtest_pricing import BacktestPricingModel
from tradeexecutor.state.state import State, TradeType
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradingstrategy.candle import GroupedCandleUniverse


class BacktestTrader:
    """Helper class to generate trades for backtesting."""

    def __init__(self,
                 start_ts: datetime.datetime,
                 state: State,
                 universe: TradingStrategyUniverse,
                 execution_model: BacktestExecutionModel,
                 pricing_model: BacktestPricingModel,
                 ):
        self.state = state
        self.ts = start_ts
        self.universe = universe
        self.execution_model = execution_model
        self.pricing_model = pricing_model
        self.nonce = 0
        self.lp_fees = 0
        self.native_token_price = 1

    def time_travel(self, timestamp: datetime.datetime):
        self.ts = timestamp

    def get_buy_price(self, pair: TradingPairIdentifier, reserve: Decimal) -> float:
        """Get the historical price for our current backtest time."""
        price = self.pricing_model.get_buy_price(self.ts, pair, reserve)
        return float(price)  # Convert from numpy.float32

    def get_sell_price(self, pair: TradingPairIdentifier, quantity: Decimal) -> float:
        """Get the historical price for our current backtest time."""
        price = self.pricing_model.get_sell_price(self.ts, pair, quantity)
        return float(price)  # Convert from numpy.float32

    def create(self, pair: TradingPairIdentifier, quantity: Optional[Decimal], reserve: Optional[Decimal], price: float) -> Tuple[TradingPosition, TradeExecution]:
        """Open a new trade."""

        # 1. Plan
        position, trade, created = self.state.create_trade(
            ts=self.ts,
            pair=pair,
            quantity=quantity,
            reserve=reserve,
            assumed_price=price,
            trade_type=TradeType.rebalance,
            reserve_currency=pair.quote,
            reserve_currency_price=1.0)

        return position, trade

    def create_and_execute(self, pair: TradingPairIdentifier, quantity: Optional[Decimal], reserve: Optional[Decimal], price: float) -> Tuple[TradingPosition, TradeExecution]:

        assert price > 0
        assert type(price) == float

        assert not(quantity and reserve), "Give only quantity (sell) or reserve (buy)"

        # 1. Plan
        position, trade = self.create(
            pair=pair,
            quantity=quantity,
            reserve=reserve,
            price=price)

        # 2. Capital allocation
        txid = hex(self.nonce)
        nonce = self.nonce
        self.state.start_execution(self.ts, trade, txid, nonce)

        # 3. Simulate tx broadcast
        self.nonce += 1
        self.state.mark_broadcasted(self.ts, trade)

        # 4. execution is dummy operation where planned execution becomes actual execution
        # Assume we always get the same execution we planned
        executed_price = trade.planned_price
        executed_quantity = trade.planned_quantity
        if trade.is_buy():
            executed_reserve = reserve
        else:
            executed_reserve = -quantity * Decimal(price)

        self.state.mark_trade_success(self.ts, trade, executed_price, executed_quantity, executed_reserve, self.lp_fees, self.native_token_price)
        return position, trade

    def buy(self, pair, reserve) -> Tuple[TradingPosition, TradeExecution]:
        assumed_price = self.get_buy_price(pair, reserve)
        return self.create_and_execute(pair, quantity=None, reserve=reserve, price=assumed_price)

    def sell(self, pair, quantity) -> Tuple[TradingPosition, TradeExecution]:
        assumed_price = self.get_sell_price(pair, quantity)
        return self.create_and_execute(pair, quantity=-quantity, reserve=None, price=assumed_price)

