"""Generate trades for backtesting."""

import datetime
from decimal import Decimal
from typing import Tuple, Optional

from tradeexecutor.backtest.backtest_execution import BacktestExecutionModel
from tradeexecutor.backtest.backtest_pricing import BacktestSimplePricingModel
from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel, BacktestRoutingState
from tradeexecutor.state.state import State, TradeType
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


class BacktestTrader:
    """Helper class to generate trades for backtesting.

    Directly generate trades without going through a strategy.
    Any trade is executed against given pair, price universe and execution model.
    """

    def __init__(self,
                 start_ts: datetime.datetime,
                 state: State,
                 universe: TradingStrategyUniverse,
                 execution_model: BacktestExecutionModel,
                 routing_model: BacktestRoutingModel,
                 pricing_model: BacktestSimplePricingModel,
                 ):
        self.state = state
        self.ts = start_ts
        self.universe = universe
        self.execution_model = execution_model
        self.pricing_model = pricing_model
        self.routing_model = routing_model
        self.nonce = 0
        self.lp_fees = 0
        self.native_token_price = 1

        # Set up routing state with dummy execution details
        execution_details = execution_model.get_routing_state_details()
        self.routing_state: BacktestRoutingState = self.routing_model.create_routing_state(universe, execution_details)

    def time_travel(self, timestamp: datetime.datetime):
        """Set the timestamp for the next executions."""
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

        assert len(self.universe.reserve_assets) == 1

        # 1. Plan
        position, trade, created = self.state.create_trade(
            ts=self.ts,
            pair=pair,
            quantity=quantity,
            reserve=reserve,
            assumed_price=price,
            trade_type=TradeType.rebalance,
            reserve_currency=self.universe.reserve_assets[0],
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

        # Run trade start, simulated broadcast, simulated execution using
        # backtest execution model
        self.execution_model.execute_trades(
            self.ts,
            self.state,
            [trade],
            self.routing_model,
            self.routing_state,
            check_balances=True)

        return position, trade

    def buy(self, pair, reserve) -> Tuple[TradingPosition, TradeExecution]:
        assumed_price = self.get_buy_price(pair, reserve)
        return self.create_and_execute(pair, quantity=None, reserve=reserve, price=assumed_price)

    def sell(self, pair, quantity) -> Tuple[TradingPosition, TradeExecution]:
        assumed_price = self.get_sell_price(pair, quantity)
        return self.create_and_execute(pair, quantity=-quantity, reserve=None, price=assumed_price)

