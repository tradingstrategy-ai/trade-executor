"""Generate trades for backtesting."""

import datetime
from decimal import Decimal
from typing import Tuple, Optional

from tradeexecutor.backtest.backtest_execution import BacktestExecution
from tradeexecutor.backtest.backtest_pricing import BacktestPricing
from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel, BacktestRoutingState
from tradeexecutor.state.state import State, TradeType
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.types import USDollarAmount, BPS, USDollarPrice
from tradeexecutor.strategy.trade_pricing import TradePricing
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
                 execution_model: BacktestExecution,
                routing_model: BacktestRoutingModel,
                 pricing_model: BacktestPricing,
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

    def get_buy_price(self, pair: TradingPairIdentifier, reserve: Decimal) -> TradePricing:
        """Get the historical price for our current backtest time."""
        return self.pricing_model.get_buy_price(self.ts, pair, reserve)

    def get_sell_price(self, pair: TradingPairIdentifier, quantity: Decimal) -> TradePricing:
        """Get the historical price for our current backtest time."""
        return self.pricing_model.get_sell_price(self.ts, pair, quantity)

    def create(self,
               pair: TradingPairIdentifier,
               quantity: Optional[Decimal],
               reserve: Optional[Decimal],
               price: float,
               planned_mid_price: Optional[USDollarPrice] = None,
               ) -> Tuple[TradingPosition, TradeExecution]:
        """Open a new trade."""

        assert len(self.universe.reserve_assets) == 1

        # 1. Plan
        position, trade, created = self.state.create_trade(
            strategy_cycle_at=self.ts,
            pair=pair,
            quantity=quantity,
            reserve=reserve,
            assumed_price=price,
            trade_type=TradeType.rebalance,
            reserve_currency=self.universe.reserve_assets[0],
            reserve_currency_price=1.0,
            planned_mid_price=planned_mid_price,
        )

        return position, trade

    def create_and_execute(self,
                           pair: TradingPairIdentifier,
                           quantity: Optional[Decimal],
                           reserve: Optional[Decimal],
                           price: float,
                           lp_fee: Optional[BPS] = None,
                           lp_fees_estimated: Optional[USDollarAmount] = None,
                           planned_mid_price: Optional[USDollarPrice] = None,
                           ) -> Tuple[TradingPosition, TradeExecution]:

        assert price > 0
        assert type(price) == float

        assert not(quantity and reserve), "Give only quantity (sell) or reserve (buy)"

        # 1. Plan
        position, trade = self.create(
            pair=pair,
            quantity=quantity,
            reserve=reserve,
            price=price,
            planned_mid_price=planned_mid_price,
        )

        trade.fee_tier = lp_fee
        trade.lp_fees_estimated = lp_fees_estimated

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
        price_structure = self.get_buy_price(pair, reserve)
        return self.create_and_execute(
            pair,
            quantity=None,
            reserve=reserve,
            price=price_structure.price,
            lp_fee=price_structure.get_fee_percentage(),
            lp_fees_estimated=price_structure.get_total_lp_fees(),
            planned_mid_price=price_structure.mid_price,
        )

    def sell(self, pair, quantity) -> Tuple[TradingPosition, TradeExecution]:
        price_structure = self.get_sell_price(pair, quantity)
        return self.create_and_execute(
            pair,
            quantity=-quantity,
            reserve=None,
            price=price_structure.price,
            lp_fee=price_structure.get_fee_percentage(),
            lp_fees_estimated=price_structure.get_total_lp_fees(),
            planned_mid_price=price_structure.mid_price)



