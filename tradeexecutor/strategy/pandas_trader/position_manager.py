"""Positions open and closing management."""

import datetime
from decimal import Decimal
from typing import List, Optional

from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeType, TradeExecution
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.pricing_model import PricingModel
from tradingstrategy.pair import DEXPair
from tradingstrategy.universe import Universe
from tradeexecutor.strategy.trading_strategy_universe import translate_trading_pair


class PositionManager:
    """Open/closing of positions.

    Abstracts the complex logic with prices and such from the strategy writer.

    Takes the price feed and current execution state as an input and
    produces the execution instructions to change positions.
    """

    def __init__(self,
                 timestamp: datetime.datetime,
                 universe: Universe,
                 state: State,
                 pricing_model: PricingModel,
                 ):

        #: TODO: Take valuation model as input

        assert pricing_model, "pricing_model is needed in order to know buy/sell price of new positions"

        self.timestamp = timestamp
        self.universe = universe
        self.state = state
        self.pricing_model = pricing_model

        reserve_currency, reserve_price = state.portfolio.get_default_reserve_currency()

        self.reserve_currency = reserve_currency

    def is_any_open(self) -> bool:
        """Do we have any positions open."""
        return len(self.state.portfolio.open_positions) > 0

    def open_1x_long(self,
                     pair: DEXPair,
                     value: USDollarAmount,
                     take_profit_pct: Optional[float]=None,
                     stop_loss_pct: Optional[float]=None,
                     ) -> List[TradeExecution]:
        """Open a long.

        - For simple buy and hold trades

        - Open a spot market buy.

        - Checks that there is not existing position - cannot increase position

        :param pair:
            Trading pair where we take the position

        :param value:
            How large position to open, in US dollar terms

        :param take_profit_pct:
            If set, set the position take profit relative
            to the current market price.
            1.0 is the current market price.
            If asset opening price is $1000, take_profit_pct=1.05
            will sell the asset when price reaches $1050.

        :param stop_loss_pct:
            If set, set the position to trigger stop loss relative to
            the current market price.
            1.0 is the current market price.
            If asset opening price is $1000, stop_loss_pct=0.95
            will sell the asset when price reaches 950.
        """

        # Translate DEXPair object to the trading pair model
        executor_pair = translate_trading_pair(pair)

        # Convert amount of reserve currency to the decimal
        # so we can have exact numbers from this point forward
        if type(value) == float:
            value = Decimal(value)

        price = self.pricing_model.get_buy_price(self.timestamp, executor_pair, value)

        reserve_asset, reserve_price = self.state.portfolio.get_default_reserve_currency()

        position, trade, created = self.state.create_trade(
            self.timestamp,
            pair=executor_pair,
            quantity=None,
            reserve=Decimal(value),
            assumed_price=price,
            trade_type=TradeType.rebalance,
            reserve_currency=self.reserve_currency,
            reserve_currency_price=reserve_price,
        )

        assert created, f"There was conflicting open position for pair: {executor_pair}"

        if take_profit_pct:
            position.take_profit = price * take_profit_pct

        if stop_loss_pct:
            position.stop_loss = price * stop_loss_pct

        self.state.visualisation.add_message(
            self.timestamp,
            f"Opened 1x long on {pair}, position value {value} USD")

        return [trade]

    def close_position(self, position: TradingPosition, trade_type: TradeType=TradeType.rebalance) -> TradeExecution:
        """Close a single position.

        :param position:
            Position to be closed

        :param trade_type:
            What's the reason to close the position
        """

        assert position.is_long(), "Only long supported for now"
        assert position.is_open(), f"Tried to close already closed position {position}"
        assert position.get_quantity() > 0

        pair = position.pair
        value = position.get_value()
        quantity = position.get_quantity()
        price = self.pricing_model.get_sell_price(self.timestamp, pair, quantity=quantity)

        reserve_asset, reserve_price = self.state.portfolio.get_default_reserve_currency()

        position2, trade, created = self.state.create_trade(
            self.timestamp,
            pair,
            -quantity,  # Negative quantity = sell all
            None,
            price,
            trade_type,
            reserve_asset,
            reserve_price,  # TODO: Harcoded stablecoin USD exchange rate
        )

        assert position == position2, "Somehow messed up the trade"

        return trade

    def close_all(self) -> List[TradeExecution]:
        """Close all open positions."""
        assert self.is_any_open(), "No positions to close"

        position: TradingPosition
        trades = []
        for position in self.state.portfolio.open_positions.values():
            trades.append(self.close_position(position))

        return trades
