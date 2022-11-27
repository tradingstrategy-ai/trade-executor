"""Positions open and closing management."""

import datetime
from decimal import Decimal
from typing import List, Optional, Union
import logging

import pandas as pd

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeType, TradeExecution
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.pricing_model import PricingModel
from tradingstrategy.pair import DEXPair
from tradingstrategy.universe import Universe
from tradeexecutor.strategy.trading_strategy_universe import translate_trading_pair


logger = logging.getLogger(__name__)


class NoSingleOpenPositionException(Exception):
    """Raised if getting the single position of the current portfolio is not successful."""


class PositionManager:
    """Open/closing of positions.

    Abstracts the complex logic with prices and such from the strategy writer.

    Takes the price feed and current execution state as an input and
    produces the execution instructions to change positions.

    Below are some recipes how to use position manager.

    Position manager is usually instiated at your `decide_trades` function as the following:

    .. code-block:: python

        from typing import List, Dict

        from tradeexecutor.state.visualisation import PlotKind
        from tradeexecutor.state.trade import TradeExecution
        from tradeexecutor.strategy.pricing_model import PricingModel
        from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
        from tradeexecutor.state.state import State
        from tradingstrategy.universe import Universe


        def decide_trades(
                timestamp: pd.Timestamp,
                universe: Universe,
                state: State,
                pricing_model: PricingModel,
                cycle_debug_data: Dict) -> List[TradeExecution]:



            # Create a position manager helper class that allows us easily to create
            # opening/closing trades for different positions
            position_manager = PositionManager(timestamp, universe, state, pricing_model)


    How to check if you have an open position using :py:meth:`is_any_open`
    and then open a new position:

    .. code-block:: python

        # List of any trades we decide on this cycle.
        # Because the strategy is simple, there can be
        # only zero (do nothing) or 1 (open or close) trades
        # decides
        trades = []

        if not position_manager.is_any_open():
            buy_amount = cash * position_size
            trades += position_manager.open_1x_long(pair, buy_amount)

        return trades

    How to check the entry price and open quantity of your latest position.
    See also :py:class:`decimal.Decimal` about arbitrary precision decimal numbers
    in Python.

    .. code-block:: python

        # Will throw an exception if there is no position open
        current_position = position_manager.get_current_position()

        # Quantity is the open amount in tokens.
        # This is expressed in Python Decimal class,
        # because Ethereum token balances are accurate up to 18 decimals
        # and this kind of accuracy cannot be expressed in floating point numbers.
        quantity = current_position.get_quantity()
        assert quantity == Decimal('0.03045760003971992547285959728')

        # The current price is the price of the trading pair
        # that was recorded on the last price feed sync.
        # This is a 64-bit floating point, as the current price
        # is always approximation based on market conditions.
        price = current_position.get_current_price()
        assert price == 1641.6263899583264

        # The opening price is the price of the first trade
        # that was made for this position. This is the actual
        # executed price of the trade, expressed as floating
        # point for the convenience.
        price = current_position.get_opening_price()
        assert price == 1641.6263899583264

    """

    def __init__(self,
                 timestamp: Union[datetime.datetime, pd.Timestamp],
                 universe: Universe,
                 state: State,
                 pricing_model: PricingModel,
                 ):

        #: TODO: Take valuation model as input

        assert pricing_model, "pricing_model is needed in order to know buy/sell price of new positions"

        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()

        self.timestamp = timestamp
        self.universe = universe
        self.state = state
        self.pricing_model = pricing_model

        reserve_currency, reserve_price = state.portfolio.get_default_reserve_currency()

        self.reserve_currency = reserve_currency

    def is_any_open(self) -> bool:
        """Do we have any positions open."""
        return len(self.state.portfolio.open_positions) > 0

    def get_current_position(self) -> TradingPosition:
        """Get the current single position.

        This is a shortcut function for trading strategies
        that operate only a single trading pair and a single position.

        :return:
            Currently open trading position

        :raise NoSingleOpenPositionError:
            If you do not have a position open or there are multiple positions open.
        """

        open_positions = self.state.portfolio.open_positions

        if len(open_positions) == 0:
            raise NoSingleOpenPositionException(f"No positions open at {self.timestamp}")

        if len(open_positions) > 1:
            raise NoSingleOpenPositionException(f"Multiple positions ({len(open_positions)}) open at {self.timestamp}")

        return next(iter(open_positions.values()))

    def open_1x_long(self,
                     pair: Union[DEXPair, TradingPairIdentifier],
                     value: USDollarAmount,
                     take_profit_pct: Optional[float]=None,
                     stop_loss_pct: Optional[float]=None,
                     notes: Optional[str]=None,
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

        :param notes:
            Human readable notes for this trade

        :return:
            A list of new trades.
            Opening a position may general several trades for complex DeFi positions,
            though usually the result contains only a single trade.

        """

        # Translate DEXPair object to the trading pair model
        if isinstance(pair, DEXPair):
            executor_pair = translate_trading_pair(pair)
        else:
            executor_pair = pair

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

    def adjust_position(self,
                        pair: TradingPairIdentifier,
                        dollar_amount_delta: USDollarAmount,
                        weight: float,
                        ) -> List[TradeExecution]:
        """Adjust holdings for a certain position.

        Used to rebalance positions.

        A new position is opened if no existing position is open.
        If everything is sold, the old position is closed

        If the rebalance is sell (`dollar_amount_delta` is negative),
        then calculate the quantity of the asset to sell based
        on the latest available market price on the position.

        This method is called by :py:func:`~tradeexecutor.strategy.pandas_trades.rebalance.rebalance_portfolio`.

        :param pair:
            Trading pair which position we adjust

        :param dollar_amount_delta:
            How much we want to increase/decrease the position

        :param weight:
            What is the weight of the asset in the new target portfolio 0....1.
            Currently only used to detect condition "sell all" instead of
            trying to match quantity/price conversion.

        :return:
            List of trades to be executed to get to the desired
            position level.
        """
        assert dollar_amount_delta != 0
        assert weight <= 1, f"Target weight cannot be over one: {weight}"
        assert weight >= 0, f"Target weight cannot be negative: {weight}"

        price = self.pricing_model.get_buy_price(self.timestamp, pair, dollar_amount_delta)

        reserve_asset, reserve_price = self.state.portfolio.get_default_reserve_currency()

        if dollar_amount_delta > 0:
            # Buy
            position, trade, created = self.state.create_trade(
                self.timestamp,
                pair=pair,
                quantity=None,
                reserve=Decimal(dollar_amount_delta),
                assumed_price=price,
                trade_type=TradeType.rebalance,
                reserve_currency=self.reserve_currency,
                reserve_currency_price=reserve_price,
            )
        else:
            # Sell
            # Convert dollar amount to quantity of the last known price
            position = self.state.portfolio.get_position_by_trading_pair(pair)
            assert position is not None, f"Assumed {pair} has open position because of attempt sell at {dollar_amount_delta} USD adjust"

            assumed_price = position.get_current_price()

            if weight != 0:
                quantity_delta = Decimal(dollar_amount_delta / assumed_price)
                assert quantity_delta < 0
            else:
                # Sell the whole position, using precise accounting
                # amount to avoid collecting dust holdings
                quantity_delta = -position.get_quantity()

            position, trade, created = self.state.create_trade(
                self.timestamp,
                pair=pair,
                quantity=quantity_delta,
                reserve=None,
                assumed_price=assumed_price,
                trade_type=TradeType.rebalance,
                reserve_currency=self.reserve_currency,
                reserve_currency_price=reserve_price,
            )

        return [trade]

    def close_position(self,
                       position: TradingPosition,
                       trade_type: TradeType=TradeType.rebalance,
                       notes: Optional[str] = None,
                       ) -> Optional[TradeExecution]:
        """Close a single position.

        The position may already have piled up selling trades.
        In this case calling `close_position()` again on the same position
        does nothing and `None` is returned.

        :param position:
            Position to be closed

        :param trade_type:
            What's the reason to close the position

        :param notes:
            Human readable notes for this trade

        :return:
            A trade that will close the position fully.
            If there is nothing left to close, return None.
        """

        assert position.is_long(), "Only long supported for now"
        assert position.is_open(), f"Tried to close already closed position {position}"

        quantity_left = position.get_live_quantity()

        if quantity_left == 0:
            # We have already generated closing trades for this position
            # earlier
            logger.warning("Tried to close position that has enough selling trades to sent it to zero: %s", position)
            return None

        pair = position.pair
        quantity = quantity_left
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
            notes=notes,
        )
        assert position == position2, "Somehow messed up the trade"

        return trade

    def close_all(self) -> List[TradeExecution]:
        """Close all open positions.

        :return:
            List of trades that will close existing positions
        """
        assert self.is_any_open(), "No positions to close"

        position: TradingPosition
        trades = []
        for position in self.state.portfolio.open_positions.values():
            trade = self.close_position(position)
            if trade:
                trades.append(trade)

        return trades

    def get_trading_pair(self, pair_id: int) -> TradingPairIdentifier:
        """Get a trading pair identifier by its internal id.

        Note that internal integer ids are not stable over
        multiple trade cycles and might be reset.
        Always use (chain id, smart contract) for persistent
        pair identifier.

        :return:
            Trading pair information
        """
        dex_pair = self.universe.universe.pairs.get_pair_by_id(pair_id)
        return translate_trading_pair(dex_pair)
