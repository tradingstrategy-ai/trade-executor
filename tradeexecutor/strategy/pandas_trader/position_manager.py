"""Positions open and closing management."""

import datetime
from decimal import Decimal
from typing import List, Optional, Union
import logging

import pandas as pd

from tradingstrategy.candle import CandleSampleUnavailable
from tradingstrategy.pair import DEXPair
from tradingstrategy.universe import Universe

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.position import TradingPosition, TriggerPriceUpdate
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeType, TradeExecution
from tradeexecutor.state.types import USDollarAmount, Percent, LeverageMultiplier
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trading_strategy_universe import translate_trading_pair, TradingStrategyUniverse
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_routing import UniswapV2SimpleRoutingModel
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_routing import UniswapV3SimpleRoutingModel
from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel
from tradeexecutor.backtest.backtest_routing import BacktestRoutingIgnoredModel
from tradeexecutor.utils.leverage_calculations import LeverageEstimate


logger = logging.getLogger(__name__)


class NoSingleOpenPositionException(Exception):
    """Raised if getting the single position of the current portfolio is not successful."""


class PositionManager:
    """An utility class to open and close new trade positions.

    `PositionManager` hides away the complex logic reason about trades.
    It is designed to be used in a trading strategy's `decide_trades()` function
    as an utility class to generate trades a list of :py:class:`TradeExecution`
    objects.

    It offers a simple interface for trading for people who are used to
    TradingView's :term:`Pine Script` or similar limited trade scripting environment.

    PositionManager helps about

    - How to have up-to-date price information

    - Setting take profit/stop loss parameters for positions

    - Converting between US dollar prices, crypto prices

    - Converting between quantity and value of a trade

    - Caring whether we have an existing position open for the trading pair already

    - Shortcut methods for trading strategies that trade only a single trading pair

    `PositionManager` takes the price feed and current execution state as an input and
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
                 universe: Universe | TradingStrategyUniverse,
                 state: State,
                 pricing_models: list[PricingModel] | PricingModel,
                 default_slippage_tolerance=0.05,  # Slippage tole
                 ):

        """Create a new PositionManager instance.
        
        Call within `decide_trades` function.
        
        :param timestamp: 
            The timestamp of the current strategy cycle
            
        :param universe: 
            Trading universe of available assets
        
        :param state: 
            Current state of the trade execution
            
        :param pricing_models:
            The list of models to estimate prices for any trades
         
        :param default_slippage_tolerance: 
            Slippage tolerance parameter set for any trades if not overriden trade-by-trade basis.

            Default to 5% slippage.

            TODO: Tighten this parameter when execution tracking works better.
            
        """

        assert pricing_models, "pricing_models is needed in order to know buy/sell price of new positions"

        if type(pricing_models) != list:
            pricing_models = [pricing_models]

        assert all(isinstance(p, PricingModel) for p in pricing_models), "pricing_models must be a list of PricingModel objects"

        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime().replace(tzinfo=None)

        self.timestamp = timestamp

        if isinstance(universe, Universe):
            # Legacy
            # Engine version 0.1 and 0.2
            self.data_universe = universe
            self.strategy_universe = None
        elif isinstance(universe, TradingStrategyUniverse):
            # Engine version 0.3
            # See tradeexecutor.strategy.engine_version
            self.strategy_universe = universe
            self.data_universe = universe.universe
        else:
            raise RuntimeError(f"Does not know the universe: {universe}")

        self.state = state

        if isinstance(pricing_models, PricingModel):
            self.pricing_models = [pricing_models]

        self.pricing_models = pricing_models
        self.default_slippage_tolerance = default_slippage_tolerance

        reserve_currency, reserve_price = state.portfolio.get_default_reserve_asset()

        self.reserve_currency = reserve_currency
        self.reserve_price = reserve_price

    @property
    def pricing_model(self):
        if len(self.pricing_models) == 1:
            return self.pricing_models[0]
        else:
            raise RuntimeError("Cannot use pricing_model property when there are multiple pricing models")

    def is_any_open(self) -> bool:
        """Do we have any positions open."""
        return len(self.state.portfolio.open_positions) > 0
    
    def is_any_open_for_pair(self, pair: TradingPairIdentifier) -> bool:
        """Do we have any positions open for the given pair.
        
        :param pair: Trading pair to check for open positions
        :return: True if there is any open position for the given pair
        """
        return len([position for position in self.state.portfolio.open_positions.values() if position.pair.pool_address == pair.pool_address]) > 0

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

    def get_current_position_for_pair(self, pair: TradingPairIdentifier) -> Optional[TradingPosition]:
        """Get the current open position for a specific trading pair.

        :return:
            Currently open trading position.

            If there is no open position return None.

        """
        return self.state.portfolio.get_position_by_trading_pair(pair)

    def get_last_closed_position(self) -> Optional[TradingPosition]:
        """Get the position that was last closed.

        If multiple positions are closed at the same time,
        return a random position.

        Example:

        .. code-block:: python

            last_position = position_manager.get_last_closed_position()
            if last_position:
                ago = timestamp - last_position.closed_at
                print(f"Last position was closed {ago}")
            else:
                print("Strategy has not decided any position before")

        :return:

            None if the strategy has not closed any positions
        """
        closed_positions = self.state.portfolio.closed_positions

        if len(closed_positions) == 0:
            return None

        return max(closed_positions.values(), key=lambda c: c.closed_at)

    def get_current_portfolio(self) -> Portfolio:
        """Return the active portfolio of the strategy."""
        return self.state.portfolio

    def get_trading_pair(self, pair_id: int) -> TradingPairIdentifier:
        """Get a trading pair identifier by its internal id.

        Note that internal integer ids are not stable over
        multiple trade cycles and might be reset.
        Always use (chain id, smart contract) for persistent
        pair identifier.

        :return:
            Trading pair information
        """
        dex_pair = self.data_universe.pairs.get_pair_by_id(pair_id)
        return translate_trading_pair(dex_pair)

    def get_pair_fee(self,
                     pair: Optional[TradingPairIdentifier] = None,
                     ) -> Optional[float]:
        """Estimate the trading/LP fees for a trading pair.

        This information can come either from the exchange itself (Uni v2 compatibles),
        or from the trading pair (Uni v3).

        The return value is used to fill the
        fee values for any newly opened trades.

        :param pair:
            Trading pair for which we want to have the fee.

            Can be left empty if the underlying exchange is always
            offering the same fee.

        :return:
            The estimated trading fee, expressed as %.

            Returns None if the fee information is not available.
            This can be different from zero fees.
        """
        return self.get_pricing_model(pair).get_pair_fee(self.timestamp, pair)

    def open_1x_long(self,
                     pair: Union[DEXPair, TradingPairIdentifier],
                     value: USDollarAmount,
                     take_profit_pct: Optional[float] = None,
                     stop_loss_pct: Optional[float] = None,
                     trailing_stop_loss_pct: Optional[float] = None,
                     stop_loss_usd: Optional[USDollarAmount] = None,
                     notes: Optional[str] = None,
                     slippage_tolerance: Optional[float] = None,
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

        :param trailing_stop_loss_pct:
            If set, set the position to trigger trailing stop loss relative to
            the current market price. Cannot be used with stop_loss_pct or stop_loss_usd.

        :param stop_loss_usd:
            If set, set the position to trigger stop loss at the given dollar price.
            Cannot be used with stop_loss_pct or trailing_stop_loss_pct.

        :param notes:
            Human readable notes for this trade

        :param slippage_tolerance:
            Slippage tolerance for this trade.

            Use :py:attr:`default_slippage_tolerance` if not set.

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

        assert value > 0, f"Negative value: {value} on {pair}"

        # Convert amount of reserve currency to the decimal
        # so we can have exact numbers from this point forward
        if type(value) == float:
            value = Decimal(value)

        pricing_model: PricingModel = self.get_pricing_model(executor_pair)

        price_structure = pricing_model.get_buy_price(self.timestamp, executor_pair, value)

        assert type(price_structure.mid_price) == float

        reserve_asset, reserve_price = self.state.portfolio.get_default_reserve_asset()

        slippage_tolerance = slippage_tolerance or self.default_slippage_tolerance

        position, trade, created = self.state.create_trade(
            self.timestamp,
            pair=executor_pair,
            quantity=None,
            reserve=Decimal(value),
            assumed_price=price_structure.price,
            trade_type=TradeType.rebalance,
            reserve_currency=self.reserve_currency,
            reserve_currency_price=reserve_price,
            lp_fees_estimated=price_structure.get_total_lp_fees(),
            pair_fee=price_structure.get_fee_percentage(),
            planned_mid_price=price_structure.mid_price,
            price_structure=price_structure,
            slippage_tolerance=slippage_tolerance,
        )

        assert created, f"There was conflicting open position for pair: {executor_pair}"

        if take_profit_pct:
            position.take_profit = price_structure.mid_price * take_profit_pct

        if stop_loss_pct is not None:
            assert 0 <= stop_loss_pct <= 1, f"stop_loss_pct must be 0..1, got {stop_loss_pct}"
            self.update_stop_loss(position, price_structure.mid_price * stop_loss_pct)

        if trailing_stop_loss_pct:
            assert stop_loss_pct is None, "You cannot give both stop_loss_pct and trailing_stop_loss_pct"
            assert 0 <= trailing_stop_loss_pct <= 1, f"trailing_stop_loss_pct must be 0..1, got {trailing_stop_loss_pct}"
            self.update_stop_loss(position, price_structure.mid_price * trailing_stop_loss_pct)
            position.trailing_stop_loss_pct = trailing_stop_loss_pct
        
        if stop_loss_usd:
            assert not stop_loss_pct, "You cannot give both stop_loss_pct and stop_loss_usd"
            assert not trailing_stop_loss_pct, "You cannot give both trailing_stop_loss_pct and stop_loss_usd"
            assert stop_loss_usd < price_structure.mid_price, f"stop_loss_usd must be less than mid_price got {stop_loss_usd} >= {price_structure.mid_price}"
            
            self.update_stop_loss(position, stop_loss_usd)

        if notes:
            position.notes = notes
            trade.notes = notes

        self.state.visualisation.add_message(
            self.timestamp,
            f"Opened 1x long on {pair}, position value {value} USD")

        return [trade]

    def adjust_position(self,
                        pair: TradingPairIdentifier,
                        dollar_delta: USDollarAmount,
                        quantity_delta: Optional[float],
                        weight: float,
                        stop_loss: Optional[Percent] = None,
                        take_profit: Optional[Percent] = None,
                        trailing_stop_loss: Optional[Percent] = None,
                        slippage_tolerance: Optional[float] = None,
                        override_stop_loss=False,
                        ) -> List[TradeExecution]:
        """Adjust holdings for a certain position.

        Used to rebalance positions.

        A new position is opened if no existing position is open.
        If everything is sold, the old position is closed

        If the rebalance is sell (`dollar_amount_delta` is negative),
        then calculate the quantity of the asset to sell based
        on the latest available market price on the position.

        This method is called by :py:func:`~tradeexecutor.strategy.pandas_trades.rebalance.rebalance_portfolio`.

        .. warning ::

            Adjust position cannot be used to close an existing position, because
            epsilons in quantity math. Use :py:meth:`close_position` for this.

        :param pair:
            Trading pair which position we adjust

        :param dollar_delta:
            How much we want to increase/decrease the position in US dollar terms.

        :param quantity_delta:
            How much we want to increase/decrease the position in the asset unit terms.

            Used only when decreasing existing positions (selling).

        :param weight:
            What is the weight of the asset in the new target portfolio 0....1.
            Currently only used to detect condition "sell all" instead of
            trying to match quantity/price conversion.

        :param stop_loss:
            Set the stop loss for the position.

            Use 0...1 based on the current mid price.
            E.g. 0.98 = 2% stop loss under the current mid price.

            Sets the initial stop loss. If you want to override
            this for an existing position you need to use `override_stop_loss` parameter.

        :param take_profit:
            Set the take profit for the position.

            Use 0...1 based on the current mid price.
            E.g. 1.02 = 2% take profit over the current mid-price.

        :param slippage_tolerance:
            Slippage tolerance for this trade.

            Use :py:attr:`default_slippage_tolerance` if not set.

        :param override_stop_loss:
            If not set and a position has already stop loss set, do not modify it.

        :return:
            List of trades to be executed to get to the desired
            position level.
        """
        assert dollar_delta != 0
        assert weight <= 1, f"Target weight cannot be over one: {weight}"
        assert weight >= 0, f"Target weight cannot be negative: {weight}"

        try:
            if dollar_delta > 0:
                dollar_delta = Decimal(dollar_delta) if isinstance(dollar_delta, float | int) else dollar_delta
                price_structure = self.get_pricing_model(pair).get_buy_price(self.timestamp, pair, dollar_delta)
            else:
                quantity_delta = Decimal(quantity_delta) if isinstance(quantity_delta, float | int) else quantity_delta
                price_structure = self.get_pricing_model(pair).get_sell_price(self.timestamp, pair, abs(quantity_delta))

        except CandleSampleUnavailable as e:
            # Backtesting cannot fetch price for an asset,
            # probably not enough data and the pair is trading early?
            data_delay_tolerance = getattr(self.get_pricing_model(pair), "data_delay_tolerance", None)
            raise CandleSampleUnavailable(
                f"Could not fetch price for {pair} at {self.timestamp}\n"
                f"\n"
                f"This is usually due to sparse candle data - trades have not been made or the blockchain was halted during the price look-up period.\n"
                f"Because there are no trades we cannot determine what was the correct asset price using {data_delay_tolerance} data tolerance delay.\n"
                f"\n"
                f"You can work around this by checking that any trading pair candles are fresh enough in your decide_trades() function\n"
                f"or increase the parameter in BacktestSimplePricingModel(data_delay_tolerance) or run_backtest_inline(data_delay_tolerance)\n"
            ) from e

        price = price_structure.price

        reserve_asset, reserve_price = self.state.portfolio.get_default_reserve_asset()

        slippage_tolerance = slippage_tolerance or self.default_slippage_tolerance

        if dollar_delta > 0:
            # Buy
            position, trade, created = self.state.create_trade(
                self.timestamp,
                pair=pair,
                quantity=None,
                reserve=Decimal(dollar_delta),
                assumed_price=price,
                trade_type=TradeType.rebalance,
                reserve_currency=self.reserve_currency,
                reserve_currency_price=reserve_price,
                planned_mid_price=price_structure.mid_price,
                lp_fees_estimated=price_structure.get_total_lp_fees(),
                pair_fee=price_structure.get_fee_percentage(),
                slippage_tolerance=slippage_tolerance,
            )
        else:
            # Sell
            # Convert dollar amount to quantity of the last known price

            assert quantity_delta is not None
            assert quantity_delta < 0, f"Received non-negative sell quantity {quantity_delta} for {pair}"

            position = self.state.portfolio.get_position_by_trading_pair(pair)
            assert position is not None, f"Assumed {pair} has open position because of attempt sell at {dollar_delta} USD adjust"

            position, trade, created = self.state.create_trade(
                self.timestamp,
                pair=pair,
                quantity=Decimal(quantity_delta),
                reserve=None,
                assumed_price=price_structure.price,
                trade_type=TradeType.rebalance,
                reserve_currency=self.reserve_currency,
                reserve_currency_price=reserve_price,
                planned_mid_price=price_structure.mid_price,
                lp_fees_estimated=price_structure.get_total_lp_fees(),
                slippage_tolerance=slippage_tolerance,
                price_structure=price_structure,
            )

        assert trade.lp_fees_estimated > 0, f"LP fees estimated: {trade.lp_fees_estimated} - {trade}"

        # Update stop loss for this position
        if stop_loss:

            assert stop_loss < 1, f"Got stop loss {stop_loss}"

            if position.stop_loss:
                # Update existing stop loss
                if override_stop_loss:
                    self.update_stop_loss(position, price_structure.mid_price * stop_loss)
                else:
                    # Do not override existing stop loss set earlier
                    pass
            else:
                # Set the initial stop loss
                self.update_stop_loss(position, price_structure.mid_price * stop_loss)

        if trailing_stop_loss:
            assert trailing_stop_loss < 1, f"Got trailing_stop_loss {trailing_stop_loss}"
            if not position.stop_loss:
                self.update_stop_loss(position, price_structure.mid_price * trailing_stop_loss)
            position.trailing_stop_loss_pct = trailing_stop_loss

        if take_profit:
            assert take_profit > 1, f"Got take profit {take_profit}"
            position.take_profit = price_structure.mid_price * take_profit

        return [trade]

    def close_spot_position(self,
                       position: TradingPosition,
                       trade_type: TradeType=TradeType.rebalance,
                       notes: Optional[str] = None,
                       slippage_tolerance: Optional[float] = None,
                       ) -> List[TradeExecution]:
        """Close a single spot market trading position.

        See :py:meth:`close_position` for usage.
        """

        assert position.is_spot_market()

        quantity_left = position.get_available_trading_quantity()

        pair = position.pair
        quantity = quantity_left
        price_structure = self.get_pricing_model(pair).get_sell_price(self.timestamp, pair, quantity=quantity)

        reserve_asset, reserve_price = self.state.portfolio.get_default_reserve_asset()

        slippage_tolerance = slippage_tolerance or self.default_slippage_tolerance

        logger.info("Preparing to close position %s, quantity %s, pricing %s, slippage tolerance %f", position, quantity, price_structure, slippage_tolerance)

        position2, trade, created = self.state.create_trade(
            self.timestamp,
            pair,
            -quantity,  # Negative quantity = sell all
            None,
            price_structure.price,
            trade_type,
            reserve_asset,
            reserve_price,  # TODO: Harcoded stablecoin USD exchange rate
            notes=notes,
            pair_fee=price_structure.get_fee_percentage(),
            lp_fees_estimated=price_structure.get_total_lp_fees(),
            planned_mid_price=price_structure.mid_price,
            position=position,
            slippage_tolerance=slippage_tolerance,
            price_structure=price_structure,
        )
        assert position == position2, f"Somehow messed up the close_position() trade.\n" \
                                      f"Original position: {position}.\n" \
                                      f"Trade's position: {position2}.\n" \
                                      f"Trade: {trade}\n" \
                                      f"Quantity left: {quantity_left}\n" \
                                      f"Price structure: {price_structure}\n" \
                                      f"Reserve asset: {reserve_asset}\n"

        return [trade]

    def close_credit_supply_position(self,
                       position: TradingPosition,
                       quantity: float | Decimal | None = None,
                       notes: Optional[str] = None,
                       trade_type: TradeType = TradeType.rebalance,
                       ) -> List[TradeExecution]:
        """Close a credit supply position

        :param position:
            Position to close.

            Must be a credit supply position.

        :param quantity:
            How much of the quantity we reduce.

            If not given close the full position.

        :return:
            New trades to be executed
        """

        assert self.strategy_universe, "Make sure trading_strategy_engine_version = 0.3. Credit supply does not work with old decide_trades()."
        pair = position.pair

        assert pair.base.underlying.is_stablecoin(), f"Non-stablecoin lending not yet implemented"
        price = 1.0

        if quantity is None:
            quantity = position.get_quantity()

        if type(quantity) == float:
            # TODO: Snap the amount to the full position size if rounding errors
            quantity = Decimal(quantity)

        # TODO: Hardcoded USD exchange rate
        reserve_asset = self.strategy_universe.get_reserve_asset()

        _, trade, _ = self.state.supply_credit(
            self.timestamp,
            pair,
            collateral_asset_price=price,
            collateral_quantity=-quantity,
            trade_type=trade_type,
            reserve_currency=reserve_asset,
            notes=notes,
            position=position,
        )
        return [trade]

    def close_position(self,
                       position: TradingPosition,
                       trade_type: TradeType | None = None,
                       notes: Optional[str] = None,
                       slippage_tolerance: Optional[float] = None,
                       ) -> List[TradeExecution]:
        """Close a single position.

        The position may already have piled up selling trades.
        In this case calling `close_position()` again on the same position
        does nothing and `None` is returned.

        :param position:
            Position to be closed

        :param trade_type:
            What's the reason to close the position

        :param notes:
            Human-readable notes for this trade

        :param slippage_tolerance:
            Slippage tolerance for this trade.

            Use :py:attr:`default_slippage_tolerance` if not set.

        :return:
            Get list of trades needed to close this position.

            return list of trades.

        """

        # assert position.is_long(), "Only long supported for now"
        assert position.is_open(), f"Tried to close already closed position {position}"

        quantity_left = position.get_available_trading_quantity()

        if quantity_left == 0:
            # We have already generated closing trades for this position
            # earlier
            logger.warning("Tried to close position that is likely already closed, as there are no tokens to sell: %s", position)
            return []

        if position.is_spot_market():

            if trade_type is None:
                trade_type = TradeType.rebalance

            return self.close_spot_position(
                position,
                trade_type,
                notes,
                slippage_tolerance
            )
        elif position.is_credit_supply():

            if trade_type is None:
                trade_type = TradeType.rebalance

            return self.close_credit_supply_position(
                position,
                trade_type=trade_type,
                notes=notes,
            )
        elif position.is_short():
            if trade_type is None:
                trade_type = TradeType.rebalance

            return self.close_short_position(
                position,
                trade_type=trade_type,
                notes=notes,
            )
        else:
            raise NotImplementedError(f"Does not know how to close: {position}")

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
                trades.extend(trade)

        return trades

    def estimate_asset_quantity(
            self,
            pair: TradingPairIdentifier,
            dollar_amount: USDollarAmount,
    ) -> float:
        """Convert dollar amount to the quantity of a token.
        
        Use the market mid-price of the timestamp.

        :param pair:
            Trading pair of which base pair we estimate.

        :param dollar_amount:
            Get the asset quantity for this many dollars.

        :return:
            Asset quantity.

            The sign of the asset quantity is the same as the sign of `dollar_amount` parameter.

            We return as float, because the exact quantity is never known due the price fluctuations and slippage.

        """
        assert dollar_amount, f"Got dollar amount: {dollar_amount}"
        timestamp = self.timestamp
        pricing_model = self.get_pricing_model(pair)
        price = pricing_model.get_mid_price(timestamp, pair)
        return float(dollar_amount / price)

    def update_stop_loss(self, position: TradingPosition, stop_loss: USDollarAmount):
        """Update the stop loss for the current position.
        
        :param position:
            Position to update. For multipair strategies, providing this parameter is strongly recommended.

        :param stop_loss:
            Stop loss in US dollar terms

        :param mid_price:
            Mid price of the pair (https://tradingstrategy.ai/glossary/mid-price). Provide when possible for most complete statistical analysis. In certain cases, it may not be easily available, so it's optional.
        """

        mid_price =  self.get_pricing_model(position.pair).get_mid_price(self.timestamp, position.pair)

        position.trigger_updates.append(TriggerPriceUpdate(
            timestamp=self.timestamp,
            stop_loss_before = position.stop_loss,
            stop_loss_after = stop_loss,
            mid_price = mid_price,
            take_profit_before = position.take_profit,
            take_profit_after = position.take_profit,  # No changes to take profit
        ))

        position.stop_loss = stop_loss

    def open_credit_supply_position_for_reserves(self, amount: USDollarAmount) -> List[TradeExecution]:
        """Move reserve currency to a credit supply position.

        :param amount:
            Amount of cash to lend out

        :return:
            List of trades that will open this credit position
        """

        lending_reserve_identifier = self.strategy_universe.get_credit_supply_pair()

        _, trade, _ = self.state.supply_credit(
            self.timestamp,
            lending_reserve_identifier,
            collateral_quantity=Decimal(amount),
            trade_type=TradeType.rebalance,
            reserve_currency=self.strategy_universe.get_reserve_asset(),
            collateral_asset_price=1.0,
        )

        return [trade]
    
    def open_short(
        self,
        pair: Union[DEXPair, TradingPairIdentifier],
        value: USDollarAmount,
        *,
        leverage: LeverageMultiplier = 1.0,
        take_profit_pct: float | None = None,
        stop_loss_pct: float | None = None,
    ) -> list[TradeExecution]:
        """Open a short position.

        NOTE: take_profit_pct and stop_loss_pct are more related to capital at risk
        percentage than to the price. So this will likely be changed in the future.

        :param pair:
            Trading pair where we take the position

        :param value:
            How much cash reserves we allocate to open this position.

            In US dollars.

            For example to open 2x short where we allocate $1000
            from our reserves, this value is $1000.

        :param leverage:
            Leverage level to use for the short position

        :param take_profit_pct:
            If set, set the position take profit relative to the current market price.
            1.0 is the current market price.
            If asset opening price is $1000, take_profit_pct=1.05
            will buy back the asset when price reaches $950.

        :param stop_loss_pct:
            If set, set the position to trigger stop loss relative to the current market price.
            1.0 is the current market price.
            If asset opening price is $1000, stop_loss_pct=0.98
            will buy back the asset when price reaches $1020.

        :return:
            List of trades that will open this credit position
        """

        if isinstance(pair, DEXPair):
            executor_pair = translate_trading_pair(pair)
        else:
            executor_pair = pair

        shorting_pair = self.strategy_universe.get_shorting_pair(executor_pair)

        # Check that pair data looks good
        assert shorting_pair.kind.is_shorting()
        assert shorting_pair.base.underlying is not None, f"Lacks underlying asset: {shorting_pair.base}"
        assert shorting_pair.quote.underlying is not None, f"Lacks underlying asset: {shorting_pair.quote}"

        if type(value) == float:
            value = Decimal(value)

        pricing_pair = shorting_pair.get_pricing_pair()  # should be effectively the same as executor_pair
        price_structure = self.pricing_model.get_sell_price(self.timestamp, pricing_pair, value)
        collateral_price = self.reserve_price
        borrowed_asset_price = price_structure.price

        estimation: LeverageEstimate = LeverageEstimate.open_short(
            starting_reserve=value,
            leverage=leverage,
            borrowed_asset_price=borrowed_asset_price,
            shorting_pair=shorting_pair,
            fee=executor_pair.fee,
        )

        logger.info("Opening a short position at timetamp %s\n"
                    "Shorting pair is %s\n"
                    "Execution pair is %s\n"
                    "Collateral amount: %s USD\n"
                    "Borrow amount: %s USD (%s %s)\n"
                    "Collateral asset price: %s %s/USD\n"
                    "Borrowed asset price: %s %s/USD (assumed execution)\n",
                    "Liquidation price: %s %s/USD\n",
                    self.timestamp,
                    shorting_pair,
                    executor_pair,
                    estimation.total_collateral_quantity,
                    estimation.borrowed_value, estimation.total_borrowed_quantity, executor_pair.base.token_symbol,
                    collateral_price, executor_pair.quote.token_symbol,
                    borrowed_asset_price, executor_pair.base.token_symbol,
                    estimation.liquidation_price, executor_pair.base.token_symbol,
                    )

        position, trade, _ = self.state.trade_short(
            self.timestamp,
            pair=shorting_pair,
            borrowed_quantity=-estimation.total_borrowed_quantity,
            collateral_quantity=value,
            borrowed_asset_price=price_structure.price,
            trade_type=TradeType.rebalance,
            reserve_currency=self.reserve_currency,
            collateral_asset_price=collateral_price,
            planned_collateral_consumption=estimation.additional_collateral_quantity,  # This is amount how much aToken is leverated besides our starting collateral
        )

        # record liquidation price into the position
        position.liquidation_price = estimation.liquidation_price

        if take_profit_pct:
            assert take_profit_pct > 1, f"Short position's take_profit_pct must be greater than 1, got {take_profit_pct}"
            position.take_profit = price_structure.mid_price * (2 - take_profit_pct)

        if stop_loss_pct is not None:
            assert 0 < stop_loss_pct < 1, f"Short position's stop_loss_pct must be 0..1, got {stop_loss_pct}"

            # calculate distance to liquidation price and make sure stoploss is far from that
            mid_price = Decimal(price_structure.mid_price)
            liquidation_distance = (estimation.liquidation_price - mid_price) / mid_price
            assert 1 - stop_loss_pct < liquidation_distance, f"stop_loss_pct must be bigger than liquidation distance {1 - liquidation_distance:.4f}, got {stop_loss_pct}"

            self.update_stop_loss(position, price_structure.mid_price * (2 - stop_loss_pct))

        return [trade]
    
    def close_short_position(
        self,
        position: TradingPosition,
        quantity: float | Decimal | None = None,
        notes: Optional[str] = None,
        trade_type: TradeType = TradeType.rebalance,
    ) -> List[TradeExecution]:
        """Close a short position

        :param position:
            Position to close.

            Must be a short position.

        :param quantity:
            How much of the quantity we reduce.

            If not given close the full position.

        :return:
            New trades to be executed
        """

        assert self.strategy_universe, "Make sure trading_strategy_engine_version = 0.3. Short does not work with old decide_trades()."
        
        # Check that pair data looks good
        pair = position.pair
        assert pair.kind.is_shorting()
        assert pair.base.underlying is not None, f"Lacks underlying asset: {pair.base}"
        assert pair.quote.underlying is not None, f"Lacks underlying asset: {pair.quote}"

        if quantity is None:
            quantity = position.get_quantity()

        if type(quantity) == float:
            # TODO: Snap the amount to the full position size if rounding errors
            quantity = Decimal(quantity)

        # TODO: Hardcoded USD exchange rate
        reserve_asset = self.strategy_universe.get_reserve_asset()
        price_structure = self.pricing_model.get_sell_price(self.timestamp, pair.underlying_spot_pair, 1)

        position, trade, _ = self.state.trade_short(
            self.timestamp,
            closing=True,
            pair=pair,
            borrowed_asset_price=price_structure.price,
            trade_type=trade_type,
            reserve_currency=self.reserve_currency,
            collateral_asset_price=1.0,
            notes=notes,
            position=position,
        )

        return [trade]
    
    def get_pricing_model(self, pair: TradingPairIdentifier) -> PricingModel:
        """Get the pricing model used by this strategy.
        
        :param pair:
            Trading pair for which we want to have the pricing model
        """
        return get_pricing_model_for_pair(pair, self.pricing_models)  

def get_pricing_model_for_pair(pair: TradingPairIdentifier | DEXPair, pricing_models: List[PricingModel], set_routing_hint: bool = True) -> PricingModel:
    """Get the pricing model for a pair.
    
    :param pair:
        Trading pair for which we want to have the pricing model


    :param pricing_models:
        List of pricing models to choose from
        
    :set_routing_hint:
        Whether to set the routing hint for the pricing model. This is useful when we have multiple pricing models that could apply to the same pair, and we want to make sure that the correct pricing model is used for the pair.


    :return:
        Pricing model for the pair
    """
    if len(pricing_models) == 1:
        return pricing_models[0]


    locked = False
    rm = None
    routing_model = None
    final_pricing_model = None

    for pricing_model in pricing_models:

        rm = pricing_model.routing_model

        if hasattr(rm, "factory_router_map"):  # uniswap v2 like
            keys = list(rm.factory_router_map.keys())
            assert len(keys) == 1, "Only one factory router map supported for now"
            factory_address = keys[0]
        elif hasattr(rm, "address_map"):  # uniswap v3 like
            factory_address = rm.address_map["factory"] 
        else:
            raise NotImplementedError("Routing model not supported")

        if factory_address.lower() == pair.exchange_address.lower() and rm.chain_id == pair.chain_id:
            if locked == True:
                raise LookupError("Multiple routing models for same exchange (on same chain) not supported")

            routing_model = rm
            final_pricing_model = pricing_model
            locked = True

    if not routing_model:
        raise NotImplementedError("Unable to find routing_model for pair, make sure to add correct routing models for the pairs that you want to trade")

    if not isinstance(routing_model, (UniswapV2SimpleRoutingModel, UniswapV3SimpleRoutingModel, BacktestRoutingModel, BacktestRoutingIgnoredModel)):
        raise NotImplementedError("Routing model not supported")
    
    # Needed in tradeexecutor/state/portfolio.py::choose_valudation_method_and_revalue_position
    if set_routing_hint:
        pair.routing_hint = routing_model.routing_hint  

    assert final_pricing_model is not None, "Unable to find pricing model for pair"

    return final_pricing_model