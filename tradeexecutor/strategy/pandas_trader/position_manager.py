"""Positions open and closing management."""

import datetime
import warnings
from decimal import Decimal
from io import StringIO
from typing import List, Optional, Union, Literal, Set
import logging

import cachetools
import pandas as pd

from tradeexecutor.utils.accuracy import QUANTITY_EPSILON
from tradingstrategy.candle import CandleSampleUnavailable
from tradingstrategy.pair import DEXPair, HumanReadableTradingPairDescription
from tradingstrategy.universe import Universe

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
from tradeexecutor.state.loan import LiquidationRisked
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.position import TradingPosition, TriggerPriceUpdate
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeType, TradeExecution, TradeFlag
from tradeexecutor.state.types import USDollarAmount, Percent, LeverageMultiplier
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trading_strategy_universe import translate_trading_pair, TradingStrategyUniverse
from tradeexecutor.utils.leverage_calculations import LeverageEstimate

logger = logging.getLogger(__name__)


#: Cache translate_trading_pair() result data structures
#:
#: See :py:meth:`PositionManager.__init__`.
#:
DEFAULT_TRADING_PAIR_CACHE = cachetools.Cache(maxsize=50000)



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

    def __init__(
        self,
        timestamp: Union[datetime.datetime, pd.Timestamp],
        universe: Universe | TradingStrategyUniverse,
        state: State,
        pricing_model: PricingModel,
        default_slippage_tolerance=0.017,
        trading_pair_cache=DEFAULT_TRADING_PAIR_CACHE,
    ):

        """Create a new PositionManager instance.
        
        Call within `decide_trades` function.
        
        :param timestamp: 
            The timestamp of the current strategy cycle
            
        :param universe: 
            Trading universe of available assets
        
        :param state: 
            Current state of the trade execution
            
        :param pricing_model:
            The model to estimate prices for any trades
         
        :param default_slippage_tolerance: 

            The max slippage tolerance parameter set for any trades if not overriden trade-by-trade basis.

            Default to 1.7% max slippage or 170 BPS.

        :param trading_pair_cache:
            Trading pair cache.

            Used to speed up trading pair look up on multipair strategies.

            See :py:meth:`get_trading_pair`.

        """

        assert pricing_model, "pricing_model is needed in order to know buy/sell price of new positions"

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
            self.data_universe = universe.data_universe
        else:
            raise RuntimeError(f"Does not know the universe: {universe}")

        assert self.data_universe is not None, "Data universe was None"

        self.state = state
        self.pricing_model = pricing_model
        self.default_slippage_tolerance = default_slippage_tolerance

        reserve_currency, reserve_price = state.portfolio.get_default_reserve_asset()

        self.reserve_currency = reserve_currency
        self.reserve_price = reserve_price
        self.trading_pair_cache = trading_pair_cache

    def is_any_open(self) -> bool:
        """Do we have any positions open.
        
        See also

        - :py:meth:`is_any_long_position_open`

        - :py:meth:`is_any_short_position_open`
        
        - :py:meth:`is_any_credit_supply_position_open`
        """
        return len(self.state.portfolio.open_positions) > 0
        
    def is_any_long_position_open(self) -> bool:
        """Do we have any long positions open.
        
        See also

        - :py:meth:`is_any_short_position_open`
        
        - :py:meth:`is_any_credit_supply_position_open`
        """
        return len([
            p for p in self.state.portfolio.open_positions.values()
            if p.is_long()
        ]) > 0
    
    def is_any_short_position_open(self) -> bool:
        """Do we have any short positions open.
        
        See also

        - :py:meth:`is_any_long_position_open`
        
        - :py:meth:`is_any_credit_supply_position_open`
        """
        return len([
            p for p in self.state.portfolio.open_positions.values()
            if p.is_short()
        ]) > 0
    
    def is_any_credit_supply_position_open(self) -> bool:
        """Do we have any credit supply positions open.
        
        See also

        - :py:meth:`is_any_long_position_open`

        - :py:meth:`is_any_short_position_open`
        """
        return len([
            p for p in self.state.portfolio.open_positions.values()
            if p.is_credit_supply()
        ]) > 0

    def get_current_cash(self) -> USDollarAmount:
        """Get the available cash in hand.

        - Cash that sits in the strategy treasury

        - Cash not in the open trading positions

        - Cash not allocated to the trading positions that are going to be opened on this cycle

        :return:
            US Dollar amount
        """
        cash = self.state.portfolio.get_current_cash()  # How much cash we have in a hand
        return cash

    def get_current_position(self) -> TradingPosition:
        """Get the current single position.

        This is a shortcut function for trading strategies
        that operate only a single trading pair and a single position.

        See also

        - :py:meth:`get_current_long_position`

        - :py:meth:`get_current_short_position`
        
        - :py:meth:`get_current_credit_supply_position`

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
    
    def _get_single_open_position_for_kind(self, kind: str) -> TradingPosition:
        """Get the current single position for the given kind.
        
        This is underlying method, do not use directly
        """
        assert kind in ["long", "short", "credit_supply"], f"Unknown kind received: {kind}"

        open_positions = [
            position
            for position in self.state.portfolio.open_positions.values()
            if any([
                kind == "long" and position.is_long(),
                kind == "short" and position.is_short(),
                kind == "credit_supply" and position.is_credit_supply(),
            ])
        ]

        if len(open_positions) == 0:
            raise NoSingleOpenPositionException(f"No {kind} position open at {self.timestamp}")

        if len(open_positions) > 1:
            raise NoSingleOpenPositionException(f"Multiple {kind} positions ({len(open_positions)}) open at {self.timestamp}")

        return open_positions[0]
    
    def get_current_long_position(self):
        """Get the current single long position.

        This is a shortcut function for trading strategies
        that operate only a single trading pair and a single long position.

        See also

        - :py:meth:`get_current_short_position`
        
        - :py:meth:`get_current_credit_supply_position`


        :return:
            Currently open long trading position

        :raise NoSingleOpenPositionError:
            If you do not have a position open or there are multiple positions open.
        """
        return self._get_single_open_position_for_kind("long")
    
    def get_current_short_position(self):
        """Get the current single short position.

        This is a shortcut function for trading strategies
        that operate only a single trading pair and a single short position.

        If you have multiple short positions open use :py:meth:`get_current_position_for_pair`
        to distinguish between them.

        .. code-block:: python

            # aave_usdc is an instance of TradingPairIdentifier
            aave_shorting_pair = strategy_universe.get_shorting_pair(aave_usdc)
            aave_short_position = position_manager.get_current_position_for_pair(aave_shorting_pair)

        See also

        - :py:meth:`get_current_long_position`
        
        - :py:meth:`get_current_credit_supply_position`

        :return:
            Currently open short trading position

        :raise NoSingleOpenPositionError:
            If you do not have a position open or there are multiple positions open.
        """
        return self._get_single_open_position_for_kind("short")

    def get_current_credit_supply_position(self):
        """Get the current single credit supply position.

        This is a shortcut function for trading strategies
        that operate only a single trading pair and a single credit supply position.

        See also

        - :py:meth:`get_current_long_position`

        - :py:meth:`get_current_short_position`


        :return:
            Currently open credit supply trading position

        :raise NoSingleOpenPositionError:
            If you do not have a position open or there are multiple positions open.
        """
        return self._get_single_open_position_for_kind("credit_supply")

    def get_current_position_for_pair(self, pair: TradingPairIdentifier) -> Optional[TradingPosition]:
        """Get the current open position for a specific trading pair.

        :return:
            Currently open trading position.

            If there is no open position return None.

        """
        return self.state.portfolio.get_position_by_trading_pair(pair)
    
    def get_closed_positions_for_pair(
        self,
        pair: TradingPairIdentifier,
        include_test_position: bool = False,
    ) -> list[TradingPosition]:
        """Get closed positions for a specific trading pair.

        :return:
            All closed trading position of a trading pair

            If there is no closed position return empty list.
        """
        return self.state.portfolio.get_closed_positions_for_pair(pair, include_test_position=include_test_position)

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

    def get_trading_pair(self, pair: int | DEXPair | HumanReadableTradingPairDescription) -> TradingPairIdentifier:
        """Get a trading pair identifier by its internal id, description or `DEXPair` data object.

        Example:

        .. code-block:: python

            # List of pair descriptions we used to look up pair metadata
            our_pairs = [
                (ChainId.centralised_exchange, "binance", "BTC", "USDT"),
                (ChainId.centralised_exchange, "binance", "ETH", "USDT"),
            ]

            # Resolve our pair metadata for our two pair strategy
            position_manager = PositionManager(timestamp, strategy_universe, state, pricing_model)
            btc_pair = position_manager.get_trading_pair(our_pairs[0])
            eth_pair = position_manager.get_trading_pair(our_pairs[1])

            position_manager.log(f"BTC pair data is: {btc_pair}")

        Note that internal integer ids are not stable over
        multiple trade cycles and might be reset.
        Always use (chain id, smart contract) for persistent
        pair identifier.

        :return:
            Trading pair identifier.

            The identifier is a pass-by-copy reference used in the strategy state internally.
        """

        cached = self.trading_pair_cache.get(pair)
        if cached is None:

            if type(pair) == int:
                pair_id = pair
                dex_pair = self.data_universe.pairs.get_pair_by_id(pair_id)
            elif type(pair) == tuple:
                dex_pair = self.data_universe.pairs.get_pair_by_human_description(pair)
            elif isinstance(pair, DEXPair):
                dex_pair = pair
            else:
                raise RuntimeError(f"Unknown trading pair reference type: {pair}")

            # Rebuild TradingPairIdentifier data structure
            cached = translate_trading_pair(dex_pair)
            self.trading_pair_cache[pair] = cached

        return cached

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
        return self.pricing_model.get_pair_fee(self.timestamp, pair)

    def open_1x_long(
        self,
         pair: Union[DEXPair, TradingPairIdentifier],
         value: USDollarAmount | Decimal,
         take_profit_pct: Optional[float] = None,
         stop_loss_pct: Optional[float] = None,
         trailing_stop_loss_pct: Optional[float] = None,
         stop_loss_usd: Optional[USDollarAmount] = None,
         notes: Optional[str] = None,
         slippage_tolerance: Optional[float] = None,
         take_profit_usd: Optional[USDollarAmount] = None,
    ) -> List[TradeExecution]:
        """Deprecated function for opening a spot position.

        Use :py:meth:`open_spot` instead.
        """
        return self.open_spot(
            pair=pair,
            value=value,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            trailing_stop_loss_pct=trailing_stop_loss_pct,
            stop_loss_usd=stop_loss_usd,
            notes=notes,
            slippage_tolerance=slippage_tolerance,
            take_profit_usd=take_profit_usd,
        )

    def open_spot(
        self,
        pair: Union[DEXPair, TradingPairIdentifier | None],
        value: USDollarAmount | Decimal,
        take_profit_pct: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
        trailing_stop_loss_pct: Optional[float] = None,
        stop_loss_usd: Optional[USDollarAmount] = None,
        notes: Optional[str] = None,
        slippage_tolerance: Optional[float] = None,
        flags: Set[TradeFlag] | None = None,
        take_profit_usd: Optional[USDollarAmount] = None,
    ) -> List[TradeExecution]:
        """Open a spot position.

        - For simple buy and hold trades

        - Open a spot market buy.

        - Checks that there is not existing position - cannot increase position

        See also

        - :py:meth:`adjust_position` if you want increase/decrease an existing position size

        - :py:meth:`close_position` if you want exit an position

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

        :param take_profit_usd:
            If set, set the position take profit at the given dollar price.
            Cannot be used with take_profit_pct.

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

        assert value > 0, f"For opening long, the value must be positive. Got: {value} on {pair}"

        # Convert amount of reserve currency to the decimal
        # so we can have exact numbers from this point forward
        if type(value) == float:
            value = Decimal(value)

        price_structure = self.pricing_model.get_buy_price(self.timestamp, executor_pair, value)

        assert type(price_structure.mid_price) == float

        reserve_asset, reserve_price = self.state.portfolio.get_default_reserve_asset()

        slippage_tolerance = slippage_tolerance or self.default_slippage_tolerance

        if not flags:
            flags = set()

        flags = {TradeFlag.open, TradeFlag.increase} | flags

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
            flags=flags,
        )

        if not created:
            msg = explain_open_position_failure(
                self.state.portfolio,
                executor_pair,
                self.timestamp,
                "opening new spot position")
            assert created, f"Opening a new position failed:\n{msg}"

        if take_profit_pct:
            position.take_profit = price_structure.mid_price * take_profit_pct

        if take_profit_usd:
            assert not take_profit_pct, "You cannot give both take_profit_pct and take_profit_usd"
            assert take_profit_usd > price_structure.mid_price, f"take_profit_usd must be more than mid_price got {take_profit_usd} <= {price_structure.mid_price}"
            position.take_profit = take_profit_usd

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

        if trade.is_buy():
            assert trade.planned_quantity > QUANTITY_EPSILON, f"Bad buy quantity: {trade}"

        logger.info("Generated trade %s to open a spot position %s", trade.get_human_description(), position.get_human_readable_name())

        return [trade]

    def adjust_position(self,
                        pair: TradingPairIdentifier,
                        dollar_delta: USDollarAmount,
                        quantity_delta: float,
                        weight: float,
                        stop_loss: Optional[Percent] = None,
                        take_profit: Optional[Percent] = None,
                        trailing_stop_loss: Optional[Percent] = None,
                        slippage_tolerance: Optional[float] = None,
                        override_stop_loss=False,
                        notes: Optional[str] = None,
                        ) -> List[TradeExecution]:
        """Adjust holdings for a certain position.

        Used to rebalance positions.

        This method rarely needs to be called directly,
        but is usually part of portfolio construction strategy
        that is using :py:class:`tradeexecutor.strategy.alpha_model.AlphaModel`.

        A new position is opened if no existing position is open.
        If everything is sold, the old position is closed

        If the rebalance is sell (`dollar_amount_delta` is negative),
        then calculate the quantity of the asset to sell based
        on the latest available market price on the position.

        .. warning ::

            Adjust position cannot be used to close an existing position, because
            epsilons in quantity math. Use :py:meth:`close_position`] for this.

        :param pair:
            Trading pair which position we adjust

        :param dollar_delta:
            How much we want to increase/decrease the position in US dollar terms.

            TODO: If you are selling the assets, you need to calculate the expected
            dollar estimate yourself at the moment.

        :param quantity_delta:
            How much we want to increase/decrease the position in the asset unit terms.

            Used only when decreasing existing positions (selling).
            Set to ``None`` if not selling.

        :param weight:
            What is the weight of the asset in the new target portfolio 0....1.
            Currently only used to detect condition "sell all" instead of
            trying to match quantity/price conversion.

            If unsure and buying, set to ``1``.

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

        :param notes:
            Human-readable plain text notes on the trade.

            Used for diagnostics.

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
                price_structure = self.pricing_model.get_buy_price(self.timestamp, pair, dollar_delta)
            else:
                quantity_delta = Decimal(quantity_delta) if isinstance(quantity_delta, float | int) else quantity_delta
                price_structure = self.pricing_model.get_sell_price(self.timestamp, pair, abs(quantity_delta))

        except CandleSampleUnavailable as e:
            # Backtesting cannot fetch price for an asset,
            # probably not enough data and the pair is trading early?
            data_delay_tolerance = getattr(self.pricing_model, "data_delay_tolerance", None)
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
                notes=notes,
            )
        else:
            # Sell
            # Convert dollar amount to quantity of the last known price

            assert quantity_delta is not None
            assert quantity_delta < 0, f"Received non-negative sell quantity {quantity_delta} for {pair}"

            # position = self.state.portfolio.get_position_by_trading_pair(pair)
            # assert position is not None, f"Assumed {pair} has open short position because of attempt sell at {dollar_delta} USD adjust, but did not get open position"

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

    def close_spot_position(
        self,
        position: TradingPosition,
        trade_type: TradeType=TradeType.rebalance,
        notes: Optional[str] = None,
        slippage_tolerance: Optional[float] = None,
        flags: Set[TradeFlag] | None = None,
    ) -> List[TradeExecution]:
        """Close a single spot market trading position.

        See :py:meth:`close_position` for usage.
        """

        assert position.is_spot_market()

        quantity_left = position.get_available_trading_quantity()

        pair = position.pair
        quantity = quantity_left
        price_structure = self.pricing_model.get_sell_price(self.timestamp, pair, quantity=quantity)

        reserve_asset, reserve_price = self.state.portfolio.get_default_reserve_asset()

        slippage_tolerance = slippage_tolerance or self.default_slippage_tolerance

        logger.info(
            "Preparing to close position %s, quantity %s, pricing %s, profit %s, slippage tolerance: %f %%",
            position,
            quantity,
            price_structure,
            position.get_unrealised_profit_usd(),
            slippage_tolerance * 100,
        )

        if not flags:
            flags = set()

        flags = {TradeFlag.close} | flags

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
            closing=True,
            flags=flags,

        )
        assert position == position2, f"Somehow messed up the close_position() trade.\n" \
                                      f"Original position: {position}.\n" \
                                      f"Trade's position: {position2}.\n" \
                                      f"Trade: {trade}\n" \
                                      f"Quantity left: {quantity_left}\n" \
                                      f"Price structure: {price_structure}\n" \
                                      f"Reserve asset: {reserve_asset}\n"

        assert trade.closing
        return [trade]

    def close_credit_supply_position(
        self,
        position: TradingPosition,
        quantity: float | Decimal | None = None,
        notes: Optional[str] = None,
        trade_type: TradeType = TradeType.rebalance,
        flags: Set[TradeFlag] | None = None,
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

        if not flags:
            flags = set()

        flags = {TradeFlag.close} | flags

        _, trade, _ = self.state.supply_credit(
            self.timestamp,
            pair,
            collateral_asset_price=price,
            collateral_quantity=-quantity,
            trade_type=trade_type,
            reserve_currency=reserve_asset,
            notes=notes,
            position=position,
            closing=True,
            flags=flags,
        )
        return [trade]

    def close_position(
        self,
        position: TradingPosition,
        trade_type: TradeType | None = None,
        notes: Optional[str] = None,
        slippage_tolerance: Optional[float] = None,
        flags: Set[TradeFlag] | None = None,
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

        assert position is not None, f"close_position() called with position == None"
        # assert position.is_long(), "Only long supported for now"
        assert position.is_open(), f"Tried to close already closed position {position}"

        quantity_left = position.get_available_trading_quantity()

        if quantity_left == 0:
            # We have already generated closing trades for this position earlier?
            # Add some debug information because these are hard to diagnose
            planned_trades = [t for t in position.trades.values() if t.is_planned()]
            planned = sum([t.get_position_quantity() for t in planned_trades])  # Sell values sum to negative
            live = position.get_quantity()  # What was the position quantity before executing any of planned trades
            logger.warning(
                "Tried to close position that is likely already closed, as there are no tokens to sell: %s.\n"
                "Quantity left zero. Planned tokens: %f, live tokens: %f\n"
                "We have existing planned trades: %s",
                position,
                planned,
                live,
                planned_trades,
            )
            return []

        if position.is_spot_market():

            if trade_type is None:
                trade_type = TradeType.rebalance

            return self.close_spot_position(
                position,
                trade_type,
                notes,
                slippage_tolerance,
                flags=flags,
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

            return self.close_short(
                position,
                trade_type=trade_type,
                notes=notes,
                flags=flags,
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
        pricing_model = self.pricing_model
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

        pair = position.pair.get_pricing_pair()
        mid_price =  self.pricing_model.get_mid_price(self.timestamp, pair)

        position.trigger_updates.append(TriggerPriceUpdate(
            timestamp=self.timestamp,
            stop_loss_before=position.stop_loss,
            stop_loss_after=stop_loss,
            mid_price=mid_price,
            take_profit_before=position.take_profit,
            take_profit_after=position.take_profit,  # No changes to take profit
        ))

        position.stop_loss = stop_loss

    def open_credit_supply_position_for_reserves(
        self,
        amount: USDollarAmount,
        flags: Set[TradeFlag] | None = None,
        notes: str | None = None,
    ) -> List[TradeExecution]:
        """Move reserve currency to a credit supply position.

        :param amount:
            Amount of cash to lend out

        :return:
            List of trades that will open this credit position
        """

        assert self.strategy_universe is not None, f"PositionManager.strategy_universe not set, data_universe is {self.data_universe}"
        assert self.strategy_universe.has_lending_data(), "open_credit_supply_position_for_reserves(): lending data not loaded"

        lending_reserve_identifier = self.strategy_universe.get_credit_supply_pair()

        if not flags:
            flags = set()
        flags = {TradeFlag.open} | flags

        position, trade, _ = self.state.supply_credit(
            self.timestamp,
            lending_reserve_identifier,
            collateral_quantity=Decimal(amount),
            trade_type=TradeType.rebalance,
            reserve_currency=self.strategy_universe.get_reserve_asset(),
            collateral_asset_price=1.0,
            flags=flags,
        )

        if notes:
            assert type(notes) == str
            position.add_notes_message(notes)
            trade.add_note(notes)

        return [trade]
    
    def open_short(
        self,
        pair: Union[DEXPair, TradingPairIdentifier],
        value: USDollarAmount,
        *,
        leverage: LeverageMultiplier = 1.0,
        take_profit_pct: float | None = None,
        stop_loss_pct: float | None = None,
        trailing_stop_loss_pct: float | None = None,
        notes: str | None = None,
        flags: Set[TradeFlag] | None = None,
    ) -> list[TradeExecution]:
        """Open a short position.

        NOTE: take_profit_pct and stop_loss_pct are more related to capital at risk
        percentage than to the price. So this will likely be changed in the future.

        :param pair:
            Trading pair where we take the position.

            For lending protocol shorts must be the underlying spot pair.

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

        :param trailing_stop_loss_pct:
            If set, set the position to trigger trailing stop loss relative to
            the current market price. Cannot be used with stop_loss_pct.

        :return:
            List of trades that will open this credit position
        """

        if isinstance(pair, DEXPair):
            executor_pair = translate_trading_pair(pair)
        else:
            executor_pair = pair

        assert executor_pair.is_spot(), f"Give a spot pair as input and we will figure out shorting pair for you. Got {executor_pair}"

        assert self.strategy_universe is not None, f"PositionManager.strategy_universe not set, data_universe is {self.data_universe}"

        shorting_pair = self.strategy_universe.get_shorting_pair(executor_pair)

        # Check that pair data looks good
        assert shorting_pair.kind.is_shorting()
        assert shorting_pair.base.underlying is not None, f"Lacks underlying asset: {shorting_pair.base}"
        assert shorting_pair.quote.underlying is not None, f"Lacks underlying asset: {shorting_pair.quote}"

        if type(value) == float:
            value = Decimal(value)

        pricing_pair = shorting_pair.get_pricing_pair()  # should be effectively the same as executor_pair
        price_structure = self.pricing_model.get_sell_price(self.timestamp, pricing_pair, Decimal(1))
        collateral_price = self.reserve_price
        borrowed_asset_price = price_structure.price

        estimation: LeverageEstimate = LeverageEstimate.open_short(
            starting_reserve=value,
            leverage=leverage,
            borrowed_asset_price=price_structure.mid_price,
            shorting_pair=shorting_pair,
            fee=executor_pair.fee,
        )

        logger.info("Opening a short position at timestamp %s\n"
                    "Shorting pair is %s\n"
                    "Execution pair is %s\n"
                    "Collateral amount: %s USD\n"
                    "Borrow amount: %s USD (%s %s)\n"
                    "Collateral asset price: %s %s/USD\n"
                    "Borrowed asset price: %s %s/USD (assumed execution)\n"
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

        if not flags:
            flags = set()

        flags = {TradeFlag.open} | flags

        position, trade, created = self.state.trade_short(
            self.timestamp,
            pair=shorting_pair,
            borrowed_quantity=-estimation.total_borrowed_quantity,
            collateral_quantity=value,
            borrowed_asset_price=borrowed_asset_price,
            trade_type=TradeType.rebalance,
            reserve_currency=self.reserve_currency,
            planned_mid_price=price_structure.mid_price,
            collateral_asset_price=collateral_price,
            planned_collateral_consumption=estimation.additional_collateral_quantity,  # This is amount how much aToken is leverated besides our starting collateral
            # TODO: planned_reserve-planned_collateral_allocation refactor later
            planned_collateral_allocation=0,
            lp_fees_estimated=estimation.lp_fees,
            notes=notes,
            flags=flags,
        )
        assert created, f"open_short() was called, but there was an existing position for pair: {executor_pair}"

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

        if trailing_stop_loss_pct:
            assert stop_loss_pct is None, "You cannot give both stop_loss_pct and trailing_stop_loss_pct"
            assert 0 < trailing_stop_loss_pct < 1, f"trailing_stop_loss_pct must be 0..1, got {trailing_stop_loss_pct}"

            # calculate distance to liquidation price and make sure stoploss is far from that
            mid_price = Decimal(price_structure.mid_price)
            liquidation_distance = (estimation.liquidation_price - mid_price) / mid_price
            assert 1 - trailing_stop_loss_pct < liquidation_distance, f"trailing_stop_loss_pct must be bigger than liquidation distance {1 - liquidation_distance:.4f}, got {trailing_stop_loss_pct}"

            self.update_stop_loss(position, price_structure.mid_price * (2 - trailing_stop_loss_pct))
            position.trailing_stop_loss_pct = trailing_stop_loss_pct

        return [trade]

    def close_short_position(
        self,
        position: TradingPosition,
        quantity: float | Decimal | None = None,
        notes: Optional[str] = None,
        trade_type: TradeType = TradeType.rebalance,
    ) -> List[TradeExecution]:
        """Legacy.

        Use :py:meth:`close_short`.

        """
        warnings.warn('This function is deprecated. Use PositionManager.close_short() instead', DeprecationWarning, stacklevel=2)
        return self.close_short(position, quantity, notes, trade_type)
    
    def close_short(
        self,
        position: TradingPosition,
        quantity: float | Decimal | None = None,
        notes: Optional[str] = None,
        trade_type: TradeType = TradeType.rebalance,
        flags: Set[TradeFlag] | None = None,
    ) -> List[TradeExecution]:
        """Close a short position

        - Buy back the shorted token

        - Release collateral and return it as cash to the reserves

        - Move any gained interest back to the reserves as well

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
        price_structure = self.pricing_model.get_buy_price(self.timestamp, pair.underlying_spot_pair, Decimal(1))

        if not flags:
            flags = set()

        flags = {TradeFlag.close, TradeFlag.reduce} | flags

        position2, trade, _ = self.state.trade_short(
            self.timestamp,
            closing=True,
            pair=pair,
            borrowed_asset_price=price_structure.price,
            trade_type=trade_type,
            reserve_currency=self.reserve_currency,
            planned_mid_price=price_structure.mid_price,
            collateral_asset_price=1.0,
            notes=notes,
            position=position,
            flags=flags,
        )

        assert position == position2, f"Somehow messed up the close_position() trade.\n" \
                                      f"Original position: {position}.\n" \
                                      f"Trade's position: {position2}.\n" \
                                      f"Trade: {trade}\n"

        assert trade.closing
        return [trade]

    def adjust_short(
        self,
        position: TradingPosition,
        new_value: USDollarAmount,
        notes: Optional[str] = None,
        trade_type: TradeType = TradeType.rebalance,
        minimum_rebalance_trade_threshold: USDollarAmount = 0.0,
        flags: Set[TradeFlag] | None = None,
    ) -> List[TradeExecution]:
        """Increase/decrease short based on the amount of collateral.

        Short adjust used in alpha model.

        - Short is already open

        - The amount of short is changing

        - We want to maintain the existing leverage

        - Any excess collateral is returned to cash reserves,
          any new collateral is moved for the cash reserves to the short

        - Cannot be used to open/close position

        See also

        - :py:meth:`open_short`

        - :py:meth:`close_short`

        :param position:
            Position to close.

            Must be a short position.

        :param new_value:
            The allocated collateral for this position after the trade in US Dollar reserves.

            The absolute amunt of reserve currency we will use for this short.

        :param quantity:
            How much of the quantity we reduce.

            If not given close the full position.

        :param price:
            The spot price of the underlying pair.

        :return:
            New trades to be executed
        """
        assert isinstance(position, TradingPosition), f"Got: {position.__class__}: {position}"

        # Check that pair data looks good
        pair = position.pair
        assert pair.kind.is_shorting()
        assert pair.base.underlying is not None, f"Lacks underlying asset: {pair.base}"
        assert pair.quote.underlying is not None, f"Lacks underlying asset: {pair.quote}"
        assert pair.underlying_spot_pair.quote.is_stablecoin(), f"Assuming stablecoin backed pair"
        assert new_value > 0, "Cannot use adjust_short() to close short position"
        assert position.is_open(), "Cannot adjust closed short position"

        underlying = pair.underlying_spot_pair

        value = position.get_value()
        delta = new_value - value

        if abs(delta) == 0:
            logger.info("Change is abs zero for %s", pair)
            return []

        if abs(delta) < minimum_rebalance_trade_threshold:
            logger.info(
                "Does not rebalance pair %s. Threshold: %f, value delta %f",
                minimum_rebalance_trade_threshold,
                delta,
             )
            return []

        if not flags:
            flags = set()

        flags = {TradeFlag.increase, TradeFlag.reduce} | flags

        state = self.state

        loan = position.loan
        assert loan is not None, f"Position did not have existing loan structure: {position}"

        reserve_currency, reserve_price = state.portfolio.get_default_reserve_asset()

        # TODO: Price impact ignored
        mid_price = self.pricing_model.get_mid_price(self.timestamp, underlying)

        logger.info(
            "Adjusting short position %s, mid price %f, delta %f USD, existing leverage %fx",
            position,
            mid_price,
            delta,
            loan.get_leverage(),
        )

        # See test_short_increase_size and test_short_decrease_size
        borrowed_quantity_delta = 0

        # See test_short_increase_size
        collateral_adjustment = Decimal(new_value - loan.get_net_asset_value())

        target_params = LeverageEstimate.open_short(
            new_value,
            loan.get_leverage(),
            mid_price,
            pair
        )

        try:
            if delta > 0:
                # Increase short
                # See test_open_and_increase_one_short_with_interest
                # import ipdb ; ipdb.set_trace()

                borrowed_quantity_delta = loan.calculate_size_adjust(collateral_adjustment)

                _, adjust_trade, _ = state.trade_short(
                    strategy_cycle_at=self.timestamp,
                    pair=pair,
                    borrowed_quantity=-borrowed_quantity_delta,
                    collateral_quantity=collateral_adjustment,
                    borrowed_asset_price=loan.borrowed.last_usd_price,
                    trade_type=TradeType.rebalance,
                    reserve_currency=reserve_currency,
                    collateral_asset_price=1.0,
                    planned_collateral_consumption=target_params.total_collateral_quantity - loan.collateral.quantity - collateral_adjustment,
                    notes=notes,
                    flags={TradeFlag.increase},
                )

            else:
                # See test_short_decrease_size

                # How much we will pay back our vToken debt
                borrowed_quantity_delta = loan.borrowed.quantity - target_params.borrowed_quantity

                reserves_released = Decimal(delta)

                _, adjust_trade, _ = state.trade_short(
                    strategy_cycle_at=self.timestamp,
                    pair=pair,
                    borrowed_quantity=borrowed_quantity_delta, # Buy back shorted tokens to decrease exposute
                    collateral_quantity=0,  # Not used when releasing reserves
                    borrowed_asset_price=loan.borrowed.last_usd_price,
                    trade_type=TradeType.rebalance,
                    reserve_currency=reserve_currency,
                    collateral_asset_price=1.0,
                    planned_collateral_allocation=reserves_released,
                    # See comments in update_short_loan()
                    planned_collateral_consumption=target_params.total_collateral_quantity - loan.collateral.quantity - reserves_released,
                    notes=notes,
                    flags={TradeFlag.reduce},
                )
        except LiquidationRisked as e:
            # Better error messag
            base_token = underlying.base.token_symbol
            raise LiquidationRisked(f"The position value adjust to new value {new_value}, delta {delta:+f} USD, delta {borrowed_quantity_delta:+f} {base_token}, would liquidate the position,") from e

        return [adjust_trade]

    def log(self, msg: str, level=logging.INFO, prefix="{self.timestamp}: "):
        """Log debug info.

        Useful to debug the backtesting when it is not making trades.

        To log a message from your `decide_trade` functions:

        .. code-block:: python

            position_manager = PositionManager(timestamp, strategy_universe, state, pricing_model)
            # ... some indicator calculation code goes here...
            position_manager.log(f"RSI current: {current_rsi_values[btc_pair]}, previous: {previous_rsi_values[btc_pair]}")

        This will create output like:

        .. code-block:: text

            INFO:tradeexecutor.strategy.pandas_trader.position_manager:2019-08-20 00:00:00: RSI current: 65.0149379533956, previous: 65.0149379533956
            INFO:tradeexecutor.strategy.pandas_trader.position_manager:2019-08-21 00:00:00: RSI current: 57.38598755909552, previous: 57.38598755909552

        To make notebook logging visible you need to pass `strategy_logging=True` to :py:func:`tradeexecutor.backtest.backtest_runner.run_backtest_inline`:

        .. code-block:: python

            from tradeexecutor.strategy.cycle import CycleDuration
            from tradeexecutor.backtest.backtest_runner import run_backtest_inline

            state, universe, debug_dump = run_backtest_inline(
                name="RSI multipair",
                engine_version="0.3",
                decide_trades=decide_trades,
                client=client,
                cycle_duration=CycleDuration.cycle_1d,
                universe=strategy_universe,
                initial_deposit=10_000,
                strategy_logging=True,
            )

        .. note::

            Any logging output will likely mess up the rendering of the backtest progress bar.

        :param msg:
            Message to log

        :param level:
            Python logging level.

            Defaults to info.

        :param prefix:
            String prefix added to each logged message.

            By default shows the strategy timestamp.
            Can use Python string formatting within PositionManager context.
        """

        if prefix:
            msg = prefix.format(self=self) + msg

        logger.log(
            level,
            msg,
        )


def explain_open_position_failure(
    portfolio: Portfolio,
    pair: TradingPairIdentifier,
    timestamp: pd.Timestamp | datetime.datetime,
    action_hint: str,
) -> str:
    """Display user friendly error message about conflicting open positions.

    - The strategy tries to open a new position,
      but there is already an existing position

    - Create a user-friendly message so that the user
      can diagnose their strategy

    :return:
        The error message
    """

    buf = StringIO()

    for pos in portfolio.open_positions.values():
        if pos.pair == pair:
            print(f"There is already open osition #{pos.position_id} is already open for {pair.get_ticker()} and the trade would conflict", file=buf)
            print(f"when strategy tried performing {action_hint} at the cycle {timestamp}.", file=buf)
            print(file=buf)
            print(f"   The trade was added to the planning list, but was cannot be executed due to the conflict.", file=buf)
            print(f"   Existing trades created for position #{pos.position_id}:", file=buf)
            for t in pos.trades.values():
                notes = f", {t.notes}" if t.notes else ""
                print(f"        {t}, opened at {t.opened_at}{notes}", file=buf)

    print("What you should check:", file=buf)
    print("- Your strategy does not have a logic error and does not try to open a position twice", file=buf)
    print("- Your strategy does not try to open and close the position in the same cycle", file=buf)
    print("- If you want to adjust the existing position size, use PositionManager.adjust_short(), ", file=buf)
    print("  adjust_position(), and such functions", file=buf)
    print("- You can fill the notes field when opening the trade to diagnose where the trade was opened", file=buf)

    return buf.getvalue()


def explain_portfolio_contents(portfolio: Portfolio) -> str:
    """Build an error message about what makes our portflio:

    :param portfolio:
    :return:
    """
    assert isinstance(portfolio, Portfolio)

    buf = StringIO()
    for pos in portfolio.open_positions.values():
        print(f"   {pos.pair.get_ticker()}, opened:{pos.opened_at} value:${pos.get_value()}", file=buf)
    return buf.getvalue()
