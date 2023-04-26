"""Portfolio state management."""

import datetime
import copy
from dataclasses import dataclass, field
from decimal import Decimal
from itertools import chain
from typing import Dict, Iterable, Optional, Tuple, List, Callable
from pandas import Timestamp

from dataclasses_json import dataclass_json

from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.trade import TradeType
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.types import USDollarAmount, BPS, USDollarPrice
from tradeexecutor.strategy.trade_pricing import TradePricing

class NotEnoughMoney(Exception):
    """We try to allocate reserve for a buy trade, but do not have cash."""


class InvalidValuationOutput(Exception):
    """Valuation model did not generate proper price value."""


class MultipleOpenPositionsWithAsset(Exception):
    """Multiple open spot positiosn for a single base asset.

    Cannot determine which one to return.
    """


@dataclass_json
@dataclass
class Portfolio:
    """Represents a trading portfolio on DEX markets.

    Multiple trading pair issue: We do not identify the actual assets we have purchased,
    but the trading pairs that we used to purchase them. Thus, each position marks
    "openness" in a certain trading pair. For example open position of WBNB-BUSD
    means we have bought X BNB tokens through WBNB-BUSD trading pair.
    But because this is DEX market, we could enter and exit through WBNB-USDT,
    WBNB-ETH, etc. and our position is not really tied to a trading pair.
    However, because all TradFi trading view the world through trading pairs,
    we keep the tradition here.
    """

    #: Each position gets it unique running counter id.
    next_position_id: int = field(default=1)

    #: Each trade gets it unique id as a running counter.
    #: Trade ids are unique across different positions.
    next_trade_id: int = field(default=1)

    #: Each balance update event gets it unique id as a running counter.
    next_balance_update_id: int = field(default=1)

    #: Currently open trading positions
    open_positions: Dict[int, TradingPosition] = field(default_factory=dict)

    #: Currently held reserve assets
    #: Token address -> reserve position mapping.
    reserves: Dict[str, ReservePosition] = field(default_factory=dict)

    #: Trades completed in the past
    closed_positions: Dict[int, TradingPosition] = field(default_factory=dict)

    #: Positions that have failed sells, or otherwise immovable and need manual clean up.
    #: Failure reasons could include
    #: - blockchain halted
    #: - ERC-20 token tax fees
    #: - rug pull token - transfer disabled
    frozen_positions: Dict[int, TradingPosition] = field(default_factory=dict)

    def is_empty(self):
        """This portfolio has no open or past trades or any reserves."""
        return len(self.open_positions) == 0 and len(self.reserves) == 0 and len(self.closed_positions) == 0

    def get_position_by_id(self, position_id: int) -> TradingPosition:
        """Get any position open/closed/frozen by id.

        Always assume the position for a `position_id` exists.

        :param position_id:
            Internal running counter id for the position inside
            this portfolio.

        :return:
            Always returns

        :throw:
            Fails with py:class:`AssertionError` if there is no such position.
        """

        assert position_id

        p1 = self.open_positions.get(position_id)
        p2 = self.closed_positions.get(position_id)
        if p2:
            # Sanity check we do not have the same position in multiple tables
            assert not p1
        p3 = self.frozen_positions.get(position_id)
        if p3:
            # Sanity check we do not have the same position in multiple tables
            assert not (p1 or p2)

        assert p1 or p2 or p3, f"Did not have position with id {position_id}"

        return p1 or p2 or p3

    def get_trade_by_id(self, trade_id: int) -> Optional[TradeExecution]:
        """Look up any trade in all positions.

        .. note ::

            Slow lookup. Only designed for testing.

        :return:
            Found trade or
        """
        for p in self.get_all_positions():
            t = p.trades.get(trade_id)
            if t is not None:
                return t

        return None

    def get_all_positions(self) -> Iterable[TradingPosition]:
        """Get open, closed and frozen, positions."""
        return chain(self.open_positions.values(), self.closed_positions.values(), self.frozen_positions.values())
    
    def get_all_positions_filtered(self) -> Iterable[TradingPosition]:
        """Get open, closed and frozen, positions filtered to remove
        repaired or failed trades.
        
        """
        
        all_positions = self.get_all_positions()
        filtered_positions = []

        for position in all_positions:
            
            # to avoid copying with same reference
            filtered_position = copy.deepcopy(position)
            filtered_position.trades = {}
            
            for key, trade in position.trades.items():
                if trade.is_repaired() or trade.is_repair_trade():
                    # These trades have quantity set to zero
                    continue

                # filter out failed trade
                if trade.executed_at is None:
                    continue
                
                # Internally negative quantities are for sells
                quantity = trade.executed_quantity

                if trade.planned_mid_price not in (0, None):
                    price = trade.planned_mid_price
                else:
                    # TODO: Legacy trades.
                    # mid_price is filled to all latest trades
                    price = trade.executed_price
                    
                assert quantity != 0, f"Got bad quantity for {trade}"
                assert (price is not None) and price > 0, f"Got invalid trade {trade.get_full_debug_dump_str()} - price is {price}"

                filtered_position.trades[key] = trade

            # if there are no trades, skip this position
            if not filtered_position.trades:
                continue

            filtered_positions.append(filtered_position)

        return filtered_positions

    def get_open_positions(self) -> Iterable[TradingPosition]:
        """Get currently open positions."""
        return self.open_positions.values()

    def get_unfrozen_positions(self) -> Iterable[TradingPosition]:
        """Get positions that have been repaired."""
        positions = chain(self.open_positions.values(), self.closed_positions.values())
        for p in positions:
            if p.is_unfrozen():
                yield p

    def get_executed_positions(self) -> Iterable[TradingPosition]:
        """Get all positions with already executed trades.

        Ignore positions that are still pending - they have only planned trades.
        """
        p: TradingPosition
        for p in self.open_positions.values():
            if p.has_executed_trades():
                yield p

    def get_open_position_for_pair(self, pair: TradingPairIdentifier) -> Optional[TradingPosition]:
        """Get Open position for a trading pair."""
        assert isinstance(pair, TradingPairIdentifier)
        pos: TradingPosition
        for pos in self.open_positions.values():
            if pos.pair.get_identifier() == pair.get_identifier():
                return pos
        return None

    def get_open_position_for_asset(self, asset: AssetIdentifier) -> Optional[TradingPosition]:
        """Get open position for a trading pair.

        - Check all open positions where asset is a base token

        :return:
            Tingle open position or None

        :raise MultipleOpenPositionsWithAsset:

            If more than one position is open

        """
        assert isinstance(asset, AssetIdentifier)

        matches = []

        pos: TradingPosition
        for pos in self.open_positions.values():
            if pos.pair.base == asset:
                matches.append(pos)

        if len(matches) > 1:
            raise MultipleOpenPositionsWithAsset(f"Querying asset: {asset} - found multipe open positions: {matches}")

        if len(matches) == 1:
            return matches[0]

        return None

    def get_open_quantities_by_position_id(self) -> Dict[str, Decimal]:
        """Return the current ownerships.

        Keyed by position id -> quantity.
        """
        return {p.get_identifier(): p.get_quantity() for p in self.open_positions.values()}

    def get_open_quantities_by_internal_id(self) -> Dict[int, Decimal]:
        """Return the current holdings in different trading pairs.

        Keyed by trading pair internal id -> quantity.
        """
        result = {}
        for p in self.open_positions.values():
            assert p.pair.internal_id, f"Did not have internal id for pair {p.pair}"
            result[p.pair.internal_id] = p.get_quantity()
        return result

    def open_new_position(self,
                          ts: datetime.datetime,
                          pair: TradingPairIdentifier,
                          assumed_price: USDollarPrice,
                          reserve_currency: AssetIdentifier,
                          reserve_currency_price: USDollarPrice) -> TradingPosition:
        """Opens a new trading position.

        - Marks the position opened.

        - Does not add any trades yet

        - Marks the current value of the portfolio at the trade opening time,
          as we need to use this for the risk calculations

        """

        portfolio_value = self.get_total_equity()

        p = TradingPosition(
            position_id=self.next_position_id,
            opened_at=ts,
            pair=pair,
            last_pricing_at=ts,
            last_token_price=assumed_price,
            last_reserve_price=reserve_currency_price,
            reserve_currency=reserve_currency,
            portfolio_value_at_open=portfolio_value,

        )
        self.open_positions[p.position_id] = p
        self.next_position_id += 1
        return p

    def get_position_by_trading_pair(self, pair: TradingPairIdentifier) -> Optional[TradingPosition]:
        """Get open position by a trading pair smart contract address identifier.

        For Uniswap-likes we use the pool address as the persistent identifier
        for each trading pair.
        """
        # https://stackoverflow.com/a/2364277/315168
        return next((p for p in self.open_positions.values() if p.pair == pair), None)
        #for p in self.open_positions.values():
            # TODO: Check with
            # if p.pair.pool_address.lower() == pair.pool_address.lower():
        #    if p.pair == pair:
        #        return p
        #return None

    def get_existing_open_position_by_trading_pair(self, pair: TradingPairIdentifier) -> Optional[TradingPosition]:
        """Get a position by a trading pair smart contract address identifier.

        The position must have already executed trades (cannot be planned position(.
        """
        assert isinstance(pair, TradingPairIdentifier), f"Got {pair}"
        for p in self.open_positions.values():
            if p.has_executed_trades():
                if p.pair.pool_address == pair.pool_address:
                    return p
        return None

    def get_positions_closed_at(self, ts: datetime.datetime) -> Iterable[TradingPosition]:
        """Get positions that were closed at a specific timestamp.

        Useful to display closed positions after the rebalance.
        """
        for p in self.closed_positions.values():
            if p.closed_at == ts:
                yield p

    def create_trade(self,
                     strategy_cycle_at: datetime.datetime,
                     pair: TradingPairIdentifier,
                     quantity: Optional[Decimal],
                     reserve: Optional[Decimal],
                     assumed_price: USDollarPrice,
                     trade_type: TradeType,
                     reserve_currency: AssetIdentifier,
                     reserve_currency_price: USDollarPrice,
                     notes: Optional[str] = None,
                     pair_fee: Optional[BPS] = None,
                     lp_fees_estimated: Optional[USDollarAmount] = None,
                     planned_mid_price: Optional[USDollarPrice] = None,
                     price_structure: Optional[TradePricing] = None,
                     position: Optional[TradingPosition] = None,
                     slippage_tolerance: Optional[float] = None,
                     ) -> Tuple[TradingPosition, TradeExecution, bool]:
        """Create a trade.

        Trade can be opened by knowing how much you want to buy (quantity) or how much cash you have to buy (reserve).

        :param strategy_cycle_at:
            The strategy cycle timestamp for which this trade was executed.

        :param trade_id:
            Trade id allocated by the portfolio

        :param quantity:
            How many units this trade does.

            Positive for buys, negative for sells in the spot market.

        :param assumed_price:
            The planned execution price.

            This is the price we expect to pay per `quantity` unit after the execution.
            This is the mid price + any LP fees included.

        :param trade_type:
            What kind of a trade is this.

        :param reserve_currency:
            Which portfolio reserve we use for this trade.

         :param reserve_currency_price:
            If the quote token is not USD, then the exchange rate between USD and quote token we assume we have.

            Actual exchange rate may depend on the execution.

        :param notes:
            Any human-readable remarks we want to tell about this trade.

        :param pair_fee:
            The fee tier from the trading pair / overriden fee.

        :param lp_fees_estimated:
            HOw much we estimate to pay in LP fees (dollar)

        :param planned_mid_price:
            What was the mid-price of the trading pair when we started to plan this trade.

        :param reserve:
            How many reserve units this trade produces/consumes.

            I.e. dollar amount for buys/sells.

        :param price_structure:
            The full planned price structure for this trade.

            The state of the market at the time of planning the trade,
            and what fees we assumed we are going to get.

        :param position:
            Override the position for the trade.

            Use for repair trades.

        :param slippage_tolerance:
            Slippage tolerance for this trade.

            See :py:attr:`tradeexecutor.state.trade.TradeExecution.slippage_tolerance` for details.

        :return:
            Tuple of entries

            - Trade position (old/new)

            - New trade

            - True if a a new position was opened
        """

        if price_structure is not None:
            assert isinstance(price_structure, TradePricing)

        if quantity is not None:
            assert reserve is None, "Quantity and reserve both cannot be given at the same time"

        if position is None:
            # Open a new position
            position = self.get_position_by_trading_pair(pair)

        if position is None:
            position = self.open_new_position(
                strategy_cycle_at,
                pair,
                assumed_price,
                reserve_currency,
                reserve_currency_price)
            created = True
        else:
            # Trade counts against old position,
            # - inc/dec size
            # - repair trades
            created = False

        assert position.pair == pair

        trade = position.open_trade(
            strategy_cycle_at,
            self.next_trade_id,
            quantity,
            reserve,
            assumed_price,
            trade_type,
            reserve_currency,
            reserve_currency_price,
            pair_fee=pair_fee,
            lp_fees_estimated=lp_fees_estimated,
            planned_mid_price=planned_mid_price,
            price_structure=price_structure,
            slippage_tolerance=slippage_tolerance,
        )

        # Update notes
        trade.notes = notes
        position.notes = notes

        # Check we accidentally do not reuse trade id somehow

        self.next_trade_id += 1

        return position, trade, created

    def get_current_cash(self) -> USDollarAmount:
        """Get how much reserve stablecoins we have."""
        return sum([r.get_value() for r in self.reserves.values()])

    def get_open_position_equity(self) -> USDollarAmount:
        """Get the value of current trading positions."""
        return sum([p.get_value() for p in self.open_positions.values()])

    def get_frozen_position_equity(self) -> USDollarAmount:
        """Get the value of trading positions that are frozen currently."""
        return sum([p.get_value() for p in self.frozen_positions.values()])

    def get_live_position_equity(self) -> USDollarAmount:
        """Get the value of current trading positions plus unexecuted trades."""
        return sum([p.get_value() for p in self.open_positions.values()])

    def get_total_equity(self) -> USDollarAmount:
        """Get the value of the portfolio based on the latest pricing.

        This is

        - Value of the positions

        plus

        - Cash in the hand
        """
        return self.get_open_position_equity() + self.get_current_cash()

    def get_unrealised_profit_usd(self) -> USDollarAmount:
        """Get the profit of currently open positions."""
        return sum([p.get_unrealised_profit_usd() for p in self.open_positions.values()])

    def get_closed_profit_usd(self) -> USDollarAmount:
        """Get the value of the portfolio based on the latest pricing."""
        return sum([p.get_total_profit_usd() for p in self.closed_positions.values()])

    def find_position_for_trade(self, trade) -> Optional[TradingPosition]:
        """Find a position tha trade belongs for."""
        return self.open_positions[trade.position_id]

    def get_reserve_position(self, asset: AssetIdentifier) -> ReservePosition:
        """Get reserves for a certain reserve asset.

        :raise KeyError:
            If we do not have reserves for the asset
        """
        return self.reserves[asset.get_identifier()]

    def get_default_reserve_position(self) -> ReservePosition:
        """Get the default reserve position.

        Assume portfolio has only one reserve asset.

        :raise AssertionError:
            If there is not exactly one reserve position
        """

        positions = list(self.reserves.values())
        assert len(positions) == 1, f"Had {len(positions)} reserve position"
        return positions[0]

    def get_equity_for_pair(self, pair: TradingPairIdentifier) -> Decimal:
        """Return how much equity allocation we have in a certain trading pair."""
        position = self.get_position_by_trading_pair(pair)
        if position is None:
            return 0
        return position.get_equity_for_position()

    def adjust_reserves(self, asset: AssetIdentifier, amount: Decimal):
        """Remove currency from reserved.

        For internal accounting of the portfolio state.

        :param asset:
            Reserve asset

        :param amount:
            Negative to reduce portfolio reserves, positive to increase
        """
        assert isinstance(amount, Decimal), f"Got amount {amount}"
        reserve = self.get_reserve_position(asset)
        assert reserve, f"No reserves available for {asset}"
        assert reserve.quantity, f"Reserve quantity missing for {asset}"
        reserve.quantity += amount

    def move_capital_from_reserves_to_trade(self, trade: TradeExecution, underflow_check=True):
        """Allocate capital from reserves to trade instance.

        Total equity of the porfolio stays the same.
        """
        assert trade.is_buy()

        reserve = trade.get_planned_reserve()
        try:
            available = self.reserves[trade.reserve_currency.get_identifier()].quantity
        except KeyError as e:
            raise RuntimeError(f"Reserve missing for {trade.reserve_currency}") from e

        # Sanity check on price calculatins
        assert abs(float(reserve) - trade.get_planned_value()) < 0.01, f"Trade {trade}: Planned value {trade.get_planned_value()}, but wants to allocate reserve currency for {reserve}"

        if underflow_check:
            if available < reserve:
                raise NotEnoughMoney(f"Not enough reserves. We have {available}, trade wants {reserve}")

        trade.reserve_currency_allocated = reserve
        self.adjust_reserves(trade.reserve_currency, -reserve)

    def return_capital_to_reserves(self, trade: TradeExecution, underflow_check=True):
        """Return capital to reserves after a sell."""
        assert trade.is_sell()
        self.adjust_reserves(trade.reserve_currency, +trade.executed_reserve)

    def has_unexecuted_trades(self) -> bool:
        """Do we have any trades that have capital allocated, but not executed yet."""
        return any([p.has_unexecuted_trades() for p in self.open_positions])

    def update_reserves(self, new_reserves: List[ReservePosition]):
        """Update current reserves.

        Overrides current amounts of reserves.

        E.g. in the case users have deposited more capital.
        """

        assert not self.has_unexecuted_trades(), "Updating reserves while there are trades in progress will mess up internal account"

        for r in new_reserves:
            self.reserves[r.get_identifier()] = r

    def check_for_nonce_reuse(self, nonce: int):
        """A helper assert to see we are not generating invalid transactions somewhere.

        :raise: AssertionError
        """
        for p in self.get_all_positions():
            for t in p.trades.values():
                for tx in t.blockchain_transactions:
                    assert tx.nonce != nonce, f"Nonce {nonce} is already being used by trade {t} with txinfo {t.tx_info}"

    def revalue_positions(self, ts: datetime.datetime, valuation_method: Callable, revalue_frozen=True):
        """Revalue all open positions in the portfolio.

        Reserves are not revalued.

        :param revalue_frozen:
            Revalue frozen positions as well
        """
        try:
            for p in self.open_positions.values():
                ts, price = valuation_method(ts, p)
                p.set_revaluation_data(ts, price)

            if revalue_frozen:
                for p in self.frozen_positions.values():
                    ts, price = valuation_method(ts, p)
                    p.set_revaluation_data(ts, price)
        except Exception as e:
            raise InvalidValuationOutput(f"Valuation model failed to output proper price: {valuation_method}: {e}") from e

    def get_default_reserve(self) -> Tuple[AssetIdentifier, USDollarAmount]:
        """Gets the default reserve currency associated with this state.

        For strategies that use only one reserve currency.
        This is the first in the reserve currency list.

        :return:
            Tuple (Reserve currency asset, its latest US dollar exchanage rate)
        """
        assert len(self.reserves) > 0, "Portfolio has no reserve currencies available"
        res_pos = next(iter(self.reserves.values()))
        return res_pos.asset, res_pos.reserve_token_price

    def get_all_trades(self) -> Iterable[TradeExecution]:
        """Iterate through all trades: completed, failed and in progress"""
        pos: TradingPosition
        for pos in self.get_all_positions():
            for t in pos.trades.values():
                yield t

    def get_first_and_last_executed_trade(self) -> Tuple[Optional[TradeExecution], Optional[TradeExecution]]:
        """Get first and last trades overall."""

        first = last = None

        for t in self.get_all_trades():
            if first is None:
                first = t
            if last is None:
                last = t
            if t.executed_at and first.executed_at and (t.executed_at < first.executed_at):
                first = t
            if t.executed_at and last.executed_at and (t.executed_at > last.executed_at):
                last = t
        return first, last
    
    def get_strategy_duration(self) -> Optional[datetime.timedelta]:
        """How long did the strategy run for."""
        first, last = self.get_first_and_last_executed_trade()
        
        if first and last:
            return last.executed_at - first.executed_at
        else:
            return None

    def get_initial_deposit(self) -> Optional[USDollarAmount]:
        """How much we invested at the beginning of a backtest.

        - Assumes we track the performance against the US dollar

        - Assume there has been only one deposit event

        - This deposit happened at the start of the backtest
        """

        if len(self.reserves) == 0:
            return 0

        assert len(self.reserves) == 1, f"Reserve assets are not defined for this state, cannot get initial deposit\n" \
                                        f"State is {self}"
        reserve = next(iter(self.reserves.values()))
        if reserve.initial_deposit:
            return float(reserve.initial_deposit) * reserve.initial_deposit_reserve_token_price
        return None

    def get_all_traded_pairs(self) -> Iterable[TradingPairIdentifier]:
        """Get all pairs for which we have or had positions."""
        already_iterated_pairs = set()
        for p in self.get_all_positions():
            if p.pair not in already_iterated_pairs:
                already_iterated_pairs.add(p.pair)
                yield p.pair

    def initialise_reserves(self, asset: AssetIdentifier):
        """Create the initial reserve currency list.

        Currently we assume there can be only one reserve currency.
        """
        assert len(self.reserves) == 0, "Reserves already initialised"
        self.reserves[asset.address] = ReservePosition(
            asset=asset,
            quantity=Decimal(0),
            last_sync_at=None,
            reserve_token_price=None,
            last_pricing_at=None,
        )
        
        
