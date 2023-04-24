"""Trading position state info."""
import datetime
import enum
import logging
from dataclasses import dataclass, field
from decimal import Decimal

from typing import Dict, Optional, List, Iterable

import numpy as np
import pandas as pd
from dataclasses_json import dataclass_json

from tradeexecutor.state.balance_update import BalanceUpdate
from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier
from tradeexecutor.state.trade import TradeType, QUANTITY_EPSILON
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.types import USDollarAmount, BPS, USDollarPrice, Percent
from tradeexecutor.strategy.trade_pricing import TradePricing
from tradeexecutor.utils.accuracy import sum_decimal


logger = logging.getLogger(__name__)


@dataclass_json
@dataclass(slots=True, frozen=True)
class TriggerPriceUpdate:
    """A position trigger prices where updated.

    Store the historical changes in trigger prices on :py:class:`TradingPosition`.

    See

    - :py:attr:`TradingPosition.trailing_stop_loss`.

    - :py:attr:`TradingPosition.trigger_updates`.
    """

    timestamp: datetime.datetime

    mid_price: USDollarAmount

    stop_loss_before: Optional[USDollarAmount]

    stop_loss_after: Optional[USDollarAmount]

    take_profit_before: Optional[USDollarAmount]

    take_profit_after: Optional[USDollarAmount]

    def __post_init__(self):
        # Currently we only support trailing stop loss upwards
        assert isinstance(self.timestamp, datetime.datetime)
        assert type(self.mid_price) == float
        if self.stop_loss_before:
            assert self.stop_loss_before < self.stop_loss_after


@dataclass_json
@dataclass(slots=True)
class TradingPosition:
    """Represents a single trading position.

    - Each position trades a single asset

    - Position is opened when the first trade is made

    - Position is closed when the last remaining quantity is sold/closed

    - Position can have its target trigger levels for :py:attr:`take_profit` and :py:attr:`stop_loss`

    - Position can have multiple trades and increase or decrease the position exposure

    - Positions are revalued outside the trades

    - Trades for the position can have different triggers: rebalance, stop los, etc.

    - Position can be marked as frozen meaning the automatic system does not how to clean it up
    """

    #: Runnint int counter primary key for positions
    position_id: int

    #: Trading pair this position is trading
    pair: TradingPairIdentifier

    #: When this position was opened
    #:
    #: Strategy tick time.
    opened_at: datetime.datetime

    #: When was the last time this position was (re)valued
    last_pricing_at: datetime.datetime

    #: Last valued price for the base token.
    #:
    #: There are two ways to receive this
    #:
    #: - When the position is opened, set to the initial buy price
    #:
    #: - When the position is revalued, set to the sell price of the position
    #:
    #: Note that this might be initially incorrect, if revaluation has not been done
    #: yet, because the buy price != sell price.
    last_token_price: USDollarAmount

    # 1.0 for stablecoins, unless out of peg, in which case can be 0.99
    last_reserve_price: USDollarAmount

    #: Which reserve currency we are going to receive when we sell the asset
    reserve_currency: AssetIdentifier

    #: List of trades taken for this position.
    #: trade_id -> Trade map
    trades: Dict[int, TradeExecution] = field(default_factory=dict)

    #: When this position was closed
    closed_at: Optional[datetime.datetime] = None

    #: Timestamp when this position was moved to a frozen state.
    #:
    #: This can happen multiple times, so is is the last time when this happened.
    #:
    #: See also :py:attr:`unfrozen_at`.
    frozen_at: Optional[datetime.datetime] = None

    #: Timestamp when this position was marked lively again
    #:
    #: Set by :py:mod:`tradeexecutor.state.repair` when the position
    #: trades are repaired and the position is moved to open or closed list.
    unfrozen_at: Optional[datetime.datetime] = None

    #: When this position had a trade last time
    last_trade_at: Optional[datetime.datetime] = None

    #: Record the portfolio value when the position was opened.
    #:
    #: This can be later used to analyse the risk of the
    #: trades. ("Max value at the risk")
    portfolio_value_at_open: Optional[USDollarAmount] = None

    #: Trigger a stop loss if this price is reached,
    #:
    #: We use mid-price as the trigger price.
    stop_loss: Optional[USDollarAmount] = None

    #: Trigger a take profit if this price is reached
    #:
    #: We use mid-price as the trigger price.
    take_profit: Optional[USDollarAmount] = None

    #: Trailing stop loss.
    #:
    #: For details see :ref:`Trailing stop loss`.
    #:
    #: Set the trailing stop as the percentage of the market price.
    #: This will update :py:attr:`stop_loss` price if the new resulting
    #: :py:attr:`stop_loss` will be higher as the previous one.
    #:
    #: Percents as the relative to the the market price,
    #: e.g. for 10% trailing stop loss set this value for 0.9.
    #:
    #: Calculated as `stop_loss = mid_price` trailing_stop_loss_pct.
    #:
    #: Updated by :py:func:`tradeexecutor.strategy.stop_loss.check_position_triggers`.
    #: For any updates you can read :py:attr:`trigger_updates`.
    #:
    trailing_stop_loss_pct: Optional[Percent] = None

    #: Human readable notes about this trade
    #:
    #: Used to mark test trades from command line.
    #: Special case; not worth to display unless the field is filled in.
    notes: Optional[str] = None

    #: All balance updates that have touched this reserve position.
    #:
    #: Generated by :py:class:`tradeexecutor.strategy.sync_model.SyncModel`.
    #:
    #: BalanceUpdate.id -> BalanceUpdate mapping
    #:
    balance_updates: Dict[int, BalanceUpdate] = field(default_factory=dict)

    #: Trigger price updates.
    #:
    #: Every time a trigger price is moved e.g. for a trailing stop loss,
    #  we make a record here for future analysis.
    #:
    #: Trigger updates are stored oldest first.
    #:
    trigger_updates: List[TriggerPriceUpdate] = field(default_factory=list)

    def __repr__(self):
        if self.is_open():
            return f"<Open position #{self.position_id} {self.pair} ${self.get_value()}>"
        else:
            return f"<Closed position #{self.position_id} {self.pair} ${self.get_first_trade().get_value()}>"

    def __hash__(self):
        return hash(self.position_id)

    def __eq__(self, other):
        """Note that we do not support comparison across different portfolios ATM."""
        assert isinstance(other, TradingPosition)
        return self.position_id == other.position_id

    def __post_init__(self):
        assert self.position_id > 0
        assert self.last_pricing_at is not None
        assert self.reserve_currency is not None

        # Note that price *can* be zero,
        # on some obscure cases when we load the state from the disk
        assert self.last_token_price >= 0, f"Token price was: {self.last_token_price}"
        assert self.last_reserve_price >= 0, f"Reserve price was: {self.last_reserve_price}"

        # Do some extra checks to avoid Pandas types in serialisation
        assert isinstance(self.opened_at, datetime.datetime)
        assert not isinstance(self.opened_at, pd.Timestamp)
        assert not isinstance(self.last_token_price, np.float32)
        assert not isinstance(self.stop_loss, np.float64)

    def is_open(self) -> bool:
        """This is an open trading position."""
        return self.closed_at is None

    def is_closed(self) -> bool:
        """This position has been closed and does not have any capital tied to it."""
        return not self.is_open()

    def is_frozen(self) -> bool:
        """This position has had a failed trade and can no longer be automatically moved around."""
        return self.frozen_at is not None

    def is_unfrozen(self) -> bool:
        """This position was frozen, but its trades were successfully repaired."""
        return self.unfrozen_at is not None

    def has_automatic_close(self) -> bool:
        """This position has stop loss/take profit set."""
        return (self.stop_loss is not None) or (self.take_profit is not None)

    def get_first_trade(self) -> TradeExecution:
        """Get the first trade for this position.

        Considers unexecuted trades.
        """
        return next(iter(self.trades.values()))

    def get_last_trade(self) -> TradeExecution:
        """Get the the last trade for this position.

        Considers unexecuted and failed trades.
        """
        return next(reversed(self.trades.values()))

    def is_long(self) -> bool:
        """Is this position long on the underlying base asset.

        We consider the position long if the first trade is buy.
        """
        assert len(self.trades) > 0, "Cannot determine if position is long or short because there are no trades"
        return self.get_first_trade().is_buy()

    def is_short(self) -> bool:
        """Is this position short on the underlying base asset."""
        return not self.is_long()

    def is_stop_loss(self) -> bool:
        """Was this position ended with stop loss trade"""
        last_trade = self.get_last_trade()
        if last_trade:
            return last_trade.is_stop_loss()
        return False

    def is_trailing_stop_loss(self) -> bool:
        """Was this position ended with a trailing stop loss trade.

        - Position was terminated with a stop loss

        - Trailing stop loss was set

        - Trailing stop loss was updated at least once
        """

        if self.trailing_stop_loss_pct is None:
            return False

        # Was not terminated at the trailing stop loss
        if not self.is_stop_loss():
            return False

        # Did we set trailing stop loss ever
        trailing_stop_set = any([True for t in self.trigger_updates if t.stop_loss_after is not None])
        return trailing_stop_set

    def is_take_profit(self) -> bool:
        """Was this position ended with take profit trade"""
        last_trade = self.get_last_trade()
        if last_trade:
            return last_trade.is_take_profit()
        return False

    def is_profitable(self):
        """This position is currently having non-zero profit."""
        return self.get_total_profit_usd() > 0

    def is_loss(self):
        """This position is currently having non-zero losses."""
        return self.get_total_profit_usd() < 0

    def has_executed_trades(self) -> bool:
        """This position represents actual holdings and has executed trades on it.

        This will return false for positions that are still planned or have zero successful trades.
        """
        t: TradeExecution
        for t in self.trades.values():
            if t.is_success():
                return True
        return False

    def has_trigger_conditions(self) -> bool:
        """Does this position need to check for stop loss/take profit."""
        return self.stop_loss is not None or self.take_profit is not None

    def get_executed_trades(self) -> Iterable[TradeExecution]:
        for t in self.trades.values():
            if t.is_success():
                yield t

    def get_name(self) -> str:
        """Get human readable name for this position"""
        return f"#{self.position_id} {self.pair.base.token_symbol}-{self.pair.quote.token_symbol}"

    def get_quantity_unit_name(self) -> str:
        """Get the unit name we label the quantity in this position"""
        return f"{self.pair.base.token_symbol}"

    def get_balance_update_quantity(self) -> Decimal:
        """Get quantity of all balance udpdates for this position.

        :return:
            How much in-kind redemption events have affected this position.

            Decimal zero epsilon noted.
        """
        return sum_decimal([b.quantity for b in self.balance_updates.values()])

    def get_quantity(self) -> Decimal:
        """Get the tied up token quantity in all successfully executed trades.

        - Does not account for trades that are currently being executed.

        - Because decimal summ might

        - Accounts for any balance update events (redemptions, interest)

        :return:
            Number of asset units held by this position.

            Rounded down to zero if the sum of
        """
        s = sum_decimal([t.get_position_quantity() for t in self.trades.values() if t.is_success()])

        if s != Decimal(0):
            assert s >= QUANTITY_EPSILON, "Safety check in floating point math triggered"

        s += self.get_balance_update_quantity()

        return Decimal(s)  # Make zero to decimal

    def get_live_quantity(self) -> Decimal:
        """Get all tied up token quantity.

        This includes

        - All executed trades

        - All planned trades for this cycle

        This gives you remaining token balance, even if there are some earlier
        sell orders that have not been executed yet.
        """
        return sum([t.get_position_quantity() for t in self.trades.values() if not t.is_failed()])

    def get_current_price(self) -> USDollarAmount:
        """Get the price of the base asset based on the latest valuation."""
        return self.last_token_price

    def get_opening_price(self) -> USDollarAmount:
        """Get the price when the position was opened."""
        assert self.has_executed_trades()
        first_trade = self.get_first_trade()
        return first_trade.executed_price

    def get_closing_price(self) -> USDollarAmount:
        """Get the price when the position was closed."""
        assert self.has_executed_trades()
        last_trade = self.get_last_trade()
        return last_trade.executed_price

    def get_equity_for_position(self) -> Decimal:
        """How many asset units this position tolds."""
        return sum_decimal([t.get_equity_for_position() for t in self.trades.values() if t.is_success()])

    def has_unexecuted_trades(self) -> bool:
        return any([t for t in self.trades.values() if t.is_pending()])

    def has_planned_trades(self) -> bool:
        return any([t for t in self.trades.values() if t.is_planned()])

    def get_identifier(self) -> str:
        """One trading pair may have multiple open positions at the same time."""
        return f"{self.pair.get_identifier()}-{self.position_id}"

    def get_successful_trades(self) -> List[TradeExecution]:
        """Get all trades that have been successfully executed and contribute to this position"""
        return [t for t in self.trades.values() if t.is_success()]

    def get_failed_trades(self) -> List[TradeExecution]:
        """Get all trades that have failed in the execution."""
        return [t for t in self.trades.values() if t.is_failed()]

    def calculate_value_using_price(self, token_price: USDollarAmount, reserve_price: USDollarAmount) -> USDollarAmount:
        """Calculate the value of this position using the given prices."""
        token_quantity = sum([t.get_equity_for_position() for t in self.trades.values() if t.is_accounted_for_equity()])
        reserve_quantity = sum([t.get_equity_for_reserve() for t in self.trades.values() if t.is_accounted_for_equity()])
        return float(token_quantity) * token_price + float(reserve_quantity) * reserve_price

    def get_value(self) -> USDollarAmount:
        """Get the position value using the latest revaluation pricing.

        If the position is closed, the value should be zero.
        """
        if self.is_closed():
            return USDollarAmount(0)
        else:
            return self.calculate_value_using_price(self.last_token_price, self.last_reserve_price)

    def get_trades_by_strategy_cycle(self, timestamp: datetime.datetime) -> Iterable[TradeExecution]:
        """Get all trades made for this position at a specific time.

        :return:
            Iterable of 0....N trades
        """
        assert isinstance(timestamp, datetime.datetime)
        for t in self.trades.values():
            if t.strategy_cycle_at == timestamp:
                yield t

    def get_unexeuted_reserve(self) -> Decimal:
        """Get the reserve currency allocated for trades.

        Assumes position can only have one reserve currency.

        Only spot buys can have unexecuted reserve.

        :return:
            Amount of capital we have allocated in trades that did not correctly execute
        """
        unexecuted = [t for t in self.trades.values() if not t.is_executed()]
        return sum(t.planned_reserve for t in unexecuted)

    def is_stop_loss_closed(self) -> bool:
        """Did this position close with stop loss."""
        last_trade = self.get_last_trade()
        return last_trade.is_stop_loss()

    def is_take_profit_closed(self) -> bool:
        """Did this position close with trake profit."""
        last_trade = self.get_last_trade()
        return last_trade.is_take_profit()

    def open_trade(self,
                   strategy_cycle_at: datetime.datetime,
                   trade_id: int,
                   quantity: Optional[Decimal],
                   reserve: Optional[Decimal],
                   assumed_price: USDollarPrice,
                   trade_type: TradeType,
                   reserve_currency: AssetIdentifier,
                   reserve_currency_price: USDollarPrice,
                   pair_fee: Optional[BPS] = None,
                   lp_fees_estimated: Optional[USDollarAmount] = None,
                   planned_mid_price: Optional[USDollarPrice] = None,
                   price_structure: Optional[TradePricing] = None,
                   slippage_tolerance: Optional[float] = None
                   ) -> TradeExecution:
        """Open a new trade on position.

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

        :param slippage_tolerance:
            Slippage tolerance for this trade.

            See :py:attr:`tradeexecutor.state.trade.TradeExecution.slippage_tolerance` for details.
        """

        if quantity is not None:
            assert reserve is None, "Quantity and reserve both cannot be given at the same time"

        if price_structure is not None:
            assert isinstance(price_structure, TradePricing)
        
        assert self.reserve_currency.get_identifier() == reserve_currency.get_identifier(), "New trade is using different reserve currency than the position has"
        assert isinstance(trade_id, int)
        assert isinstance(strategy_cycle_at, datetime.datetime)

        if reserve is not None:
            planned_reserve = reserve
            planned_quantity = reserve / Decimal(assumed_price)
        else:
            planned_quantity = quantity
            planned_reserve = abs(quantity * Decimal(assumed_price))

        trade = TradeExecution(
            trade_id=trade_id,
            position_id=self.position_id,
            trade_type=trade_type,
            pair=self.pair,
            opened_at=strategy_cycle_at,
            planned_quantity=planned_quantity,
            planned_price=assumed_price,
            planned_reserve=planned_reserve,
            reserve_currency=self.reserve_currency,
            planned_mid_price=planned_mid_price,
            fee_tier=pair_fee,
            lp_fees_estimated=lp_fees_estimated,
            price_structure=price_structure,
            slippage_tolerance=slippage_tolerance,
        )
        self.trades[trade.trade_id] = trade
        return trade

    def has_trade(self, trade: TradeExecution):
        """Check if a trade belongs to this position."""
        if trade.position_id != self.position_id:
            return False
        return trade.trade_id in self.trades

    def has_buys(self) -> bool:
        """Does is position have any spot buys."""
        for t in self.trades.values():
            if t.is_buy():
                return True
        return False

    def has_sells(self) -> bool:
        """Does is position have any spot sells."""
        for t in self.trades.values():
            if t.is_sell():
                return True
        return False

    def can_be_closed(self) -> bool:
        """There are no tied tokens in this position."""
        return self.get_equity_for_position() == 0

    def get_total_bought_usd(self) -> USDollarAmount:
        """How much money we have used on buys"""
        return sum([t.get_value() for t in self.trades.values() if t.is_success() if t.is_buy()])

    def get_total_sold_usd(self) -> USDollarAmount:
        """How much money we have received on sells"""
        return sum([t.get_value() for t in self.trades.values() if t.is_success() if t.is_sell()])

    def get_buy_quantity(self) -> Decimal:
        """How many units we have bought total"""
        return sum_decimal([t.get_position_quantity() for t in self.trades.values() if t.is_success() if t.is_buy()])

    def get_sell_quantity(self) -> Decimal:
        """How many units we have sold total"""
        return sum_decimal([abs(t.get_position_quantity()) for t in self.trades.values() if t.is_success() if t.is_sell()])

    def get_net_quantity(self) -> Decimal:
        """The difference in the quantity of assets bought and sold to date."""
        return self.get_quantity()

    def get_average_buy(self) -> Optional[USDollarAmount]:
        """Calculate average buy price.

        :return: None if no buys
        """
        q = float(self.get_buy_quantity())
        if not q:
            return None
        return self.get_total_bought_usd() / q

    def get_average_sell(self) -> Optional[USDollarAmount]:
        """Calculate average buy price.

        :return: None if no sells
        """
        q = float(self.get_sell_quantity())
        return self.get_total_sold_usd() / q

    def get_price_at_open(self) -> USDollarAmount:
        """Get the price of the position at open.

        Include only the first trade that opened the position.
        Calculate based on the executed price.
        """
        first_trade =self.get_first_trade()
        return first_trade.executed_price

    def get_quantity_at_open(self) -> Decimal:
        """Get the quanaity of the asset the position at open.

        Include only the first trade that opened the position.
        Calculate based on the executed price.
        """
        first_trade = self.get_first_trade()
        return first_trade.get_position_quantity()

    def get_average_price(self) -> Optional[USDollarAmount]:
        """The average price paid for all assets on the long or short side.

        :return: None if no executed trades
        """
        if self.is_long():
            return self.get_average_buy()
        else:
            return self.get_average_sell()

    def get_realised_profit_usd(self) -> Optional[USDollarAmount]:
        """Calculates the profit & loss (P&L) that has been 'realised' via two opposing asset transactions in the Position to date.

        :return: profit in dollar or None if no opposite trade made
        """
        assert self.is_long(), "TODO: Only long supported"
        if self.get_sell_quantity() == 0:
            return None
        return (self.get_average_sell() - self.get_average_buy()) * float(self.get_sell_quantity())

    def get_unrealised_profit_usd(self) -> USDollarAmount:
        """Calculate the position unrealised profit.

        Calculates the profit & loss (P&L) that has yet to be 'realised'
        in the remaining non-zero quantity of assets, due to the current
        market price.

        :return: profit in dollar
        """
        avg_price = self.get_average_price()
        if avg_price is None:
            return 0
        return (self.get_current_price() - avg_price) * float(self.get_net_quantity())

    def get_total_profit_usd(self) -> USDollarAmount:
        """Realised + unrealised profit."""
        realised_profit = self.get_realised_profit_usd() or 0
        unrealised_profit = self.get_unrealised_profit_usd() or 0
        total_profit = realised_profit + unrealised_profit
        return total_profit

    def get_total_profit_percent(self) -> float:
        """How much % we have made profit so far.

        :return: 0 if profit calculation cannot be made yet
        """
        assert self.is_long(), f"Profit pct for shorts unimplemented, got {self}, first trade was {self.get_first_trade()}"
        profit = self.get_total_profit_usd()
        bought = self.get_total_bought_usd()
        if bought == 0:
            return 0
        return profit / bought

    def get_total_profit_at_timestamp(self, timestamp: datetime.datetime) -> USDollarAmount:
        """Get the profit of the position what it was at a certain point of time.

        Include realised and unrealised profit.

        :param timestamp:
            Include all traeds before and including at this timestamp.
        """
        raise NotImplementedError()

    def get_freeze_reason(self) -> str:
        """Return the revert reason why this position is frozen.

        Get the revert reason of the last blockchain transaction, assumed to be swap,
        for this trade.
        """
        assert self.is_frozen()
        return self.get_last_trade().blockchain_transactions[-1].revert_reason

    def get_last_tx_hash(self) -> Optional[str]:
        """Get the latest transaction performed for this position.

        It's the tx of the trade that was made for this position.

        TODO: Deprecate
        """
        t = self.get_last_trade()
        if not t:
            return None
        return t.blockchain_transactions[-1].tx_hash

    def set_revaluation_data(self, last_pricing_at: datetime.datetime, last_token_price: float):
        assert isinstance(last_pricing_at, datetime.datetime)
        assert not isinstance(last_pricing_at, pd.Timestamp)
        assert isinstance(last_token_price, float)
        assert not isinstance(last_pricing_at, np.float32)
        assert last_pricing_at.tzinfo is None
        self.last_pricing_at = last_pricing_at
        self.last_token_price = last_token_price

    def get_value_at_open(self) -> USDollarAmount:
        """How much the position had value tied after its open.

        Calculate the value after the first trade.
        """
        assert len(self.trades) > 0, "No trades available"
        return self.get_first_trade().get_executed_value()
    
    def get_value_at_close(self) -> USDollarAmount:
        """How much the position had value tied after its close.

        Calculate the value after the last trade
        """
        assert len(self.trades) > 0, "No trades available"
        return self.get_last_trade().get_executed_value()

    def get_capital_tied_at_open_pct(self) -> float:
        """Calculate how much portfolio capital was risk when this position was opened.

        - This is based on the opening values,
          any position adjustment after open is ignored

        - Assume capital is tied to the position and we can never release it.

        - Assume no stop loss is used, or it cannto be trigged

        See also :py:meth:`get_loss_risk_at_open_pct`.

        :return:
            Percent of the portfolio value
        """
        assert self.portfolio_value_at_open, "Portfolio value at position open was not recorded"
        return self.get_value_at_open() / self.portfolio_value_at_open

    def get_loss_risk_at_open(self) -> USDollarAmount:
        """What is the maximum risk of this position.

        The maximum risk is the amount of portfolio we can lose at one position.
        It is calculated as `position stop loss / position total size`.
        We assume stop losses always trigged perfectly and we do not lose
        (too much) on the stop loss trigger.

        :return:
            Dollar value of the risked capital
        """
        assert self.is_long(), "Only long positions supported"
        assert self.stop_loss, f"Stop loss price must be set to calculate the maximum risk"
        # Calculate how much value we can lose
        price_diff = ( self.get_price_at_open() - self.stop_loss)
        risked_value = price_diff * float(self.get_quantity_at_open())
        return risked_value

    def get_loss_risk_at_open_pct(self) -> float:
        """What is the maximum risk of this position.

        Risk relative to the portfolio size.

        See also :py:meth:`get_loss_risk_at_open_pct`.

        :return:
            Percent of total portfolio value
        """
        if self.portfolio_value_at_open:
            return self.get_loss_risk_at_open() / self.portfolio_value_at_open
        else:
            # Old invalid data
            return 0
    
    def get_realised_profit_percent(self) -> float:
        """Calculated life-time profit over this position."""
        
        assert not self.is_open()
        buy_value = self.get_buy_value()
        sell_value = self.get_sell_value()
        return sell_value / buy_value - 1
    
    def get_duration(self) -> datetime.timedelta | None:
        """How long this position was held.
        :return: None if the position is still open
        """
        if self.is_closed():
            return self.closed_at - self.opened_at  
        else:
            return None
    
    def get_total_lp_fees_paid(self) -> int:
        """Get the total amount of swap fees paid in the position. Includes all trades."""
        
        lp_fees_paid = 0

        for trade in self.trades.values():
            if type(trade.lp_fees_paid) == list:
                lp_fees_paid += sum(filter(None,trade.lp_fees_paid))
            else:
                lp_fees_paid += trade.lp_fees_paid or 0

        return lp_fees_paid
    
    def get_buy_value(self) -> USDollarAmount:
        """Get the total value of the position when it was bought."""
        
        return sum(t.get_executed_value() for t in self.trades.values() if t.is_buy())
    
    def get_sell_value(self) -> USDollarAmount:
        """Get the total value of the position when it was sold."""
        return sum(t.get_executed_value() for t in self.trades.values() if t.is_sell())
    
    def has_bad_data_issues(self) -> bool:
        """Do we have legacy / incompatible data issues."""
        
        for t in self.trades.values():
            if t.planned_mid_price in {0, None}:  # Old data
                return True
            
        return False
    
    def get_max_size(self) -> USDollarAmount:
        """Get the largest size of this position over the time"""
        cur_size = 0
        max_size = 0

        for t in self.trades.values():
            executed_value = t.get_executed_value()
            
            # skip trade if we don't have the executed value
            if not executed_value:
                continue
            
            if t.is_buy():
                cur_size += executed_value
            else:
                cur_size -= executed_value
            
            if cur_size > max_size:
                max_size = cur_size
        
        return max_size

    def get_trade_count(self) -> int:
        """Get the number of trades in this position."""
        return len(self.trades)

    def get_orignal_stop_loss(self) -> Optional[USDollarPrice]:
        """Get the original stop loss value when this position was opened.

        Setting :py:attr:`trailing_stop_loss` will cause `stop_loss` to be updated.
        We can still fetch the original stop loss from :py:attr:`trigger_updates`.

        :return:
            The original dollar price of the stop loss
        """

        # Stop loss not used
        if self.stop_loss is None:
            return None

        # We have at least 1 dynamic update
        if self.trigger_updates:
            return self.trigger_updates[0].stop_loss_before

        # Static stop loss
        return self.stop_loss


class PositionType(enum.Enum):
    token_hold = "token_hold"
    lending_pool_hold = "lending_pool_hold"