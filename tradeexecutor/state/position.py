"""Trading position state info."""
import datetime
import enum
from dataclasses import dataclass, field
from decimal import Decimal

from typing import Dict, Optional, List, Iterable

import numpy as np
import pandas as pd
from dataclasses_json import dataclass_json

from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier
from tradeexecutor.state.trade import TradeType
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.types import USDollarAmount


@dataclass_json
@dataclass
class TradingPosition:

    #: Runnint int counter primary key for positions
    position_id: int

    #: Trading pair this position is trading
    pair: TradingPairIdentifier

    # type: PositionType
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

    #: Timestamp when this position was moved to a frozen state
    frozen_at: Optional[datetime.datetime] = None

    last_trade_at: Optional[datetime.datetime] = None

    #: Trigger a stop loss if this price is reached
    stop_loss: Optional[USDollarAmount] = None

    #: Trigger a take profit if this price is reached
    take_profit: Optional[USDollarAmount] = None

    #: Human readable notes about this trade
    #:
    #: Used to mark test trades from command line.
    #: Special case; not worth to display unless the field is filled in.
    notes: Optional[str] = None

    def __repr__(self):
        if self.is_open():
            return f"<Open position #{self.position_id} {self.pair} ${self.get_value()}>"
        else:
            return f"<Closed position #{self.position_id} {self.pair} ${self.get_first_trade().get_value()}>"

    def __post_init__(self):
        assert self.position_id > 0
        assert self.last_pricing_at is not None
        assert self.reserve_currency is not None
        # Note that price *can* be zero,
        # on some obscure cases when we load the state from the disk
        assert self.last_token_price >= 0
        assert self.last_reserve_price >= 0

        # Do some extra checks to avoid Pandas types in serialisation
        assert isinstance(self.opened_at, datetime.datetime)
        assert not isinstance(self.opened_at, pd.Timestamp)
        assert not isinstance(self.last_token_price, np.float32)

    def is_open(self) -> bool:
        """This is an open trading position."""
        return self.closed_at is None

    def is_closed(self) -> bool:
        """This position has been closed and does not have any capital tied to it."""
        return not self.is_open()

    def is_frozen(self) -> bool:
        """This position has had a failed trade and can no longer be automatically moved around."""
        return self.frozen_at is not None

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

    def needs_real_time_price(self) -> bool:
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

    def get_quantity(self) -> Decimal:
        """Get the tied up token quantity in all successfully executed trades.

        Does not account for trades that are currently being executd.
        """
        return sum([t.get_position_quantity() for t in self.trades.values() if t.is_success()])

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

    def get_equity_for_position(self) -> Decimal:
        # TODO: Rename to get_quantity_for_position
        return sum([t.get_equity_for_position() for t in self.trades.values() if t.is_success()])

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

    def calculate_value_using_price(self, token_price: USDollarAmount, reserve_price: USDollarAmount) -> USDollarAmount:
        token_quantity = sum([t.get_equity_for_position() for t in self.trades.values() if t.is_accounted_for_equity()])
        reserve_quantity = sum([t.get_equity_for_reserve() for t in self.trades.values() if t.is_accounted_for_equity()])
        return float(token_quantity) * token_price + float(reserve_quantity) * reserve_price

    def get_value(self) -> USDollarAmount:
        """Get the position value using the latest revaluation pricing.

        If the position is closed, the value should be zero.
        """
        return self.calculate_value_using_price(self.last_token_price, self.last_reserve_price)

    def is_stop_loss_closed(self) -> bool:
        """Did this position close with stop loss."""
        last_trade = self.get_last_trade()
        return last_trade.is_stop_loss()

    def is_take_profit_closed(self) -> bool:
        """Did this position close with trake profit."""
        last_trade = self.get_last_trade()
        return last_trade.is_take_profit()

    def open_trade(self,
                   ts: datetime.datetime,
                   trade_id: int,
                   quantity: Optional[Decimal],
                   reserve: Optional[Decimal],
                   assumed_price: USDollarAmount,
                   trade_type: TradeType,
                   reserve_currency: AssetIdentifier,
                   reserve_currency_price: USDollarAmount) -> TradeExecution:
        """Open a new trade on position.

        Trade can be opened by knowing how much you want to buy (quantity) or how much cash you have to buy (reserve).
        """

        if quantity is not None:
            assert reserve is None, "Quantity and reserve both cannot be given at the same time"

        assert self.reserve_currency.get_identifier() == reserve_currency.get_identifier(), "New trade is using different reserve currency than the position has"
        assert isinstance(trade_id, int)
        assert isinstance(ts, datetime.datetime)

        if reserve is not None:
            planned_reserve = reserve
            planned_quantity = reserve / Decimal(assumed_price)
        else:
            planned_quantity = quantity
            planned_reserve = quantity * Decimal(assumed_price) if quantity > 0 else 0

        trade = TradeExecution(
            trade_id=trade_id,
            position_id=self.position_id,
            trade_type=trade_type,
            pair=self.pair,
            opened_at=ts,
            planned_quantity=planned_quantity,
            planned_price=assumed_price,
            planned_reserve=planned_reserve,
            reserve_currency=self.reserve_currency,
        )
        self.trades[trade.trade_id] = trade
        return trade

    def has_trade(self, trade: TradeExecution):
        """Check if a trade belongs to this position."""
        if trade.position_id != self.position_id:
            return False
        return trade.trade_id in self.trades

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
        return sum([t.get_position_quantity() for t in self.trades.values() if t.is_success() if t.is_buy()])

    def get_sell_quantity(self) -> Decimal:
        """How many units we have sold total"""
        return sum([abs(t.get_position_quantity()) for t in self.trades.values() if t.is_success() if t.is_sell()])

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

class PositionType(enum.Enum):
    token_hold = "token_hold"
    lending_pool_hold = "lending_pool_hold"