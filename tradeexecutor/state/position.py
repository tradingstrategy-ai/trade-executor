"""Trading position state info."""
import datetime
import logging
import pprint
import statistics
import warnings
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from itertools import chain

from typing import Dict, Optional, List, Iterable, Tuple, Set, Literal, TypedDict

import numpy as np
import pandas as pd
from dataclasses_json import dataclass_json

from tradingstrategy.utils.time import ZERO_TIMEDELTA

from tradeexecutor.state.balance_update import BalanceUpdate, BalanceUpdateCause, DEFAULT_YEAR
from tradeexecutor.state.generic_position import GenericPosition, BalanceUpdateEventAlreadyAdded
from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier, TradingPairKind
from tradeexecutor.state.loan import Loan
from tradeexecutor.state.trade import TradeType, TradeFlag
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.types import USDollarAmount, BPS, USDollarPrice, Percent, LeverageMultiplier, LegacyDataException
from tradeexecutor.state.valuation import ValuationUpdate
from tradeexecutor.strategy.dust import get_close_epsilon_for_pair
from tradeexecutor.strategy.execution_context import ExecutionMode
from tradeexecutor.strategy.lending_protocol_leverage import create_short_loan, update_short_loan, create_credit_supply_loan, update_credit_supply_loan
from tradeexecutor.strategy.pnl import calculate_pnl
from tradeexecutor.strategy.trade_pricing import TradePricing
from tradeexecutor.utils.accuracy import sum_decimal, QUANTITY_EPSILON
from tradingstrategy.lending import LendingProtocolType
from tradeexecutor.utils.leverage_calculations import LeverageEstimate


logger = logging.getLogger(__name__)


#: If a token position helds less than this absolute amount of token
#: consider closing it as dust
CLOSED_POSITION_DUST_EPSILON = 0.0001


class PositionOtherData(TypedDict):
    """Position data that is not relevant for most positions."""

    marked_down_at: datetime.datetime


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

    mid_price: Optional[USDollarAmount] = None

    stop_loss_before: Optional[USDollarAmount] = None

    stop_loss_after: Optional[USDollarAmount] = None

    take_profit_before: Optional[USDollarAmount] = None

    take_profit_after: Optional[USDollarAmount] = None

    def __post_init__(self):
        # Currently we only support trailing stop loss upwards
        assert isinstance(self.timestamp, datetime.datetime)

        if self.mid_price:
            assert type(self.mid_price) == float


@dataclass_json
@dataclass(slots=True)
class TradingPosition(GenericPosition):
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

    #: When was the last time this position was (re)valued.
    #:
    #: Read with :py:meth:`get_last_valued_at`.
    #:
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

    #: 1.0 for stablecoins, unless out of peg, in which case can be 0.99
    last_reserve_price: USDollarAmount

    #: Which reserve currency we are going to receive when we sell the asset
    reserve_currency: AssetIdentifier

    #: List of trades taken for this position.
    #: trade_id -> Trade map
    trades: Dict[int, TradeExecution] = field(default_factory=dict)

    #: List of trigger trades waiting to be taken
    #: trade_id -> Trade map
    #:
    #: SEe also old style :py:attr:`stop_loss`, :py:attr:`trailing_stop_loss`, :py:attr:`trigged_updates`.
    #:
    pending_trades: Dict[int, TradeExecution] = field(default_factory=dict)

    #: List of trigger trades that have expried
    #: trade_id -> Trade map
    expired_trades: Dict[int, TradeExecution] = field(default_factory=dict)

    #: When this position was closed
    #:
    #: Execution time of the trade or wall-clock time if not available.
    #:
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

    #: Timestamp when this position was set pending for a market limit.
    #:
    #: Cleared back to `None` if the position opens.
    #:
    #: See also :py:meth:`is_pending`
    #:
    pending_since_at: Optional[datetime.datetime] = None

    #: Trigger a stop loss if this price is reached,
    #:
    #: We use mid-price as the trigger price.
    #:
    #: .. note:: This should not be updated directly, but via :py:func:`tradeexecutor.strategy.pandas_trader.position_manager.update_stop_loss`.
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
    #: Special case; not worth to display unless the field is filled in.
    #:
    #: - May contain multiple newline separated messages
    #:
    #: - Used to mark test trades from command line.
    #:
    #: - Used to add log information about frozen and unfrozen positions
    #:
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
    #: There is no record made if there are no trigger updates change.
    #: For example, for trailing stop loss, there is no record added,
    #: if the price did not move upwards, causing the stop loss level to move.
    #:
    #: See also py:attr:`pending_trades` where new-style partial take profit trades are placed.
    #:
    trigger_updates: List[TriggerPriceUpdate] = field(default_factory=list)

    #: Valuation updates.
    #:
    #: Every time a trigger price is moved e.g. for a trailing stop loss,
    #  we make a record here for future analysis.
    #:
    #: Trigger updates are stored oldest first.
    #:
    #: See also :py:attr:`last_token_price` and :py:attr:`last_pricing_at`
    #: legacy attributes.
    #:
    valuation_updates: List[ValuationUpdate] = field(default_factory=list)

    #: The loan underlying the position leverage or credit supply.
    #:
    #: Applicable for
    #:
    #: - short/long positions using lending protocols
    #:
    #: - credit supply (collateral without borrow)
    #:
    #: This reflects the latest :py:attr:`tradeexecutor.state.trade.TradeExecution.executed_loan`
    #: of a successful trade. This object is updated with accrued interest information from on-chain
    #: data outside trades. If the position does not have successfully executed trades yet,
    #: this is ``None``.
    #:
    loan: Optional[Loan] = None

    #: What is the liquidation price for this position.
    #: If the price goes below this, the position is liquidated.
    #:
    #: Applicable for
    #:
    #: - short/long positions using lending protocols
    #:
    #: TODO: When this is set and when this is updated.
    #:
    liquidation_price: USDollarAmount | None = None

    #: Misc bag of data, not often needed
    other_data: PositionOtherData = field(default_factory=dict)

    def __repr__(self):
        if self.is_pending():
            return f"<Pending position #{self.position_id} {self.pair} ${self.get_value()}>"
        elif self.is_frozen():
            return f"<Frozen position #{self.position_id} {self.pair} ${self.get_value()}>"
        elif self.is_open():
            return f"<Open position #{self.position_id} {self.pair} ${self.get_value()}>"
        else:
            return f"<Closed position #{self.position_id} {self.pair} ${self.get_first_trade().get_value()}>"

    def __hash__(self):
        return hash(self.position_id)

    def __eq__(self, other):
        """Note that we do not support comparison across different portfolios ATM."""
        assert isinstance(other, TradingPosition), f"Expected TradingPosition, got {type(other)}: {other}"
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

    def get_human_readable_name(self) -> str:
        return f"Trading position #{self.position_id} for {self.pair.get_ticker()}"

    def get_debug_dump(self, char_limit=1000) -> str:
        """Return class contents for logging.

        :param char_limit:
            Display maximum of this many characters.

        :return:
            Indented JSON-like content
        """
        msg = pprint.pformat(asdict(self), width=160)
        if len(msg) > char_limit:
            msg = msg[:char_limit] + "..."
        return msg

    def get_human_summary(self) -> dict:
        """Get the human readable debug dump.

        :return:
            Human readable dict
        """

        return {
            "Position id": self.position_id,
            "Pair": self.pair.get_ticker(),
            "Type": self.pair.kind.name,
            "Trade count": self.get_trade_count(),
            "PnL %": self.get_unrealised_and_realised_profit_percent() * 100,
            "PnL USD": self.get_realised_profit_usd(),
            "Started at": self.get_first_trade().executed_at,
            "Ended at": self.get_last_trade().executed_at,
            "Duration": self.get_duration(),
            "Entry price": self.get_opening_price(),
            "Exit price": self.get_closing_price(),
            "Quantity (open)": self.get_first_trade().executed_quantity,
        }

    def is_open(self) -> bool:
        """This is an open trading position."""
        return self.closed_at is None

    def is_closed(self) -> bool:
        """This position has been closed and does not have any capital tied to it.

        See also :py:meth:`is_reduced`:
        """
        return not self.is_open()

    def is_test(self) -> bool:
        """The position was opened and closed by perform-test-trade command.

        The trade and the position should not be counted in the statistics.
        """
        return any(TradeFlag.test_trade in t.flags for t in self.trades.values())

    def is_frozen(self) -> bool:
        """This position has had a failed trade and can no longer be automatically moved around.

        After the position is unfrozen the flag goes away.
        """
        return (self.frozen_at is not None) and not self.is_unfrozen()

    def is_unfrozen(self) -> bool:
        """This position was frozen, but its trades were successfully repaired."""
        return self.unfrozen_at is not None

    def is_repaired(self) -> bool:
        """This position contains repaired trades.

        - Any of the trades is repaired

        - Does not check if broken but unrepaired trades
        """
        return any(t.is_repaired() for t in self.trades.values())

    def is_marked_down(self) -> bool:
        """Position value was forcefully set to zero.

        See :py:meth:`mark_down`.
        """
        return self.other_data.get("marked_down_at") is not None

    def has_automatic_close(self) -> bool:
        """This position has stop loss/take profit set."""
        return (self.stop_loss is not None) or (self.take_profit is not None)

    def get_first_trade(self) -> TradeExecution:
        """Get the first trade for this position.

        Considers unexecuted trades.
        """
        return next(iter(chain(self.trades.values(), self.pending_trades.values())))

    def get_last_trade(self) -> TradeExecution:
        """Get the the last trade for this position.

        Considers unexecuted and failed trades.
        """
        return next(reversed(self.trades.values()))

    def is_spot(self) -> bool:
        """Is this a spot market position."""
        assert len(self.trades) + len(self.pending_trades) > 0, "Cannot determine if position is long or short because there are no trades"
        return self.get_first_trade().is_spot()

    def is_vault(self) -> bool:
        """Is this a vault shares position."""
        return self.pair.is_vault()

    def is_multi_stage(self) -> bool:
        return self.other_data.get("multi_stage") is not None

    def is_multi_stage_in_process(self) -> bool:
        """Do we have pending vault deposits or redeems in progress which prevents us to start other operations"""
        last_trade = self.get_last_trade()
        ticket_state = last_trade.get_multi_stage()
        return ticket_state.is_in_progress()

    def get_multi_stage_phase(self) -> Literal["deposit", "redeem"]:
        assert self.is_vault()
        last_trade = self.get_last_trade()
        ticket_state = last_trade.get_multi_stage()
        assert ticket_state.is_in_progress()
        if self.is_buy():
            return "deposit"
        else:
            return "redeem"

    def is_long(self) -> bool:
        """Is this position long on the underlying base asset.

        We consider the position long if the first trade is buy.

        - This includes spot buy.
        - This includes holding vault shares
        """
        assert len(self.trades) + len(self.pending_trades) > 0, "Cannot determine if position is long or short because there are no trades"
        return self.pair.is_spot() or self.pair.is_long() or self.pair.is_vault()

    def is_short(self) -> bool:
        """Is this position short on the underlying base asset."""
        return self.pair.is_short()

    def is_leverage(self) -> bool:
        """Is this leveraged/loan backed position."""
        return self.pair.is_leverage()

    def is_loan_based(self) -> bool:
        """The profit for this trading pair is loan based.."""
        return self.pair.is_leverage() or self.pair.is_credit_supply()

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

    def is_credit_supply(self):
        """This is a trading position for gaining interest by lending out reserve currency."""
        return self.pair.kind == TradingPairKind.credit_supply

    def is_spot_market(self) -> bool:
        """Alias for :py:meth:`is_spot`."""
        return self.is_spot()

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

    def is_pending(self) -> bool:
        """This position is waiting for a condition to trigger to make it open.

        - The position is hyphotetical in `pending_positions` list

        - Will realise if market limit is reached
        """
        return self.pending_since_at is not None

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
        """Does this position need to check for a trigger condition.

        - stop loss/take profit hardcoded options

        - other triggers

        :return:
            True if ewe need to check for the triggers in the trigger udpdate
        """
        return self.stop_loss is not None or \
            self.take_profit is not None or \
            self.pending_trades

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

    def get_balance_update_events(self) -> Iterable[BalanceUpdate]:
        return self.balance_updates.values()

    def get_base_token_balance_update_quantity(self) -> Decimal:
        """Get quantity of all balance updates for this position.

        - How much non-trade events have changed our base token balance

        - This includes interest events and accounting corrections

        :return:
            How much in-kind redemption events have affected this position.

            Decimal zero epsilon noted.
        """
        base = self.pair.base
        return sum_decimal([b.quantity for b in self.balance_updates.values() if b.asset == base])

    def get_quantity(self, planned=False) -> Decimal:
        """Get the tied up token quantity in all successfully executed trades.

        - Does not account for trades that are currently being executed (in started,
          or planned state).

        - Does some fixing for rounding errors in the form of epsilon checks

        - Accounts for any balance update events (redemptions, interest, accounting corrections)

        For interest positions

        - The underlying principle is calculated as sum of trades e.g.
          how many deposit or redemption trades we did for Aave reserves

        - The accrued interest can be read from balance update events

        :param planned:
            Include the quantity of the planned trades that are going to be executed on this cycle.

        :return:
            Number of asset units held by this position.

            Rounded down to zero if the sum of
        """
        if planned:
            trades = sum_decimal([t.get_position_quantity() for t in self.trades.values() if t.is_success() or t.is_planned()])
        else:
            trades = sum_decimal([t.get_position_quantity() for t in self.trades.values() if t.is_success()])
        direct_balance_updates = self.get_base_token_balance_update_quantity()

        # Because short position is modelled as negative quantity,
        # any added interest payments must make the position more negative
        if self.is_short():
            s = trades - direct_balance_updates
        else:
            s = trades + direct_balance_updates

        # TODO:
        # We should not have math that ends up with a trading position with dust left,
        # tough this might not always hold the case
        if s != Decimal(0):
            # assert abs(s) >= QUANTITY_EPSILON, f"Epsilon dust safety check in floating point math triggered. Quantity: {s}. Epsilon: {QUANTITY_EPSILON}."
            if abs(s) <= QUANTITY_EPSILON:
                return Decimal(0)

        # Always convert zero to decimal
        return Decimal(s)

    def get_pending_quantity(self) -> Decimal:
        """Get the quantity locked up in market limit trades."""
        q = sum_decimal([t.get_position_quantity() for t in self.pending_trades.values() if t.is_success() or t.is_planned()])
        return q

    def get_redeemed(self) -> Decimal:
        """Get amount of position reduced via in-kind redemptions.

        - Also include any other account corrections

        Alias for :py:meth:`get_base_token_balance_update_quantity`

        :return:
            A negative number, the amount of quantiy redeemed from this position
        """
        assert self.is_spot()
        return self.get_base_token_balance_update_quantity()

    def get_available_trading_quantity(
        self,
        include_pending_trades: bool = False,
    ) -> Decimal:
        """Get token quantity still availble for the trades in this strategy cycle.

        This includes

        - All executed trades

        - All planned trades for this cycle that have already reduced/increased
          amounts for this position

        This gives you remaining token balance, even if there are some earlier
        sell orders that have not been executed yet.
        """

        if self.is_pending() or include_pending_trades:
            planned = self.get_pending_quantity()
        else:
            # this will be checked when stoploss is triggered
            # so we need to exclude pending trades (e.g. partial tp trades)
            planned = sum([
                t.get_position_quantity()
                for t in self.trades.values()
                if t.is_planned() and not t.is_partial_take_profit()
            ])

        live = self.get_quantity()  # What was the position quantity before executing any of planned trades
        # Temporary logging to track down SAND token errors
        # logger.info("get_available_trading_quantity(): Figuring out available position size to trade. Planned quantity: %s, live quantity: %s", planned, live)
        return planned + live

    def get_current_price(self) -> USDollarAmount:
        """Get the price of the base asset based on the latest valuation."""
        return self.last_token_price

    def get_opening_price(self) -> USDollarPrice:
        """Get the price when the position was opened.

        :return:
            Get the executed opening price of the position.

            If the first trade was not success, return 0.0.
        """
        assert self.has_executed_trades()
        first_trade = self.get_first_trade()
        return first_trade.executed_price or 0.0

    def get_closing_price(self) -> USDollarPrice:
        """Get the price when the position was closed."""
        assert self.has_executed_trades()
        last_trade = self.get_last_trade()
        return last_trade.executed_price

    def get_current_price(self) -> USDollarPrice:
        """Get the last recorded and cached market price for the position base asset."""
        return self.last_token_price

    def get_quantity_old(self) -> Decimal:
        """How many asset units this position tolds.

        TODO: Remove this

        Alias for :py:meth:`get_quantity`
        """
        return self.get_quantity()

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

    def calculate_value_using_price(
            self,
            token_price: USDollarAmount,
            reserve_price: USDollarAmount,
            include_interest=True,
    ) -> USDollarAmount:
        """Calculate the value of this position using the given prices."""

        token_quantity = sum([t.get_equity_for_position() for t in self.trades.values() if t.is_accounted_for_equity()])

        if include_interest:
            raise NotImplementedError()

        reserve_quantity = sum([t.get_equity_for_reserve() for t in self.trades.values() if t.is_accounted_for_equity()])

        direct_balance_updates = self.get_base_token_balance_update_quantity()

        value = float(token_quantity) * token_price + float(reserve_quantity) * reserve_price

        if self.is_short():
            value -= float(direct_balance_updates) * token_price
        else:
            value += float(direct_balance_updates) * token_price

        return value

    def get_equity(self) -> USDollarAmount:
        """Get equity tied to this position.

        TODO: Use :py:meth:`TradingPosition.loan.get_net_asset_value` for collateral based positions.

        :return:
            How much equity we have tied in this position.

            TODO: Does not work for collateral positions.
        """

        match self.pair.kind:
            case TradingPairKind.spot_market_hold | TradingPairKind.vault:
                return self.calculate_value_using_price(
                    self.last_token_price,
                    self.last_reserve_price,
                    include_interest=False,
                )
            case _:
                # TODO: U
                return 0

    def get_value(self, include_interest=True) -> USDollarAmount:
        """Get the current net asset value of this position.

        If the position is closed, the value should be zero

        .. note:: This is the planned value, so may be different fr om the executed value.

        :param include_interest:
            Include accrued interest in the valuation.

            This will add any interest earned/lost in loans,
            plus their repayments.

        :return:
            The value of the position if any remaining open amount
            would be completely closed/unwind.
        """

        if include_interest:
            value = self.get_accrued_interest_with_repayments()
        else:
            value = 0

        if self.is_closed():
            # Closed positions do not have any value left,
            # outside its accrued interest
            return value

        match self.pair.kind:
            case TradingPairKind.spot_market_hold | TradingPairKind.vault:

                value += self.calculate_value_using_price(
                    self.last_token_price,
                    self.last_reserve_price,
                    include_interest=False,
                )

            case TradingPairKind.lending_protocol_short | TradingPairKind.credit_supply:
                # Value for leveraged positions is net asset value from its two loans
                return self.get_loan_based_nav(include_interest=include_interest)
            case _:
                raise NotImplementedError(f"Does not know how to value position for {self.pair}")

        return value

    def get_loan_based_nav(self, include_interest=True, include_fees=True) -> USDollarAmount:
        """Calculate net asset value (NAV) for a loan based position.

        :param include_interest:
            Should interest should be included in the NAV

        :param include_fees:
            TODO

        :return:
            Zero if this position is not yet opened.

            When the first trade of position is executed,
            :py:attr:`loan` attribute becomes available.
        """
        assert self.is_loan_based(), f"Not loan based position: {self}"
        if not self.loan:
            return 0.0

        nav = self.loan.get_net_asset_value(include_interest)
        return nav

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

    def is_properly_opened(self) -> bool:
        """Did we manage to open this position correctly.

        :return:
            True if the opening trade was correctly executed.

            Might also return False for legacy data
        """

        open_trade = self.get_first_trade()
        return open_trade.is_success()

    def open_trade(
            self,
           strategy_cycle_at: datetime.datetime | None,
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
           slippage_tolerance: Optional[float] = None,
           portfolio_value_at_creation: Optional[USDollarAmount] = None,
           leverage: Optional[LeverageMultiplier]=None,
           closing: Optional[bool] = False,
           planned_collateral_consumption: Optional[Decimal] = None,
           planned_collateral_allocation: Optional[Decimal] = None,
           exchange_name: Optional[str] = None,
           flags: Optional[Set[TradeFlag]] = None,
        ) -> TradeExecution:
        """Open a new trade on position.

        Trade can be opened by knowing how much you want to buy (quantity) or how much cash you have to buy (reserve).

        :param strategy_cycle_at:
            The strategy cycle timestamp for which this trade was executed.

            Might not be available for the accounting corrections done offline.

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

        :param portfolio_value_at_creation:
            Record the portfolio's value when this posistion was opened.

            Will be later used for risk metrics calculations and such.

        :param flags:
            Flags set on the trade.
        """

        # Done in State.create_trade()
        # if quantity is not None:
        #    assert reserve is None, "Quantity and reserve both cannot be given at the same time"

        pair = self.pair

        if price_structure is not None:
            assert isinstance(price_structure, TradePricing)

        assert self.reserve_currency.get_identifier() == reserve_currency.get_identifier(), "New trade is using different reserve currency than the position has"
        assert isinstance(trade_id, int)

        if flags is None:
            flags = set()
        assert isinstance(flags, set), f"Got: {flags}"
        assert all([isinstance(f, TradeFlag) for f in flags])

        if strategy_cycle_at is not None:
            assert isinstance(strategy_cycle_at, datetime.datetime)

        # Set lending market estimated quantities
        match pair.kind:
            case TradingPairKind.lending_protocol_short:

                if len(self.trades) == 0:

                    # Open a new short

                    assert reserve is not None, "Both reserve and quantity needs to be given for lending protocol short open"
                    assert quantity is not None, "Both reserve and quantity needs to be given for lending protocol short open"
                    assert not closing, "Cannot close position not yet open"

                    # Automatically calculate the amount of collateral increase for this trade sizes
                    if planned_collateral_consumption is None:
                        planned_collateral_consumption = -quantity * Decimal(assumed_price)

                else:

                    if closing:

                        # Close the short
                        assert self.is_open()

                        assert reserve is None, "reserve calculated automatically when closing a short position"
                        # assert quantity is None, "quantity calculated automatically when closing a short position"
                        assert not planned_collateral_consumption, "planned_collateral_consumption set automatically when closing a short position"

                        # Pay back all the debt and its interest
                        if not quantity:
                            quantity = self.loan.get_borrowed_principal_and_interest_quantity()

                        leverage_estimate = LeverageEstimate.close_short(
                            start_collateral=self.loan.collateral.quantity,
                            start_borrowed=self.loan.borrowed.quantity,
                            close_size=quantity,
                            borrowed_asset_price=planned_mid_price,
                            fee=self.pair.get_pricing_pair().fee,
                        )

                        # Release collateral is the current collateral
                        reserve = 0

                        # We need to use USD from the collateral to pay back the loan
                        planned_collateral_consumption = leverage_estimate.additional_collateral_quantity

                        # TODO: stablecoin 1:1 USD assumption here

                        #
                        # We cash out accrued interest when closing the position.
                        # - You can have positive and negative interest on both vToken and aToken
                        # - We assume vToken expenses (interest) is paid from the collateral
                        # -
                        #

                        accured_interest = self.loan.collateral_interest.last_accrued_interest

                        # TODO: pass a flag to the function to decide if we want to withdraw all the interest or not
                        # Any leftover USD from the collateral is released to the reserves
                        planned_collateral_allocation = -leverage_estimate.total_collateral_quantity - accured_interest

                        lp_fees_estimated = leverage_estimate.lp_fees

                    else:

                        # Increase/decrease the position size
                        assert quantity is not None, "For increasing/reducing short position quantity must be given"

                        if planned_collateral_consumption is None:
                            # TODO: Explain / check if this default makes sense
                            planned_collateral_consumption = -quantity * Decimal(self.loan.borrowed.last_usd_price)

                        planned_collateral_allocation = planned_collateral_allocation

                assert reserve_currency_price, f"Collateral price missing"
                assert assumed_price, f"Short token price missing"

                planned_reserve = reserve or Decimal(0)
                planned_quantity = (quantity or Decimal(0))

                # From now on, we need meaningful values for math
                planned_collateral_consumption = planned_collateral_consumption or Decimal(0)
                planned_collateral_allocation = planned_collateral_allocation or Decimal(0)

            case TradingPairKind.spot_market_hold | TradingPairKind.vault:
                # Set spot market estimated quantities
                if reserve is not None:
                    planned_reserve = reserve
                    planned_quantity = reserve / Decimal(assumed_price)
                else:
                    planned_quantity = quantity
                    planned_reserve = abs(quantity * Decimal(assumed_price))
            case TradingPairKind.credit_supply:
                if trade_type != TradeType.repair:
                    assert reserve, "You must give reserve"
                assert quantity is not None, "You must give quantity"
                planned_reserve = reserve
                planned_quantity = quantity

            case _:
                raise NotImplementedError(f"Does not know how to calculate quantities for open a trade on: {pair}")

        # TODO: Legacy compatibility.
        # Remove boolean when adding flags to the codebase is complete.
        if closing:
            flags.add(TradeFlag.close)

        trade = TradeExecution(
            trade_id=trade_id,
            position_id=self.position_id,
            trade_type=trade_type,
            pair=pair,
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
            portfolio_value_at_creation=portfolio_value_at_creation,
            leverage=leverage,
            reserve_currency_exchange_rate=reserve_currency_price,
            planned_collateral_allocation=planned_collateral_allocation,
            planned_collateral_consumption=planned_collateral_consumption,
            exchange_name=exchange_name,
            closing=closing,
            flags=flags,
        )

        self.trades[trade.trade_id] = trade

        # Initialise interest tracking data structure
        if pair.kind.is_interest_accruing():
            if pair.kind == TradingPairKind.credit_supply:
                assert pair.kind == TradingPairKind.credit_supply, "Only credit supply supported for now"
                if trade_type != TradeType.repair:
                    if self.loan is None:
                        assert trade.is_buy(), f"Opening credit position is should be modelled as buy, we got: {trade}"
                        trade.planned_loan_update = create_credit_supply_loan(self, trade, strategy_cycle_at)
                    else:
                        trade.planned_loan_update = update_credit_supply_loan(
                            loan=self.loan.clone(),
                            position=self,
                            trade=trade,
                            timestamp=strategy_cycle_at,
                        )
                else:
                    print("Repair trade, no loan update or creation")

            elif pair.kind.is_leverage():
                assert pair.get_lending_protocol() == LendingProtocolType.aave_v3, "Unsupported protocol"
                if pair.kind.is_shorting():
                    if not self.loan:
                        # Opening the position, create the first loan
                        trade.planned_loan_update = create_short_loan(
                            self,
                            trade,
                            strategy_cycle_at,
                        )
                    else:
                        # Loan is being increased/reduced
                        trade.planned_loan_update = update_short_loan(
                            self.loan.clone(),
                            self,
                            trade,
                        )

                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError(f"Don't know how to deal with {pair}")

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

    def is_about_to_close(self) -> bool:
        """This position will close during this trade execution cycle.

        - It has planned trades taking the position to zero
        """
        epsilon = get_close_epsilon_for_pair(self.pair)
        return self.get_quantity(planned=True) < epsilon

    def can_be_closed(self) -> bool:
        """There are no tied tokens in this position.

        Perform additional check for token amount dust caused by rounding errors.

        See also

        - :py:meth:`is_closed`
        """

        if self.closed_at:
            # Already closed
            return False

        if self.is_vault():
            # Morpho Spark USDC workaround.
            # The vault maxRedeem() returns an amount of shares that has a rounding error.
            # Thus we are left we one share token that cannot be redeemed.
            # See estimate_4626_deposit() for details.
            epsilon = get_close_epsilon_for_pair(self.pair)
        else:
            # Try to get sane default epsilon
            epsilon = get_close_epsilon_for_pair(self.pair)

        quantity = self.get_quantity()

        if self.is_credit_supply() or self.is_vault():
            logger.info(
                "can_be_closed(): epsilon debug. Pair: %s, quantity: %s, epsilon: %s",
                self.pair,
                quantity,
                epsilon,
            )

        # VELVET HACK: Quantity can go to below zero, because te last trade
        # got in last minute deposit and executed more than we thought we have
        # aBasUSDC HACK: Quantity can also go below zero due to rounding/epsilon
        if self.is_spot() or self.is_credit_supply():
            if quantity <= 0:
                return True

        return abs(quantity) <= epsilon

    def get_last_updated_at(self) -> datetime.datetime:
        """When did we update this position last time.

        - What was the last timestamp when this position data was mutated

        :return:
            UTC time
        """
        time_vars = (self.opened_at, self.closed_at, self.last_pricing_at, self.last_trade_at)
        # https://stackoverflow.com/a/37311978/315168
        return max(filter(None, time_vars))

    def get_profit_timeline_timestamp(self) -> datetime.datetime:
        """Where to place this position on a profit timeline.

        - If the position is closed, place it at its closing data

        :return:
            UTC time
        """
        return self.get_last_updated_at()

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
        """The difference in the quantity of assets bought and sold to date.

        .. note::

            To be deprecated. Please use :py:method:`get_quantity` instead.
        """
        return self.get_quantity()

    def get_average_buy(self) -> Optional[USDollarAmount]:
        """Calculate average buy price.

        :return: ``0.0`` if no buys
        """
        q = float(self.get_buy_quantity())
        if not q:
            return 0.0
        return self.get_total_bought_usd() / q

    def get_average_sell(self) -> Optional[USDollarAmount]:
        """Calculate average buy price.

        :return:
            ``0.0`` if no sell trades that would have completed successfully
        """
        q = float(self.get_sell_quantity())
        if not q:
            return 0.0
        return self.get_total_sold_usd() / q

    def get_price_at_open(self) -> USDollarAmount:
        """Legacy.
        """
        warnings.warn('This function is deprecated. Use TradingPosition.get_opening_price() instead', DeprecationWarning, stacklevel=2)
        return self.get_opening_price()

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
        if self.is_long() or self.is_credit_supply():
            return self.get_average_buy()
        else:
            return self.get_average_sell()

    def is_reduced(self) -> bool:
        """Is any of the position closed.

        The position is reduced towards close if it contains opposite trades.

        See also :py:meth:`is_closed`.
        """
        sells = any([t.is_sell() for t in self.trades.values()])
        buys = any([t.is_buy() for t in self.trades.values()])
        return sells and buys

    def get_realised_profit_usd(
            self,
            include_interest=True) -> Optional[USDollarAmount]:
        """Calculates the profit & loss (P&L) that has been 'realised' via two opposing asset transactions in the Position to date.

        - Profit is calculated as the diff of avg buy and sell price times quantity

        - Any avg buy and sell contains all fees we have paid in included in the price,
          so we do not need to add them to profit here

        - See also :py:meth:`get_realised_profit_percent`.

        - Always returns zero for frozen positions

        .. note ::

            This function does not account for in-kind redemptions or any other account corrections.
            Please use :py:meth:`get_realised_profit_percent` if possible.

        :param include_interest:
            Include any accrued interest in PnL.

        :return:
            Profit in dollar.

            `None` if the position lacks any realised profit (contains only unrealised).
        """

        if self.is_frozen():
            return 0

        if not self.is_reduced():
            return None

        if self.is_reduced():
            if self.is_spot_market():
                sells = self.get_average_sell()
                buys = self.get_average_buy()
                sell_unit = self.get_sell_quantity()
                if sell_unit != 0:
                    trade_profit = (sells - buys) * float(sell_unit)
                else:
                    # We do not have successful sells (trades failed) so the
                    # realised profit is zero
                    trade_profit = 0

            elif self.is_credit_supply():

                # NOTE: assume that credit position is using stablecoins
                # so the price wouldn't change overtime
                trade_profit = 0

            else:

                sells = self.get_average_sell()
                buys = self.get_average_buy()
                buy_quantity = self.get_buy_quantity()

                trade_profit = (sells - buys) * float(buy_quantity)
        else:
            # No closes yet, only unrealised PnL
            trade_profit = 0.0

        if include_interest:
            trade_profit += self.get_claimed_interest()  # Profit gained from collateral interest
            trade_profit -= self.get_repaid_interest()  # Loss made from borrowed asset interest payments

        return trade_profit

    def get_unrealised_profit_usd(self, include_interest=True, account_redemptions=True) -> USDollarAmount:
        """Calculate the position unrealised profit.

        Calculates the profit & loss (P&L) that has yet to be 'realised'
        in the remaining non-zero quantity of assets, due to the current
        market price.

        :param include_interest:
            The interest accrued on this position is includedin the result

        :param account_redemptions:
            Any in-kind redemptions are reduced from the net euqantity.

        :return:
            profit in dollar
        """
        avg_price = self.get_average_price()
        if avg_price is None:
            return 0

        # get_quantity() accounts for redemptions
        net_quantity = self.get_quantity()

        if account_redemptions:
            # Simple accounting model where open positions cannot be touched by redemptions
            unrealised_equity = (self.get_current_price() - avg_price) * float(net_quantity)
        else:
            # Simple accounting model where open positions cannot be touched by redemptions
            unrealised_equity = (self.get_current_price() - avg_price) * float(net_quantity)

        if include_interest:
            return unrealised_equity + self.get_accrued_interest()

        return unrealised_equity

    def get_total_profit_usd(self) -> USDollarAmount:
        """Realised + unrealised profit."""
        realised_profit = self.get_realised_profit_usd() or 0
        unrealised_profit = self.get_unrealised_profit_usd() or 0
        total_profit = realised_profit + unrealised_profit
        return total_profit

    def get_total_profit_percent(
        self,
        calculation_method: Literal["cumulative", "legacy"] = "legacy",
        end_at: datetime.datetime | None = None,
    ) -> Percent:
        """How much % we have made profit so far.

        TODO: Legacy method. Use :py:meth:`get_unrealised_and_realised_profit_percent` instead.

        :param calculation_method:
            Use cumulative.

            Legacy is for backwards compatibility.

        :return:
            0 if profit calculation cannot be made yet
        """

        match calculation_method:
            case "legacy":

                if self.is_long():
                    profit = self.get_total_profit_usd()
                    bought = self.get_total_bought_usd()
                    if bought == 0:
                        return 0
                    return profit / bought
                else:
                    # TODO: this is not correct yet since it doesn't factor in interest
                    profit = self.get_total_profit_usd()
                    sold = self.get_total_sold_usd()
                    if sold == 0:
                        return 0
                    return profit / sold

            case "cumulative":
                profit_data = calculate_pnl(self, end_at=end_at)
                return profit_data.profit_pct

    def calculate_total_profit_percent_annualised(
        self,
        end_at: datetime.datetime | None = None,
        calculation_method: Literal["cumulative", "legacy"] = "legacy",
    ) -> Percent:
        """Calculate the annualised profit percentage for this position.

        :param end_at:
            The end date for the annualisation calculation if the position is open..

        :return:
            Annualised profit percentage
        """

        if self.is_closed():
            assert end_at, "Position is still open, but no end date given"

        duration = (end_at - self.opened_at)
        annual_periods = datetime.timedelta(days=365) / duration
        profit = self.get_total_profit_percent(calculation_method=calculation_method, end_at=end_at)

        # Calculate the return factor (1 + profit rate)

        return_factor = 1 + profit

        annualized_factor = return_factor ** annual_periods
        # Convert back to percentage
        annualized_profit_percent = (annualized_factor - 1)
        return annualized_profit_percent

    def get_total_profit_at_timestamp(self, timestamp: datetime.datetime) -> USDollarAmount:
        """Get the profit of the position what it was at a certain point of time.

        Include realised and unrealised profit.

        :param timestamp:
            Include all traeds before and including at this timestamp.
        """
        raise NotImplementedError()

    def get_freeze_reason(self) -> Optional[str]:
        """Return the revert reason why this position is frozen.

        Get the revert reason of the last blockchain transaction, assumed to be swap,
        for this trade.

        If this position has been unfrozen, then return the last freeze reason.

        :return:
            Revert message (cleaned) or None
        """
        assert self.is_frozen(), f"Asked for freeze reason, but position not frozen: {self}"

        if len(self.get_last_trade().blockchain_transactions) == 0:
            logger.warning("Position frozen: Last trade did not have any blockchain transactions: %s", self)
            for t in self.trades.values():
                logger.warning("Trade #%d: %s", t.trade_id, t)
            return "Could not extract freeze reason"

        t: TradeExecution
        for t in reversed(self.trades.values()):
            reason = t.get_revert_reason()
            if reason:
                return reason

        return None

    def get_last_tx_hash(self) -> Optional[str]:
        """Get the latest transaction performed for this position.

        It's the tx of the trade that was made for this position.

        TODO: Deprecate
        """
        t = self.get_last_trade()
        if not t:
            return None

        # TODO: Not sure what is going on here
        if len(t.blockchain_transactions) == 0:
            logger.warning("Trade does not have transactions: %s", self)
            return None

        return t.blockchain_transactions[-1].tx_hash

    def get_last_valued_at(self) -> datetime.datetime:
        """Get the timestamp when this position was last valued"""

        if self.is_loan_based():
            # Loan-based positions are updated by update_interest()
            # and it does not always update the token price (e.g. aUSDC).
            # The timestamp for these positions must be thus taken from the interest update events.
            timestamps = []
            if self.loan.collateral and self.loan.collateral_interest.last_updated_at:
                timestamps.append(self.loan.collateral_interest.last_updated_at)
            if self.loan.borrowed and self.loan.borrowed_interest.last_updated_at:
                # Credit position does not have borrowed part of the loan
                timestamps.append(self.loan.borrowed_interest.last_updated_at)
            # TODO: Need to be smarter here?
            logger.info("Credit position timestamps: %s", timestamps)
            return max(timestamps)
        elif self.valuation_updates:
            # We have generated valuation update event for this position.
            # Use data from it/
            return self.valuation_updates[-1].valued_at
        else:
            # Spot tokens
            # TODO: Legacy. Fix everything to use the same path.
            return self.last_pricing_at

    def revalue_base_asset(self, last_pricing_at: datetime.datetime, last_token_price: USDollarPrice):
        """Update position token prices.

        TODO: Legacy. See :py:attr:`valuation_updates`.
        """
        assert isinstance(last_pricing_at, datetime.datetime)
        assert not isinstance(last_pricing_at, pd.Timestamp)

        assert isinstance(last_token_price, float), f"Expected price as float, got {last_token_price.__class__}"
        assert not isinstance(last_pricing_at, np.float32)

        assert last_pricing_at.tzinfo is None
        self.last_pricing_at = last_pricing_at
        self.last_token_price = last_token_price

        if self.loan and self.is_short():
            self.loan.borrowed.revalue(last_token_price, last_pricing_at)

        return self.get_value()

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

    def get_capital_tied_at_open_pct(self) -> Percent:
        """Calculate how much portfolio capital was risk when this position was opened.

        - This is based on the opening values,
          any position adjustment after open is ignored

        - Assume capital is tied to the position and we can never release it.

        - Assume no stop loss is used, or it cannto be trigged

        See also :py:meth:`get_loss_risk_at_open_pct`.

        :return:
            Percent of the portfolio value
        """
        if not self.portfolio_value_at_open:
            raise LegacyDataException(f"Portfolio value at position open was not recorded for {self}")

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

        # Failed trade, or legacy data
        if not self.is_properly_opened():
            return 0.0

        # assert self.is_long(), "Only long positions supported"
        assert self.stop_loss, f"Stop loss price must be set to calculate the maximum risk"
        # Calculate how much value we can lose

        opening_price = self.get_opening_price()
        if opening_price is None:
            # Legacy data
            return 0.0

        price_diff = abs(opening_price - self.stop_loss)
        risked_value = price_diff * float(self.get_quantity_at_open())
        return risked_value

    def get_loss_risk_at_open_pct(self) -> Percent:
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

    def get_realised_profit_percent(self) -> Percent:
        """Calculated life-time profit over this position.

        Calculate how many percent profit this position made,
        relative to all trades taken over the life time of the position.

        See also

        - :py:meth:`get_realised_profit_usd`

        - :py:meth:`get_unrealised_profit_usd`

        See :ref:`profitability` for more details.

        :return:
            If the position made 1% profit returns 0.01.

            Return ``0`` if the position profitability cannot be calculated,
            e.g. due to broken trades.
        """
        if self.is_spot():
            # This is the new code path that takes account in-kind redemptions
            # and redefines the meaning of realised profit
            return self.get_unrealised_and_realised_profit_percent(include_unrealised=False)
        elif self.is_long() or self.is_credit_supply():
            # Legacy path
            # TODO: Check if we need to use lending-based calculations here
            total_bought = self.get_total_bought_usd()
            if total_bought == 0:
                return 0
            return (self.get_realised_profit_usd() or 0)/total_bought
        elif self.is_short():
            # Legacy path
            # TODO: Check if we need to use lending-based calculations here
            total_sold = self.get_total_sold_usd()
            if total_sold == 0:
                return 0
            return (self.get_realised_profit_usd() or 0)/total_sold
        else:
            # TODO: Some legacy code paths end here?
            # raise NotImplementedError(f"Should not never happen as for non-spot positions we use leverage-based profit calculation: {self}")
            return 0

    def estimate_gained_interest(
        self,
        interest_period: Literal["position", "year"] = "position",
    ):
        """Calculate avg % interest we have earned over time for a credit supply position.

        - Slow, as we need to iterate over all balance update events

        - See also: py:meth:`get_unrealised_and_realised_profit_percent`

        :return:
            Average interest over time.

            Regardless of position increase/decrease.

            Return 0 if no datapoints.
        """

        try:
            # Unrolled get_effective_yearly_yield()
            yearly_interest = statistics.mean(
                float(b.quantity / b.old_balance) / ((b.block_mined_at - b.previous_update_at) / DEFAULT_YEAR)
                for b in self.balance_updates.values()
                if (b.cause == BalanceUpdateCause.interest) and (b.previous_update_at is not None) and (b.previous_update_at != b.block_mined_at)
            )

            match interest_period:
                case "year":
                    return yearly_interest
                case "position":
                    return yearly_interest * self.get_duration(partial=True) / DEFAULT_YEAR
                case _:
                    raise NotImplementedError(f"Unknown divider")

        except statistics.StatisticsError:
            # Zero data points
            return 0

    def get_unrealised_and_realised_profit_percent(
        self,
        valuation_price=None,
        include_unrealised=True,
    ) -> Percent:
        """Calculated unrealised PnL for this position.

        This is an estimation of the profit % assuming the position would be completely closed
        with the current price.

        - This function accounts for in-kind redemptions affecting the position

        - Currently only long positions supportd

        See also

        - :py:meth:`get_realised_profit_percent`

        - :py:meth:`get_total_profit_percent` (don't use, legacy)

        - py:meth:`get_unrealised_and_realised_profitability_percent_credit`

        :param valuation_price:
            Valuate the unrealised portion of tokens at this price

        :param include_unrealised:
            Include the unrealised PnL as the part of the result

        :return:
            The profitability of this position currently.

            Return 0.05 for a position that is 5% in profit.

            Return 0 if you ask for realised profit and none of the position is yet closed.
        """

        if self.is_long():
            redemptions = self.get_redeemed()  # Negative
            total_bought = self.get_total_bought_usd()
            if total_bought == 0:
                return 0

            unrealised_equity = 0
            if include_unrealised:
                quantity_left_sell = self.get_quantity()
                if quantity_left_sell:

                    if not valuation_price:
                        valuation_price = self.last_token_price

                    # Marked down positions need to have special handling
                    if not self.is_marked_down():
                        assert valuation_price, f"Cannot value unrealised PnL without an explicit valuation price for the unsold portion.\n" \
                                                f"Position was: {self}, quantity left: {quantity_left_sell}, valuation price: {valuation_price}"
                    unrealised_equity = valuation_price * float(quantity_left_sell)

            avg_price = self.get_average_price()
            # redemption_adjustment = avg_price * float(redemptions)
            profit = ((self.get_realised_profit_usd() or 0) + (self.get_unrealised_profit_usd() or 0))

            buy_quantity = self.get_buy_quantity()

            sell_quantity = self.get_sell_quantity()
            redeem_adjusted_buy_quantity = buy_quantity + redemptions
            buy_volume = sum([t.get_value() for t in self.trades.values() if t.is_buy()])
            sell_volume = sum([t.get_value() for t in self.trades.values() if t.is_sell()])
            average_buy = buy_volume / float(buy_quantity)
            average_sell = (sell_volume / float(sell_quantity)) if sell_quantity else 0

            # Predent redemptions where there not in the first palce
            adjusted_buy_volume = buy_volume + (average_buy * float(redemptions))
            adjusted_sell_volume = sell_volume

            if adjusted_buy_volume == 0:
                # We do not have any buys, so we would give zero divider
                return 0

            if redemptions < 0:  # Some broken data might break redemptions amount, in legacy tests
                assert adjusted_buy_volume <= buy_volume, f"Adjusted buy volume: {adjusted_buy_volume}, buy volume: {buy_volume}"
                assert adjusted_sell_volume <= sell_volume

            # How much have we closed this position
            if not include_unrealised:

                # Position has account corrections
                divider = (buy_quantity + redemptions)
                if divider == 0:
                    return 0

                pct_closed = float(sell_quantity / divider)
                if pct_closed == 0:
                    return 0

                # if pct_closed not in (0, 1):
                #    import ipdb ; ipdb.set_trace()

                return sell_volume / (adjusted_buy_volume * pct_closed) - 1

                # return (average_sell - average_buy) * buy_quantity * pct_closed
                # adjusted_profit_pct = 1 - (adjusted_buy_volume-adtest_legacy_calculate_all_statisticsjusted_sell_volume)/(adjusted_buy_volume*pct_closed)
            else:

                adjusted_profit_pct = (adjusted_sell_volume + unrealised_equity - adjusted_buy_volume) / (adjusted_buy_volume)

            # average_sell = self.get_total_sold_usd() / float(sell_quantity)
            # unadjusted_profit_usd = (average_sell - average_buy) / float(buy_quantity)
            # buy_volume = self.get_average_buy() * float(redeem_adjusted_buy_quantity)
            return adjusted_profit_pct
            #else:
            #    # No in-kind redemptions, the simple accounting path
            #    return ((self.get_realised_profit_usd() or 0) + (self.get_unrealised_profit_usd() or 0)) / total_bought
        else:
            raise NotImplementedError(f"get_unrealised_and_realised_profit_percent() supports only long positions ATM, got {self}")

    def get_unrealised_profit_pct(self) -> Percent:
        """Get the current profit of this position, minus any netflow.

        .. warning::

            This function is only tested for spot positions with a single trade.
            It may fail to calculate other kind of positions.

        - Calculate based on avg buy and sell

        - For the unrealised portion, calculate the expected close

        See also

        - :py:meth:`get_realised_profit_percent`

        :return:
            Estimated position profit in percent, based on avg trade prices.

            E.g. if the position gained 100 USD -> 110 USD return 0.1.

            If profit cannot be calculated yet (zero trades executed),
            return 0.
        """

        realised_profit = self.get_unrealised_profit_usd()
        if realised_profit is None:
            realised_profit = 0

        if self.is_long():
            total_bought = self.get_total_bought_usd()
            if total_bought == 0:
                return 0
            return realised_profit/total_bought
        else:
            total_sold = self.get_total_sold_usd()
            if total_sold == 0:
                return 0
            return realised_profit/total_sold

    def get_size_relative_realised_profit_percent(self) -> Percent:
        """Calculated life-time profit over this position.

        The profit is scaled to the % of the position size relative to the portfolio
        to account for max capital allocation for the position.

        - TODO: This does not work for positions that have capital added over time or strategies that have overlapping positions

        See :ref:`profitability` for more details.

        :return:
            If the 50% aloocation position made 1% profit returns 1.005.
        """
        return self.get_realised_profit_percent() * self.get_capital_tied_at_open_pct()

    def get_size_relative_unrealised_or_realised_profit_percent(self) -> Percent:
        """Calculated life-time profit over this position, including unrealised PnL.

        The profit is scaled to the % of the position size relative to the portfolio
        to account for max capital allocation for the position.

        - TODO: This does not work for positions that have capital added over time

        - TODO: Only correctly support unrealised PnL for spot

        :return:
            If the 50% aloocation position made 1% profit returns 1.005.
        """

        if self.is_credit_supply():
            # Credit position profitability is avg interest %
            return self.estimate_gained_interest()
        if self.is_spot():
            # Calculate with in-kind redemption support
            return self.get_unrealised_and_realised_profit_percent() * self.get_capital_tied_at_open_pct()
        else:
            # Legacy fallback
            return self.get_size_relative_realised_profit_percent()

    def get_size_relative_profit_percent(self) -> Percent:
        """Calculated life-time profit over this position.

        Both realised and unrealised profit.

        :return:
            If the position made 1% profit returns 1.01.
        """
        return self.get_realised_profit_percent() * self.get_capital_tied_at_open_pct()

    def get_duration(
        self,
        partial=False,
        execution_mode: ExecutionMode=ExecutionMode.real_trading,
        end_at: datetime.datetime = None,
    ) -> datetime.timedelta | None:
        """How long this position was held.

        :param partial:
            Return duration until now for open positions.

            Assumes the position is still open.

        :param end_at:
            What is the timestamp at the end of the backtest.

            Not needed for live trading, uses current time.

        :param execution_mode:
            Are we live trading or backtesting.

        :return:
            None if the position is still open if partial is not set
        """
        if self.is_closed():
            return self.closed_at - self.opened_at

        if partial:
            if execution_mode.is_backtesting():
                assert end_at, f"get_duration(): You must give end_at timestamp in backtesting"
            else:
                end_at = datetime.datetime.utcnow()

            return end_at - self.opened_at

        return None

    def get_total_lp_fees_paid(self) -> USDollarAmount:
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
        """Get the largest size of this position over time

        NOTE: This metric doesn't work for positions with more than 2 trades
        i.e: positions which have been increased and reduced in size
        """
        return self.get_first_trade().get_executed_value()

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

    def get_original_planned_price(self) -> USDollarPrice:
        """Get the US-dollar price for a position that does not have any executed price.

        - The position was left in a broken state after the first trade failed to execute

        - We will still have the price from the first trade we thought
          we were going to get

        - Always use :py:attr:`TradingPosition.last_token_price` if available.

        - Applies to spot positions only

        :return:
            PlannedUS dollar spot price of the first trade
        """
        assert self.is_spot()
        first_trade = self.get_first_trade()
        return first_trade.planned_price

    def calculate_quantity_usd_value(self, quantity: Decimal) -> USDollarAmount:
        """Calculate value of asset amount using the latest known price.

        - If we do not have price data, use the first planned trade
        """
        if quantity == 0:
            return 0

        price = self.last_token_price
        if price is None:
            # See test_cli_correct_account_price_missing
            price = self.get_original_planned_price()

        return float(quantity) * price

    def add_notes_message(self, msg: str):
        """Add a new message to the notes field.

        Messages are newline separated.
        """

        if not msg:
            return

        if self.notes is None:
            self.notes = ""

        self.notes += msg
        self.notes += "\n"

    def add_balance_update_event(self, event: BalanceUpdate):
        """Include a new balance update event

        :raise BalanceUpdateEventAlreadyAdded:
            In the case of a duplicate and event id is already used.
        """
        if event.balance_update_id in self.balance_updates:
            raise BalanceUpdateEventAlreadyAdded(f"Duplicate balance update: {event}")

        self.balance_updates[event.balance_update_id] = event

    def calculate_accrued_interest_quantity(self, asset: AssetIdentifier) -> Decimal:
        """Calculate the gained interest in tokens.

        This is done as the sum of all interest events.

        This is also denormalised as `position.interest.accrued_interest`.

        :param asset:
            aToken/vToken for which we calculate the interest for

        :return:
            Number of quote tokens this position has gained interest.
        """

        assert asset.underlying is not None, "asset argument must be aToken/vToken"

        return sum_decimal([
            b.quantity
            for b in self.balance_updates.values()
            if b.cause == BalanceUpdateCause.interest and b.asset == asset
        ])

    def get_accrued_interest(self) -> USDollarAmount:
        """Get the USD value of currently net accrued interest for this position so far.

        Get any unclaimed interest on this position. After position is closed,
        all remaining accrued interest is claimed.

        See :py:meth:`get_accrued_interest_with_repayments` to account any interest payments.

        - The accrued interest is included as the position accounting item until the position is completely closed

        - When the position is completed closed,
          the accured interest tokens are traded and moved to reserves

        - After position is closed calling `get_accrued_interest()`
          should return zero

        TODO: This might not work correctly for partially closed positions.

        :return:
            Net interest PnL in USD.

            Positive if we have earned interest, negative if we have paid it.
        """

        if self.loan is not None:
            return self.loan.get_net_interest()
        return 0.0

    def get_accrued_interest_with_repayments(self) -> USDollarAmount:
        """Return the net from collateral and borrowed interest plus their interest payments.

        - See also :py:meth:`get_accrued_interest`

        """
        return self.get_accrued_interest()

    def get_claimed_interest(self) -> USDollarAmount:
        """How much interest we have claimed from this position and moved back to reserves.

        See also

        - :py:meth:`get_accrued_interest` for the life-time interest accumulation.

        - :py:meth:`Loan.get_net_asset_value` for notes about loan interest tracking

        """
        interest = sum([t.get_claimed_interest() for t in self.trades.values() if t.is_success()])
        return interest

    def get_repaid_interest(self) -> USDollarAmount:
        """How much interest payments we have made in total.

        See also

        - :py:meth:`get_claimed_interest`.

        - :py:meth:`Loan.get_net_asset_value` for notes about loan interest tracking
        """
        interest = sum([t.get_repaid_interest() for t in self.trades.values() if t.is_success()])
        return interest

    def get_borrowed(self) -> USDollarAmount:
        """Get the amount of outstanding loans we have."""
        return self.loan.borrowed.get_usd_value()

    def get_collateral(self) -> USDollarAmount:
        """Get the amount of outstanding loans we have."""
        return self.loan.collateral.get_usd_value()

    def get_held_assets(self) -> Iterable[Tuple[AssetIdentifier, Decimal]]:
        """Get the assumption of the portfolio asset amount for the account correction / accounting double check.

        :return:
            (Asset id, amount) iterables
        """
        assert self.is_open() or self.is_frozen()
        if self.is_spot():
            yield self.pair.base, self.get_quantity()
        elif self.is_credit_supply():
            yield self.loan.collateral.asset, self.loan.get_collateral_quantity()
        elif self.is_short():
            yield self.loan.collateral.asset, self.loan.get_collateral_quantity()
            yield self.loan.borrowed.asset, self.loan.get_borrowed_quantity()
        else:
            raise AssertionError(f"Unsupported: {self}")

    def get_annualised_credit_interest(self) -> Percent:
        """Get the annualised interest for a credit position.

        See also: :py:meth:`get_annualised_profit`

        :return:
            Annualised interest.

            Return 0 if the position does not have a duration, or its still open.
        """
        assert self.is_credit_supply(), f"Only works with credit positions, got {self}"

        duration = self.get_duration()
        value_at_open = self.get_value_at_open()
        claimed_interest = self.get_claimed_interest()

        if duration is None or value_at_open == 0:
            return 0.0

        return (claimed_interest / value_at_open) * datetime.timedelta(days=365) / duration

    def calculate_annualised_profit(self, duration: datetime.timedelta) -> Percent:
        """Get the annualised profit for any position position.

        See also: :py:meth:`get_annualised_credit_interest`

        :param duration:
            See :py:meth:`get_duration`

        :return:
            Annualised profit.

            Return 0 if the position does not have a duration, or its still open.
        """
        assert isinstance(duration, datetime.timedelta), f"Got: {duration.__class__}"
        assert duration != ZERO_TIMEDELTA, f"Got: {duration}"
        return self.get_unrealised_and_realised_profit_percent() * datetime.timedelta(days=365) / duration

    def mark_down(self):
        """Manually set position value to zero.

        - Must be done to unsellable assets like scam coins
        """
        self.other_data["marked_down_at"] = datetime.datetime.utcnow()
        self.add_notes_message(f"Marked down to zero manually, last price was {self.last_token_price}, last value was: {self.get_value()}")
        self.last_token_price = 0
        self.last_pricing_at = datetime.datetime.utcnow()
        self.closed_at = datetime.datetime.utcnow()
        self.unfrozen_at = datetime.datetime.utcnow()
