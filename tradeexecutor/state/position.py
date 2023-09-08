"""Trading position state info."""
import datetime
import enum
import logging
import pprint
from dataclasses import dataclass, field, asdict
from decimal import Decimal

from typing import Dict, Optional, List, Iterable

import numpy as np
import pandas as pd
from dataclasses_json import dataclass_json

from tradeexecutor.state.balance_update import BalanceUpdate, BalanceUpdateCause
from tradeexecutor.state.generic_position import GenericPosition, BalanceUpdateEventAlreadyAdded
from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier, TradingPairKind
from tradeexecutor.state.interest import Interest
from tradeexecutor.state.loan import Loan
from tradeexecutor.state.trade import TradeType, QUANTITY_EPSILON
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.types import USDollarAmount, BPS, USDollarPrice, Percent, LeverageMultiplier
from tradeexecutor.strategy.dust import get_dust_epsilon_for_pair
from tradeexecutor.strategy.lending_protocol_leverage import create_short_loan, plan_loan_update_for_short, create_credit_supply_loan, update_credit_supply_loan
from tradeexecutor.strategy.trade_pricing import TradePricing
from tradeexecutor.utils.accuracy import sum_decimal
from tradingstrategy.lending import LendingProtocolType

logger = logging.getLogger(__name__)


#: If a token position helds less than this absolute amount of token
#: consider closing it as dust
CLOSED_POSITION_DUST_EPSILON = 0.0001


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
        
        if self.stop_loss_before:
            assert self.stop_loss_before < self.stop_loss_after


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

    #: 1.0 for stablecoins, unless out of peg, in which case can be 0.99
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
    #: - Used to add log information abotu frozen and unfrozen positions
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
    trigger_updates: List[TriggerPriceUpdate] = field(default_factory=list)

    #: The loan underlying the position leverage or crdit supply.
    #:
    #: Applicable for
    #:
    #: - short/long positions using lending protocols
    #:
    #: - credit supply (collateral without borrow)
    #:
    loan: Optional[Loan] = None

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

    def get_human_readable_name(self) -> str:
        return f"Trading position #{self.position_id} for {self.pair.get_ticker()}"

    def get_debug_dump(self) -> str:
        """Return class contents for logging.

        :return:
            Indented JSON-like content
        """
        return pprint.pformat(asdict(self), width=160)

    def is_open(self) -> bool:
        """This is an open trading position."""
        return self.closed_at is None

    def is_closed(self) -> bool:
        """This position has been closed and does not have any capital tied to it.

        See also :py:meth:`is_reduced`:
        """
        return not self.is_open()

    def is_frozen(self) -> bool:
        """This position has had a failed trade and can no longer be automatically moved around.

        After the position is unfrozen the flag goes away.
        """
        return (self.frozen_at is not None) and not self.is_unfrozen()

    def is_unfrozen(self) -> bool:
        """This position was frozen, but its trades were successfully repaired."""
        return self.unfrozen_at is not None

    def is_repaired(self) -> bool:
        """This position was frozen, but its trades were successfully repaired.

        Alias for :py:meth:`is_unfrozen`.
        """
        return self.is_unfrozen()

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

    def is_spot_market(self):
        """This is a spot market position."""
        return self.pair.kind == TradingPairKind.spot_market_hold

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

    def get_quantity(self) -> Decimal:
        """Get the tied up token quantity in all successfully executed trades.

        - Does not account for trades that are currently being executed.

        - Does some fixing for rounding errors in the form of epsilon checks

        - Accounts for any balance update events (redemptions, interest, accounting corrections)

        For interest positions

        - The underlying principle is calculated as sum of trades e.g.
          how many deposit or redemption trades we did for Aave reserves

        - The accrued interest can be read from balance update events

        :return:
            Number of asset units held by this position.

            Rounded down to zero if the sum of
        """
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
            assert abs(s) >= QUANTITY_EPSILON, f"Epsilon dust safety check in floating point math triggered. Quantity: {s}. Epsilon: {QUANTITY_EPSILON}."

        # Always convert zero to decimal
        return Decimal(s)

    def get_available_trading_quantity(self) -> Decimal:
        """Get token quantity still availble for the trades in this strategy cycle.

        This includes

        - All executed trades

        - All planned trades for this cycle that have already reduced/increased
          amounts for this position

        This gives you remaining token balance, even if there are some earlier
        sell orders that have not been executed yet.
        """
        planned = sum([t.get_position_quantity() for t in self.trades.values() if t.is_planned()])
        live = self.get_quantity()
        return planned + live

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

        return float(token_quantity) * token_price + float(reserve_quantity) * reserve_price

    def get_equity(self) -> USDollarAmount:
        """Get equity tied to this position.

        TODO: Use :py:meth:`TradingPosition.loan.get_net_asset_value` for collateral based positions.

        :return:
            How much equity we have tied in this position.

            TODO: Does not work for collateral positions.
        """

        match self.pair.kind:
            case TradingPairKind.spot_market_hold:
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
            case TradingPairKind.spot_market_hold | TradingPairKind.credit_supply:

                value += self.calculate_value_using_price(
                    self.last_token_price,
                    self.last_reserve_price,
                    include_interest=False,
                )

            case TradingPairKind.lending_protocol_short:
                # Value for leveraged positions is net asset value from its two loans
                return self.loan.get_net_asset_value(include_interest)
            case _:
                raise NotImplementedError(f"Does not know how to value position for {self.pair}")

        return value

    def get_loan_based_nav(self, include_interest=True, include_fees=True):
        """Calculate NAV for a lona based position."""
        nav = self.loan.get_net_asset_value(include_interest)
        # TODO: Do we need to include fees
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

    def open_trade(self,
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
        """

        # Done in State.create_trade()
        # if quantity is not None:
        #    assert reserve is None, "Quantity and reserve both cannot be given at the same time"

        pair = self.pair

        if price_structure is not None:
            assert isinstance(price_structure, TradePricing)
        
        assert self.reserve_currency.get_identifier() == reserve_currency.get_identifier(), "New trade is using different reserve currency than the position has"
        assert isinstance(trade_id, int)

        if strategy_cycle_at is not None:
            assert isinstance(strategy_cycle_at, datetime.datetime)

        # Set lending market estimated quantities
        match pair.kind:
            case TradingPairKind.lending_protocol_short:

                if len(self.trades) == 0:
                    assert reserve is not None, "Both reserve and quantity needs to be given for lending protocol short open"
                    assert quantity is not None, "Both reserve and quantity needs to be given for lending protocol short open"
                    assert not closing, "Cannot close position not yet open"

                    # Automatically calculate the amount of collateral increase for this trade sizes
                    if planned_collateral_consumption is None:
                        planned_collateral_consumption = -quantity * Decimal(assumed_price)

                else:

                    if closing:
                        assert reserve is None, "reserve calculated automatically when closing a short position"
                        assert quantity is None, "quantity calculated automatically when closing a short position"
                        assert not planned_collateral_consumption, "planned_collateral_consumption set automatically when closing a short position"

                        # Pay back all the debt and its interest
                        quantity = self.loan.get_borrowed_principal_and_interest_quantity()

                        # Release collateral is the current collateral
                        reserve = 0

                        # We need to use USD from the collateral to pay back the loan
                        planned_collateral_consumption = -quantity * Decimal(self.loan.borrowed.last_usd_price)

                        # Any leftover USD from the collateral is released to the reserves
                        planned_collateral_allocation = -(self.loan.collateral.quantity + planned_collateral_consumption)

                    else:
                        assert quantity is not None, "For increasing/reducing short position quantity must be given"
                        planned_collateral_consumption = -quantity * Decimal(self.loan.borrowed.last_usd_price)
                        planned_collateral_allocation = planned_collateral_allocation

                assert reserve_currency_price, f"Collateral price missing"
                assert assumed_price, f"Short token price missing"

                planned_reserve = reserve or Decimal(0)
                planned_quantity = (quantity or Decimal(0))

                # From now on, we need meaningful values for math
                planned_collateral_consumption = planned_collateral_consumption or Decimal(0)
                planned_collateral_allocation = planned_collateral_allocation or Decimal(0)

            case TradingPairKind.spot_market_hold:
                # Set spot market estimated quantities
                if reserve is not None:
                    planned_reserve = reserve
                    planned_quantity = reserve / Decimal(assumed_price)
                else:
                    planned_quantity = quantity
                    planned_reserve = abs(quantity * Decimal(assumed_price))
            case TradingPairKind.credit_supply:
                assert reserve, "You must give reserve"
                assert quantity, "You must give quantity"
                planned_reserve = reserve
                planned_quantity = quantity

            case _:
                raise NotImplementedError(f"Does not know how to calculate quantities for open a trade on: {pair}")

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
        )

        self.trades[trade.trade_id] = trade

        # Initialise interest tracking data structure
        if pair.kind.is_interest_accruing():
            if pair.kind == TradingPairKind.credit_supply:
                assert pair.kind == TradingPairKind.credit_supply, "Only credit supply supported for now"
                if self.loan is None:
                    assert trade.is_buy(), "Opening credit position is modelled as buy"
                    trade.planned_loan_update = create_credit_supply_loan(self, trade, strategy_cycle_at)
                else:
                    trade.planned_loan_update = update_credit_supply_loan(self, trade, strategy_cycle_at)

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
                        trade.planned_loan_update = plan_loan_update_for_short(
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

    def can_be_closed(self) -> bool:
        """There are no tied tokens in this position.

        Perform additional check for token amount dust caused by rounding errors.
        """
        epsilon = get_dust_epsilon_for_pair(self.pair)
        quantity = self.get_quantity()
        return abs(quantity) <= epsilon

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

        :return:
            ``None`` if no sell trades that would have completed successfully
        """
        q = float(self.get_sell_quantity())
        if q == 0:
            return None
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

        :param include_interest:
            Include any accrued interest in PnL.

        :return:
            Profit in dollar.

            `None` if the position lacks any realised profit (contains only unrealised).
        """

        if not self.is_reduced():
            return None

        if self.is_reduced():
            if self.is_spot_market():
                sells = self.get_average_sell() or 0.0
                buys = self.get_average_buy() or 0.0
                sell_unit = self.get_sell_quantity()
                if sell_unit != 0:
                    trade_profit = (sells - buys) * float(sell_unit)
                else:
                    # We do not have successful sells (trades failed) so the
                    # realised profit is zero
                    trade_profit = 0
            else:
                trade_profit = (self.get_average_sell() - self.get_average_buy()) * float(self.get_buy_quantity())
        else:
            # No closes yet, only unrealised PnL
            trade_profit = 0.0

        if include_interest:
            trade_profit += self.get_claimed_interest()  # Profit gained from collateral interest
            trade_profit -= self.get_repaid_interest()  # Loss made from borrowed asset interest payments

        return trade_profit

    def get_unrealised_profit_usd(self, include_interest=True) -> USDollarAmount:
        """Calculate the position unrealised profit.

        Calculates the profit & loss (P&L) that has yet to be 'realised'
        in the remaining non-zero quantity of assets, due to the current
        market price.

        :return:
            profit in dollar
        """
        avg_price = self.get_average_price()
        if avg_price is None:
            return 0

        unrealised_equity = (self.get_current_price() - avg_price) * float(self.get_net_quantity())

        if include_interest:
            return unrealised_equity + self.get_accrued_interest() - self.get_claimed_interest() + self.get_repaid_interest()

        return unrealised_equity

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
        if self.is_long():
            profit = self.get_total_profit_usd()
            bought = self.get_total_bought_usd()
            if bought == 0:
                return 0
            return profit / bought
        else:
            # TODO: this is not correct yet since it doesn't factor in interest
            profit = -self.get_total_profit_usd()
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

    def revalue_base_asset(self, last_pricing_at: datetime.datetime, last_token_price: USDollarPrice):
        """Update position token prices."""
        assert isinstance(last_pricing_at, datetime.datetime)
        assert not isinstance(last_pricing_at, pd.Timestamp)

        assert isinstance(last_token_price, float), f"Expected price as float, got {last_token_price.__class__}"
        assert not isinstance(last_pricing_at, np.float32)

        assert last_pricing_at.tzinfo is None
        self.last_pricing_at = last_pricing_at
        self.last_token_price = last_token_price

        if self.loan:
            assert self.is_short()
            self.loan.borrowed.revalue(last_token_price, last_pricing_at)

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
        assert self.portfolio_value_at_open, f"Portfolio value at position open was not recorded for {self}"
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

        See :ref:`profitability` for more details.

        :return:
            If the position made 1% profit returns 1.01.
        """
        
        assert not self.is_open(), "Cannot calculate realised profit for open positions"
        buy_value = self.get_buy_value()
        sell_value = self.get_sell_value()

        if buy_value == 0:
            # Repaired trade
            return 0

        return sell_value / buy_value - 1

    def get_size_relative_realised_profit_percent(self) -> Percent:
        """Calculated life-time profit over this position.

        Calculate how many percent this profit made profit,
        adjusted to the position size compared to the available
        strategy equity at the opening of the position.

        This is mostly useful to calculate the strategy performance
        independent of funding deposits and redemptions.

        See :ref:`profitability` for more details.

        :return:
            If the position made 1% profit returns 1.01.
        """
        return self.get_realised_profit_percent() * self.get_capital_tied_at_open_pct()

    def get_duration(self) -> datetime.timedelta | None:
        """How long this position was held.
        :return: None if the position is still open
        """
        if self.is_closed():
            return self.closed_at - self.opened_at  
        else:
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

    def calculate_quantity_usd_value(self, quantity: Decimal) -> USDollarAmount:
        """Calculate value of asset amount using the latest known price."""
        if quantity == 0:
            return 0
        assert self.last_token_price, f"Asset price not available when calculating price for quantity: {quantity}"
        return float(quantity) * self.last_token_price

    def add_notes_message(self, msg: str):
        """Add a new message to the notes field.

        Messages are newline separated.
        """
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

        :return:
            Number of quote tokens this position has gained interest
        """
        return sum_decimal([
            b.quantity
            for b in self.balance_updates.values()
            if b.cause == BalanceUpdateCause.interest and b.asset == asset
        ])

    def get_accrued_interest(self) -> USDollarAmount:
        """Get the USD value of currently net accrued interest for this position so far.

        See :py:meth:`get_accrued_interest_with_repayments` to account any interest payments.

        - The accrued interest is included as the position accounting item until the position is completely closed

        - When the position is completed closed,
          the accured interest tokens are traded and moved to reserves

        - After position is closed calling `get_accrued_interest()` Keeps returning
          the lifetime interest of the position.

        - See :py:meth:`get_claimed_interest` to get the interest that has
          been moved to reserves from this position.


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
        return self.get_accrued_interest() - self.get_claimed_interest() + self.get_repaid_interest()

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

