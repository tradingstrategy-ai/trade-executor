"""Trigger order structure.

- Simulate market limit orders and such

- Stop loss and take profit are hardcoded into :py:class:`tradeexecutor.state.trade.TradeExecution`
  and not currently part of :py:class:`Trigger` data structure

"""
import datetime
import enum
from decimal import Decimal
from dataclasses import dataclass
from typing import TypeAlias

from tradeexecutor.state.types import USDollarPrice
from tradingstrategy.types import USDollarAmount, Percent


#: Take profit / stop less levels are (price, quantity) tuples
#:
#: - Base asset price as US Dollar
#: - Quantity in base asset units
#: - Close flag: True if this trade should close the position
#:
PartialTradeLevel: TypeAlias = tuple[USDollarAmount | datetime.timedelta | datetime.datetime, Decimal, bool]


class TriggerType(enum.Enum):
    market_limit = "market_limit"

    #: Execute take profit and partially close position
    #:
    #: This is also used when closing the position with time (e.g. 24h since open)
    #:
    take_profit_partial = "take_profit_partial"

    #: Execute take profit and close position fully
    take_profit_full = "take_profit_full"


class TriggerCondition(enum.Enum):

    #: Market mid-price goes above a level
    cross_above = "cross_above"

    #: Market mid-price below above a level
    cross_below = "cross_below"

    #: Execute regardless of a price when deadline reached.
    #:
    #: E.g. close after midnight.
    #:
    timed_absolute = "timed_absolute"

    #: Close based on opening time
    #:
    timed_relative_to_open = "timed_relative_to_open"



@dataclass(frozen=False, slots=True)
class Trigger:
    """Trigger data for market orders.

    Triggers can be on both on

    - Unopened positions, see :py:attr:`tradeeexecutor.state.portfolio.Portfolio.pending_positions`
      where the trigger is on the trade that will open the position (market limit order case)

    - Opened positions, see :py:attr:`tradeeexecutor.state.position.TradingPosition.pending_positions`
      where the trigger order is on the increase/reduce position (partial take profit case)

    - Trigger orders can be created to make trades to happen outside the strategy decision cycle

    - Market limit order is the most famous trigger order from the TradFi markets,
      used to enter into breakout positions

    - Triggers are executed in the s

    - Nested structure inside :py:class:`tradeexecutor.state.trade.TradeExecution`

    - Any price structure estimations on trigger trades is based on the time of the trigger creation,
      and maybe very different when the trigger is executed

    """

    #: Do we trigger when price crossed above or below
    type: TriggerType

    #: When this trigger is executed
    #:
    condition: TriggerCondition

    #: When to take action
    price: USDollarAmount | None = None

    #: Always executed at a certain time.
    #:
    #: See `TriggerType.timed_absolute`
    #:
    triggering_at: datetime.datetime | None = None

    #: Always executed at a certain time.
    #:
    #: See `TriggerType.timed_relative_to_open`
    #:
    triggering_at_delta: datetime.timedelta | None = None

    #: After expiration, this trade execution is removed from the hanging queue
    expires_at: datetime.datetime | None = None

    #: When this trigger was marked as expired
    expired_at: datetime.datetime | None = None

    #: When the trigger happened
    #:
    #:
    triggered_at: datetime.datetime | None = None

    def __repr__(self):
        return f"<Trigger {self.type.value} {self.condition.value} at {self.price} expires at {self.expires_at}>"

    def __post_init__(self):
        if self.expires_at is not None:
            assert isinstance(self.expires_at, datetime.datetime)

        if type(self.price) == int:
            self.price = float(self.price)

        if self.price is not None:
            assert type(self.price) == float, f"Price not a float: {type(self.price)}"

        if self.triggering_at:
            assert isinstance(self.triggering_at, datetime.datetime)

        if self.triggering_at_delta:
            assert isinstance(self.triggering_at_delta, datetime.timedelta)

        assert isinstance(self.type, TriggerType)
        assert isinstance(self.condition, TriggerCondition)

    def is_expired(self, ts: datetime.datetime) -> bool:
        """This trigger has expired.

        Expired triggers must not be executed, and must be moved to past triggers.
        """
        if not self.expires_at:
            return False
        return ts > self.expires_at

    def is_executed(self) -> bool:
        """This trigged is already executed."""
        return self.triggered_at is not None

    def is_triggering(
        self,
        market_price: USDollarPrice,
        timestamp: datetime.datetime,
        position_open_time: datetime.datetime,
    ) -> bool:
        """Is the given price triggering a tride."""

        match self.condition:
            case TriggerCondition.cross_above:
                if market_price >= self.price:
                    return True
            case TriggerCondition.cross_below:
                if market_price <= self.price:
                    return True
            case TriggerCondition.timed_absolute:
                return timestamp > self.triggering_at
            case TriggerCondition.timed_relative_to_open:
                return timestamp > position_open_time + self.triggering_at_delta
            case _:
                raise NotImplementedError(f"Unknown condition: {self}")

        return False



