"""Trigger order structure.

- Simulate market limit orders and such

- Stop loss and take profit are hardcoded into :py:class:`tradeexecutor.state.trade.TradeExecution`
  and not currently part of :py:class:`Trigger` data structure

"""
import datetime
import enum
from dataclasses import dataclass

from tradingstrategy.types import USDollarAmount, Percent


class TriggerType(enum.Enum):
    market_limit = "market_limit"

    #: Execute take profit and partially close position
    take_profit_partial = "take_profit_partial"

    #: Execute take profit and close position fully
    take_profit_full = "take_profit_full"


class TriggerCondition(enum.Enum):
    cross_above = "cross_above"
    cross_below = "cross_below"



@dataclass(frozen=False, slots=True)
class Trigger:
    """Trigger data for market orders.

    Triggers can be on both on

    - Unopened positions, see :py:attr:`tradeeexecutor.state.portfolio.Portfolio.pending_positions`
      where the trigger is on the trade that will open the position (market limit order case)

    - Opened positions, see :py:attr:`tradeeexecutor.state.portfolio.Portfolio.pending_positions`
      where the trigger order is on the increase/reduce position (partial take profit case)

    - Trigger orders can be created to make trades to happen outside the strategy decision cycle

    - Market limit order is the most famous trigger order from the TradFi markets,
      used to enter into breakout positions

    - Triggers are executed in the s

    - Nested structure inside :py:class:`tradeexecutor.state.trade.TradeExecution`

    """

    #: Do we trigger when price crossed above or below
    type: TriggerType

    #: When this trigger is executed
    #:
    condition: TriggerCondition

    #: When to take action
    price: USDollarAmount | None

    #: After expiration, this trade execution is removed from the hanging queue
    expires_at: datetime.datetime | None

    #: When the trigger happened
    #:
    #:
    triggered_at: datetime.datetime | None = None

    #: How much of the position to execute.
    #:
    #: Only with the partial take profits
    #:
    relative_size: Percent | None = None

    def __repr__(self):
        return f"<Trigger {self.type.value} {self.condition.value} at {self.price} expires at {self.expires_at} with size {self.relative_size}>"

    def __post_init__(self):
        if self.expiration is not None:
            assert isinstance(self.expiration, datetime.datetime)

        if self.price_level is not None:
            assert type(self.price_level) == float

        assert isinstance(self.type, TriggerType)

        if self.relative_size:
            assert self.type == TriggerType.take_profit_partial, "Size supported only for take_profit_partial"

    def is_expired(self, ts: datetime.datetime) -> bool:
        """This trigger has expired.

        Expired triggers must not be executed, and must be moved to past triggers.
        """
        return ts > self.expiration

    def is_executed(self) -> bool:
        """This trigged is already executed."""
        return self.triggered_at is not None


