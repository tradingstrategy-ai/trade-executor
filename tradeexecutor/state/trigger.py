"""Trigger order structure.

- Simulate market limit orders and such

- Stop loss and take profit are hardcoded into :py:class:`tradeexecutor.state.trade.TradeExecution`
  and not currently part of :py:class:`Trigger` data structure

"""
import datetime
import enum

from tradingstrategy.types import USDollarAmount


class TriggerType(enum.Enum):

    cross_above = "cross_above"

    cross_below = "cross_below"


class Trigger:
    """Trigger data for market orders.

    - Trigger orders can be created to make trades to happen outside the strategy decision cycle

    - Market limit order is the most famous trigger order from the TradFi markets,
      used to enter into breakout positions

    - Triggers are executed in the s

    - Nested structure inside :py:class:`tradeexecutor.state.trade.TradeExecution`
    """

    #: Do we trigger when price crossed above or below
    type: TriggerType

    #: When to take action
    price_level: USDollarAmount | None

    #: After expiration, this trade execution is removed from the hanging queue
    expiration: datetime.datetime | None

    #: When the trigger happened
    triggered_at: datetime.datetime | None = None

    def __post_init__(self):
        if self.expiration is not None:
            assert isinstance(self.expiration, datetime.datetime)

        if self.price_level is not None:
            assert type(self.price_level) == float

    def is_expired(self, ts: datetime.datetime) -> bool:
        return ts > self.expiration


