"""Valuation models for the portfolio.

Valuation models estimate the value of the portfolio.

This is important for

- Investors understanding if they are profit or loss

- Accounting (taxes)

For the simplest case, we take all open positions and estimate their sell
value at the open market.
"""
import datetime
from typing import Protocol, Tuple

from tradingstrategy.types import USDollarAmount

from tradeexecutor.state.position import TradingPosition
from tradeexecutor.strategy.pricing_model import PricingModel



class ValuationModel(Protocol):
    """Revalue a current position.

    TODO: See if this should be moved inside state module, as it is referred by state.revalue_positions.
    """

    def __call__(self,
                 ts: datetime.datetime,
                 position: TradingPosition) -> Tuple[datetime.datetime, USDollarAmount]:
        """

        :param ts:
            When to revalue. Used in backesting. Live strategies may ignore.
        :param position:
            Open position
        :return:
            (revaluation date, price) tuple.
            Note that revaluation date may differ from the wantead timestamp if
            there is no data available.

        """
        assert isinstance(ts, datetime.datetime)
        pair = position.pair

        assert position.is_long(), "Short not supported"

        quantity = position.get_quantity()
        # Cannot do pricing for zero quantity
        if quantity == 0:
            return ts, 0.0

        price = self.pricing_model.get_sell_price(ts, pair, position)

        return ts, price


class ValuationModelFactory(Protocol):
    """Creates a valuation method.

    - Valuation method is recreated for each cycle

    - Valuation method takes `PricingModel` as an input

    - Called after the pricing model has been established for the cycle
    """

    def __call__(self, pricing_model: PricingModel) -> ValuationModel:
        pass