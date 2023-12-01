"""Value model based on their selling price on generic routing"""
import datetime

from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing
from tradeexecutor.strategy.valuation import ValuationModel


class GenericValuation(ValuationModel):
    """Position is valued depending on its type.

    - Spot: What is the current spot close price,
      ask directly from the chain

    - Leveraged position: Use the refreshed loan
      value from the last `sync_interests()`,
      but don't ask anything from the chain here
    """

    def __init__(self, pricing_model: GenericPricing):
        assert isinstance(pricing_model, GenericPricing)
        self.pricing_model = pricing_model

    def __call__(
            self,
            ts: datetime.datetime,
            position: TradingPosition,
    ) -> tuple[datetime.datetime, USDollarAmount]:

        pair = position.pair

        if pair.is_leverage():
            # Get the latest NAV of the loan based position
            loan = position.loan
            nav = loan.get_net_asset_value(include_interest=True)
            ts = loan.collateral.last_pricing_at
            return ts, nav
        elif pair.is_spot():
            # Cannot do pricing for zero quantity
            quantity = position.get_quantity()
            if quantity == 0:
                return ts, 0.0
            price_structure = self.pricing_model.get_sell_price(ts, pair, quantity)
            return ts, price_structure.price
        else:
            raise NotImplementedError(f"Does not know how to value position {position}")


def generic_valuation_factory(pricing_model):
    return GenericValuation(pricing_model)