"""Value model based on their selling price on generic routing"""
import datetime
from typing import Dict

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.state.valuation import ValuationUpdate
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing
from tradeexecutor.strategy.generic.pair_configurator import PairConfigurator
from tradeexecutor.strategy.generic.routing_function import RoutingFunction, default_route_chooser, UnroutableTrade
from tradeexecutor.strategy.valuation import ValuationModel
from tradingstrategy.pair import PandasPairUniverse


class GenericValuation(ValuationModel):
    """Position is valued depending on its type.

    - Spot: What is the current spot close price,
      ask directly from the chain

    - Leveraged position: Use the refreshed loan
      value from the last `sync_interests()`,
      but don't ask anything from the chain here
    """

    def __init__(
            self,
            pair_configurator: PairConfigurator,
    ):
        self.pair_configurator = pair_configurator

    def __call__(
            self,
            ts: datetime.datetime,
            position: TradingPosition,
    ) -> ValuationUpdate:
        config = self.pair_configurator.get_pair_config(position.pair)
        return config.valuation_model(ts, position)


def generic_valuation_factory(pricing_model):
    return GenericValuation(pricing_model)