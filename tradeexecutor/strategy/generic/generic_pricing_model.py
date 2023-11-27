"""Pricing model that multiplexes requests to different protocols."""

import datetime
from _decimal import Decimal
from typing import Dict, Optional

from tradingstrategy.pair import PandasPairUniverse

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.types import USDollarPrice
from tradeexecutor.strategy.generic.routing_function import RoutingFunction, default_route_chooser, UnroutableTrade
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trade_pricing import TradePricing


class GenericPricingModel(PricingModel):
    """Get a price for the asset from multiple protocols.

    - Each protocol has its own pricing model instance,
      e.g. :py:class:`tradeexecutor.ethereum.uniswap_v3.uniswap_v3_live_pricing.UniswapV3LivePricing`

    - Map a trading pair to an underlying protocol

    - Ask the protocol-specific pricing model the trade price
    """

    def __init__(
            self,
            pair_universe: PandasPairUniverse,
            routes: Dict[str, PricingModel],
            routing_function: RoutingFunction = default_route_chooser,
    ):
        self.pair_universe = pair_universe
        self.routes = routes
        self.routing_function = routing_function

    def route(self, pair: TradingPairIdentifier) -> PricingModel:
        router_name = self.routing_function(self.pair_universe, pair)
        if router_name is None:
            raise UnroutableTrade(
                f"Cannot route: {pair}\n"
                f"Using routing function: {self.routing_function}"
                f"Available routes: {list(self.routes.keys())}"
            )

        route = self.routes.get(router_name)
        if route is None:
            raise UnroutableTrade(
                f"Router not available: {pair}\n"
                f"Trade routing function give us a route: {router_name}, but it is not configured\n"
                f"Available routes: {list(self.routes.keys())}"
            )

        return route

    def get_sell_price(self,
                       ts: datetime.datetime,
                       pair: TradingPairIdentifier,
                       quantity: Optional[Decimal]) -> TradePricing:
        route = self.route(pair)
        return route.get_sell_price(ts, pair, quantity)

    def get_buy_price(self,
                      ts: datetime.datetime,
                      pair: TradingPairIdentifier,
                      reserve: Optional[Decimal]
                      ) -> TradePricing:
        route = self.route(pair)
        return route.get_buy_price(ts, pair, reserve)

    def get_mid_price(self,
                      ts: datetime.datetime,
                      pair: TradingPairIdentifier) -> USDollarPrice:
        route = self.route(pair)
        return route.get_mid_price(ts, pair)

    def get_pair_fee(self,
                     ts: datetime.datetime,
                     pair: TradingPairIdentifier,
                     ) -> Optional[float]:
        route = self.route(pair)
        return route.get_pair_fee(ts, pair)