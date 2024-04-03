"""Pricing model that multiplexes requests to different protocols."""

import datetime
from _decimal import Decimal
from typing import Dict, Optional

from tradeexecutor.strategy.generic.pair_configurator import PairConfigurator

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.types import USDollarPrice, Percent
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trade_pricing import TradePricing


class GenericPricing(PricingModel):
    """Get a price for the asset from multiple protocols.

    - Each protocol has its own pricing model instance,
      e.g. :py:class:`tradeexecutor.ethereum.uniswap_v3.uniswap_v3_live_pricing.UniswapV3LivePricing`

    - Map a trading pair to an underlying protocol

    - Ask the protocol-specific pricing model the trade price
    """


    def __init__(
            self,
            pair_configurator: PairConfigurator,
    ):
        self.pair_configurator = pair_configurator

    def route(self, pair: TradingPairIdentifier) -> PricingModel:
        return self.pair_configurator.get_pricing(pair)

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

    def set_trading_fee_override(
            self,
            trading_fee_override: Percent | None
    ):
        # TODO: Finish API description and such
        for config in self.pair_configurator.configs.values():
            config.pricing_model.set_trading_fee_override(trading_fee_override)
