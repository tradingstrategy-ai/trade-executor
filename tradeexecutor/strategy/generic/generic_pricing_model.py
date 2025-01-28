"""Pricing model that multiplexes requests to different protocols."""

import logging
import datetime
from _decimal import Decimal
from typing import Optional

import pandas as pd

from tradeexecutor.ethereum.ethereum_protocol_adapters import EthereumPairConfigurator
from tradeexecutor.strategy.generic.pair_configurator import PairConfigurator

from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier
from tradeexecutor.state.types import USDollarPrice, Percent, USDollarAmount, TokenAmount
from tradeexecutor.strategy.pricing_model import PricingModel, PricingModelFactory
from tradeexecutor.strategy.trade_pricing import TradePricing


logger = logging.getLogger(__name__)


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
        data_delay_tolerance=pd.Timedelta("2d"),
        exchange_rate_pairs: dict[AssetIdentifier, TradingPairIdentifier] | None = None,
    ):
        self.pair_configurator = pair_configurator
        self.data_delay_tolerance = data_delay_tolerance
        self.exchange_rate_pairs = exchange_rate_pairs
        self.trading_fee_override = None

    def get_exchange_rate(
        self,
        timestamp: datetime.datetime,
        quote_asset: AssetIdentifier
    ) -> USDollarPrice:
        """Get WETH exchange rate.

        - Always use a predefined pool

        :param quote_asset:
            Usually WETH

        :return:
            WETH price in dollars
        """

        if quote_asset.is_stablecoin():
            return 1.0
        assert self.exchange_rate_pairs is not None, "GenericPricing.exchange_rate_pairs not configured"

        routed_pair = self.exchange_rate_pairs.get(quote_asset)

        if not routed_pair:
            raise KeyError(f"No exchange rate route for {quote_asset}")

        try:
            return self.get_mid_price(timestamp, routed_pair)
        except Exception as e:
            raise RuntimeError(f"Could not get {quote_asset.token_symbol} exchange rate using routed pair {routed_pair}") from e

    def route(self, pair: TradingPairIdentifier) -> PricingModel:
        return self.pair_configurator.get_pricing(pair)

    def get_sell_price(self,
                       ts: datetime.datetime,
                       pair: TradingPairIdentifier,
                       quantity: Optional[Decimal]) -> TradePricing:
        route = self.route(pair)
        return route.get_sell_price(ts, pair, quantity)

    def get_buy_price(
        self,
        ts: datetime.datetime,
        pair: TradingPairIdentifier,
        reserve: Optional[Decimal]
    ) -> TradePricing:
        route = self.route(pair)
        return route.get_buy_price(ts, pair, reserve)

    def get_mid_price(
        self,
        ts: datetime.datetime,
        pair: TradingPairIdentifier
    ) -> USDollarPrice:
        route = self.route(pair)
        mid_price = route.get_mid_price(ts, pair)
        return mid_price

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
        self.trading_fee_override = trading_fee_override

    def get_usd_tvl(
        self,
        timestamp: datetime.datetime | None,
        pair: TradingPairIdentifier
    ) -> USDollarAmount:
        # TODO: Temporary debug
        route = self.route(pair)
        logger.info("get_usd_tvl(): routing %s to %s", pair, route)
        return route.get_usd_tvl(timestamp, pair)

    def get_quote_token_tvl(
        self,
        timestamp: datetime.datetime | None,
        pair: TradingPairIdentifier
    ) -> TokenAmount:
        route = self.route(pair)
        return route.get_quote_token_tvl(timestamp, pair)


class EthereumGenericPricingFactory(PricingModelFactory):
    """Create Ethereum pricing routing tables.

    - Support Uniswap v2 and v3 routing in a mixed environment
    """
    def __init__(self, web3):
        self.web3 = web3

    def __call__(
        self,
        execution_model,
        universe,
        routing_model,
    ):
        pair_configurator = EthereumPairConfigurator(self.web3, universe)
        return GenericPricing(pair_configurator)

