"""This GenericPricingModel class follows the PricingModel interface, and is able to choose between different pricing models based on the trading pair. This is useful for when we want to use different pricing models for different pairs, such as Uniswap v2 for ETH/DAI and Uniswap v3 for ETH/USDC.

TODO: check if this needs to be moved to a different file location (yep)
TODO: check if docstrings need to be copied from the PricingModel interface
"""
import datetime
from decimal import Decimal, ROUND_DOWN
from typing import Optional, List

from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trade_pricing import TradePricing
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.types import USDollarPrice
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_routing import UniswapV2SimpleRoutingModel
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_routing import UniswapV3SimpleRoutingModel
from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel, BacktestRoutingIgnoredModel

from tradingstrategy.pair import DEXPair


# TODO remove hardcoding. Also used in eth_pricing_model.py
LP_FEE_VALIDATION_EPSILON = 0.001

class GenericPricingModel(PricingModel):
    
    def __init__(
        self,
        pricing_models: list[PricingModel],
    ):
        assert all(isinstance(model, PricingModel) for model in pricing_models), "pricing_models must be a list of PricingModel objects"
        
        self.pricing_models = pricing_models
        
    
    def get_buy_price(self, ts: datetime.datetime, pair: TradingPairIdentifier, reserve: Optional[Decimal]) -> TradePricing:
        """Get live price for a buy on relevant exchange.
        
        :param ts:
            Timestamp for which we want to get the price
        
        :param reserve:
            The buy size in quote token e.g. in dollars
            
        :return:
            Price for one reserve unit e.g. a dollar
        """
        pricing_model = get_pricing_model_for_pair(pair, self.pricing_models)
        return pricing_model.get_buy_price(ts, pair, reserve)
    
    def get_sell_price(self, ts: datetime.datetime, pair: TradingPairIdentifier, quantity: Decimal) -> TradePricing:
        """Get live price for a sell on relevant exchange.
        
        :param ts:
            Timestamp for which we want to get the price
        
        :param reserve:
            The buy size in quote token e.g. in dollars
            
        :return:
            Price for one reserve unit e.g. a dollar
        """
        pricing_model = get_pricing_model_for_pair(pair, self.pricing_models)
        return pricing_model.get_sell_price(ts, pair, quantity)
    
    def get_mid_price(self,
                      ts: datetime.datetime,
                      pair: TradingPairIdentifier) -> USDollarPrice:
        """Get the mid-price for an asset.

        Mid price is an non-trddeable price between the best ask
        and the best pid.

        :param ts:
            Timestamp. Ignored for live pricing models.

        :param pair:
            Which trading pair price we query.

        :return:
            The mid price for the pair at a timestamp.
        """
        pricing_model = get_pricing_model_for_pair(pair, self.pricing_models)
        return pricing_model.get_mid_price(ts, pair)
    
    def quantize_base_quantity(self, pair: TradingPairIdentifier, quantity: Decimal, rounding=ROUND_DOWN) -> Decimal:
        """Convert any base token quantity to the native token units by its ERC-20 decimals."""
        pricing_model = get_pricing_model_for_pair(pair, self.pricing_models)
        return pricing_model.quantize_base_quantity(pair, quantity, rounding)

    def get_pair_fee(self,
                     ts: datetime.datetime,
                     pair: TradingPairIdentifier,
                     ) -> Optional[float]:
        """Estimate the trading/LP fees for a trading pair.

        This information can come either from the exchange itself (Uni v2 compatibles),
        or from the trading pair (Uni v3).

        The return value is used to fill the
        fee values for any newly opened trades.

        :param ts:
            Timestamp of the trade. Note that currently
            fees do not vary over time, but might
            do so in the future.

        :param pair:
            Trading pair for which we want to have the fee.

            Can be left empty if the underlying exchange is always
            offering the same fee.

        :return:
            The estimated trading fee, expressed as %.

            Returns None if the fee information is not available.
            This can be different from zero fees.
        """
        pricing_model = get_pricing_model_for_pair(pair, self.pricing_models)
        return pricing_model.get_pair_fee(ts, pair)
    

def get_pricing_model_for_pair(pair: TradingPairIdentifier | DEXPair, pricing_models: List[PricingModel], set_routing_hint: bool = True) -> PricingModel:
    """Get the pricing model for a pair.
    
    :param pair:
        Trading pair for which we want to have the pricing model


    :param pricing_models:
        List of pricing models to choose from
        
    :set_routing_hint:
        Whether to set the routing hint for the pricing model. This is useful when we have multiple pricing models that could apply to the same pair, and we want to make sure that the correct pricing model is used for the pair.


    :return:
        Pricing model for the pair
    """
    if len(pricing_models) == 1:
        return pricing_models[0]


    locked = False
    rm = None
    routing_model = None
    final_pricing_model = None

    for pricing_model in pricing_models:

        rm = pricing_model.routing_model

        if hasattr(rm, "factory_router_map"):  # uniswap v2 like
            keys = list(rm.factory_router_map.keys())
            assert len(keys) == 1, "Only one factory router map supported for now"
            factory_address = keys[0]
        elif hasattr(rm, "address_map"):  # uniswap v3 like
            factory_address = rm.address_map["factory"] 
        else:
            raise NotImplementedError("Routing model not supported")

        if factory_address.lower() == pair.exchange_address.lower() and rm.chain_id == pair.chain_id:
            if locked == True:
                raise LookupError("Multiple routing models for same exchange (on same chain) not supported")

            routing_model = rm
            final_pricing_model = pricing_model
            locked = True

    if not routing_model:
        raise NotImplementedError("Unable to find routing_model for pair, make sure to add correct routing models for the pairs that you want to trade")

    if not isinstance(routing_model, (UniswapV2SimpleRoutingModel, UniswapV3SimpleRoutingModel, BacktestRoutingModel, BacktestRoutingIgnoredModel)):
        raise NotImplementedError("Routing model not supported")
    
    # Needed in tradeexecutor/state/portfolio.py::choose_valudation_method_and_revalue_position
    if set_routing_hint:
        pair.routing_hint = routing_model.routing_hint  

    assert final_pricing_model is not None, "Unable to find pricing model for pair"

    return final_pricing_model