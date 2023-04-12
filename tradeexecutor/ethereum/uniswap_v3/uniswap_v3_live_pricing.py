"""Uniswap v3 live pricing.

Directly asks Uniswap v3 asset price from Uniswap pair contract
and JSON-RPC API.
"""
import logging
import datetime
from decimal import Decimal
from typing import Optional, Dict

from web3 import Web3

from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_execution import UniswapV3ExecutionModel
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_routing import UniswapV3SimpleRoutingModel, route_tokens, get_uniswap_for_pair
from tradeexecutor.ethereum.eth_pricing_model import EthereumPricingModel
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.strategy.trade_pricing import TradePricing
from tradingstrategy.pair import PandasPairUniverse

from eth_defi.uniswap_v3.price import UniswapV3PriceHelper
from eth_defi.uniswap_v3.deployment import UniswapV3Deployment

logger = logging.getLogger(__name__)


class UniswapV3LivePricing(EthereumPricingModel):
    """Always pull the latest dollar price for an asset from Uniswap v3 deployment.

    Supports

    - Two-way BUSD -> Cake

    - Three-way trades BUSD -> BNB -> Cake

    ... within a single, uniswap v3 like exchange.

    .. note ::

        If a trade quantity/currency amount is not given uses
        a "default small value" that is 0.1. Depending on the token,
        this value might be too much/too little, as Uniswap
        fixed point math starts to break for very small amounts.
        For example, for USDC trade 10 cents is already quite low.

    More information

    - `About ask and bid <https://www.investopedia.com/terms/b/bid-and-ask.asp>`_:
    """

    def __init__(self,
                 web3: Web3,
                 pair_universe: PandasPairUniverse,
                 routing_model: UniswapV3SimpleRoutingModel,
                 very_small_amount=Decimal("0.10")):

        assert isinstance(routing_model, UniswapV3SimpleRoutingModel)

        self.uniswap_cache: Dict[TradingPairIdentifier, UniswapV3Deployment] = {}

        super().__init__(
            web3,
            pair_universe,
            routing_model,
            very_small_amount
        )

    def get_pair_fee_multiplier(self, ts, pair):
        """Uniswap V3 pairs get fees in raw format e.g. 3000 instead of 0.3%"""
        return super().get_pair_fee(ts, pair)/1_000_000
    
    def get_uniswap(self, target_pair: TradingPairIdentifier) -> UniswapV3Deployment:
        """Helper function to speed up Uniswap v3 deployment resolution."""
        if target_pair not in self.uniswap_cache:
            self.uniswap_cache[target_pair] = get_uniswap_for_pair(
                self.web3, 
                self.routing_model.address_map, 
                target_pair
            )
        return self.uniswap_cache[target_pair]
    
    def get_price_helper(self, target_pair: TradingPairIdentifier) -> UniswapV3PriceHelper:
        uniswap_v3 = self.get_uniswap(target_pair)
        return UniswapV3PriceHelper(uniswap_v3)

    def get_sell_price(self,
                       ts: datetime.datetime,
                       pair: TradingPairIdentifier,
                       quantity: Optional[Decimal],
                       ) -> USDollarAmount:
        """Get live price on Uniswap."""

        if quantity is None:
            quantity = Decimal(self.very_small_amount)

        assert isinstance(quantity, Decimal)

        target_pair, intermediate_pair = self.routing_model.route_pair(self.pair_universe, pair)

        base_addr, quote_addr, intermediate_addr = route_tokens(target_pair, intermediate_pair)

        # In three token trades, be careful to use the correct reserve token
        quantity_raw = target_pair.base.convert_to_raw_amount(quantity)

        # See eth_defi.uniswap_v2.fees.estimate_sell_received_amount_raw
        if intermediate_pair:
            path = [base_addr, intermediate_addr, quote_addr] 
            fees = [intermediate_pair.fee, target_pair.fee]
            total_fee_pct = 1 - (1-fees[0]) * (1-fees[1])
        else:
            path = [base_addr, quote_addr]
            fees = [target_pair.fee]
            total_fee_pct = 1 - (1-fees[0])
                        
        raw_fees = [int(fee * 1_000_000) for fee in fees]
        
        price_helper = self.get_price_helper(target_pair)
        received_raw = price_helper.get_amount_out(
            amount_in=quantity_raw,
            path=path,
            fees=raw_fees
        ) 

        
        if intermediate_pair:
            received = intermediate_pair.quote.convert_to_decimal(received_raw)
            path = [intermediate_pair, target_pair]
            
        else:
            received = target_pair.quote.convert_to_decimal(received_raw)
            price = float(received / quantity)
            path = [target_pair]
        
        price = float(received / quantity)
        
        if intermediate_pair:
            mid_price = price * (1 + self.get_pair_fee_multiplier(ts, target_pair)) * (1 + self.get_pair_fee_multiplier(ts, intermediate_pair.fee))
        else:
            mid_price = price * (1 + self.get_pair_fee_multiplier(ts, target_pair))
        
        assert price <= mid_price, f"Bad pricing: {price}, {mid_price}"

        lp_fee = float(quantity) * total_fee_pct
        
        #lp_fee = (mid_price - price) * float(quantity)
        
        #assert lp_fee == float(quantity) * total_fee_pct
        
        return TradePricing(
            price=price,
            mid_price=mid_price,
            lp_fee=[lp_fee],
            pair_fee=fees,
            side=False,
            path=path
        )
        

    def get_buy_price(self,
                       ts: datetime.datetime,
                       pair: TradingPairIdentifier,
                       reserve: Optional[Decimal],
                       ) -> USDollarAmount:
        """Get live price on Uniswap.

        :param reserve:
            The buy size in quote token e.g. in dollars

        :return: Price for one reserve unit e.g. a dollar
        """

        if reserve is None:
            reserve = Decimal(self.very_small_amount)
        else:
            assert isinstance(reserve, Decimal), f"Reserve must be decimal, got {reserve.__class__}: {reserve}"

        target_pair, intermediate_pair = self.routing_model.route_pair(self.pair_universe, pair)

        base_addr, quote_addr, intermediate_addr = route_tokens(target_pair, intermediate_pair)

        # In three token trades, be careful to use the correct reserve token
        if intermediate_pair is not None:
            reserve_raw = intermediate_pair.quote.convert_to_raw_amount(reserve)
            self.check_supported_quote_token(intermediate_pair)
            
            path = [quote_addr, intermediate_addr, base_addr]
            
            fees = [intermediate_pair.fee, target_pair.fee]
            
            total_fee_pct = 1 - (1 - fees[0]) * (1-fees[1])
        else:
            reserve_raw = target_pair.quote.convert_to_raw_amount(reserve)
            self.check_supported_quote_token(pair)
            
            path = [quote_addr, base_addr] 
            
            fees = [target_pair.fee]
            
            total_fee_pct = 1 - (1 - fees[0])

        raw_fees = [int(fee * 1_000_000) for fee in fees]

        # See eth_defi.uniswap_v2.fees.estimate_buy_received_amount_raw
        price_helper = self.get_price_helper(target_pair)
        token_raw_received = price_helper.get_amount_out(
            amount_in=reserve_raw,
            path=path,
            fees=raw_fees
        )

        token_received = target_pair.base.convert_to_decimal(token_raw_received)
        
        fee = self.get_pair_fee_multiplier(ts, pair)
        assert fee is not None, "Uniswap v3 fee data missing"

        price = float(reserve / token_received)

        lp_fee = float(reserve) * total_fee_pct

        # TODO: Verify mid_price calculation

        if intermediate_pair:        
            mid_price = price * (1 - self.get_pair_fee_multiplier(ts, intermediate_pair)) * (1 - self.get_pair_fee_multiplier(ts, target_pair))
            
            path = [intermediate_pair, target_pair]
        else:
            mid_price = price * (1 - fee)
            
            path = [target_pair]

        assert price >= mid_price, f"Bad pricing: {price}, {mid_price}"
        
        return TradePricing(
            price=float(price),
            mid_price=float(mid_price),
            lp_fee=[lp_fee],
            pair_fee=fees,
            market_feed_delay=datetime.timedelta(seconds=0),
            side=True,
            path=path
        )
        


def uniswap_v3_live_pricing_factory(
        execution_model: ExecutionModel,
        universe: TradingStrategyUniverse,
        routing_model: UniswapV3SimpleRoutingModel,
        ) -> UniswapV3LivePricing:

    assert isinstance(universe, TradingStrategyUniverse)
    assert isinstance(execution_model, (UniswapV3ExecutionModel)), f"Execution model not compatible with this execution model. Received {execution_model}"
    assert isinstance(routing_model, UniswapV3SimpleRoutingModel), f"This pricing method only works with Uniswap routing model, we received {routing_model}"
    web3 = execution_model.web3
    return UniswapV3LivePricing(
        web3,
        universe.universe.pairs,
        routing_model
    )
