"""Uniswap v2 live pricing.

Directly asks Uniswap v2 asset price from Uniswap pair contract
and JSON-RPC API.
"""
import logging
import datetime
from decimal import Decimal
from typing import Optional, Dict

from IPython.testing.plugin.pytest_ipdoctest import ipdoctest_namespace
from web3 import Web3

from eth_defi.uniswap_v2.deployment import UniswapV2Deployment
from eth_defi.uniswap_v2.fees import estimate_buy_received_amount_raw, estimate_sell_received_amount_raw

from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_execution import UniswapV2Execution
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_execution_v0 import UniswapV2ExecutionModelVersion0
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_routing import UniswapV2Routing, route_tokens, get_uniswap_for_pair
from tradeexecutor.ethereum.eth_pricing_model import EthereumPricingModel, LP_FEE_VALIDATION_EPSILON
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.dummy import DummyExecutionModel
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.trade_pricing import TradePricing
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse

from tradingstrategy.pair import PandasPairUniverse


logger = logging.getLogger(__name__)


class UniswapV2LivePricing(EthereumPricingModel):
    """Always pull the latest dollar price for an asset from Uniswap v2 deployment.

    Supports

    - Two-way BUSD -> Cake

    - Three-way trades BUSD -> BNB -> Cake

    ... within a single exchange.

    .. note ::

        If a trade quantity/currency amount is not given uses
        a "default small value" that is 0.1. Depending on the token,
        this value might be too much/too little, as Uniswap
        fixed point math starts to break for very small amounts.
        For example, for USDC trade 10 cents is already quite low.

    More information

    - `About ask and bid <https://www.investopedia.com/terms/b/bid-and-ask.asp>`_:
    """

    def __init__(
        self,
        web3: Web3,
        pair_universe: PandasPairUniverse,
        routing_model: UniswapV2Routing,
        very_small_amount=Decimal("0.10"),
        epsilon: Optional[float] = LP_FEE_VALIDATION_EPSILON,
    ):

        assert isinstance(routing_model, UniswapV2Routing)

        self.uniswap_cache: Dict[TradingPairIdentifier, UniswapV2Deployment] = {}

        super().__init__(
            web3,
            pair_universe,
            routing_model,
            very_small_amount,
            epsilon,
        )

    def get_uniswap(self, target_pair: TradingPairIdentifier) -> UniswapV2Deployment:
        """Helper function to speed up Uniswap v2 deployment resolution."""
        if target_pair not in self.uniswap_cache:
            self.uniswap_cache[target_pair] = get_uniswap_for_pair(self.web3, self.routing_model.factory_router_map, target_pair)
        return self.uniswap_cache[target_pair]

    def get_sell_price(
        self,
        ts: datetime.datetime,
        pair: TradingPairIdentifier,
        quantity: Optional[Decimal],
    ) -> TradePricing:
        """Get live price on Uniswap."""

        if quantity is None:
            quantity = Decimal(self.very_small_amount)

        assert isinstance(quantity, Decimal)

        target_pair, intermediate_pair = self.routing_model.route_pair(self.pair_universe, pair)

        base_addr, quote_addr, intermediate_addr = route_tokens(target_pair, intermediate_pair)

        uniswap = self.get_uniswap(target_pair)

        # In three token trades, be careful to use the correct reserve token
        quantity_raw = target_pair.base.convert_to_raw_amount(quantity)
        
        fee = self.get_pair_fee(ts, pair)
        assert fee is not None, f"Uniswap v2 fee data missing: {uniswap}"

        bps_fee = int(fee * 10000)

        received_raw = estimate_sell_received_amount_raw(
            uniswap,
            base_addr,
            quote_addr,
            quantity_raw,
            intermediate_token_address=intermediate_addr,
            fee=bps_fee,
        )

        
        if intermediate_pair:
            received = intermediate_pair.quote.convert_to_decimal(received_raw)
            
            fee2 = self.get_pair_fee(ts, intermediate_pair)
            assert fee2 == fee, "Pairs for Uniswap V2 should have same fee"
            fees = [fee, fee2]
            
            total_fee_pct = 1 - (1-fees[0]) * (1-fees[1])
        else:
            received = target_pair.quote.convert_to_decimal(received_raw)
            
            fees = [fee]
            
            total_fee_pct = 1 - (1 - fees[0])

        price = float(received / quantity)
            
        if intermediate_pair:
            mid_price = price / (1 - fee) / (1 - fee)
            
            path = [intermediate_pair, target_pair]
        else:
            mid_price = price / (1 - fee)
            
            path = [target_pair]
            
        
        lp_fee = float(quantity) * total_fee_pct
            
        assert price <= mid_price, f"Bad pricing: {price}, {mid_price}"

        # self.validate_mid_price_for_sell(lp_fee, mid_price, price, quantity)

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
                       ) -> TradePricing:
        """Get live price on Uniswap.
        :param reserve:
            The buy size in quote token e.g. in dollars
        :return:
            Price for one reserve unit e.g. a dollar
        """

        if reserve is None:
            reserve = Decimal(self.very_small_amount)
        else:
            assert isinstance(reserve, Decimal), f"Reserve must be decimal, got {reserve.__class__}: {reserve}"

        target_pair, intermediate_pair = self.routing_model.route_pair(self.pair_universe, pair)

        base_addr, quote_addr, intermediate_addr = route_tokens(target_pair, intermediate_pair)

        try:
            uniswap = get_uniswap_for_pair(self.web3, self.routing_model.factory_router_map, target_pair)
        except Exception as e:
            raise RuntimeError(f"We do not have Uniswap router configured for pair {target_pair} on exchange {target_pair.exchange_address}, factory_router_map is {self.routing_model.factory_router_map}") from e
        
        fee = self.get_pair_fee(ts, pair)
        assert fee is not None, f"Uniswap v2 fee data missing: exchange:{uniswap} pair:{pair}"
        
        # In three token trades, be careful to use the correct reserve token
        if intermediate_pair:
            reserve_raw = intermediate_pair.quote.convert_to_raw_amount(reserve)
            self.check_supported_quote_token(intermediate_pair)
            
        else:
            reserve_raw = target_pair.quote.convert_to_raw_amount(reserve)
            self.check_supported_quote_token(pair)

        bps_fee = int(fee * 10000)

        # Calculate base token received
        token_raw_received = estimate_buy_received_amount_raw(
            uniswap,
            base_addr,
            quote_addr,
            reserve_raw,
            intermediate_token_address=intermediate_addr,
            fee=bps_fee,
        )

        token_received = target_pair.base.convert_to_decimal(token_raw_received)

        price = float(reserve / token_received)
        
        if intermediate_pair:
            fee2 = self.get_pair_fee(ts, intermediate_pair)

            assert fee2 == fee, f"Pairs for Uniswap V2 should have same fee. Intermediate has: {fee2}, expected {fee}. Pair {intermediate_pair}"

            # TODO: Verify calculation
            mid_price = price * (1 - fee) * (1 - fee)
            
            path = [intermediate_pair, target_pair]
            
            fees = [fee, fee2]
            
            total_fee_pct = 1 - (1-fees[0]) * (1-fees[1])
        else:
            mid_price = price * (1 - fee)
            
            path = [target_pair]
            
            fees = [fee]
            
            total_fee_pct = 1 - (1 - fees[0])
            
        # Reserve is not necessarily a dollar amount (quote token doesn't have to be dollars), so we need to calculate
        lp_fee = float(reserve) * total_fee_pct

        assert price >= mid_price, f"Bad pricing: {price}, {mid_price}"

        # self.validate_mid_price_for_buy(lp_fee, price, mid_price, reserve)

        return TradePricing(
            price=float(price),
            mid_price=float(mid_price),
            lp_fee=[lp_fee],
            pair_fee=fees,
            market_feed_delay=datetime.timedelta(seconds=0),
            side=True,
            path=path
        )



def uniswap_v2_live_pricing_factory(
        execution_model: ExecutionModel,
        universe: TradingStrategyUniverse,
        routing_model: UniswapV2Routing) -> UniswapV2LivePricing:

    assert isinstance(universe, TradingStrategyUniverse)
    assert isinstance(execution_model, (UniswapV2ExecutionModelVersion0, UniswapV2Execution, DummyExecutionModel)), f"Execution model not compatible with this execution model. Received {execution_model}"
    assert isinstance(routing_model, UniswapV2Routing), f"This pricing method only works with Uniswap routing model, we received {routing_model}"

    web3 = execution_model.web3
    return UniswapV2LivePricing(
        web3,
        universe.data_universe.pairs,
        routing_model)

