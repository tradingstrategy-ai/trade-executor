"""Uniswap v2 live pricing.

Directly asks Uniswap v2 asset price from Uniswap pair contract
and JSON-RPC API.
"""
import logging
import datetime
from decimal import Decimal, ROUND_DOWN
from typing import Optional, Dict

from web3 import Web3

from eth_defi.uniswap_v2.deployment import UniswapV2Deployment
from eth_defi.uniswap_v2.pair import fetch_pair_details
from tradeexecutor.ethereum.uniswap_v2_execution import UniswapV2ExecutionModel
from tradeexecutor.ethereum.uniswap_v2_execution_v0 import UniswapV2ExecutionModelVersion0
from tradeexecutor.ethereum.uniswap_v2_routing import UniswapV2SimpleRoutingModel, route_tokens, get_uniswap_for_pair
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.execution_model import ExecutionModel

from eth_defi.uniswap_v2.fees import estimate_buy_price_decimals, estimate_sell_price_decimals, \
    estimate_buy_received_amount_raw, estimate_sell_received_amount_raw
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.pricing_model import PricingModel, TradePricing
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, translate_trading_pair
from tradingstrategy.pair import PandasPairUniverse


logger = logging.getLogger(__name__)


class UniswapV2LivePricing(PricingModel):
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

    def __init__(self,
                 web3: Web3,
                 pair_universe: PandasPairUniverse,
                 routing_model: UniswapV2SimpleRoutingModel,
                 very_small_amount=Decimal("0.10")):

        assert isinstance(web3, Web3)
        assert isinstance(routing_model, UniswapV2SimpleRoutingModel)
        assert isinstance(pair_universe, PandasPairUniverse)

        self.web3 = web3
        self.pair_universe = pair_universe
        self.very_small_amount = very_small_amount
        self.routing_model = routing_model

        self.trading_fee = routing_model.trading_fee
        if(self.trading_fee is None):
            logger.warning("No trading fee provided, web3-ethereum-defi defaults to trading fee of 30 bps")

        self.uniswap_cache: Dict[TradingPairIdentifier, UniswapV2Deployment] = {}

        assert isinstance(self.very_small_amount, Decimal)

    def get_uniswap(self, target_pair: TradingPairIdentifier) -> UniswapV2Deployment:
        """Helper function to speed up Uniswap v2 deployment resolution."""
        if target_pair not in self.uniswap_cache:
            self.uniswap_cache[target_pair] = get_uniswap_for_pair(self.web3, self.routing_model.factory_router_map, target_pair)
        return self.uniswap_cache[target_pair]

    def get_pair_for_id(self, internal_id: int) -> Optional[TradingPairIdentifier]:
        """Look up a trading pair.

        Useful if a strategy is only dealing with pair integer ids.

        :return:
            None if the price data is not available
        """
        pair = self.pair_universe.get_pair_by_id(internal_id)
        if not pair:
            return None
        return translate_trading_pair(pair)

    def check_supported_quote_token(self, pair: TradingPairIdentifier):
        assert pair.quote.address == self.routing_model.reserve_token_address, f"Quote token {self.routing_model.reserve_token_address} not supported for pair {pair}, pair tokens are {pair.base.address} - {pair.quote.address}"

    def get_sell_price(self,
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

        if(self.trading_fee is not None):
            received_raw = estimate_sell_received_amount_raw(
                uniswap,
                base_addr,
                quote_addr,
                quantity_raw,
                intermediate_token_address=intermediate_addr,
                fee=self.trading_fee
            )
        else:
            received_raw = estimate_sell_received_amount_raw(
                uniswap,
                base_addr,
                quote_addr,
                quantity_raw,
                intermediate_token_address=intermediate_addr,
            )

        if intermediate_pair is not None:
            received = intermediate_pair.quote.convert_to_decimal(received_raw)
        else:
            received = target_pair.quote.convert_to_decimal(received_raw)

        fee = self.get_pair_fee(ts, pair)
        assert fee is not None, f"Uniswap v2 fee data missing: {uniswap}"

        price = float(received / quantity)

        # TODO: Verify calculation
        mid_price = price * (1 + fee)

        assert price <= mid_price, f"Bad pricing: {price}, {mid_price}"

        lp_fee = (mid_price - price) * float(quantity)

        return TradePricing(
            price=price,
            mid_price=mid_price,
            lp_fee=lp_fee,
            pair_fee=fee,
            side=False,
        )

    def get_buy_price(self,
                       ts: datetime.datetime,
                       pair: TradingPairIdentifier,
                       reserve: Optional[Decimal],
                       ) -> TradePricing:
        """Get live price on Uniswap.

        TODO: Fees are incorrectly calculated in the case of multipair routing

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

        uniswap = get_uniswap_for_pair(self.web3, self.routing_model.factory_router_map, target_pair)

        # In three token trades, be careful to use the correct reserve token
        if intermediate_pair is not None:
            reserve_raw = intermediate_pair.quote.convert_to_raw_amount(reserve)
            self.check_supported_quote_token(intermediate_pair)
        else:
            reserve_raw = target_pair.quote.convert_to_raw_amount(reserve)
            self.check_supported_quote_token(pair)

        if self.trading_fee is not None:
            token_raw_received = estimate_buy_received_amount_raw(
                uniswap,
                base_addr,
                quote_addr,
                reserve_raw,
                intermediate_token_address=intermediate_addr,
                fee=self.trading_fee
            )
        else:
            token_raw_received = estimate_buy_received_amount_raw(
                uniswap,
                base_addr,
                quote_addr,
                reserve_raw,
                intermediate_token_address=intermediate_addr
            )
        token_received = target_pair.base.convert_to_decimal(token_raw_received)

        fee = self.get_pair_fee(ts, pair)
        assert fee is not None, f"Uniswap v2 fee data missing: {uniswap}"

        price = float(reserve / token_received)

        lp_fee = float(reserve) * fee

        # TODO: Verify calculation
        mid_price = price * (1 - fee)

        assert price >= mid_price, f"Bad pricing: {price}, {mid_price}"

        return TradePricing(
            price=float(price),
            mid_price=float(mid_price),
            lp_fee=lp_fee,
            pair_fee=fee,
            market_feed_delay=datetime.timedelta(seconds=0),
            side=True,
        )

    def get_mid_price(self,
                      ts: datetime.datetime,
                      pair: TradingPairIdentifier) -> USDollarAmount:
        """Get the mid price from Uniswap pool.

        Gets tricky, because we calculate dollar mid-price, not
        quote token midprice.
        """

        # TODO: Use native Uniswap router functions to get the mid price
        # Here we are using a hack)
        bp = self.get_buy_price(ts, pair, self.very_small_amount)
        sp = self.get_sell_price(ts, pair, self.very_small_amount)
        return (bp.price + sp.price) / 2

    def quantize_base_quantity(self, pair: TradingPairIdentifier, quantity: Decimal, rounding=ROUND_DOWN) -> Decimal:
        """Convert any base token quantity to the native token units by its ERC-20 decimals."""
        assert isinstance(pair, TradingPairIdentifier)
        decimals = pair.base.decimals
        return Decimal(quantity).quantize((Decimal(10) ** Decimal(-decimals)), rounding=ROUND_DOWN)

    def get_pair_fee(self,
                     ts: datetime.datetime,
                     pair: TradingPairIdentifier,
                     ) -> Optional[float]:
        """Uniswap v2 compatibles have fixed fee across the exchange."""
        return self.routing_model.get_default_trading_fee()


def uniswap_v2_live_pricing_factory(
        execution_model: ExecutionModel,
        universe: TradingStrategyUniverse,
        routing_model: UniswapV2SimpleRoutingModel) -> UniswapV2LivePricing:

    assert isinstance(universe, TradingStrategyUniverse)
    assert isinstance(execution_model, (UniswapV2ExecutionModelVersion0, UniswapV2ExecutionModel)), f"Execution model not compatible with this execution model. Received {execution_model}"
    assert isinstance(routing_model, UniswapV2SimpleRoutingModel), f"This pricing method only works with Uniswap routing model, we received {routing_model}"

    web3 = execution_model.web3
    return UniswapV2LivePricing(
        web3,
        universe.universe.pairs,
        routing_model)

