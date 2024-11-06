"""Asset pricing model for Uniswap V2 and V3 like exchanges."""

import abc
import datetime
import functools
import logging
from decimal import Decimal, ROUND_DOWN
from typing import Callable, Optional
from web3 import Web3

from tradeexecutor.ethereum.tvl import fetch_uni_v2_v3_quote_token_tvl
from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier
from tradeexecutor.state.types import USDollarAmount, TokenAmount
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.routing import RoutingModel
from tradeexecutor.strategy.universe_model import StrategyExecutionUniverse
from tradeexecutor.strategy.trading_strategy_universe import translate_trading_pair
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trade_pricing import TradePricing
from tradeexecutor.ethereum.routing_model import EthereumRoutingModel

from eth_defi.uniswap_v2.deployment import UniswapV2Deployment
from eth_defi.uniswap_v3.deployment import UniswapV3Deployment

from tradingstrategy.pair import PandasPairUniverse


logger = logging.getLogger(__name__)


deployment_types = (UniswapV2Deployment | UniswapV3Deployment)


#: TODO: No good data yet for the value used here
#:
LP_FEE_VALIDATION_EPSILON = 0.001


class EthereumPricingModel(PricingModel):
    """Get a price for the asset.

    Needed for various aspects

    - Revaluate portfolio positiosn

    - Estimate buy/sell price for the live trading so we can calculate slippage

    - Get the historical price in backtesting

    Timestamp is passed to the pricing method. However we expect it only be honoured during
    the backtesting - live execution may always use the latest price.

    .. note ::

        For example, in futures markets there could be different fees
        on buy/sell transctions.

    Used by UniswapV2LivePricing and UniswapV3LivePricing
    """

    def __init__(
        self,
        web3: Web3,
        pair_universe: PandasPairUniverse,
        routing_model: EthereumRoutingModel,
        very_small_amount: Decimal,
        epsilon: Optional[float] = LP_FEE_VALIDATION_EPSILON,
    ):

        assert isinstance(web3, Web3)
        assert isinstance(pair_universe, PandasPairUniverse), f"Expected PandasPairUniverse, got {pair_universe.__class__}"

        self.web3 = web3
        self.pair_universe = pair_universe
        self.very_small_amount = very_small_amount
        self.routing_model = routing_model
        self.epsilon = epsilon

        assert isinstance(self.very_small_amount, Decimal)

    @functools.lru_cache(maxsize=4)
    def _find_exchange_rate_usd_pair(
        self,
        token: AssetIdentifier,
    ) -> TradingPairIdentifier:
        """Find a trading pair we can use to convert quote token to USD."""
        for pair in self.pair_universe.iterate_pairs():
            if pair.base_token_address == token.address and pair.quote_token_symbol in ("USDT", "USDC"):
                return translate_trading_pair(pair)

        raise RuntimeError(f"Pair universe does not contain USDT/USDC pair for token: {token}")

    def get_pair_for_id(self, internal_id: int) -> Optional[TradingPairIdentifier]:
        """Look up a trading pair.

        Useful if a strategy is only dealing with pair integer ids.

        :return:
            None if the price data is not available
        """
        pair = self.pair_universe.get_pair_by_id(internal_id)
        return translate_trading_pair(pair) if pair else None
    
    def check_supported_quote_token(self, pair: TradingPairIdentifier):
        assert pair.quote.address == self.routing_model.reserve_token_address, f"Quote token {self.routing_model.reserve_token_address} not supported for pair {pair}, pair tokens are {pair.base.address} - {pair.quote.address}"
        
    def get_mid_price(self,
                      ts: datetime.datetime,
                      pair: TradingPairIdentifier) -> USDollarAmount:
        """Get the mid price from Uniswap pool.

        Gets tricky, because we calculate dollar mid-price, not
        quote token midprice.
        
        Mid price is an non-trddeable price between the best ask
        and the best pid.

        :param ts:
            Timestamp. Ignored for live pricing models.

        :param pair:
            Which trading pair price we query.

        :return:
            The mid price for the pair at a timestamp.
        """

        # TODO: Use native Uniswap router functions to get the mid price
        # Here we are using a hack)
        bp = self.get_buy_price(ts, pair, self.very_small_amount)
        sp = self.get_sell_price(ts, pair, self.very_small_amount)
        return (bp.price + sp.price) / 2

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
        return pair.fee
    
    def validate_mid_price_for_sell(self, lp_fee, mid_price, price, quantity):
        """Validate the mid price calculation for a sell trade.

        Should basically have:
            lp_fee = (mid_price - price)/mid_price * float(quantity)
        
        :param lp_fee:
            The fee that is paid to the LPs.
        
        :param mid_price:
            The mid price of the pair.

        :param price:
            The price of the trade.

        :param quantity:
            The quantity of the trade.
        """
        raise NotImplementedError("Cannot use mid-price here")

        #value = lp_fee - (mid_price - price)/mid_price * float(quantity)
        #assert abs(value) < self.epsilon, f"Bad lp fee calculation: {lp_fee}, {mid_price}, {price}, {quantity}.\n" \
        #                             f"Value {value} < epsilon {self.epsilon}\n"

    def validate_mid_price_for_buy(self, lp_fee, price, mid_price, reserve):
        """Validate the mid price calculation for a buy trade.

        Should basically have:
            lp_fee = (price - mid_price)/price * float(reserve)
        
        :param lp_fee:
            The fee that is paid to the LPs.
        
        :param price:
            The price of the trade.

        :param mid_price:
            The mid price of the pair.

        :param reserve:
            The reserve of the trade.
        """
        raise NotImplementedError("Cannot use mid-price here")
        #assert lp_fee - (price - mid_price)/price * float(reserve) < self.epsilon, f"Bad lp fee calculation: {lp_fee}, {mid_price}, {price}, {reserve}"
    
    @abc.abstractmethod
    def get_uniswap(self, target_pair: TradingPairIdentifier) -> deployment_types:
        """Helper function to speed up Uniswap v2 or v3 deployment resolution."""
    
    @abc.abstractmethod
    def get_sell_price(self,
                       ts: datetime.datetime,
                       pair: TradingPairIdentifier,
                       quantity: Optional[Decimal]) -> TradePricing:
        """Get the sell price for an asset.

        :param ts:
            When to get the price.
            Used in backtesting.
            Live models may ignore.

        :param pair:
            Trading pair we are intereted in

        :param quantity:
            If the sel quantity is known, get the price with price impact.

        :return:
            Price structure for the trade.
        """

    @abc.abstractmethod
    def get_buy_price(self,
                      ts: datetime.datetime,
                      pair: TradingPairIdentifier,
                      reserve: Optional[Decimal]
                      ) -> TradePricing:
        """Get the sell price for an asset.

        :param ts:
            When to get the price.
            Used in backtesting.
            Live models may ignore.

        :param pair:
            Trading pair we are intereted in

        :param reserve:
            If the buy token quantity quantity is known,
            get the buy price with price impact.

        :return:
            Price structure for the trade.
        """

    def get_usd_tvl(
        self,
        timestamp: datetime.datetime | None,
        pair: TradingPairIdentifier
    ) -> USDollarAmount:
        """Get TVL in a pool.

        - Read directly from Uniswap v2/v3 pool over JSON-RPC

        - Convert to USD using some pair in our pair universe that provides /WETH exchange rate

        :param timestamp:
            Ignore, always get the latest.
        """

        quote_token_tvl = self.get_quote_token_tvl(timestamp, pair)

        # No exchange rate needed
        if pair.quote.is_stablecoin():
            return float(quote_token_tvl)

        # Find exchange rate pool
        exchange_rate_pair = self._find_exchange_rate_usd_pair(pair.quote)

        # Get price at the exchange rate pool
        exchange_rate_price_data = self.get_buy_price(
            timestamp,
            exchange_rate_pair,
            Decimal(1)
        )

        mid_price = exchange_rate_price_data.mid_price
        logger.info("TVL exchange rate pair is %s, and rate is %s", pair, mid_price)
        return float(mid_price) * float(quote_token_tvl)

    def get_quote_token_tvl(
        self,
        timestamp: datetime.datetime | None,
        pair: TradingPairIdentifier
    ) -> TokenAmount:
        """Get TVL in a pool.

        - Read directly from Uniswap v2/v3 pool over JSON-RPC

        :param timestamp:
            Ignore, always get the latest.
        """
        logger.info("Fetching quote token TVL for %s", pair)
        return fetch_uni_v2_v3_quote_token_tvl(
            self.web3,
            pair,
        )

#: This factory creates a new pricing model for each trade cycle.
#: Pricing model depends on the trading universe that may change for each strategy tick,
#: as new trading pairs appear.
#: Thus, we need to reconstruct pricing model as the start of the each tick.
#:
#: TODO: Convert to protocol
#:
PricingModelFactory = Callable[[ExecutionModel, StrategyExecutionUniverse, RoutingModel], PricingModel]

