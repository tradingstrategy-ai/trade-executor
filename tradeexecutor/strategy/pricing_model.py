"""Asset pricing model."""

import abc
import datetime
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Callable, Optional
from web3 import Web3

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.types import USDollarAmount, BPS, USDollarPrice
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.routing import RoutingModel
from tradeexecutor.strategy.universe_model import StrategyExecutionUniverse
from tradeexecutor.strategy.trading_strategy_universe import translate_trading_pair
from tradeexecutor.ethereum.uniswap_v2_routing import UniswapV2SimpleRoutingModel, UniswapV2Deployment
from tradeexecutor.ethereum.uniswap_v3_routing import UniswapV3SimpleRoutingModel, UniswapV3Deployment

from tradingstrategy.pair import PandasPairUniverse

@dataclass(slots=True, frozen=True)
class TradePricing:
    """Describe price results for a price query.

    - Each price result is tied to quantiy/amount

    - Each price result gets a split that describes liquidity provider fees

    A helper class to deal with problems of accounting and estimation of prices on Uniswap like exchange.
    """

    #: The price we expect this transaction to clear.
    #:
    #: This price has LP fees already deducted away from it.
    #: It may or may not include price impact if liquidity data was available
    #: for the pricing model.
    price: USDollarPrice

    #: The "fair" market price during the transaction.
    #:
    #: This is the mid price - no LP fees, price impact,
    #: etc. included.
    mid_price: USDollarPrice

    #: How much liquidity provider fees we are going to pay on this trade.
    #:
    #: Set to None if data is not available.
    lp_fee: Optional[USDollarAmount] = None

    #: What was the LP fee % used as the base of the calculations.
    #:
    pair_fee: Optional[BPS] = None

    #: How old price data we used for this estimate
    #:
    market_feed_delay: Optional[datetime.timedelta] = None

    #: Is this buy or sell trade.
    #:
    #: True for buy.
    #: False for sell.
    #: None for Unknown.
    side: Optional[bool] = None

    def __repr__(self):
        fee = self.pair_fee or 0
        return f"<TradePricing:{self.price} mid:{self.mid_price} fee:{fee:.4f}%>"

    def __post_init__(self):
        """Validate parameters.

        Make sure we don't slip in e.g. NumPy types.
        """
        assert type(self.price) == float
        assert type(self.mid_price) == float
        if self.lp_fee is not None:
            assert type(self.lp_fee) == float
        if self.pair_fee is not None:
            assert type(self.pair_fee) == float, f"Got fee: {self.pair_fee} {type(self.pair_fee)} "
        if self.market_feed_delay is not None:
            assert isinstance(self.market_feed_delay, datetime.timedelta)

        # Do safety checks for the price calculation
        if self.side is not None:
            if self.side:
                assert self.price >= self.mid_price, f"Got bad buy pricing: {self.price} > {self.mid_price}"
            if not self.side:
                assert self.price <= self.mid_price, f"Got bad sell pricing: {self.price} < {self.mid_price}"

routing_types = UniswapV2SimpleRoutingModel | UniswapV3SimpleRoutingModel

class PricingModel(abc.ABC):
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

    """

    def __init__(self,
                 web3: Web3,
                 pair_universe: PandasPairUniverse,
                 routing_model: routing_types,
                 very_small_amount: Decimal):

        assert isinstance(web3, Web3)
        assert isinstance(pair_universe, PandasPairUniverse)

        self.web3 = web3
        self.pair_universe = pair_universe
        self.very_small_amount = very_small_amount
        self.routing_model = routing_model

        assert isinstance(self.very_small_amount, Decimal)
    
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

    def quantize_base_quantity(self, pair: TradingPairIdentifier, quantity: Decimal, rounding=ROUND_DOWN) -> Decimal:
        """Convert any base token quantity to the native token units by its ERC-20 decimals."""
        assert isinstance(pair, TradingPairIdentifier)
        decimals = pair.base.decimals
        return Decimal(quantity).quantize((Decimal(10) ** Decimal(-decimals)), rounding=rounding)
    
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
    
    @abc.abstractmethod
    def get_uniswap(self, target_pair: TradingPairIdentifier) -> UniswapV3Deployment:
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
        

#: This factory creates a new pricing model for each trade cycle.
#: Pricing model depends on the trading universe that may change for each strategy tick,
#: as new trading pairs appear.
#: Thus, we need to reconstruct pricing model as the start of the each tick.
PricingModelFactory = Callable[[ExecutionModel, StrategyExecutionUniverse, RoutingModel], PricingModel]

