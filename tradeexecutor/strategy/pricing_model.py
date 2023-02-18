"""Asset pricing model."""

import abc
import datetime
from logging import getLogger
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Callable, Optional, List

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.types import USDollarAmount, BPS, USDollarPrice
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.routing import RoutingModel
from tradeexecutor.strategy.universe_model import StrategyExecutionUniverse


logger = getLogger(__name__)


@dataclass(slots=True, frozen=True)
class TradePricing:
    """Describe price results for a price query.
    
    - One TradePricing instance can represent multiple swaps if there 
    is an intermediary pair

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
    #:
    #: Can be specified as single value or list, will be converted to list regardless 
    lp_fee: Optional[List[USDollarAmount]] = None

    #: What was the LP fee % used as the base of the calculations.
    #:
    #: Can be specified as single value or list, will be converted to list regardless
    pair_fee: Optional[List[BPS]] = None

    #: How old price data we used for this estimate
    #:
    market_feed_delay: Optional[datetime.timedelta] = None

    #: Is this buy or sell trade.
    #:
    #: True for buy.
    #: False for sell.
    #: None for Unknown.
    side: Optional[bool] = None
    
    #: Path of the trade
    #: One trade can have multiple swaps if there is an intermediary pair.
    path: Optional[List[TradingPairIdentifier]] = None
    
    @property
    def pair_fee(self):
        return self._pair_fee
    
    @property
    def lp_fee(self):
        return self._lp_fee
    
    @property
    def path(self):
        return self._path
    
    @pair_fee.setter
    def pair_fee(self, value):
        if type(value) != list:
            lst = [value]

        if all(lst):
            assert [
                type(fee) in {float, int} for fee in lst
            ], "pair_fee elements must be float or int."
        else:
            logger.warn("pair_fee provided with falsy values")

        self._pair_fee = value

    @lp_fee.setter
    def lp_fee(self, value):
        if type(value) != list:
            lst = [value]

        if all(lst):
            assert [
                type(fee) == float for fee in lst
            ], "lp_fee elements must be float."
        else:
            logger.warn("lp_fee provided with falsy values")

        self._lp_fee = value
        
    @path.setter
    def path(self, value):
        assert type(value) == list, "Path must be provided as a list"
        
        assert [type(address) == TradingPairIdentifier for address in self.path], "path must be provided as a list of TradePairIdentifier" 
        
        self._path = value
    
    def __repr__(self):
        if not self.pair_fee:
            fee_list = [0]
        else:
            fee_list = [fee or 0 for fee in self.pair_fee]
            
        return f"<TradePricing:{self.price} mid:{self.mid_price} fee:{format_fees_percentage(fee_list)}>"
    
    def __post_init__(self):
        """Validate parameters.

        Make sure we don't slip in e.g. NumPy types.
        """
        assert type(self.price) == float
        assert type(self.mid_price) == float
        
        if self.market_feed_delay is not None:
            assert isinstance(self.market_feed_delay, datetime.timedelta)

        # Do safety checks for the price calculation
        if self.side is not None:
            if self.side:
                assert self.price >= self.mid_price, f"Got bad buy pricing: {self.price} > {self.mid_price}"
            if not self.side:
                assert self.price <= self.mid_price, f"Got bad sell pricing: {self.price} < {self.mid_price}"
            
    def get_total_lp_fees(self):
        """:returns: The total lp fees paid (dollars) for the trade."""
        return sum(self.lp_fee)

def format_fees_percentage(fees: list[BPS]) -> str:
    """Returns string of formatted fees
    
    e.g. fees = [0.03, 0.005]
    => 0.3000% 0.0500%
    
    :param fees:
        list of lp fees in float (multiplier) format
        
    :returns:
        formatted str
    """
    _fees = [fee or 0 for fee in fees]
    strFormat = len(_fees) * '{:.4f}% '
    return strFormat.format(*_fees)
    
    
def format_fees_dollars(fees: list[USDollarAmount] | USDollarAmount) -> str:
    """Returns string of formatted fees
    
    :param fees:
        Can either be a list of fees or a single fee
    
    e.g. fees = [30, 50]
    => $30.00 $50.00
    
    :param fees:
        list of fees paid in absolute value (dollars)
    
    :returns:
        formatted str
    """
    
    if type(fees) != list:
        return f"${fees:.2f}"
    
    _fees = [fee or 0 for fee in fees]
    strFormat = len(_fees) * '${:.2f} '
    return strFormat.format(*_fees)

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

    @abc.abstractmethod
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

    @abc.abstractmethod
    def quantize_base_quantity(self, pair: TradingPairIdentifier, quantity: Decimal, rounding=ROUND_DOWN) -> Decimal:
        """Convert any base token quantity to the native token units by its ERC-20 decimals."""

    @abc.abstractmethod
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

#: This factory creates a new pricing model for each trade cycle.
#: Pricing model depends on the trading universe that may change for each strategy tick,
#: as new trading pairs appear.
#: Thus, we need to reconstruct pricing model as the start of the each tick.
PricingModelFactory = Callable[[ExecutionModel, StrategyExecutionUniverse, RoutingModel], PricingModel]
