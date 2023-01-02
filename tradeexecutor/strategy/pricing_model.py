"""Asset pricing model."""

import abc
import datetime
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Callable, Optional

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.types import USDollarAmount, BPS, USDollarPrice
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.routing import RoutingModel
from tradeexecutor.strategy.universe_model import StrategyExecutionUniverse


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

    def __post_init__(self):
        """Validate parameters.

        Make sure we don't slip in e.g. NumPy types.
        """
        assert type(self.price) == float
        assert type(self.mid_price) == float
        if self.lp_fee is not None:
            assert type(self.lp_fee) == float
        if self.pair_fee is not None:
            assert type(self.pair_fee) == float
        if self.market_feed_delay is not None:
            assert isinstance(self.market_feed_delay, datetime.timedelta)


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

