"""Asset pricing model."""

import abc
import datetime
from logging import getLogger
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Callable, Optional

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.price_impact import PriceImpactEstimation
from tradeexecutor.state.types import USDollarPrice, Percent, USDollarAmount
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.routing import RoutingModel
from tradeexecutor.strategy.universe_model import StrategyExecutionUniverse
from tradeexecutor.strategy.trade_pricing import TradePricing



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
        """Get the buy price for an asset.

        :param ts:
            When to get the price.
            Used in backtesting.
            Live models may ignore.

        :param pair:
            Trading pair we are intereted in

        :param reserve:
            If the buy token quantity is known,
            get the buy price with price impact.

        :return:
            Price structure for the trade.
        """

    @abc.abstractmethod
    def get_mid_price(self,
                      ts: datetime.datetime,
                      pair: TradingPairIdentifier) -> USDollarPrice:
        """Get the mid-price     for an asset.

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
    def get_pair_fee(
        self,
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

    def quantize_base_quantity(self, pair: TradingPairIdentifier, quantity: Decimal, rounding=ROUND_DOWN) -> Decimal:
        """Convert any base token quantity to the native token units by its ERC-20 decimals."""
        assert isinstance(pair, TradingPairIdentifier)
        decimals = pair.base.decimals
        return Decimal(quantity).quantize((Decimal(10) ** Decimal(-decimals)), rounding=rounding)

    def set_trading_fee_override(
            self,
            trading_fee_override: Percent | None
    ):
        """Set the trading fee override.

        - Override the trading fee for all tradingpairs

        - Allows to simulate different price levels
          and price impacts, based on the historical data with a different fee tier

        - Only useful for backtesting - in the live execution you pay whatever fees you are given by the venue

        :param trading_fee_override:
            The new fee tier.

            Example: `0.0030` for 30 BPS.

            Set ``None`` to disable and use the trading fee from the source data.
        """
        raise NotImplementedError()


#: This factory creates a new pricing model for each trade cycle.
#: Pricing model depends on the trading universe that may change for each strategy tick,
#: as new trading pairs appear.
#: Thus, we need to reconstruct pricing model as the start of the each tick.
PricingModelFactory = Callable[[ExecutionModel, StrategyExecutionUniverse, RoutingModel], PricingModel]
