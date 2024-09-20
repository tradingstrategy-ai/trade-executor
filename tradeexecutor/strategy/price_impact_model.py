import abc
import datetime

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.price_impact import PriceImpactEstimation
from tradeexecutor.state.types import Percent, USDollarAmount


class PriceImpactModel(abc.ABC):
    """Estimate an impact of a single trade.

    - We are going to take a hit when taking liquidity out of the market

    Estimate this based on either

    - capped fixed amount (no data needed)
    - historical real data (EVM archive node),
    - historical estimation (based on TVL)
    - live real data (EVM node)
    """

    def get_acceptable_size_for_buy(
        self,
        timestamp: datetime.datetime | None,
        pair: TradingPairIdentifier,
        acceptable_price_impact: Percent,
        size: USDollarAmount,
    ) -> PriceImpactEstimation:
        raise NotImplementedError()

    def get_acceptable_size_for_sell(
        self,
        timestamp: datetime.datetime | None,
        pair: TradingPairIdentifier,
        acceptable_price_impact: Percent,
        quantity: USDollarAmount,
    ) -> PriceImpactEstimation:
        raise NotImplementedError()

