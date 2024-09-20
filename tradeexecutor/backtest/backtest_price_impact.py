import datetime

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.price_impact import PriceImpactEstimation, PriceImpactSide
from tradeexecutor.state.types import USDollarAmount, Percent
from tradeexecutor.strategy.price_impact_model import PriceImpactModel


class FixedCappedSizeBacktestPriceImpact(PriceImpactModel):
    """Always cap the max size of a single trade to a fixed dollar amount.

    -
    """

    def __init__(
        self,
        fixed_price_impact: Percent | None = None,
        capped_size: USDollarAmount | None = None,
    ):
        self.capped_size = capped_size
        self.fixed_price_impact = fixed_price_impact
        assert 0 < self.fixed_price_impact < 1

    def get_acceptable_size_for_buy(
        self,
        timestamp: datetime.datetime | None,
        pair: TradingPairIdentifier,
        acceptable_price_impact: Percent,
        size: USDollarAmount,
    ) -> PriceImpactEstimation:
        return PriceImpactEstimation(
            side=PriceImpactSide.buy,
            timestamp=timestamp,
            pair=pair,
            block_number=None,
            asked_size=size,
            accepted_size=size * (1 - self.fixed_price_impact),
            estimated_price_impact=self.fixed_price_impact,
            acceptable_price_impact=acceptable_price_impact,
        )

    def get_acceptable_size_for_sell(
        self,
        timestamp: datetime.datetime | None,
        pair: TradingPairIdentifier,
        max_price_impact: Percent,
        quantity: USDollarAmount,
    ) -> PriceImpactEstimation:
        raise NotImplementedError()
