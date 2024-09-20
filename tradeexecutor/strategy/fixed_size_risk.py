"""Fixed size and unlimited trade size risking."""

import datetime
from decimal import Decimal

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.size_risk import SizeRisk
from tradeexecutor.state.types import USDollarAmount, TokenAmount
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.size_risk_model import SizeRiskModel, SizingType

#: We assume we are too rich to trade 10M trades/positions
UNLIMITED_CAP: USDollarAmount = 9_999_999


class FixedSizeRiskModel(SizeRiskModel):
    """A trade sizer that assumes unlimited market depth.

    - Always get the trade you ask for, unless
      a trade or a position hits the maximum value

    """

    def __init__(
        self,
        pricing_model: PricingModel,
        per_trade_cap: USDollarAmount = UNLIMITED_CAP,
        per_position_cap: USDollarAmount = UNLIMITED_CAP,
    ):
        """

        :param per_trade_cap:
            Maximum US dollar value of a single trade, or unlimited.
        """
        self.pricing_model = pricing_model
        self.per_trade_cap = per_trade_cap
        self.per_position_cap = per_position_cap

    def get_acceptable_size_for_buy(
        self,
        timestamp: datetime.datetime | None,
        pair: TradingPairIdentifier,
        asked_size: USDollarAmount,
    ) -> SizeRisk:
        accepted_size = min(self.per_trade_cap, asked_size)
        capped = accepted_size == self.per_trade_cap
        return SizeRisk(
            timestamp=timestamp,
            type=SizingType.buy,
            pair=pair,
            path=[pair],
            asked_size=asked_size,
            accepted_size=accepted_size,
            capped=capped,
        )

    def get_acceptable_size_for_sell(
        self,
        timestamp: datetime.datetime | None,
        pair: TradingPairIdentifier,
        asked_quantity: TokenAmount,
    ) -> SizeRisk:
        assert isinstance(asked_quantity, Decimal)
        mid_price = self.pricing_model.get_mid_price(timestamp, pair)
        asked_value = asked_quantity * mid_price
        max_value = min(self.per_trade_cap, asked_value)
        capped = max_value == self.per_trade_cap
        accepted_quantity = Decimal(max_value / mid_price)
        return SizeRisk(
            timestamp=timestamp,
            type=SizingType.sell,
            pair=pair,
            path=[pair],
            asked_quantity=asked_quantity,
            accepted_quantity=accepted_quantity,
            capped=capped,
        )

    def get_acceptable_size_for_position(
        self,
        timestamp: datetime.datetime | None,
        pair: TradingPairIdentifier,
        asked_value: USDollarAmount,
    ) -> SizeRisk:
        accepted_size = min(self.per_position_cap, asked_value)
        capped = accepted_size == self.per_position_cap
        return SizeRisk(
            timestamp=timestamp,
            type=SizingType.hold,
            pair=pair,
            path=[pair],
            asked_size=asked_value,
            accepted_size=accepted_size,
            capped=capped,
        )

