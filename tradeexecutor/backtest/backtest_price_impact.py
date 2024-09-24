import datetime

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.trade_size import TradeSize, TradeSizeSide
from tradeexecutor.state.types import USDollarAmount, Percent, TokenAmount
from tradeexecutor.strategy.pricing_model import PricingModel

from tradeexecutor.strategy.trade_sizer import TradeSizer

#: Too rich to trade 10M
UNLIMITED_CAP = 9_999_999


class UncappedTradeSizer(TradeSizer):
    """A trade sizer that assumes unlimited market depth.

    - Always get the trade you ask for

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
    ) -> TradeSize:
        return TradeSize(
            timestamp=timestamp,
            side=TradeSizeSide.buy,
            pair=pair,
            path=[pair],
            asked_size=asked_size,
            accepted_size=min(self.per_trade_cap, asked_size),
        )

    def get_acceptable_size_for_sell(
        self,
        timestamp: datetime.datetime | None,
        pair: TradingPairIdentifier,
        max_price_impact: Percent,
        quantity: TokenAmount,
    ) -> TradeSize:
        raise NotImplementedError()

    def get_acceptable_size_for_position():
        pass

    def get_value(self, quantity: TokenAmount):
        self.pricing

