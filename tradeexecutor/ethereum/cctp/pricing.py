"""CCTP bridge pricing model.

CCTP bridge pricing is always 1:1 with zero fees since USDC is
transferred between chains at face value via Circle's protocol.
"""

import datetime
import logging
from decimal import Decimal

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.types import USDollarPrice
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trade_pricing import TradePricing

logger = logging.getLogger(__name__)


class CctpBridgePricingModel(PricingModel):
    """CCTP bridge pricing — always 1:1, zero fee.

    USDC transferred via CCTP maintains its value 1:1 across chains.
    There are no LP fees or price impact for bridge transfers.
    """

    def get_sell_price(
        self,
        ts: datetime.datetime,
        pair: TradingPairIdentifier,
        quantity: Decimal | None,
    ) -> TradePricing:
        assert pair.is_cctp_bridge(), f"Not a CCTP bridge pair: {pair}"
        return TradePricing(
            price=1.0,
            mid_price=1.0,
            lp_fee=[0.0],
            pair_fee=[0.0],
            side=False,
            path=[pair],
            read_at=ts,
        )

    def get_buy_price(
        self,
        ts: datetime.datetime,
        pair: TradingPairIdentifier,
        reserve: Decimal | None,
    ) -> TradePricing:
        assert pair.is_cctp_bridge(), f"Not a CCTP bridge pair: {pair}"
        return TradePricing(
            price=1.0,
            mid_price=1.0,
            lp_fee=[0.0],
            pair_fee=[0.0],
            side=True,
            path=[pair],
            read_at=ts,
        )

    def get_mid_price(
        self,
        ts: datetime.datetime,
        pair: TradingPairIdentifier,
    ) -> USDollarPrice:
        return 1.0

    def get_pair_fee(
        self,
        ts: datetime.datetime,
        pair: TradingPairIdentifier,
    ) -> float | None:
        return 0.0
