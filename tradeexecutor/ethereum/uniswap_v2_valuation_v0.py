"""Protoype model of Uniswap v2 pricing.

.. warning::

    Deprecated. Will be removed.
"""

import datetime
from typing import Tuple

from eth_defi.uniswap_v2.deployment import UniswapV2Deployment
from eth_defi.uniswap_v2.fees import estimate_sell_received_amount_raw
from tradeexecutor.ethereum.uniswap_v2_live_pricing import UniswapV2LivePricing
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.valuation import ValuationModel


class UniswapV2PoolValuationMethodV0(ValuationModel):
    """Legacy valuation methdd.
    """

    def __init__(self, uniswap: UniswapV2Deployment):
        self.uniswap = uniswap

    def __call__(self,
                 ts: datetime.datetime,
                 position: TradingPosition) -> Tuple[datetime.datetime, USDollarAmount]:
        assert isinstance(ts, datetime.datetime)
        pair = position.pair

        assert position.is_long(), "Short not supported"

        quantity = position.get_quantity()
        # Cannot do pricing for zero quantity
        if quantity == 0:
            return ts, 0.0

        raw_price = estimate_sell_received_amount_raw(
            self.uniswap,
            pair.base.address,
            pair.quote.address,
            pair.base.convert_to_raw_amount(quantity)
        )

        price_in_dec = pair.quote.convert_to_decimal(raw_price)

        return ts, float(price_in_dec / quantity)


def uniswap_v2_sell_valuation_factory(pricing_model):
    return UniswapV2PoolRevaluator(pricing_model)