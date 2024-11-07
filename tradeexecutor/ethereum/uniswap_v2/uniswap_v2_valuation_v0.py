"""Protoype model of Uniswap v2 pricing.

.. warning::

    Deprecated. Will be removed.
"""

import datetime
from typing import Tuple

from eth_defi.uniswap_v2.deployment import UniswapV2Deployment
from eth_defi.uniswap_v2.fees import estimate_sell_received_amount_raw
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.state.valuation import ValuationUpdate
from tradeexecutor.strategy.valuation import ValuationModel


class UniswapV2PoolValuationMethodV0(ValuationModel):
    """Legacy valuation methdd.
    """

    def __init__(self, uniswap: UniswapV2Deployment):
        self.uniswap = uniswap

    def __call__(self,
                 ts: datetime.datetime,
                 position: TradingPosition) -> ValuationUpdate:
        assert isinstance(ts, datetime.datetime)
        pair = position.pair

        old_price = position.last_token_price

        assert position.is_long(), "Short not supported"

        quantity = position.get_quantity()
        assert quantity > 0

        raw_price = estimate_sell_received_amount_raw(
            self.uniswap,
            pair.base.address,
            pair.quote.address,
            pair.base.convert_to_raw_amount(quantity)
        )

        new_price = pair.quote.convert_to_decimal(raw_price)

        old_value = position.get_value()
        new_value = position.revalue_base_asset(ts, float(new_price))

        evt = ValuationUpdate(
            created_at=ts,
            position_id=position.position_id,
            valued_at=ts,
            old_value=old_value,
            new_value=new_value,
            old_price=old_price,
            new_price=new_price,
            quantity=quantity,
        )

        position.valuation_updates.append(evt)

        return evt


def uniswap_v2_sell_valuation_factory(pricing_model):
    return UniswapV2PoolValuationMethodV0(pricing_model)