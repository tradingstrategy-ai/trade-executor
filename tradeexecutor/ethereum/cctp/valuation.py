"""CCTP bridge position valuation model.

Values bridged USDC positions at 1:1 USD value since CCTP
maintains the full value of USDC across chains.
"""

import datetime
import logging

from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.valuation import ValuationUpdate
from tradeexecutor.strategy.valuation import ValuationModel

logger = logging.getLogger(__name__)


class CctpBridgeValuationModel(ValuationModel):
    """Value CCTP bridge positions at 1:1 USD.

    Bridged USDC is always worth its face value.
    The position value equals the quantity of USDC bridged.
    """

    def __call__(
        self,
        ts: datetime.datetime,
        position: TradingPosition,
    ) -> ValuationUpdate:
        assert position.pair.is_cctp_bridge(), f"Not a CCTP bridge position: {position}"

        quantity = position.get_quantity()
        old_value = position.get_value()
        old_price = position.last_token_price
        new_price = 1.0  # USDC is always 1:1
        new_value = float(quantity)

        position.last_token_price = new_price
        position.last_pricing_at = ts

        return ValuationUpdate(
            created_at=ts,
            position_id=position.position_id,
            valued_at=ts,
            old_value=old_value,
            new_value=new_value,
            old_price=old_price,
            new_price=new_price,
            quantity=quantity,
        )
