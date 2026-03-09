"""Valuation and pricing models for Hypercore native vault positions.

Hypercore vault equity is queried via the Hyperliquid info API,
not from on-chain contracts. The position tracks a single "unit"
(quantity=1) with the price equal to the vault equity in USDC.
"""

import datetime
import logging
from decimal import Decimal
from typing import Callable

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.valuation import ValuationUpdate
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.valuation import ValuationModel

logger = logging.getLogger(__name__)


class HypercoreVaultPricing(PricingModel):
    """Pricing model for Hypercore vault positions.

    Returns the vault equity as the per-unit price.
    The position has quantity=1, so value = 1 × equity = equity.
    """

    def __init__(self, value_func: Callable[[TradingPairIdentifier], Decimal]):
        self.value_func = value_func

    def get_sell_price(self, ts, pair, quantity):
        equity = self.value_func(pair)
        return float(equity)

    def get_buy_price(self, ts, pair, reserve):
        equity = self.value_func(pair)
        return float(equity)

    def get_mid_price(self, ts, pair):
        equity = self.value_func(pair)
        return float(equity)

    def get_pair_fee(self, ts, pair):
        return 0.0

    def get_pair_for_id(self, internal_id):
        raise NotImplementedError("Hypercore vault pricing does not support pair lookup by ID")


class HypercoreVaultValuator(ValuationModel):
    """Re-value Hypercore vault positions using the Hyperliquid info API.

    Queries the vault equity for the Safe address and updates the
    position price accordingly. The position model is:

    - quantity = 1 (one "unit" of this vault position)
    - price = equity in USDC
    - value = 1 × equity = equity
    """

    def __init__(self, value_func: Callable[[TradingPairIdentifier], Decimal]):
        self.value_func = value_func

    def __call__(
        self,
        ts: datetime.datetime,
        position: TradingPosition,
    ) -> ValuationUpdate:
        assert position.is_vault(), f"Not a vault position: {position}"

        position.last_pricing_at = ts

        try:
            equity = self.value_func(position.pair)
        except Exception as e:
            logger.error(
                "Failed to get Hypercore vault equity for position %s: %s",
                position, e,
            )
            raise

        old_price = position.last_token_price
        old_value = position.get_value()
        new_price = float(equity)
        new_value = position.revalue_base_asset(ts, new_price)

        evt = ValuationUpdate(
            created_at=ts,
            position_id=position.position_id,
            valued_at=ts,
            old_value=old_value,
            new_value=new_value,
            old_price=old_price,
            new_price=new_price,
            quantity=position.get_quantity(),
        )

        position.last_token_price = new_price

        logger.info(
            "Hypercore vault position %s, valuation updated: equity=$%.2f, old=$%.2f, new=$%.2f",
            position, equity, old_value, new_value,
        )

        return evt
