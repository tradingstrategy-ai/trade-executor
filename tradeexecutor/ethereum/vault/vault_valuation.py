"""Value model based on a vault mid-share price.

Assume we can redeem all shares without slippage.
"""
import datetime
from typing import Tuple

from tradeexecutor.ethereum.vault.vault_live_pricing import VaultPricing
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.valuation import ValuationModel


class VaultValuator(ValuationModel):
    """Re-value assets based on what vault tells us."""

    def __init__(self, pricing_model: VaultPricing):
        super().__init__(pricing_model)
        assert isinstance(pricing_model, VaultPricing)

    def __call__(
        self,
        ts: datetime.datetime,
        position: TradingPosition
    ) -> Tuple[datetime.datetime, USDollarAmount]:
        assert position.is_vault()
        shares_amount = position.get_quantity()
        return self.pricing_model.get_sell_price(ts, position.pair, shares_amount)


def vault_valuation_factory(pricing_model):
    return VaultValuator(pricing_model)