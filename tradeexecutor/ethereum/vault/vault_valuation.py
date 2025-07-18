"""Value model based on a vault mid-share price.

Assume we can redeem all shares without slippage.
"""
import datetime
import logging

from tradeexecutor.ethereum.vault.vault_live_pricing import VaultPricing
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.valuation import ValuationUpdate
from tradeexecutor.strategy.valuation import ValuationModel


logger = logging.getLogger(__name__)

class VaultValuator(ValuationModel):
    """Re-value assets based on what vault tells us."""

    def __init__(self, pricing_model: VaultPricing):
        assert isinstance(pricing_model, VaultPricing)
        self.pricing_model = pricing_model

    def __call__(
        self,
        ts: datetime.datetime,
        position: TradingPosition
    ) -> ValuationUpdate:
        assert position.is_vault()
        shares_amount = position.get_quantity()

        position.last_pricing_at = ts

        if shares_amount == 0:
            # Frozen position, deposit has failed?
            logger.warning(f"Creating null valuation update: Shares amount must be greater than 0, got {shares_amount} for position {position}")
            evt = ValuationUpdate(
                created_at=ts,
                position_id=position.position_id,
                valued_at=ts,
                old_value=0,
                new_value=0,
                old_price=0,
                new_price=0,
                quantity=shares_amount,
            )
            position.last_token_price = 0
            return evt

        assert position.pair.quote.is_stablecoin(), f"Vault position {position} must be a stablecoin pair, got {position.pair}"

        price_structure = self.pricing_model.get_sell_price(ts, position.pair, shares_amount)
        old_price = position.last_token_price
        old_value = position.get_value()
        new_price = price_structure.price
        new_value = position.revalue_base_asset(ts, float(new_price))
        evt = ValuationUpdate(
            created_at=ts,
            position_id=position.position_id,
            valued_at=ts,
            old_value=old_value,
            new_value=new_value,
            old_price=old_price,
            new_price=new_price,
            quantity=shares_amount,
        )
        logger.info("Vault position %s, valuation updated: %s", position, evt)
        position.last_token_price = new_price
        return evt


def vault_valuation_factory(pricing_model):
    return VaultValuator(pricing_model)