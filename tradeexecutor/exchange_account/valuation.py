"""Exchange account position valuation model.

Valuation model for exchange account positions (Derive, Hyperliquid, etc.)
using a configurable account value function.
"""

import datetime
import logging
from decimal import Decimal

from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.valuation import ValuationUpdate
from tradeexecutor.strategy.valuation import ValuationModel
from tradeexecutor.exchange_account.pricing import ExchangeAccountPricingModel

logger = logging.getLogger(__name__)


class ExchangeAccountValuator(ValuationModel):
    """Revalue exchange account positions using configurable account value function.

    Works with any exchange that can provide account value in USD.
    The account value function is injected via the pricing model,
    making this valuator protocol-agnostic.

    Example:

    .. code-block:: python

        from tradeexecutor.exchange_account.pricing import ExchangeAccountPricingModel
        from tradeexecutor.exchange_account.valuation import ExchangeAccountValuator
        from tradeexecutor.exchange_account.derive import create_derive_account_value_func

        # Create Derive-specific account value function
        clients = {subaccount_id: derive_client}
        account_value_func = create_derive_account_value_func(clients)

        # Create pricing and valuation models
        pricing = ExchangeAccountPricingModel(account_value_func)
        valuator = ExchangeAccountValuator(pricing)

        # Revalue position
        update = valuator(datetime.utcnow(), position)
    """

    def __init__(self, pricing_model: ExchangeAccountPricingModel):
        """Initialise valuation model.

        :param pricing_model:
            ExchangeAccountPricingModel with configured account value function
        """
        assert isinstance(pricing_model, ExchangeAccountPricingModel), \
            f"Expected ExchangeAccountPricingModel, got {type(pricing_model)}"
        self.pricing_model = pricing_model

    def __call__(
        self,
        ts: datetime.datetime,
        position: TradingPosition,
    ) -> ValuationUpdate:
        """Revalue exchange account position by querying exchange API.

        The position value is the total account value in USD from the exchange,
        which includes collateral and unrealised PnL from all positions.

        :param ts:
            Timestamp for valuation
        :param position:
            Exchange account position to revalue
        :return:
            ValuationUpdate event with new value
        """
        assert position.is_exchange_account(), f"Not an exchange account position: {position}"

        tracked_amount = position.get_quantity()
        position.last_pricing_at = ts

        if tracked_amount == 0:
            logger.warning(
                "Creating null valuation: amount is 0 for position %d",
                position.position_id,
            )
            return ValuationUpdate(
                created_at=ts,
                position_id=position.position_id,
                valued_at=ts,
                old_value=0,
                new_value=0,
                old_price=0,
                new_price=0,
                quantity=tracked_amount,
            )

        try:
            # Query exchange API for account value
            api_value = self.pricing_model.get_account_value(position.pair)

            # Check for reconciliation issues (drift between tracked and actual)
            balance_diff = api_value - tracked_amount
            if abs(balance_diff) > tracked_amount * Decimal("0.01"):
                # >1% drift
                logger.warning(
                    "Balance drift detected for position %d: "
                    "API=%.2f, Tracked=%.2f, Diff=%.2f",
                    position.position_id,
                    api_value,
                    tracked_amount,
                    balance_diff,
                )

            old_price = position.last_token_price
            old_value = position.get_value()
            new_price = 1.0  # Always 1:1 for USD denominated
            new_value = float(api_value)

            position.last_token_price = new_price

            evt = ValuationUpdate(
                created_at=ts,
                position_id=position.position_id,
                valued_at=ts,
                old_value=old_value,
                new_value=new_value,
                old_price=old_price,
                new_price=new_price,
                quantity=api_value,
            )

            logger.info(
                "Exchange account position %d revalued: %.2f -> %.2f",
                position.position_id,
                old_value,
                new_value,
            )
            return evt

        except Exception as e:
            # API failure - keep last valuation
            logger.error(
                "Failed to revalue exchange account position %d: %s",
                position.position_id,
                e,
            )
            old_value = position.get_value()
            return ValuationUpdate(
                created_at=ts,
                position_id=position.position_id,
                valued_at=ts,
                old_value=old_value,
                new_value=old_value,
                old_price=position.last_token_price,
                new_price=position.last_token_price,
                quantity=tracked_amount,
            )


def exchange_account_valuation_factory(pricing_model: ExchangeAccountPricingModel) -> ExchangeAccountValuator:
    """Factory function for creating ExchangeAccountValuator.

    :param pricing_model:
        ExchangeAccountPricingModel with configured account value function
    :return:
        ExchangeAccountValuator instance
    """
    return ExchangeAccountValuator(pricing_model)
