"""Freqtrade position valuation model."""

import datetime
import logging
from decimal import Decimal

from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.valuation import ValuationUpdate
from tradeexecutor.strategy.valuation import ValuationModel
from tradeexecutor.strategy.freqtrade.freqtrade_pricing import FreqtradePricingModel

logger = logging.getLogger(__name__)


class FreqtradeValuator(ValuationModel):
    """Revalue Freqtrade positions based on current balance from API.

    Queries the Freqtrade REST API to get the current account balance
    and reconciles it against the executor's internal tracking.
    """

    def __init__(self, pricing_model: FreqtradePricingModel):
        """Initialize valuation model.

        Args:
            pricing_model: FreqtradePricingModel for API access
        """
        assert isinstance(pricing_model, FreqtradePricingModel)
        self.pricing_model = pricing_model

    def __call__(
        self,
        ts: datetime.datetime,
        position: TradingPosition,
    ) -> ValuationUpdate:
        """Revalue Freqtrade position by querying API.

        Compares the API balance with executor's tracked amount and
        warns if there's drift >1% (which could indicate manual deposits
        or trading losses not yet accounted for).

        Args:
            ts: Timestamp for valuation
            position: Freqtrade position to revalue

        Returns:
            ValuationUpdate event with new value
        """
        assert position.is_freqtrade(), f"Not a Freqtrade position: {position}"

        # Get current tracked quantity (sum of deposits - withdrawals)
        tracked_amount = position.get_quantity()

        position.last_pricing_at = ts

        if tracked_amount == 0:
            # Position closed or failed deposit
            logger.warning(
                f"Creating null valuation: amount is 0 for position {position.position_id}"
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
            # Query Freqtrade API for current balance
            api_balance = self.pricing_model._get_freqtrade_balance(position.pair)

            # Check for reconciliation issues
            balance_diff = api_balance - tracked_amount
            if abs(balance_diff) > tracked_amount * Decimal("0.01"):
                # >1% drift
                logger.warning(
                    f"Balance drift detected for position {position.position_id}: "
                    f"API={api_balance}, Tracked={tracked_amount}, "
                    f"Diff={balance_diff}"
                )

            # Update position value with API balance
            old_price = position.last_token_price
            old_value = position.get_value()
            new_price = 1.0  # Always 1:1 for reserve currency

            # Set quantity to API balance (reconciliation)
            new_value = float(api_balance)
            position.last_token_price = new_price

            evt = ValuationUpdate(
                created_at=ts,
                position_id=position.position_id,
                valued_at=ts,
                old_value=old_value,
                new_value=new_value,
                old_price=old_price,
                new_price=new_price,
                quantity=api_balance,  # Use API balance as truth
            )

            logger.info(
                f"Freqtrade position {position.position_id} revalued: "
                f"{old_value:.2f} -> {new_value:.2f}"
            )
            return evt

        except Exception as e:
            # API failure - keep last valuation
            logger.error(f"Failed to revalue Freqtrade position {position.position_id}: {e}")
            # Return no-change valuation
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


def freqtrade_valuation_factory(pricing_model):
    """Factory function for creating FreqtradeValuator."""
    return FreqtradeValuator(pricing_model)
