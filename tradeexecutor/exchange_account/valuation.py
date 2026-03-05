"""Exchange account position valuation model.

Valuation model for exchange account positions (Derive, Hyperliquid, etc.)
using a configurable account value function.

Creates ``BalanceUpdate`` events directly on positions so that
``position.get_value()`` reflects the exchange API value regardless
of which sync model is active (Lagoon, hot wallet, etc.).
"""

import datetime
import logging
from decimal import Decimal

from eth_defi.compat import native_datetime_utc_now

from tradeexecutor.state.balance_update import BalanceUpdate, BalanceUpdateCause, BalanceUpdatePositionType
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

    Creates a ``BalanceUpdate`` event on the position each time the
    API value differs from the tracked quantity, so that
    ``position.get_value()`` (which sums trade quantities +
    balance updates) stays in sync with the exchange.

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
        update = valuator(native_datetime_utc_now(), position)
    """

    def __init__(self, pricing_model: ExchangeAccountPricingModel):
        """Initialise valuation model.

        :param pricing_model:
            ExchangeAccountPricingModel with configured account value function
        """
        assert isinstance(pricing_model, ExchangeAccountPricingModel), \
            f"Expected ExchangeAccountPricingModel, got {type(pricing_model)}"
        self.pricing_model = pricing_model

    def _create_balance_update(
        self,
        ts: datetime.datetime,
        position: TradingPosition,
        diff: Decimal,
        tracked_amount: Decimal,
    ) -> BalanceUpdate:
        """Create a BalanceUpdate event to adjust position quantity.

        Uses the position's own balance_updates dict to derive a
        locally-unique event ID (max existing key + 1).  This avoids
        needing access to ``state.portfolio.next_balance_update_id``.

        :param ts:
            Timestamp of the valuation.
        :param position:
            The exchange account position.
        :param diff:
            Value change (new_value - tracked_value).
        :param tracked_amount:
            The tracked quantity before this update.
        :return:
            The new BalanceUpdate event (already stored on the position).
        """
        event_id = max(position.balance_updates.keys(), default=0) + 1

        evt = BalanceUpdate(
            balance_update_id=event_id,
            position_type=BalanceUpdatePositionType.open_position,
            cause=BalanceUpdateCause.vault_flow,
            asset=position.pair.base,
            block_mined_at=ts,
            strategy_cycle_included_at=ts,
            chain_id=position.pair.base.chain_id,
            old_balance=tracked_amount,
            usd_value=float(diff),
            quantity=diff,
            owner_address=None,
            tx_hash=None,
            log_index=None,
            position_id=position.position_id,
            block_number=None,
            notes=f"Exchange account valuation: {position.pair.get_exchange_account_protocol()}",
        )

        position.balance_updates[evt.balance_update_id] = evt
        return evt

    def __call__(
        self,
        ts: datetime.datetime,
        position: TradingPosition,
    ) -> ValuationUpdate:
        """Revalue exchange account position by querying exchange API.

        The position value is the total account value in USD from the exchange,
        which includes collateral and unrealised PnL from all positions.

        Creates a ``BalanceUpdate`` event on the position when the API
        value differs from the tracked quantity, ensuring
        ``position.get_value()`` returns the correct value.

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

        try:
            # Query exchange API for account value
            api_value = self.pricing_model.get_account_value(position.pair)

            old_price = position.last_token_price
            old_value = position.get_value()
            # Exchange account quantity represents USD value directly,
            # so the per-unit price is always 1.0
            new_price = 1.0

            # Create a BalanceUpdate to adjust the position quantity
            # so that get_value() reflects the API value
            diff = api_value - tracked_amount
            if diff != 0:
                self._create_balance_update(ts, position, diff, tracked_amount)

            position.last_token_price = new_price
            new_value = float(api_value)

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
