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
from tradeexecutor.strategy.position_internal_share_price import (
    create_share_price_state_for_exchange_account,
    update_share_price_state_for_balance_update,
)
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

    def __init__(self, pricing_model: ExchangeAccountPricingModel, web3=None):
        """Initialise valuation model.

        :param pricing_model:
            ExchangeAccountPricingModel with configured account value function
        :param web3:
            Optional Web3 instance for capturing the current block number.
            When provided, the block is captured once per valuation call
            and passed to the account value function as ``block_identifier``,
            then persisted in ``ValuationUpdate.block_number`` and
            ``BalanceUpdate.block_number`` for audit.
        """
        assert isinstance(pricing_model, ExchangeAccountPricingModel), \
            f"Expected ExchangeAccountPricingModel, got {type(pricing_model)}"
        self.pricing_model = pricing_model
        self.web3 = web3

    def _create_balance_update(
        self,
        valued_at: datetime.datetime,
        position: TradingPosition,
        diff: Decimal,
        tracked_amount: Decimal,
        strategy_cycle_ts: datetime.datetime,
        block_number: int | None = None,
    ) -> BalanceUpdate:
        """Create a BalanceUpdate event to adjust position quantity.

        Uses the position's own balance_updates dict to derive a
        locally-unique event ID (max existing key + 1).  This avoids
        needing access to ``state.portfolio.next_balance_update_id``.

        :param valued_at:
            Wall clock time when the exchange API was queried.
        :param position:
            The exchange account position.
        :param diff:
            Value change (new_value - tracked_value).
        :param tracked_amount:
            The tracked quantity before this update.
        :param strategy_cycle_ts:
            The strategy cycle timestamp that triggered this valuation.
            May differ significantly from wall clock time during manual
            CLI runs (e.g. ``trade-executor start --max-cycles 1``).
        :param block_number:
            Block at which the exchange API value was read.
        :return:
            The new BalanceUpdate event (already stored on the position).
        """
        event_id = max(position.balance_updates.keys(), default=0) + 1

        evt = BalanceUpdate(
            balance_update_id=event_id,
            position_type=BalanceUpdatePositionType.open_position,
            cause=BalanceUpdateCause.vault_flow,
            asset=position.pair.base,
            block_mined_at=valued_at,
            strategy_cycle_included_at=strategy_cycle_ts,
            chain_id=position.pair.base.chain_id,
            old_balance=tracked_amount,
            usd_value=float(diff),
            quantity=diff,
            owner_address=None,
            tx_hash=None,
            log_index=None,
            position_id=position.position_id,
            block_number=block_number,
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
            Strategy cycle timestamp that triggered this valuation.
            May differ significantly from wall clock time when the cycle
            is run manually from the CLI (e.g. ``trade-executor start
            --max-cycles 1``) or when the executor starts up and the
            cycle timestamp snaps to midnight.

            We use wall clock time (not this value) for ``valued_at``
            and ``last_pricing_at`` because the exchange API data is
            fetched in real-time. This keeps Lagoon's freshness guard
            happy.

        :param position:
            Exchange account position to revalue
        :return:
            ValuationUpdate event with new value
        """
        assert position.is_exchange_account(), f"Not an exchange account position: {position}"

        tracked_amount = position.get_quantity()

        # Capture block once — used for both the GMX read and state persistence
        block_number = self.web3.eth.block_number if self.web3 else None

        try:
            # Query exchange API for account value, forwarding block_identifier
            api_value = self.pricing_model.get_account_value(
                position.pair, block_identifier=block_number,
            )

            # Only advance freshness timestamps after a successful fetch.
            # Wall clock time is used (not the cycle timestamp ``ts``)
            # because the exchange API data is real-time. The cycle
            # timestamp can lag wall clock by hours during manual CLI
            # runs, which would trip Lagoon's valuation freshness guard.
            now = native_datetime_utc_now()
            position.last_pricing_at = now

            old_price = position.last_token_price
            old_value = position.get_value()
            # Exchange account quantity represents USD value directly,
            # so the per-unit price is always 1.0
            new_price = 1.0

            # Create a BalanceUpdate to adjust the position quantity
            # so that get_value() reflects the API value
            diff = api_value - tracked_amount
            if diff != 0:
                self._create_balance_update(
                    now, position, diff, tracked_amount,
                    strategy_cycle_ts=ts, block_number=block_number,
                )

            # Initialise or update internal share price state.
            # The first successful valuation establishes the initial capital
            # (share_price_state is None until then because placeholder trades
            # from open_exchange_account_position are excluded).
            # Subsequent valuations adjust the share price as value changes.
            total_value = float(api_value)
            if position.share_price_state is None:
                if total_value > 0:
                    position.share_price_state = create_share_price_state_for_exchange_account(
                        total_value, now,
                    )
            elif diff != 0:
                last_bu = position.balance_updates[max(position.balance_updates)]
                position.share_price_state = update_share_price_state_for_balance_update(
                    position.share_price_state, last_bu,
                )

            position.last_token_price = new_price
            new_value = float(api_value)

            evt = ValuationUpdate(
                created_at=ts,
                position_id=position.position_id,
                valued_at=now,
                old_value=old_value,
                new_value=new_value,
                old_price=old_price,
                new_price=new_price,
                quantity=api_value,
                block_number=block_number,
            )
            position.valuation_updates.append(evt)

            logger.info(
                "Exchange account position %d revalued: %.2f -> %.2f (block=%s)",
                position.position_id,
                old_value,
                new_value,
                block_number,
            )
            return evt

        except Exception as e:
            # API failure — preserve the previous freshness timestamps
            # so Lagoon's guard still catches stale data after an outage.
            # Do NOT update last_pricing_at here.
            logger.error(
                "Failed to revalue exchange account position %d: %s",
                position.position_id,
                e,
            )
            old_value = position.get_value()
            evt = ValuationUpdate(
                created_at=ts,
                position_id=position.position_id,
                valued_at=position.last_pricing_at,
                old_value=old_value,
                new_value=old_value,
                old_price=position.last_token_price,
                new_price=position.last_token_price,
                quantity=tracked_amount,
            )
            position.valuation_updates.append(evt)
            return evt


def exchange_account_valuation_factory(pricing_model: ExchangeAccountPricingModel, web3=None) -> ExchangeAccountValuator:
    """Factory function for creating ExchangeAccountValuator.

    :param pricing_model:
        ExchangeAccountPricingModel with configured account value function
    :param web3:
        Optional Web3 instance for block number tracking.
    :return:
        ExchangeAccountValuator instance
    """
    return ExchangeAccountValuator(pricing_model, web3=web3)
