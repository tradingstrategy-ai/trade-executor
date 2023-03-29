"""Synchrone deposits/withdrawals of the portfolio.

Syncs the external portfolio changes from a (blockchain) source.
See ethereum/hotwallet_sync.py for details.
"""

import datetime
from decimal import Decimal
from typing import List

from tradeexecutor.ethereum.wallet import ReserveUpdateEvent, logger
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.reserve import ReservePosition


class DummyWalletSyncer:
    """Simulate a wallet events with a fixed balance set in the beginning."""

    def __init__(self, initial_deposit_amount: Decimal = Decimal(0)):
        assert isinstance(initial_deposit_amount, Decimal)
        self.initial_deposit_amount = initial_deposit_amount
        self.initial_deposit_processed_at = None

    def __call__(self, portfolio: Portfolio, ts: datetime.datetime, supported_reserves: List[AssetIdentifier]) -> List[ReserveUpdateEvent]:
        """Process the backtest initial deposit.

        The backtest wallet is credited once at the start.
        """

        if not self.initial_deposit_processed_at:
            self.initial_deposit_processed_at = ts

            assert len(supported_reserves) == 1

            reserve_token = supported_reserves[0]

            # Generate a deposit event
            evt = ReserveUpdateEvent(
                asset=reserve_token,
                past_balance=Decimal(0),
                new_balance=self.initial_deposit_amount,
                updated_at=ts
            )

            # Update state
            apply_sync_events(portfolio, [evt])

            return [evt]
        else:
            return []


def apply_sync_events(portfolio: Portfolio, new_reserves: List[ReserveUpdateEvent], default_price=1.0):
    """Apply deposit and withdraws on reserves in the portfolio.

    .. note ::

        This is only used with deprecated :py:data:`tradeexecutor.strategy.sync_model.SyncMethodV0`

    :param default_price: Set the reserve currency price for new reserves.
    """

    for evt in new_reserves:

        res_pos = portfolio.reserves.get(evt.asset.get_identifier())
        if res_pos is not None:
            # Update existing
            res_pos.quantity = evt.new_balance
            res_pos.last_sync_at = evt.updated_at
            logger.info("Portfolio reserve synced. Asset: %s", evt.asset)
        else:
            # Initialise new reserve position
            res_pos = ReservePosition(
                asset=evt.asset,
                quantity=evt.new_balance,
                last_sync_at=evt.updated_at,
                reserve_token_price=default_price,
                last_pricing_at=evt.updated_at,
                initial_deposit_reserve_token_price=default_price,
                initial_deposit=evt.new_balance,
            )
            portfolio.reserves[res_pos.get_identifier()] = res_pos
            logger.info("Portfolio reserve created. Asset: %s", evt.asset)


