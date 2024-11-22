"""Synchrone deposits/withdrawals of the portfolio.

Syncs the external portfolio changes from a (blockchain) source.
See ethereum/hotwallet_sync.py for details.
"""

import datetime
from decimal import Decimal
from typing import List

from tradeexecutor.ethereum.balance_update import apply_reserve_update_events
from tradeexecutor.ethereum.wallet import ReserveUpdateEvent
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.identifier import AssetIdentifier


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
            apply_reserve_update_events(portfolio, [evt])

            return [evt]
        else:
            return []


