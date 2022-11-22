import datetime
from decimal import Decimal
from typing import List

from tradeexecutor.backtest.simulated_wallet import SimulatedWallet
from tradeexecutor.ethereum.wallet import ReserveUpdateEvent
from tradeexecutor.state.sync import apply_sync_events
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.identifier import AssetIdentifier


class BacktestSyncer:
    """Simulate deposit events to the backtest wallet."""

    def __init__(self, wallet: SimulatedWallet, initial_deposit_amount: Decimal):
        assert isinstance(initial_deposit_amount, Decimal)
        self.wallet = wallet
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

            # Update wallet
            self.wallet.update_balance(reserve_token.address, self.initial_deposit_amount)

            # Update state
            apply_sync_events(portfolio, [evt])

            return [evt]
        else:
            return []

