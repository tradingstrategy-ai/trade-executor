"""Old backtest wallet top up code."""

import datetime
from _decimal import Decimal
from typing import List

from tradeexecutor.backtest.simulated_wallet import SimulatedWallet
from tradeexecutor.ethereum.wallet import ReserveUpdateEvent
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.ethereum.reserve_update import apply_sync_events


class BacktestSyncer:
    """LEGACY backtest sync model.

    Simulate deposit events to the backtest wallet.

    .. warning::

        Does not correctly fire any balance update events.
        Can be used to backtest with a fixed initial amount only.
    """

    def __init__(self, wallet: SimulatedWallet, initial_deposit_amount: Decimal):
        assert isinstance(initial_deposit_amount, Decimal)
        self.wallet = wallet
        self.initial_deposit_amount = initial_deposit_amount
        self.initial_deposit_processed_at = None

    def __call__(self, state: State, ts: datetime.datetime, supported_reserves: List[AssetIdentifier]) -> List[ReserveUpdateEvent]:
        """Process the backtest initial deposit.

        The backtest wallet is credited once at the start.
        """
        assert isinstance(state, State)

        portfolio = state.portfolio

        if not self.initial_deposit_processed_at:
            self.initial_deposit_processed_at = ts

            assert len(supported_reserves) == 1

            reserve_token = supported_reserves[0]

            # Generate a deposit event
            evt = ReserveUpdateEvent(
                asset=reserve_token,
                past_balance=Decimal(0),
                new_balance=self.initial_deposit_amount,
                updated_at=ts,
                mined_at=ts,
            )

            # Update wallet
            self.wallet.update_token_info(reserve_token)
            self.wallet.update_balance(reserve_token.address, self.initial_deposit_amount)

            # Update state
            apply_sync_events(state, [evt])

            return [evt]
        else:
            return []
