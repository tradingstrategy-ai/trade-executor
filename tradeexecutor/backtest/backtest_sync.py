import datetime
from decimal import Decimal
from typing import List, Optional

from tradeexecutor.backtest.simulated_wallet import SimulatedWallet
from tradeexecutor.ethereum.wallet import ReserveUpdateEvent
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.strategy.sync_model import SyncModel

from tradeexecutor.testing.dummy_wallet import apply_sync_events


class BacktestSyncer:
    """LEGACY backtest sync model.

    Simulate deposit events to the backtest wallet."""

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



class BacktestSyncModel(SyncModel):
    """Backtest sync model.

    Simulate deposit events to the backtest wallet."""

    def __init__(self, wallet: SimulatedWallet, initial_deposit_amount: Decimal):
        assert isinstance(initial_deposit_amount, Decimal)
        self.wallet = wallet
        self.initial_deposit_amount = initial_deposit_amount
        self.initial_deposit_processed_at = None

    def sync_initial(self, state: State):
        """Set u[ initial sync details."""

        deployment = state.sync.deployment

        deployment.chain_id = None
        deployment.address = None
        deployment.block_number = None
        deployment.tx_hash = None
        deployment.block_mined_at = None
        deployment.vault_token_name = None
        deployment.vault_token_symbol = None

    def sync_treasury(self,
                 strategy_cycle_ts: datetime.datetime,
                 state: State,
                 supported_reserves: Optional[List[AssetIdentifier]] = None
                 ):
        """Apply the balance sync before each strategy cycle.

        .. warning::

            Old legacy code with wrong return signature compared to the parent class
        """

        portfolio = state.portfolio

        # TODO: Move this code eto sync_initial()
        if not self.initial_deposit_processed_at:
            self.initial_deposit_processed_at = strategy_cycle_ts

            assert len(supported_reserves) == 1

            reserve_token = supported_reserves[0]

            # Generate a deposit event
            evt = ReserveUpdateEvent(
                asset=reserve_token,
                past_balance=Decimal(0),
                new_balance=self.initial_deposit_amount,
                updated_at=strategy_cycle_ts
            )

            # Update wallet
            self.wallet.update_balance(reserve_token.address, self.initial_deposit_amount)

            # Update state
            apply_sync_events(portfolio, [evt])

            # Set synced flag
            # TODO: fix - wrong event type
            state.sync.treasury.last_updated_at = strategy_cycle_ts
            state.sync.treasury.balance_update_refs = []

            return []
        else:
            return []

    def create_transaction_builder(self) -> None:
        return None
