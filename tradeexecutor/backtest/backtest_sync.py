import datetime
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional

from tradeexecutor.backtest.simulated_wallet import SimulatedWallet
from tradeexecutor.ethereum.wallet import ReserveUpdateEvent
from tradeexecutor.state.balance_update import BalanceUpdate
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.sync_model import SyncModel

from tradeexecutor.testing.dummy_wallet import apply_sync_events


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
                updated_at=ts
            )

            # Update wallet
            self.wallet.update_balance(reserve_token.address, self.initial_deposit_amount)

            # Update state
            apply_sync_events(state, [evt])

            return [evt]
        else:
            return []


@dataclass
class FundFlowEvent:
    """One simulated deposit/redemption in backtest.

    - Events can be triggered any time with :py:meth:`BacktestSyncModel.simulate_funding`

    - Fund flow is added to the reserves during :py:meth:`BacktestSyncModel.sync_treasury`
      as it would be with live trading
    """


    timestamp: datetime.datetime

    amount: Decimal


class BacktestSyncModel(SyncModel):
    """Backtest sync model.

    Simulate deposit events to the backtest wallet."""

    def __init__(self, wallet: SimulatedWallet, initial_deposit_amount: Decimal):
        assert isinstance(initial_deposit_amount, Decimal)
        self.wallet = wallet

        #: Simulated deposit/redemption events pending to be processed
        self.fund_flow_queue: List[FundFlowEvent] = [
            FundFlowEvent(datetime.datetime.utcnow(), initial_deposit_amount)
        ]

    def sync_initial(self, state: State):
        """Set up the initial sync details.

        For backtesting these are irrelevant.
        """
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
                 ) -> List[BalanceUpdate]:
        """Apply the balance sync before each strategy cycle.

        .. warning::

            Old legacy code with wrong return signature compared to the parent class
        """

        assert len(supported_reserves) == 1
        reserve_token = supported_reserves[0]
        balance_update_events = []

        for funding_event in self.fund_flow_queue:

            # Update wallet
            self.wallet.update_balance(reserve_token.address, funding_event.amount)

            # Generate a deposit event
            evt = ReserveUpdateEvent(
                asset=reserve_token,
                past_balance=Decimal(0),
                new_balance=self.wallet.get_balance(reserve_token.address),
                updated_at=strategy_cycle_ts
            )
            # Update state
            balance_update_events = apply_sync_events(state, [evt])

        # Clear our pending funding simulation events
        self.fund_flow_queue = []

        return balance_update_events

    def simulate_funding(self, timestamp: datetime.datetime, amount: Decimal):
        """Simulate a funding flow event.

        Call for the test to cause deposit or redemption for the backtest.
        The event goes to a queue and is processed in next `tick()`
        through `sync_portfolio()`.

        :param amount:
            Positive for deposit, negative for redemption

        """
        self.fund_flow_queue.append(
            FundFlowEvent(timestamp, amount)
        )

    def create_transaction_builder(self) -> None:
        """Backtesting does not need to care about how to build blockchain transactions."""
