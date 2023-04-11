"""Strategy deposit and withdrawal syncing."""

import datetime
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from tradeexecutor.ethereum.tx import TransactionBuilder
from tradeexecutor.ethereum.wallet import ReserveUpdateEvent
from tradeexecutor.state.balance_update import BalanceUpdate, BalanceUpdateCause, BalanceUpdatePositionType
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.state import State
from tradeexecutor.state.sync import BalanceEventRef

# Prototype sync method that is not applicable to the future production usage
SyncMethodV0 = Callable[[Portfolio, datetime.datetime, List[AssetIdentifier]], List[ReserveUpdateEvent]]


class SyncModel(ABC):
    """Abstract class for syncing on-chain fund movements event to the strategy treasury."""

    def get_vault_address(self) -> Optional[str]:
        """Get the vault address we are using"""
        return None

    @abstractmethod
    def sync_initial(self, state: State):
        """Initialize the vault connection."""
        pass

    @abstractmethod
    def sync_treasury(self,
                 strategy_cycle_ts: datetime.datetime,
                 state: State,
                 supported_reserves: Optional[List[AssetIdentifier]] = None
                 ) -> List[BalanceUpdate]:
        """Apply the balance sync before each strategy cycle.

        :param strategy_cycle_ts:
            The current strategy cycle.

            Resevers are synced before executing the strategy cycle.

        :param state:
            Current state of the execution.

        :param supported_reverses:
            List of assets the strategy module wants to use as its reserves.

            May be None in testing.

        :return:
            List of balance updates detected.

            - Deposits

            - Redemptions

        """

    @abstractmethod
    def create_transaction_builder(self) -> Optional[TransactionBuilder]:
        """Creates a transaction builder instance to make trades against this asset management model.

        Only needed when trades are being executed.

        :return:
            Depending on the asset management mode.

            - :py:class:`tradeexecutor.ethereum.tx.HotWalletTransactionBuilder`

            - :py:class:`tradeexecutor.ethereum.enzyme.tx.EnzymeTransactionBuilder`
        """


class DummySyncModel(SyncModel):
    """Do nothing sync model.

    - There is no integration to external systems

    - Used in unit testing
    """

    def __init__(self):
        self.fake_sync_done = False

    def sync_initial(self, state: State):
        pass

    def sync_treasury(self,
                 strategy_cycle_ts: datetime.datetime,
                 state: State,
                 supported_reserves: Optional[List[AssetIdentifier]] = None
                 ) -> List[BalanceUpdate]:
        if not self.fake_sync_done:
            state.sync.treasury.last_updated_at = datetime.datetime.utcnow()
            state.sync.treasury.last_cycle_at = strategy_cycle_ts
            ref = BalanceEventRef(0, datetime.datetime.utcnow(), BalanceUpdateCause.deposit, BalanceUpdatePositionType.open_position, None)
            state.sync.treasury.balance_update_refs = [ref]  # TODO: Get rid of the checks in on_clock()
            return []
        return []

    def create_transaction_builder(self) -> Optional[TransactionBuilder]:
        return None