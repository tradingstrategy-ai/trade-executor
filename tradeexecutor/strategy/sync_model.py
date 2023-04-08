"""Strategy deposit and withdrawal syncing."""

import datetime
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from tradeexecutor.ethereum.wallet import ReserveUpdateEvent
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.state import State


# Prototype sync method that is not applicable to the future production usage
SyncMethodV0 = Callable[[Portfolio, datetime.datetime, List[AssetIdentifier]], List[ReserveUpdateEvent]]


class SyncModel(ABC):
    """Abstract class for syncing on-chain fund movements event to the strategy treasury."""

    def get_vault_address(self) -> Optional[str]:
        """Get the vault address we are using"""

    @abstractmethod
    def sync_initial(self, state: State):
        """Initialize the vault connection."""
        pass

    @abstractmethod
    def sync_treasury(self,
                 strategy_cycle_ts: datetime.datetime,
                 state: State,
                 supported_reserves: Optional[List[AssetIdentifier]] = None
                 ):
        """Apply the balance sync before each strategy cycle.

        :param strategy_cycle_ts:
            The current strategy cycle.

            Resevers are synced before executing the strategy cycle.

        :param state:
            Current state of the execution.

        :param supported_reverses:
            List of assets the strategy module wants to use as its reserves.

            May be None in testing.
        """


