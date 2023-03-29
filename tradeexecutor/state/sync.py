""""Store information about caught up chain state."""
from abc import ABC, abstractmethod
import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Protocol

from dataclasses_json import dataclass_json

from tradeexecutor.ethereum.wallet import ReserveUpdateEvent
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.state import State

from tradingstrategy.chain import ChainId


@dataclass_json
@dataclass
class Deployment:
    """Information for the strategy deployment.

    - Capture information about the vault deployment in the strategy's persistent state

    - This information can be later used to look up information (e.g deposit transactions)

    - This information can be later used to look up verify data
    """

    #: Which chain we are deployed
    chain_id: ChainId

    #: Vault smart contract address
    #:
    #: For hot wallet execution, the address of the hot wallet
    address: str

    #: When the vault was deployed
    #:
    #: Not available for hot wallet based strategies
    deployment_block_number: Optional[int]

    #: When the vault was deployed
    #:
    #: Not available for hot wallet based strategies
    deployment_transaction: Optional[str]

    #: UTC block timestamp of the vault deployment tx
    #:
    #: Not available for hot wallet based strategies
    deployment_timestamp: Optional[datetime.datetime]


@dataclass_json
@dataclass
class Treasury:
    """State of syncind deposits and redemptions from the chain.

    """

    #: The strategy cycle timestamp for which we run the last sync
    #:
    last_updated: Optional[datetime.datetime] = None

    #: What is the last processed block for deposit
    last_scanned_block_for_deposits: int = 0

    #: What is the last processed block for redempetions
    last_scanned_block_for_redemptions: int = 0

    #: List of Solidity deposit/withdraw events that we have correctly accounted in the strategy balances.
    #:
    #: Contains Solidity event logs for processed transactions
    processed_events: List[dict] = field(default_factory=list)


@dataclass_json
@dataclass
class Sync:
    """On-chain sync state.

    - Store persistent information about the vault on transactions we have synced,
      so that the strategy knows its available capital

    - Updated before the strategy execution step
    """

    deployment: Deployment = field(default_factory=Deployment)

    treasury: Treasury = field(default_factory=Treasury)


# Prototype sync method that is not applicable to the future production usage
SyncMethodV0 = Callable[[Portfolio, datetime.datetime, List[AssetIdentifier]], List[ReserveUpdateEvent]]


class SyncMethod(ABC):
    """Describe the sync adapter."""

    @abstractmethod
    def sync_initial(self, state: State):
        """Initialize the vault connection."""
        pass

    @abstractmethod
    def sync_treasuty(self,
                 strategy_cycle_ts: datetime.datetime,
                 state: State,
                 ):
        """Apply the balance sync before each strategy cycle."""
        pass
