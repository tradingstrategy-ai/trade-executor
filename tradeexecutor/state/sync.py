""""Store information about caught up chain state.

- Treasury understanding is needed in order to reflect on-chain balance changes to the strategy execution

- Most treasury changes are deposits and redemptions

- Interest rate events also change on-chain treasury balances

- See :py:mod:`tradeexecutor.strategy.sync_model` how to on-chain treasuty
"""
import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Iterable, Dict

from dataclasses_json import dataclass_json

from tradingstrategy.chain import ChainId

from tradeexecutor.state.balance_update import BalanceUpdate, BalanceUpdateCause, BalanceUpdatePositionType


@dataclass_json
@dataclass
class Deployment:
    """Information for the strategy deployment.

    - Capture information about the vault deployment in the strategy's persistent state

    - This information can be later used to look up information (e.g deposit transactions)

    - This information can be later used to look up verify data
    """

    #: Which chain we are deployed
    chain_id: Optional[ChainId] = None

    #: Vault smart contract address
    #:
    #: For hot wallet execution, the address of the hot wallet
    address: Optional[str] = None

    #: When the vault was deployed
    #:
    #: Not available for hot wallet based strategies
    block_number: Optional[int] = None

    #: When the vault was deployed
    #:
    #: Not available for hot wallet based strategies
    tx_hash: Optional[str] = None

    #: UTC block timestamp of the vault deployment tx
    #:
    #: Not available for hot wallet based strategies
    block_mined_at: Optional[datetime.datetime] = None

    #: Vault name
    #:
    #: Enzyme vault name - same as vault toke name
    vault_token_name: Optional[str] = None

    #: Vault token symbol
    #:
    #: Enzyme vault name - same as vault toke name
    vault_token_symbol: Optional[str] = None


@dataclass_json
@dataclass
class BalanceEventRef:
    """Register the balance event in the treasury model."""

    balance_event_id: int

    updated_at: datetime.datetime

    cause: BalanceUpdateCause

    position_type: BalanceUpdatePositionType

    position_id: Optional[int]


@dataclass_json
@dataclass
class Treasury:
    """State of syncind deposits and redemptions from the chain.

    """

    #: Wall clock time. timestamp for which we run the last sync
    #:
    #: Wall clock time, at the beginning on the sync cycle.
    last_updated_at: Optional[datetime.datetime] = None

    #: The strategy cycle timestamp for which we run the last sync
    #:
    #: Wall clock time, at the beginning on the sync cycle.
    last_cycle_at: Optional[datetime.datetime] = None

    #: What is the last processed block for deposit
    #:
    #: 0 = not scanned yet
    last_block_scanned: Optional[int] = None

    #: List of refernces to all balance update events.
    #:
    #: The actual balance update content is stored on the position itself.
    balance_update_refs: List[BalanceEventRef] = field(default_factory=list)


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

    def is_initialised(self) -> bool:
        """Have we scanned the initial deployment event for the sync model."""
        return self.deployment.block_number is not None

