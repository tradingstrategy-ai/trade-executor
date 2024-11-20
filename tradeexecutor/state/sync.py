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

from tradeexecutor.state.interest_distribution import InterestDistributionOperation
from tradeexecutor.state.types import USDollarAmount, BlockNumber
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

    #: When the initialisation was complete
    #:
    initialised_at: Optional[datetime.datetime] = None

    def __repr__(self):
        return f"<Deployment chain:{self.chain_id.name} address:{self.address} name:{self.vault_token_name} token:{self.vault_token_symbol}>"


@dataclass_json
@dataclass
class BalanceEventRef:
    """Register the balance event in the treasury model.

    Balance updates can happen for

    - Treasury

    - Open trading positions

    We maintain a list of balance update references across all positions using :py:class:`BalanceEventRef`.
    This allows us quickly to calculate net inflow/outflow.
    """

    #: Balance event id we are referring to
    balance_event_id: int

    #: When this update was made.
    #:
    #: Strategy cycle timestamp when the deposit/redemption was included
    #: in the strategy treasury.
    #:
    #: It might be outside the cycle frequency if treasuries were processed
    #: in a cron job outside the cycle for slow moving strategies.
    #:
    strategy_cycle_included_at: Optional[datetime.datetime]

    #: Cause of the event
    cause: BalanceUpdateCause

    #: Reserve currency or underlying position
    position_type: BalanceUpdatePositionType

    #: Which trading positions were affected
    position_id: Optional[int]

    #: How much this deposit/redemption was worth
    #:
    #: Used for deposit/redemption inflow/outflow calculation.
    #: This is the asset value from our internal price keeping at the time of the event.
    #:
    usd_value: Optional[USDollarAmount]

    @staticmethod
    def from_balance_update_event(evt: BalanceUpdate) -> "BalanceEventRef":
        """Create a reference to a balance update event stored elsewhere in the state."""
        ref = BalanceEventRef(
            balance_event_id=evt.balance_update_id,
            strategy_cycle_included_at=evt.strategy_cycle_included_at,
            cause=evt.cause,
            position_type=evt.position_type,
            position_id=evt.position_id,
            usd_value=evt.usd_value,
        )
        return ref

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

    #: List of refences to all balance update events.
    #:
    #: The actual balance update content is stored on the position itself.
    balance_update_refs: List[BalanceEventRef] = field(default_factory=list)

    def __repr__(self):
        return f"<Treasury updated:{self.last_updated_at} cycle:{self.last_cycle_at} block scanned:{self.last_block_scanned or 0:,} refs:{len(self.balance_update_refs)}>"



@dataclass_json
@dataclass
class Accounting:
    """State of accounting corrections.

    """

    #: Wall clock time. timestamp for which we run the last sync
    #:
    #: Wall clock time, at the beginning on the sync cycle.
    last_updated_at: Optional[datetime.datetime] = None

    #: What is the last processed block for deposit
    #:
    #: 0 = not scanned yet
    last_block_scanned: Optional[int] = None

    #: List of refernces to all balance update events.
    #:
    #: The actual balance update content is stored on the position itself.
    balance_update_refs: List[BalanceEventRef] = field(default_factory=list)



@dataclass_json
@dataclass(slots=True)
class InterestSync:
    """Track the interest sync for on-chain rebase tokens.
    """

    #: When did we perform the last interest sync all all assets
    #:
    last_sync_at: datetime.datetime | None = None

    #: Block number when we synced the portfolio
    #:
    #: Backtesting does not use block numbers and has this always set to ``None``.
    #:
    last_sync_block: BlockNumber | None = None

    #: Last operation we did.
    #:
    #: Data is not needed. Stored only for diagnostics purposes.
    #: Always overwritten in the next sync.
    #:
    last_distribution: InterestDistributionOperation | None = None


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

    accounting: Accounting = field(default_factory=Accounting)

    interest: InterestSync = field(default_factory=InterestSync)

    def is_initialised(self) -> bool:
        """Have we scanned the initial deployment event for the sync model."""
        return self.deployment.block_number is not None

