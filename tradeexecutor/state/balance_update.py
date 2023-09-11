"""Balance update data.

.. note ::

    These are not used by legacy wallet sync model, but only vault based wallets.

"""

import datetime
import enum
from _decimal import Decimal
from dataclasses import dataclass, field
from typing import Optional

from dataclasses_json import dataclass_json

from tradeexecutor.state.identifier import AssetIdentifier
from tradingstrategy.types import USDollarAmount, Percent


class BalanceUpdateCause(enum.Enum):

    #: Reserve was deposited in the vault
    deposit = "deposit"

    #: User redeemed assets
    redemption = "redemption"

    #: Position value has change due to accrued interest
    #:
    #:
    #:
    interest = "interest"

    #: Accounting correction from on-chain balances to the state (internal ledger)
    #:
    correction = "correction"


class BalanceUpdatePositionType(enum.Enum):
    reserve = "reserve"
    open_position = "open_position"


@dataclass_json
@dataclass
class   BalanceUpdate:
    """Processed balance update event.

    Events that are generated on

    - Deposits

    - Redemptions

    - Interest payments

    Events are stored in :py:class:`TradingPosition` and :py:class:`ReservePosition` by their id.

    Events are referred in :py:class:`tradeexecutor.sync.Treasury`..
    """

    #: Allocated from portfolio
    #:
    #: This id is referred in :py:class:`tradeexecutor.state.position.TradingPosition` and :py:class:`tradeexecutor.state.reserve.ReservePosition`
    balance_update_id: int

    #: What caused the balance update event to happen
    cause: BalanceUpdateCause

    #: What kind of position this event modified
    position_type: BalanceUpdatePositionType

    #: Asset that was updated
    #:
    #: If this an interest event, this is aToken/vToken asset
    #:
    asset: AssetIdentifier

    #: When the balance event was generated
    #:
    #: The block mined timestamp
    block_mined_at: datetime.datetime

    #: When balance event was included to the strategy's treasury.
    #:
    #: The strategy cycle timestamp.
    #:
    #: It might be outside the cycle frequency if treasuries were processed
    #: in a cron job outside the cycle for slow moving strategies.
    #:
    #: For accounting corrections this is set to `None`.
    #:
    strategy_cycle_included_at: datetime.datetime | None

    #: Chain that updated the balance
    chain_id: int

    #: What was delta of the asset.
    #:
    #: Positive for deposits, negative for redemptions.
    #:
    quantity: Decimal

    #: What was the total of the asset in the position before this event was applied.
    #:
    old_balance: Decimal

    #: How much this deposit/redemption was worth
    #:
    #: Used for deposit/redemption inflow/outflow calculation.
    #: This is the asset value from our internal price keeping at the time of the event.
    #:
    usd_value: USDollarAmount

    #: Wall clock time when this event was created
    #:
    created_at: datetime.datetime | None = field(default_factory=datetime.datetime.utcnow)

    #: What was the event time of the previous update.
    #:
    #: This allows us to calculate the effective interest rate
    #: between the update cycles.
    #:
    #: This is the same as :py:attr:`block_mined_at` of the previous event.
    #:
    previous_update_at: datetime.datetime | None = None

    #: Investor address that the balance update is related to
    #:
    owner_address: Optional[str] = None

    #: Transaction that updated the balance
    #:
    #: Set None for interested calculation updates
    tx_hash: Optional[str] = None

    #: Log that updated the balance
    #:
    #: Set None for interest rate updates
    log_index: Optional[int] = None

    #: If this update was for open position
    #:
    #: Set None for reserve updates
    position_id: Optional[int] = None

    #: Human-readable notes regarding this event
    #:
    notes: Optional[str] = None

    #: Block number related to the event.
    #:
    #: Not always available.
    #:
    block_number: int | None = None

    def __post_init__(self):
        assert self.quantity != 0, "Balance update cannot be zero: {self}"

        if self.previous_update_at:
            assert self.previous_update_at <= self.block_mined_at, f"Travelling back in time: {self.previous_update_at} - {self.block_mined_at}"

    def __repr__(self):
        if self.position_id:
            position_name = f"position #{self.position_id}"
        else:
            position_name = "strategy reserves"

        return f"<BalanceUpdate #{self.balance_update_id} {self.cause.name} {self.quantity} for {position_name} at block {self.block_mined_at and self.block_mined_at:,}>"

    def __eq__(self, other: "BalanceUpdate"):
        assert isinstance(other, BalanceUpdate), f"Got {other}"
        match self.cause:
            case BalanceUpdateCause.deposit:
                return self.chain_id == other.chain_id and self.tx_hash == other.tx_hash and self.log_index == other.log_index
            case BalanceUpdateCause.redemption:
                return self.chain_id == other.chain_id and self.tx_hash == other.tx_hash and self.log_index == other.log_index and self.asset == other.asset
            case _:
                raise RuntimeError("Unsupported")

    def __hash__(self):
        match self.cause:
            case BalanceUpdateCause.deposit:
                return hash((self.chain_id, self.tx_hash, self.log_index))
            case BalanceUpdateCause.redemption:
                return hash((self.chain_id, self.tx_hash, self.log_index, self.asset.address))
            case _:
                raise RuntimeError("Unsupported")

    def is_reserve_update(self) -> bool:
        """Return whether this event updates reserve balance or open position balance"""
        return self.position_type == BalanceUpdatePositionType.reserve

    def get_update_period(self) -> datetime.timedelta | None:
        """How long it was between this event and previous sync event.

        :return:
            None if only inital update made
        """
        if not self.previous_update_at:
            return None

        return (self.block_mined_at - self.previous_update_at)

    def get_effective_yearly_yield(self, year=datetime.timedelta(days=360)) -> Percent | None:
        """How much we are gaining % yearly.

        - Based on the this balance update and the previous balance update

        - Mostly useful for interest rate events

        - Calculated in tokens (exchange rate immune)

        :return:
            1-based interest.

            E.g. 1.02 for 2% yearly gained interest. 0.9 for 10% yearly paid interest.

            Positive if we are gaining interest, negative if we are paying interest.

            ``None``  if no update period available
        """

        period = self.get_update_period()
        if not period:
            return None
        gain = self.quantity / self.old_balance
        return float(gain) / (period / year)
