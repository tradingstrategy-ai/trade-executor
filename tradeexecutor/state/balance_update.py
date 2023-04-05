"""Balance update data.

.. note ::

    These are not used by legacy wallet sync model, but only vault based wallets.

"""

import datetime
import enum
from _decimal import Decimal
from dataclasses import dataclass
from typing import Optional

from dataclasses_json import dataclass_json

from tradeexecutor.state.identifier import AssetIdentifier


class BalanceUpdateType(enum.Enum):
    deposit = "deposit"
    redemption = "redemption"
    interest = "interest"


class BalanceUpdatePositionType(enum.Enum):
    reserve = "reserve"
    open_position = "open_position"


@dataclass_json
@dataclass
class BalanceUpdate:
    """Processed balance update event.

    Events that are generated on

    - Deposits

    - Redemptions

    - Interest payments

    Events are stored in :py:class:`tradeexecutor.sync.Treasury`.
    They are referred in :py:class:`TradingPosition` and :py:class:`ReservePosition` by their id.
    """

    #: Allocated from portfolio
    #:
    #: This id is referred in :py:class:`tradeexecutor.state.position.TradingPosition` and :py:class:`tradeexecutor.state.reserve.ReservePosition`
    balance_update_id: int

    type: BalanceUpdateType

    position_type: BalanceUpdatePositionType

    #: Asset that was updated
    #:
    #:
    asset: AssetIdentifier

    #: When the update happened
    #:
    #: The block mined timestamp
    block_mined_at: datetime.datetime

    #: Chain that updated the balance
    chain_id: int

    #: What was the position balance before update
    #:
    past_quantity: Decimal

    #: What was the position balance after update
    #:
    new_quantity: Decimal

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

    def __eq__(self, other: "BalanceUpdate"):
        assert isinstance(other, BalanceUpdate), f"Got {other}"
        match self.type:
            case BalanceUpdateType.deposit:
                return self.chain_id == other.chain_id and self.tx_hash == other.tx_hash and self.log_index == other.log_index
            case BalanceUpdateType.redemption:
                return self.chain_id == other.chain_id and self.tx_hash == other.tx_hash and self.log_index == other.log_index and self.asset == other.asset
            case _:
                raise RuntimeError("Unsupported")

    def __hash__(self):
        match self.type:
            case BalanceUpdateType.deposit:
                return hash((self.chain_id, self.tx_hash, self.log_index))
            case BalanceUpdateType.redemption:
                return hash((self.chain_id, self.tx_hash, self.log_index, self.asset.address))
            case _:
                raise RuntimeError("Unsupported")

    def is_reserve_update(self) -> bool:
        """Return whether this event updates reserve balance or open position balance"""
        return self.position_type == BalanceUpdatePositionType.reserve

    @property
    def quantity(self) -> Decimal:
        """How much this event modified the position balance."""
        return self.new_quantity - self.past_quantity
