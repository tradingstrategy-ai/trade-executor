"""Interest tracking data structures."""
import datetime
from dataclasses import dataclass
from decimal import Decimal

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass(slots=True)
class Interest:
    """Interest data tracking for positions that where the amount changes over time.

    - Credit positions (depositing in Aave reservess)

    - Longs / shorts
    """

    #: How many tokens we deposited to this position at amount.
    #:
    #: Use this only for calculation verifications,
    #: because the amount can be increased/reduced over time
    #:
    opening_amount: Decimal

    #: How many atokens/votkens we had on the previous read.
    #:
    #: This is principal + interest.
    #:
    last_atoken_amount: Decimal

    #: When the denormalised data was last updated.
    #:
    #: Wall clock time.
    #:
    last_updated_at: datetime.datetime

    #: When the denormalised data was last updated.
    #:
    #: Event time (block mined timestamp).
    #:
    last_event_at: datetime.datetime

    #: How much interest we have gained
    #:
    #:
    last_accrued_interest: Decimal

    #: Block number for the update
    #:
    #: When was the last time we read aToken balance.
    #:
    last_updated_block_number: int | None = None

    def __repr__(self):
        return f"<Interest, current principal + interest {self.last_atoken_amount}>"

    def __post_init__(self):
        assert isinstance(self.opening_amount, Decimal)
        assert isinstance(self.last_accrued_interest, Decimal)

    @staticmethod
    def open_new(opening_amount: Decimal) -> "Interest":
        assert opening_amount > 0
        return Interest(
            opening_amount=opening_amount,
            last_updated_at=datetime.datetime.utcnow(),
            last_event_at=datetime.datetime.utcnow(),
            last_accrued_interest=Decimal(0),
            last_atoken_amount=opening_amount,
        )


