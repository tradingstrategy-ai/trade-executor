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

    #: How many tokens we had at the previous read
    # TODO: not sure if this is needed
    #:
    last_token_amount: Decimal

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
    last_updated_block_number: int | None = None

    def __post_init__(self):
        assert isinstance(self.opening_amount, Decimal)
        assert isinstance(self.last_accrued_interest, Decimal)


