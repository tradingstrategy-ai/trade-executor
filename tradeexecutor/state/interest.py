"""Interest tracking."""
import datetime
from dataclasses import dataclass
from decimal import Decimal

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass(slots=True, frozen=True)
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

    #: When the denormalised data was last updated.
    #:
    #: Wall clock time.
    #:
    last_amount_updated_at: datetime.datetime

    #: How much interest we have gained
    #:
    #:
    last_accrued_interest: Decimal

    def __post_init__(self):
        assert isinstance(self.opening_amount, Decimal)
        assert isinstance(self.last_accrued_interest, Decimal)