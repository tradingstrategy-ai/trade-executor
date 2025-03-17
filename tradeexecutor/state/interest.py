"""Interest tracking data structures."""
import logging
import datetime
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict

from dataclasses_json import dataclass_json

from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.types import BlockNumber
from tradeexecutor.utils.accuracy import ZERO_DECIMAL, QUANTITY_EPSILON, INTEREST_QUANTITY_EPSILON


logger = logging.getLogger(__name__)


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
    #: Absolute number of on-chain balance.
    #:
    #: This is principal + interest.
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
    #: When was the last time we read aToken balance.
    #:
    last_updated_block_number: int | None = None

    #: How much repayments this loan has received.
    #:
    #: - If this is collateral, then this is how much interest we have claimed
    #:
    #: - If this is borrow, then this is how much interest we have paid back
    #:
    #: TODO: This must be reset when there is change to underlying aToken/vToken amount
    #  e.g. when partially closing a position.
    #:
    interest_payments: Decimal = ZERO_DECIMAL

    #: If we repair/reset this interest tracked, when this happened.
    #:
    reset_at: datetime.datetime | None = None

    def __repr__(self):
        return f"<Interest, current principal + interest {self.last_token_amount}, current tracked interest gains {self.last_accrued_interest}>"

    def __post_init__(self):
        assert isinstance(self.opening_amount, Decimal)
        assert isinstance(self.last_accrued_interest, Decimal)

    def get_principal_and_interest_quantity(self) -> Decimal:
        """Return how many tokens exactly we have on the loan.

        Assuming any aToken/vToken will be fully converted to the underlying.
        """
        return self.last_token_amount

    @staticmethod
    def open_new(opening_amount: Decimal, timestamp: datetime.datetime) -> "Interest":
        assert opening_amount > 0
        return Interest(
            opening_amount=opening_amount,
            last_updated_at=timestamp,
            last_event_at=timestamp,
            last_accrued_interest=Decimal(0),
            last_token_amount=opening_amount,
        )

    def get_remaining_interest(self) -> Decimal:
        """GEt the amount of interest this position has still left.

        This is total lifetime interest + repayments / claims.
        """
        return self.last_accrued_interest - self.interest_payments

    def claim_interest(self, quantity: Decimal):
        """Update interest claims from profit from accuring interest on collateral/"""
        self.interest_payments += quantity

    def repay_interest(self, quantity: Decimal):
        """Update interest payments needed to maintain the borrowed debt."""
        self.interest_payments += quantity

    def adjust(self, delta: Decimal, epsilon: Decimal = INTEREST_QUANTITY_EPSILON):
        """Adjust the quantity on this loan.

        Used when doing increase/reduce shorts to get a new amount.
        With safety checks.

        :param delta:
            Positive: increase amount, negative decrease amount.

        :param epsilon:
            Consider
        """


        if delta < 0 and abs(delta) < epsilon:
            logger.warning(
                "Ignoring negative change in the interest amount. We are: %s, delta %s, epsilon %s",
                self,
                delta,
                epsilon,
            )
            delta = 0

        self.last_token_amount += delta

        if abs(self.last_token_amount) < epsilon:
            self.last_token_amount = ZERO_DECIMAL

        assert self.last_token_amount >= 0, f"last_token_amount cannot go negative. Got {self.last_token_amount} on {self}, delta was {delta}, epsilon was {epsilon}"

    def reset(
        self,
        amount: Decimal,
        block_timestamp: datetime.datetime,
        block_number: BlockNumber,
    ):
        """Reset the loan token amount.

        - Used in the manual account correction

        """
        self.last_token_amount = amount
        self.last_updated_at = datetime.datetime.utcnow()
        self.last_event_at = block_timestamp
        self.last_accrued_interest = Decimal(0)
        self.last_updated_block_number = block_number
        self.interest_payments = Decimal(0)
        self.reset_at = datetime.datetime.utcnow()
        assert self.last_token_amount >= 0, f"last_token_amount cannot go negative. Got {self.last_token_amount} on {self}, delta was {delta}, epsilon was {epsilon}"




