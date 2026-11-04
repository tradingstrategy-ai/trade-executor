"""Observed, uninvested USDC in a Safe-owned Lighter pool account."""

import datetime
from dataclasses import dataclass
from decimal import Decimal

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class LighterIdleCollateral:
    """Cash included in NAV, but unavailable to Safe-funded trades.

    This observation belongs to the executor state. It is not an exchange
    account position and does not include pool shares or withdrawal claims.
    """

    custody_chain_id: int
    safe_address: str
    account_index: int
    amount: Decimal
    observed_at: datetime.datetime
    valid: bool = True

    def get_identifier(self) -> str:
        """Stable public identity, independent of API-key rotation."""
        return f"{self.custody_chain_id}:{self.safe_address.lower()}:{self.account_index}"
