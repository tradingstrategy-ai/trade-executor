"""Strategy reserve currency management."""

import datetime
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from dataclasses_json import dataclass_json

from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.types import USDollarAmount


@dataclass_json
@dataclass
class ReservePosition:
    """Manage reserve currency of a portfolio.

    - One portfolio can have multiple reserve currencies,
      but currently the code is simplified to handle only one reserve currency

    """

    #: What is our reserve currency
    asset: AssetIdentifier

    #: How much reserves we have currently
    quantity: Decimal

    #: When we processed deposits/withdraws last time
    last_sync_at: datetime.datetime

    #: What was the US dollar exchange rate of our reserves
    reserve_token_price: USDollarAmount

    #: When we fetched the US dollar exchange rate of our reserves last time
    last_pricing_at: datetime.datetime

    #: What was the first deposit amount.
    #: Used to shortcut the backtest performance benchmark.
    #: TODO: Remove optional in future versions.
    initial_deposit: Optional[Decimal] = None

    #: What was the first deposit exchange rate.
    #: Used to shortcut the backtest performance benchmark.
    #: TODO: Remove optional in future versions.
    initial_deposit_reserve_token_price: Optional[USDollarAmount] = None

    def __post_init__(self):
        assert self.asset.decimals > 0, f"Looks like we have inproper reserve asset: {self.asset}"

    def get_identifier(self) -> str:
        return self.asset.get_identifier()

    def get_current_value(self) -> USDollarAmount:
        return float(self.quantity) * self.reserve_token_price