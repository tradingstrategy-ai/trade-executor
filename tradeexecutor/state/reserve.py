import datetime
from dataclasses import dataclass
from decimal import Decimal

from dataclasses_json import dataclass_json

from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.types import USDollarAmount


@dataclass_json
@dataclass
class ReservePosition:
    asset: AssetIdentifier
    quantity: Decimal
    last_sync_at: datetime.datetime

    reserve_token_price: USDollarAmount
    last_pricing_at: datetime.datetime

    def __post_init__(self):
        assert self.asset.decimals > 0, f"Looks like we have inproper reserve asset: {self.asset}"

    def get_identifier(self) -> str:
        return self.asset.get_identifier()

    def get_current_value(self) -> USDollarAmount:
        return float(self.quantity) * self.reserve_token_price