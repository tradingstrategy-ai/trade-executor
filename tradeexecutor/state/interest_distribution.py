"""Distribute gained asset interest across positions holding those assets.

Data structures used in interest distribution.
"""

import datetime

from dataclasses import dataclass, field, asdict
from decimal import Decimal
from pprint import pformat
from typing import Set, Dict, List
import logging

from dataclasses_json import dataclass_json

from tradeexecutor.state.identifier import AssetIdentifier, AssetWithTrackedValue, AssetFriendlyId
from tradeexecutor.state.loan import LoanSide
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.types import USDollarPrice, Percent
from tradingstrategy.types import PrimaryKey

logger = logging.getLogger(__name__)


@dataclass_json
@dataclass(slots=True)
class InterestDistributionEntry:
    """Map interest distribution across different trading positions.

    A helper class to help us to distribute accrued interest across different
    related trading positions.

    The lifetime of an instance is one sync_interests() run.
    """

    #: Which side we are on
    side: LoanSide

    #: The trading position containing this asset
    position: TradingPosition

    #: Can be either loan.collateral or loan.borrower
    #:
    #: This is a pass-by-reference copy from Loan instance
    #:
    tracker: AssetWithTrackedValue

    #: All weight are normalised to 0...1 based on loan token amount.
    weight: Decimal = None

    #: The updated price for the tracked asset
    #:
    #: To update US dollar based prices of interests,
    #: we need to know any change in the asset prices.
    #:
    price: USDollarPrice = None

    def __repr__(self):
        return f"<InterestDistributionEntry {self.side.name} {self.position.pair.get_ticker()} {self.weight * 100}%>"

    @property
    def asset(self) -> AssetIdentifier:
        """Amount of tracked asset in tokens"""
        return self.tracker.asset

    @property
    def quantity(self) -> Decimal:
        """Amount of tracked asset in tokens"""
        return self.tracker.quantity

    def get_diagnostics_data(self) -> str:
        """Get debug dump of this instance."""
        data = asdict(self)
        data["position"] = self.position.get_human_readable_name()  # Don't dump full position data
        return pformat(data)


@dataclass_json
@dataclass(slots=True)
class AssetInterestData:
    """Per-asset data we track in interest calculations."""

    #: Portfolio total quantity of interest bearing assets before the update.
    #:
    total: Decimal = field(default=Decimal(0))

    #: Calculated the effective interest rate for this asset.
    #:
    #: What was the rate based on the operation duration and on-chain balance change.
    #:
    effective_rate: Percent = None

    #: All entries that appleid for this asset
    entries: list[InterestDistributionEntry] = field(default_factory=list)


@dataclass_json
@dataclass(slots=True)
class InterestDistributionOperation:
    """One interest update batch we do."""

    #: Starting period of time span for which we calculate the interest.
    #:
    #: Timestamp of the previously synced block.
    #:
    start: datetime.datetime

    #: Ending period of time span for which we calculate the interest
    #:
    #: Timestamp of the block end range
    #:
    end: datetime.datetime

    #: All interest bearing assets we have across positions
    assets: Set[AssetIdentifier]

    #: Asset interest data entries keyed (chain id, address) tuples
    #:
    asset_interest_data: Dict[AssetFriendlyId, AssetInterestData]

    #: All entries we need to update.
    #:
    #: One or two entries per position accruing interest.
    #:
    entries: List[InterestDistributionEntry]

    #: Not used
    effective_rate: Dict[int, Percent]

    @property
    def duration(self) -> datetime.timedelta:
        """The time span for which we updated the accrued interest."""
        return self.end - self.start

    def get_interest_data(self, asset: AssetIdentifier) -> AssetInterestData | None:
        return self.asset_interest_data.get(asset.get_identifier())

    def get_diagnostics_data(self) -> str:
        """Get debug dump of this instance."""
        data = asdict(self)
        data["asset_interest_data"] = f"{len(data['asset_interest_data'])} entries"
        data["entries"] = f"{len(data['entries'])} entries"
        return pformat(data)
