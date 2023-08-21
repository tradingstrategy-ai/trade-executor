"""Loan management.

When you do a leveraged position by borrowing assets.
"""
from decimal import Decimal
from dataclasses import dataclass
from typing import TypeAlias

import dataclasses_json
from dataclasses_json import dataclass_json

from tradeexecutor.state.identifier import AssetIdentifier, AssetWithTrackedValue
from tradeexecutor.state.types import LeverageMultiplier, USDollarAmount

#: Health Factor: hF=dC/d, if lower than 1, the account can be liquidated
HealthFactor: TypeAlias = float


@dataclass_json
@dataclass
class Loan:
    """Borrowed out assets.

    See also

    - :py:class:`tradeexcutor.state.position.TradingPosition` for tracking a long/short

    - :py:class:`tradeexcutor.state.interest.Interest` for position interest calculations

    - `1delta documentation <https://docs.1delta.io/lenders/metrics>`__
    """

    #: What collateral we used for this loan
    #:
    #: This is aToken for Aave
    #:
    collateral: AssetWithTrackedValue

    #: What collateral we used for this loan
    #:
    #: This is vToken for Aave
    #:
    borrowed: AssetWithTrackedValue

    def get_nav(self) -> USDollarAmount:
        """TODO: From 1delta terminology"""
        return self.collateral.get_usd_value() - self.borrowed.get_usd_value()

    def get_leverage(self) -> LeverageMultiplier:
        """TODO: From 1delta terminology"""
        return self.collateral.get_usd_value() / self.get_nav()

    def get_free_margin(self) -> USDollarAmount:
        raise NotImplementedError()

    def get_health_factor(self) -> HealthFactor:
        raise NotImplementedError()

    def get_max_size(self) -> USDollarAmount:
        raise NotImplementedError()