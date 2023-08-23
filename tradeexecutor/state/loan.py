"""Loan data structures.

- When you do short and leveraged position by borrowing assets

"""
import copy
from _decimal import Decimal
from dataclasses import dataclass
from typing import TypeAlias

from dataclasses_json import dataclass_json

from tradeexecutor.state.identifier import AssetIdentifier, AssetWithTrackedValue, TradingPairIdentifier
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

    - `leverage calculations <https://amplifiedlambda.com/leverage-with-defi-calculations/>`__
    """

    #: Our trading pair data for this position
    pair: TradingPairIdentifier

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

    def clone(self) -> "Loan":
        """Clone this data structure for mutating."""
        return copy.deepcopy(self)

    def get_net_asset_value(self) -> USDollarAmount:
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

    def get_collateral_factor(self) -> float:
        return self.pair.collateral_factor

    def get_loan_to_value(self):
        """Get LTV of this loan.

        LTV should stay below the liquidation threshold.
        For Aave ETH the liquidation threshold is 80%.
        """
        return self.borrowed.get_usd_value() / self.collateral.get_usd_value()

    def calculate_collateral_for_target_ltv(
            self,
            target_ltv: float,
            borrowed_quantity: Decimal | float,
    ) -> Decimal:
        """Calculate the collateral amount we need to hit a target LTV.

        Assuming our debt stays the same, how much collateral we need
        to hit the target LTV.

        .. note ::

            Watch out for rounding/epsilon errors.

        :return:
            US dollars worth of collateral needed
        """
        borrowed_usd = self.borrowed.last_usd_price  * float(borrowed_quantity)
        usd_value = borrowed_usd/ target_ltv
        return Decimal(usd_value / self.collateral.last_usd_price)


