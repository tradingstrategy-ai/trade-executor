"""Loan data structures.

- When you do short and leveraged position by borrowing assets

"""
import copy
import math
from _decimal import Decimal
from dataclasses import dataclass
from typing import TypeAlias

from dataclasses_json import dataclass_json

from tradeexecutor.state.identifier import AssetIdentifier, AssetWithTrackedValue, TradingPairIdentifier
from tradeexecutor.state.types import LeverageMultiplier, USDollarAmount

#: Health Factor: hF=dC/d, if lower than 1, the account can be liquidated
#:
#: Health factor is infinite for loans that do not borrow
#: (doing only credit supply for interest).
#:
HealthFactor: TypeAlias = float


class LiquidationRisked(Exception):
    """The planned loan health factor is too low.

    You would be immediately liquidated if the parameter
    changes would be applied.
    """


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

    def __repr__(self):
        return f"<Loan, borrowed ${self.borrowed.get_usd_value()} {self.borrowed.asset.token_symbol} for collateral ${self.collateral.get_usd_value()}, at leverage {self.get_leverage()}>"

    def clone(self) -> "Loan":
        """Clone this data structure for mutating."""
        return copy.deepcopy(self)

    def get_net_asset_value(self) -> USDollarAmount:
        """What's the withdrawable amount of the position is closed."""
        return self.collateral.get_usd_value() - self.borrowed.get_usd_value()

    def get_leverage(self) -> LeverageMultiplier:
        """TODO: From 1delta terminology"""
        return self.collateral.get_usd_value() / self.get_net_asset_value()

    def get_free_margin(self) -> USDollarAmount:
        raise NotImplementedError()

    def get_health_factor(self) -> HealthFactor:
        if self.borrowed.quantity == 0:
            return math.inf
        return self.collateral.asset.liquidation_threshold * self.collateral.get_usd_value() / self.borrowed.get_usd_value()

    def get_max_size(self) -> USDollarAmount:
        raise NotImplementedError()

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

        :param borrowed_quantity:
            What is expected outstanding loan amount

        :return:
            US dollars worth of collateral needed
        """
        borrowed_usd = self.borrowed.last_usd_price  * float(borrowed_quantity)
        usd_value = borrowed_usd/ target_ltv
        return Decimal(usd_value / self.collateral.last_usd_price)

    def calculate_collateral_for_target_leverage(
            self,
            leverage: LeverageMultiplier,
            borrowed_quantity: Decimal | float,
    ) -> Decimal:
        """Calculate the collateral amount we need to hit a target leverage.

        Assuming our debt stays the same, how much collateral we need
        to hit the target LTV.

        .. code-block:: text

            col / (col - borrow) = leverage
            col = (col - borrow) * leverage
            col = col * leverage - borrow * leverage
            col - col * leverage = - borrow * levereage
            col(1 - leverage) = - borrow * leverage
            col = -(borrow * leverage) / (1 - leverage)

        See also :py:func:`calculate_leverage_for_target_size`

        :param borrowed_quantity:
            What is expected outstanding loan amount

        :return:
            US dollars worth of collateral needed
        """
        borrowed_usd = self.borrowed.last_usd_price  * float(borrowed_quantity)
        usd_value = -(borrowed_usd * leverage) / (1 - leverage)
        return Decimal(usd_value / self.collateral.last_usd_price)

    def check_health(self, desired_health_factor=1):
        """Check if this loan is healthy.

        Health factor must stay above 1 or you get liquidated.

        :raise LiquidationRisked:
            If the loan would be instantly liquidated

        """
        health_factor = self.get_health_factor()
        if health_factor <= desired_health_factor:
            raise LiquidationRisked(
                f"You would be liquidated with health factor {health_factor}.\n"
                f"Desired health factor is {desired_health_factor}.\n"
                f"Collateral {self.collateral.get_usd_value()} USD.\n"
                f"Borrowed {self.borrowed.quantity} {self.borrowed.asset.token_symbol} {self.borrowed.get_usd_value()} USD.\n"
            )


def calculate_leverage_for_target_size(
    position_size: USDollarAmount,
    leverage: LeverageMultiplier,
) -> USDollarAmount:
    """Calculate the collateral amount we need to hit a target leverage when opening a position.

    :param position_size:
        How large is the position size in USC

    :return:
        US dollars worth of collateral needed
    """
    borrowed_usd = position_size * leverage
    usd_value = -(borrowed_usd * leverage) / (1 - leverage)
    return usd_value

