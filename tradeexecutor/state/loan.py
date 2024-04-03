"""Loan data structures.

- When you do short and leveraged position by borrowing assets

"""
import copy
import enum
import math
from _decimal import Decimal
from dataclasses import dataclass
from typing import TypeAlias, Tuple, Literal

from dataclasses_json import dataclass_json

from tradeexecutor.state.identifier import (
    AssetIdentifier, AssetWithTrackedValue, TradingPairIdentifier, 
    TradingPairKind,
)
from tradeexecutor.state.interest import Interest
from tradeexecutor.state.types import LeverageMultiplier, USDollarAmount, USDollarPrice
from tradeexecutor.utils.accuracy import ZERO_DECIMAL, ensure_exact_zero
from tradeexecutor.utils.leverage_calculations import LeverageEstimate

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


class LoanSide(enum.Enum):

    collateral = "collateral"

    borrowed = "borrowed"


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

    #: Tracker for collateral interest events
    #:
    collateral_interest: Interest

    #: What collateral we used for this loan
    #:
    #: This is vToken for Aave.
    #:
    #: Not set if the loan is only for credit supply position.
    #:
    borrowed: AssetWithTrackedValue | None = None

    #: Tracker for borrowed asset interest events
    #:
    borrowed_interest: Interest | None = None

    def __repr__(self):
        asset_symbol = self.borrowed.asset.token_symbol if self.borrowed else ""
        return f"<Loan, borrowed {self.get_borrowed_quantity()} {asset_symbol} ${self.get_borrow_value()} for collateral ${self.get_collateral_value()}, at leverage {self.get_leverage()}, borrow price: {self.borrowed.last_usd_price if self.borrowed else 0}, collateral price: {self.collateral.last_usd_price}>"

    def clone(self) -> "Loan":
        """Clone this data structure for mutating.

        Used when increasing/reducing shorts,
        as we copy the existing on-chain data
        and then apply our delta in :py:func:`tradeexecutor.state.position.TradingPosition.open_trade`
        on the top of this.
        """
        return copy.deepcopy(self)

    def get_tracked_asset(self, asset: AssetIdentifier) -> Tuple[LoanSide | None, AssetWithTrackedValue | None]:
        """Get one side of the loan.

        :param asset:
            Asset this loan is tracking

        :return:
            Colleteral tracker, borrowed tracked or ``None`` if this loan does not track an asset.
        """

        if asset == self.collateral.asset:
            return LoanSide.collateral, self.collateral
        elif asset == self.borrowed.asset:
            return LoanSide.borrowed, self.borrowed
        else:
            return None, None

    def get_collateral_interest(self) -> USDollarAmount:
        """How much interest we have received on collateral."""
        return float(self.collateral_interest.get_remaining_interest()) * self.collateral.last_usd_price

    def get_collateral_value(self, include_interest=True) -> USDollarAmount:
        """How much value the collateral for this loan has.

        .. warning::

            TODO: Does not account for repaid interest at this point.
            Please use TradingPosition functions to get amounts with repaid interest.

        """
        if include_interest:
            return self.collateral.get_usd_value() + self.get_collateral_interest()
        return self.collateral.get_usd_value()

    def get_collateral_quantity(self) -> Decimal:
        """Get abs number of atokens we have."""
        return self.collateral_interest.last_token_amount

    def get_borrowed_quantity(self) -> Decimal:
        """Get abs number of vtokens we have."""
        if not self.borrowed:
            return 0
        return self.borrowed_interest.last_token_amount

    def get_borrow_value(self, include_interest=True) -> USDollarAmount:
        """Get the outstanding debt amount.

        :param include_interest:
            With interest.
        """
        if not self.borrowed:
            return 0
        if include_interest:
            return self.borrowed.get_usd_value() + self.get_borrow_interest()
        return self.borrowed.get_usd_value()

    def get_borrow_interest(self) -> USDollarAmount:
        """How much interest we have paid on borrows.

        :return:
            Always positive
        """
        if self.borrowed:
            return float(self.borrowed_interest.get_remaining_interest()) * self.borrowed.last_usd_price
        return 0

    def get_borrowed_principal_and_interest_quantity(self) -> Decimal:
        """Get how much borrow there is left to repay.

        - Round to zero
        """
        total = self.borrowed.quantity + self.borrowed_interest.last_accrued_interest
        return ensure_exact_zero(total)

    def get_net_interest(self) -> USDollarAmount:
        """How many dollars of interest we have accumulated.

        - We gain money on collateral

        - We lost money by maintaining borrow

        """
        if self.borrowed:
            # Margined trading
            return self.get_collateral_interest() - self.get_borrow_interest()
        else:
            # Credit supply
            return self.get_collateral_interest()

    def get_net_asset_value(self, include_interest=True) -> USDollarAmount:
        """What's the withdrawable amount of the position is closed.

        .. warning ::

            For closed position the behavior here is a bit weird.

            The value reflects any reminder of interest that was paid
            off when the position was closed. It is negative if borrowing costed
            more than collateral interest gained.

            This is later fixed in :py:meth:`TradingPosition.get_claimed_interest()`
            and the position accounts the difference in the final trade that pays
            principal + interest back.

        :return:
            The current net asset value or remaining interest that was paid off when the position was closed.
        """

        if self.borrowed:
            # Margined trading
            return self.get_collateral_value(include_interest) - self.get_borrow_value(include_interest)
        else:
            # Credit supply
            return self.get_collateral_value(include_interest)

    def get_leverage(self) -> LeverageMultiplier:
        """How leveraged this loan is.

        Using formula ``(collateral / (collateral - borrow))``.

        :return:
            Zero if the loan has zero net asset value.
        """

        if self.get_net_asset_value() == 0 or not self.borrowed:
            return 0

        return self.borrowed.get_usd_value() / self.get_net_asset_value()

    def get_free_margin(self) -> USDollarAmount:
        raise NotImplementedError()

    def get_health_factor(self) -> HealthFactor:
        """Get loan health factor.

        Safety of your deposited collateral against the borrowed assets and its underlying value.

        If the health factor goes below 1, the liquidation of your collateral might be triggered.
        """
        if self.borrowed is None or self.get_borrowed_principal_and_interest_quantity() == 0:
            # Already closed
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

    def claim_interest(self, quantity: Decimal | None = None) -> Decimal:
        """Claim intrest from this position.

        Interest should be moved to reserves.

        :param quantity:
            How many reserve tokens worth to claim.

            If not given claim all accrued interest.

        :return:
            Claimed interest in reserve tokens
        """

        if not quantity:
            quantity = self.collateral_interest.last_accrued_interest

        self.collateral_interest.claim_interest(quantity)

        return quantity

    def repay_interest(self, quantity: Decimal | None = None) -> Decimal:
        """Repay interest for this position.

        Pay any open interest on vToken position.

        :param quantity:
            How many vTokens worth of interest we pay.

            If not given assume any remaining open interest.

        :return:
            Repaid interest.
        """

        if not quantity:
            quantity = self.borrowed_interest.last_accrued_interest

        self.borrowed_interest.repay_interest(quantity)

        return quantity

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
        usd_value = borrowed_usd / target_ltv
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

            nav = col - borrow
            leverage = borrow / nav
            leverage = col / nav - 1

            borrow = nav * leverage
            col = borrow + nav
            col = borrow + borrow / leverage
            col = borrow * (1 + 1 / leverage)

        See also :py:func:`calculate_leverage_for_target_size`

        :param borrowed_quantity:
            What is expected outstanding loan amount

        :return:
            US dollars worth of collateral needed
        """
        borrowed_usd = self.borrowed.last_usd_price  * float(borrowed_quantity)
        usd_value = borrowed_usd * (1 + 1 / leverage)
        return Decimal(usd_value / self.collateral.last_usd_price)

    def calculate_size_adjust(
        self,
        collater_adjust: Decimal,
        borrowed_asset_price: USDollarPrice = None,
        leverage: LeverageMultiplier = None,
    ) -> Decimal:
        """Calculate the collateral amount we need to hit a target leverage.

        Assume ``collateral_adjust`` amount of collateral
        is deposited/withdrawn. Calculate the amount of borrowed
        token we need to trade to

        :param collateral_adjust:
            Are we adding or removing collateral.

        :param borrowed_asset_price:
            The currentprice of the borrowed asset.

            If not given use the cached value.

        :param leverage:
            The target leverage level.

            If not given use the existing leverage.

        :return:
            Positive to buy more borrowed token, negative to sell more.
        """
        if not leverage:
            leverage = self.get_leverage()

        if not borrowed_asset_price:
            borrowed_asset_price = self.borrowed.last_usd_price

        if collater_adjust > 0:
            estimate = LeverageEstimate.open_short(
                collater_adjust,
                leverage,
                borrowed_asset_price,
                self.pair,
            )
            return estimate.borrowed_quantity
        else:
            raise NotImplementedError(f"Not implemented for negative collateral adjust, received {collater_adjust}")

    def check_health(self, desired_health_factor=1):
        """Check if this loan is healthy.

        Health factor must stay above 1 or you get liquidated.

        :raise LiquidationRisked:
            If the loan would be instantly liquidated

        """
        health_factor = self.get_health_factor()
        if health_factor <= desired_health_factor:
            raise LiquidationRisked(
                f"Loan health factor: {health_factor:.2f}.\n"
                f"You would be liquidated.\n"
                f"Desired health factor is {desired_health_factor:.2f}.\n"
                f"Collateral {self.collateral.get_usd_value()} USD.\n"
                f"Borrowed {self.borrowed.quantity} {self.borrowed.asset.token_symbol} {self.borrowed.get_usd_value()} USD.\n"
            )

