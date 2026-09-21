"""Pure cash allocation policy for external exchange accounts."""

from dataclasses import dataclass
from decimal import Decimal


class ExchangeCashManagementError(ValueError):
    """Raised when exchange cash-management inputs are invalid."""


@dataclass(frozen=True, slots=True)
class ExchangeCashSnapshot:
    """Latest balances used to choose an exchange cash transfer."""

    #: USDC held by the Safe after the most recent treasury synchronisation.
    safe_usdc: Decimal

    #: Free USDC collateral that the exchange permits withdrawing.
    exchange_available_usdc: Decimal

    #: USDC awaiting Lagoon deposit settlement.
    pending_deposits_usdc: Decimal

    #: USDC required to settle the asynchronous Lagoon redemption queue.
    pending_redemptions_usdc: Decimal


@dataclass(frozen=True, slots=True)
class ExchangeCashManagementDecision:
    """The transfer selected by :class:`ExchangeCashManager`."""

    #: ``deposit``, ``withdraw``, or ``None`` when no movement is needed.
    direction: str | None

    #: USDC amount to move in ``direction``.
    amount_usdc: Decimal

    @property
    def should_transfer(self) -> bool:
        """Return whether the decision creates a transfer."""
        return self.direction is not None and self.amount_usdc > 0


@dataclass(frozen=True, slots=True)
class ExchangeCashManager:
    """Choose a Safe-to-exchange or exchange-to-Safe cash transfer.

    :param safe_cash_buffer_usdc:
        USDC to retain in the Safe after automatic cash management.
    :param free_collateral_buffer_usdc:
        Free collateral to retain in the exchange account after a withdrawal.
    :param minimum_transfer_usdc:
        Smallest custody movement worth submitting to the exchange.
    """

    safe_cash_buffer_usdc: Decimal
    free_collateral_buffer_usdc: Decimal
    minimum_transfer_usdc: Decimal

    def __post_init__(self) -> None:
        """Validate the static cash-management policy."""
        for name, value in {
            "safe_cash_buffer_usdc": self.safe_cash_buffer_usdc,
            "free_collateral_buffer_usdc": self.free_collateral_buffer_usdc,
            "minimum_transfer_usdc": self.minimum_transfer_usdc,
        }.items():
            _validate_non_negative_decimal(name, value)

    def decide(
        self,
        snapshot: ExchangeCashSnapshot,
        *,
        transfer_pending: bool = False,
    ) -> ExchangeCashManagementDecision:
        """Return a deterministic transfer decision without network access.

        :param snapshot:
            Latest Safe, exchange, and Lagoon queue balances.
        :param transfer_pending:
            Whether an earlier exchange custody transfer is still unconfirmed.
        :return:
            The single safe transfer to submit, or an empty decision.
        """
        for name, value in {
            "safe_usdc": snapshot.safe_usdc,
            "exchange_available_usdc": snapshot.exchange_available_usdc,
            "pending_deposits_usdc": snapshot.pending_deposits_usdc,
            "pending_redemptions_usdc": snapshot.pending_redemptions_usdc,
        }.items():
            _validate_non_negative_decimal(name, value)
        if transfer_pending:
            return ExchangeCashManagementDecision(None, Decimal(0))

        desired_safe_usdc = max(
            snapshot.pending_redemptions_usdc - snapshot.pending_deposits_usdc,
            Decimal(0),
        ) + self.safe_cash_buffer_usdc
        if snapshot.safe_usdc < desired_safe_usdc:
            shortfall = desired_safe_usdc - snapshot.safe_usdc
            withdrawable = max(
                Decimal(0),
                snapshot.exchange_available_usdc - self.free_collateral_buffer_usdc,
            )
            amount = min(shortfall, withdrawable)
            if amount >= self.minimum_transfer_usdc:
                return ExchangeCashManagementDecision("withdraw", amount)
            return ExchangeCashManagementDecision(None, Decimal(0))

        amount = snapshot.safe_usdc - desired_safe_usdc
        if amount >= self.minimum_transfer_usdc:
            return ExchangeCashManagementDecision("deposit", amount)
        return ExchangeCashManagementDecision(None, Decimal(0))


def _validate_non_negative_decimal(name: str, value: Decimal) -> None:
    """Validate one monetary policy or balance value."""
    if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
        raise ExchangeCashManagementError(
            f"{name} must be a finite non-negative Decimal"
        )
