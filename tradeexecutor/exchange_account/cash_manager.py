"""Pure cash allocation policy for external exchange accounts."""

from dataclasses import dataclass
from decimal import Decimal


#: Default maximum wait for an external exchange withdrawal.
DEFAULT_EXCHANGE_WITHDRAWAL_TIMEOUT_SECONDS = 30 * 60


class ExchangeCashManagementError(ValueError):
    """Raised when exchange cash-management inputs are invalid."""


@dataclass(frozen=True, slots=True)
class ExchangeCashManagementInput:
    """Balances and limits used to choose an exchange cash transfer."""

    #: USDC held by the Safe after the most recent treasury synchronisation.
    safe_usdc: Decimal

    #: Free USDC collateral that the exchange permits withdrawing.
    exchange_available_usdc: Decimal

    #: USDC required to settle the asynchronous vault redemption queue.
    pending_redemptions_usdc: Decimal

    #: USDC retained in the Safe after automatic cash management.
    safe_cash_buffer_usdc: Decimal

    #: Free collateral retained in the exchange account after a withdrawal.
    free_collateral_buffer_usdc: Decimal

    #: Smallest custody movement worth submitting to the exchange.
    minimum_transfer_usdc: Decimal


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


class ExchangeCashManager:
    """Choose a Safe-to-exchange or exchange-to-Safe cash transfer."""

    def decide(
        self,
        inputs: ExchangeCashManagementInput,
        *,
        transfer_pending: bool = False,
    ) -> ExchangeCashManagementDecision:
        """Return a deterministic transfer decision without network access.

        :param inputs:
            Latest Safe, exchange, and redemption balances with policy limits.
        :param transfer_pending:
            Whether an earlier exchange custody transfer is still unconfirmed.
        :return:
            The single safe transfer to submit, or an empty decision.
        """
        self._validate(inputs)
        if transfer_pending:
            return ExchangeCashManagementDecision(None, Decimal(0))

        desired_safe_usdc = inputs.pending_redemptions_usdc + inputs.safe_cash_buffer_usdc
        if inputs.safe_usdc < desired_safe_usdc:
            shortfall = desired_safe_usdc - inputs.safe_usdc
            withdrawable = max(
                Decimal(0),
                inputs.exchange_available_usdc - inputs.free_collateral_buffer_usdc,
            )
            amount = min(shortfall, withdrawable)
            if amount >= inputs.minimum_transfer_usdc:
                return ExchangeCashManagementDecision("withdraw", amount)
            return ExchangeCashManagementDecision(None, Decimal(0))

        amount = inputs.safe_usdc - desired_safe_usdc
        if amount >= inputs.minimum_transfer_usdc:
            return ExchangeCashManagementDecision("deposit", amount)
        return ExchangeCashManagementDecision(None, Decimal(0))

    @staticmethod
    def _validate(inputs: ExchangeCashManagementInput) -> None:
        """Reject negative and non-finite monetary inputs."""
        values = {
            "safe_usdc": inputs.safe_usdc,
            "exchange_available_usdc": inputs.exchange_available_usdc,
            "pending_redemptions_usdc": inputs.pending_redemptions_usdc,
            "safe_cash_buffer_usdc": inputs.safe_cash_buffer_usdc,
            "free_collateral_buffer_usdc": inputs.free_collateral_buffer_usdc,
            "minimum_transfer_usdc": inputs.minimum_transfer_usdc,
        }
        for name, value in values.items():
            if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
                raise ExchangeCashManagementError(
                    f"{name} must be a finite non-negative Decimal"
                )
