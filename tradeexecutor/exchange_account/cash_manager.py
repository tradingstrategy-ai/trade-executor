"""Pure cash allocation policy for external exchange accounts."""

from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal


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
            amount = round_down_to_usdc(min(shortfall, withdrawable))
            if amount >= self.minimum_transfer_usdc:
                return ExchangeCashManagementDecision("withdraw", amount)
            return ExchangeCashManagementDecision(None, Decimal(0))

        amount = round_down_to_usdc(snapshot.safe_usdc - desired_safe_usdc)
        if amount >= self.minimum_transfer_usdc:
            return ExchangeCashManagementDecision("deposit", amount)
        return ExchangeCashManagementDecision(None, Decimal(0))


def _validate_non_negative_decimal(name: str, value: Decimal) -> None:
    """Validate one monetary policy or balance value."""
    if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
        raise ExchangeCashManagementError(
            f"{name} must be a finite non-negative Decimal"
        )


@dataclass(frozen=True, slots=True)
class SubAccountBalance:
    """Latest balances of one exchange sub-account."""

    #: Account equity, including unrealised profit and loss.
    equity: Decimal

    #: Collateral the exchange permits moving out of the account.
    free: Decimal


#: Smallest USDC unit Lighter and Ethereum USDC can move.
USDC_PRECISION = Decimal("0.000001")


def round_down_to_usdc(amount: Decimal) -> Decimal:
    """Round an amount down to whole USDC units, so it can be submitted and waited for exactly."""
    return amount.quantize(USDC_PRECISION, rounding=ROUND_DOWN)


def plan_sub_account_rebalance(
    *,
    master_free_usdc: Decimal,
    sub_accounts: dict[str, SubAccountBalance],
    target_weights: dict[str, Decimal],
    minimum_transfer_usdc: Decimal,
    rebalance_band_usdc: Decimal,
    transfer_fee_usdc: Decimal = Decimal(0),
) -> list[tuple[str, Decimal]]:
    """Plan one kind of capital movement towards sub-account target weights.

    Capital is split between sub-accounts by ``target_weights`` of the
    investable capital: the master's free collateral plus every managed
    sub-account's equity. Any weight left unassigned stays on the master.
    Pass only the sub-accounts that have a target weight: an account without
    one is unmanaged, and its capital is neither counted nor moved.

    The plan contains either withdrawals or fundings, never both, so no
    movement in a cycle depends on another one in the same cycle:

    - Overweight sub-accounts are withdrawn to the Safe, capped by their free
      collateral, because margin in use cannot move. A later cycle deposits
      the proceeds to the master.
    - Otherwise underweight sub-accounts are funded from the master's free
      collateral, most underweight first, with the transfer fee budgeted on
      top of each amount.

    Accounts within ``rebalance_band_usdc`` of their target and movements
    below ``minimum_transfer_usdc`` are skipped. The function is pure and
    idempotent: a converged snapshot returns no movements.

    :param master_free_usdc:
        Master account free collateral available for funding, net of any
        buffer the caller keeps on the master.
    :param sub_accounts:
        Balances keyed by sub-account id, in declaration order.
    :param target_weights:
        Target share of investable capital per sub-account id.
    :param minimum_transfer_usdc:
        Smallest movement worth submitting.
    :param rebalance_band_usdc:
        Tolerated distance from a target before moving capital.
    :param transfer_fee_usdc:
        Fee charged on top of each funding transfer.
    :return:
        ``(sub_account_id, amount)`` pairs. A negative amount withdraws from
        the sub-account to the Safe; a positive amount funds it from the
        master.
    """
    for name, value in {
        "master_free_usdc": master_free_usdc,
        "minimum_transfer_usdc": minimum_transfer_usdc,
        "rebalance_band_usdc": rebalance_band_usdc,
        "transfer_fee_usdc": transfer_fee_usdc,
    }.items():
        _validate_non_negative_decimal(name, value)
    unknown = set(target_weights) - set(sub_accounts)
    if unknown:
        raise ExchangeCashManagementError(f"Target weights for unknown sub-accounts: {sorted(unknown)}")
    if any(weight < 0 for weight in target_weights.values()) or sum(target_weights.values(), Decimal(0)) > 1:
        raise ExchangeCashManagementError("Sub-account target weights must be non-negative and sum to at most one")

    investable = master_free_usdc + sum((balance.equity for balance in sub_accounts.values()), Decimal(0))
    targets = {account_id: target_weights.get(account_id, Decimal(0)) * investable for account_id in sub_accounts}

    withdrawals = []
    for account_id, balance in sub_accounts.items():
        excess = balance.equity - targets[account_id]
        if excess > rebalance_band_usdc:
            amount = round_down_to_usdc(min(excess, balance.free))
            if amount >= minimum_transfer_usdc and amount > 0:
                withdrawals.append((account_id, -amount))
    if withdrawals:
        return withdrawals

    fundings = []
    budget = master_free_usdc
    deficits = {account_id: targets[account_id] - balance.equity for account_id, balance in sub_accounts.items()}
    # Most underweight first; the stable sort keeps declaration order for ties.
    for account_id in sorted(deficits, key=lambda account_id: -deficits[account_id]):
        deficit = deficits[account_id]
        if deficit <= rebalance_band_usdc:
            continue
        amount = round_down_to_usdc(min(deficit, budget - transfer_fee_usdc))
        if amount < minimum_transfer_usdc or amount <= 0:
            continue
        fundings.append((account_id, amount))
        budget -= amount + transfer_fee_usdc
    return fundings
