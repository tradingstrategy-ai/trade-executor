"""Withdrawal-time externalised vault fees, derived from the existing trade ledger."""

import datetime
from decimal import Decimal

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.strategy.vault_fee_data import get_vault_fee_internalised


#: Key in TradeExecution.other_data for request snapshots and realised deductions.
FEE_KEY = "backtest_vault_externalised_fees"
#: Annualisation convention for management fees: 365.25 days, expressed in seconds.
SECONDS_IN_YEAR = Decimal("31557600")


def externalised_rates(pair: TradingPairIdentifier) -> tuple[Decimal, Decimal] | None:
    """Resolve the rates shared by backtest quotes, valuation and withdrawals.

    Prefer the attached metadata and pinned protocol evidence for older bundles,
    falling back to persisted pair fields after state reload. Internalised fees
    are already reflected in share prices.
    Completely absent legacy metadata preserves the old fee-free behaviour;
    after internalised/known-zero cases are excluded, partial or invalid inputs
    must fail rather than understate externalised costs.

    :param pair:
        Vault pair with fee mode and fractional annual management/performance rates.
    :return:
        Management and performance rates as Decimals, or ``None`` for a non-vault,
        internalised fees, known zero rates or wholly absent legacy fee metadata.
    :raises AssertionError:
        Fee mode is unknown, a required rate is missing, or rates are non-finite
        or outside the inclusive range zero to one.
    """
    if not pair.is_vault():
        return None
    metadata = pair.get_vault_metadata()
    mode = get_vault_fee_internalised(metadata) if metadata else pair.other_data.get("vault_fee_internalised")
    management = metadata.management_fee if metadata else pair.get_management_fee()
    performance = metadata.performance_fee if metadata else pair.get_performance_fee()
    if mode is True or (management == 0 and performance == 0):
        return None
    # Legacy hand-built universes without fee metadata retain their previous behaviour.
    if mode is None and management is None and performance is None:
        return None
    assert mode is False, f"Unknown vault fee mode: {pair}"
    assert management is not None and performance is not None, f"Missing externalised vault fees: {pair}"
    rates = Decimal(str(management)), Decimal(str(performance))
    assert all(r.is_finite() and 0 <= r <= 1 for r in rates), f"Invalid vault fees: {pair}: {rates}"
    return rates


def vault_exit_rate(pair: TradingPairIdentifier) -> float:
    """Keep fixed exit costs on the existing sell-tax path, separately from manager fees.

    Explicit legacy share-token taxes take precedence. Otherwise the pair getter
    exposes only supported loaded exit fees, independently of the manager-fee
    mode; unsupported costs must be refused by the selecting strategy.

    :param pair:
        Pair whose base share token may carry an explicitly configured sell tax.
    :return:
        Fraction of proceeds charged on exit, or zero when no tax is configured.
    """
    return pair.get_sell_tax() or 0


def remaining_basis(position: TradingPosition, timestamp: datetime.datetime) -> tuple[Decimal, Decimal, Decimal]:
    """Return remaining shares, invested principal and principal-years at ``timestamp``.

    Successful buys accrue from share settlement; successful reductions remove
    basis proportionally. Pending requests leave basis intact until settlement,
    so the existing ledger also supports quotes and restart without another balance.

    :param position:
        Position with its complete successful acquisition/reduction ledger.
    :param timestamp:
        Naive UTC valuation or settlement time, at or after the ledger's trades.
    :return:
        Remaining share quantity, denomination-token principal excluding entry
        costs, and principal multiplied by holding years (365.25 days per year).
    """
    quantity = principal = exposure = Decimal(0)
    for trade in sorted(position.get_successful_trades(), key=lambda t: t.executed_at):
        shares = abs(trade.get_position_quantity())
        if trade.is_buy():
            # executed_price includes an entry charge; mid-price is the invested share value.
            price = trade.executed_price / (1 + (trade.pair.base.get_buy_tax() or 0))
            cost = shares * Decimal(str(price))
            quantity += shares
            principal += cost
            age = Decimal(str((timestamp - trade.executed_at).total_seconds())) / SECONDS_IN_YEAR
            exposure += cost * max(age, Decimal(0))
        elif trade.is_sell():
            assert quantity > 0, f"Reduction without acquisition: {trade}"
            remaining = max(Decimal(0), (quantity - shares) / quantity)
            principal *= remaining
            exposure *= remaining
            quantity -= shares
    return quantity, principal, exposure


def calculate_vault_fees(position: TradingPosition, timestamp: datetime.datetime, quantity: Decimal,
                         gross_price: float, exit_rate: float = 0,
                         snapshot: dict[str, str | int] | None = None) -> dict[str, str | int] | None:
    """Calculate net withdrawal and its fee components in denomination units.

    Vault operators may deduct externalised fees before withdrawal. This simulator
    currently collects them only on withdrawal. Future work may model management
    fee deductions weekly or monthly; periodic collection is not simulated here.
    Constant snapshot rates, principal-based management and proportional cost basis
    are simulation assumptions, not protocol-specific high-water-mark accounting.

    Quotes and withdrawals share this calculation, which does not mutate state.
    Queued withdrawals save the result at request time and pass it back at
    settlement to preserve basis and rates while management fees keep accruing.

    :param position:
        Vault position supplying the pair metadata and successful trade ledger.
    :param timestamp:
        Naive UTC quote, request or settlement time.
    :param quantity:
        Non-negative number of shares being reduced. With a snapshot, this must
        be the same requested reduction whose basis the snapshot records.
    :param gross_price:
        Denomination-token value per share before manager fees or exit charges.
    :param exit_rate:
        Fractional fixed exit charge, applied once after management/performance fees.
    :param snapshot:
        Prior result for this request, or ``None`` to derive proportional basis
        from the current ledger. The snapshot fixes principal, principal-years,
        fee rates and their accrual timestamp; gross proceeds use the new price.
    :return:
        JSON-compatible record with version, ISO timestamp and decimal strings
        for basis, rates, quantities, fee components and net proceeds. Monetary
        values use denomination-token units. ``None`` means no externalised fee
        calculation applies or the unsnapshotted ledger has no shares.
    :raises AssertionError:
        Rates, amounts or metadata are invalid, or a fresh reduction exceeds holdings.
    """
    rates = (Decimal(snapshot["management_rate"]), Decimal(snapshot["performance_rate"])) if snapshot else externalised_rates(position.pair)
    if rates is None:
        if not exit_rate:
            return None
        rates = Decimal(0), Decimal(0)
    assert quantity >= 0
    if snapshot:
        principal = Decimal(snapshot["principal"])
        exposure = Decimal(snapshot["principal_years"])
        elapsed = Decimal(str((timestamp - datetime.datetime.fromisoformat(snapshot["timestamp"])).total_seconds())) / SECONDS_IN_YEAR
        exposure += principal * max(elapsed, Decimal(0))
    else:
        held, principal, exposure = remaining_basis(position, timestamp)
        if held == 0:
            return None
        assert quantity <= held + Decimal("0.00000001"), f"Withdrawal exceeds held shares: {quantity} > {held}"
        fraction = min(quantity / held, Decimal(1))
        principal *= fraction
        exposure *= fraction
    gross = quantity * Decimal(str(gross_price))
    assert gross.is_finite() and gross >= 0
    exit_fee_rate = Decimal(str(snapshot.get("exit_rate", exit_rate) if snapshot else exit_rate))
    assert exit_fee_rate.is_finite() and 0 <= exit_fee_rate <= 1
    nominal_management = exposure * rates[0]
    # A loss cannot create a negative payout; retain the uncapped accrual in the
    # record so the amount actually collected remains distinguishable from it.
    management = min(gross, nominal_management)
    performance = max(gross - principal - nominal_management, Decimal(0)) * rates[1]
    after_fees = gross - management - performance
    exit_fee = after_fees * exit_fee_rate
    values = dict(principal=principal, principal_years=exposure, gross=gross,
                  nominal_management=nominal_management, management=management,
                  performance=performance, exit_fee=exit_fee, net=after_fees - exit_fee,
                  management_rate=rates[0], performance_rate=rates[1], exit_rate=exit_fee_rate, quantity=quantity)
    return {"version": 1, "timestamp": timestamp.isoformat(), **{k: str(v) for k, v in values.items()}}
