"""Refresh dedicated pool-account cash using the existing executor state."""

from decimal import Decimal

from eth_defi.compat import native_datetime_utc_now
from eth_defi.lighter.execution import fetch_execution_account
from eth_defi.lighter.session import LighterSession
from eth_defi.lighter.valuation import fetch_lighter_account_by_index
from eth_defi.vault.deposit_redeem import VaultRequestInvariantViolation

from tradeexecutor.ethereum.lighter.lighter_pool_valuation import LighterPoolPricing, LighterPoolValuator
from tradeexecutor.exchange_account.lighter import LIGHTER_PROTOCOL
from tradeexecutor.state.lighter import LighterIdleCollateral
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeStatus


def collateral_snapshot(account: dict) -> tuple:
    """Compare economic inputs without volatile aggregate-equity telemetry."""
    return (
        account.get("l1_address", "").lower(),
        account.get("collateral"),
        account.get("available_balance"),
        sorted((row["public_pool_index"], row["shares_amount"], row["principal_amount"], row["entry_usdc"], row["entry_timestamp"]) for row in account.get("shares", [])),
        account.get("positions"),
    )


def refresh_lighter_idle_collateral(
    state: State,
    session: LighterSession,
    account_index: int,
    safe_address: str,
) -> None:
    """Observe cash and held shares together before publishing portfolio NAV.

    Pending operations must first be advanced by the shared settlement sweep.
    The API cannot provide an Ethereum-block-pinned snapshot. Read the account
    again after valuation and refuse a changing snapshot instead of mixing two
    sequencer states. No private API credentials are used.
    """
    now = native_datetime_utc_now()
    observation = LighterIdleCollateral(1, safe_address, account_index, Decimal(0), now, valid=False)
    key = observation.get_identifier()
    old = state.portfolio.lighter_idle_collateral.get(key)
    if old is None and state.portfolio.lighter_idle_collateral:
        raise VaultRequestInvariantViolation("Configured Lighter account differs from saved collateral custody")
    if old is not None:
        old.valid = False
    account, equity = fetch_execution_account(session, account_index)
    # The public account endpoint identifies the L1 owner on mainnet. Tests
    # exercise the same field rather than accepting an unauthenticated index.
    if str(account.get("l1_address", "")).lower() != safe_address.lower():
        raise VaultRequestInvariantViolation("Lighter account does not belong to the Ethereum Safe")
    if account["positions"]:
        raise VaultRequestInvariantViolation("A pool account cannot also hold perpetual positions")
    if not equity.collateral.is_finite() or equity.collateral < 0:
        raise VaultRequestInvariantViolation("Invalid Lighter idle collateral")
    positions = [p for p in state.portfolio.get_open_and_frozen_positions() if p.pair.is_lighter_pool()]
    if any(
        p.is_exchange_account()
        and p.pair.get_exchange_account_protocol() == LIGHTER_PROTOCOL
        and str(p.pair.get_exchange_account_id()) == str(account_index)
        for p in state.portfolio.get_open_and_frozen_positions()
    ):
        raise VaultRequestInvariantViolation("Dedicated Lighter pool collateral cannot share exchange-account valuation")
    for position in positions:
        if any(t.get_status() not in {TradeStatus.success, TradeStatus.repaired, TradeStatus.planned, TradeStatus.expired} for t in position.trades.values()):
            raise VaultRequestInvariantViolation("Reconcile pending Lighter pool operations before posting NAV")
    holdings = {int(row["public_pool_index"]): int(row["shares_amount"]) for row in account["shares"] if row["shares_amount"]}
    recorded = {p.pair.get_lighter_pool_index(): int(p.get_quantity()) for p in positions if p.get_quantity()}
    if holdings != recorded:
        raise VaultRequestInvariantViolation(f"Lighter pool holdings differ from executor state: expected raw shares {recorded}, observed {holdings}; reconcile before posting NAV")
    pricing = LighterPoolPricing(session)
    valuator = LighterPoolValuator(pricing, account_index=account_index)
    for position in positions:
        valuator(now, position, account_snapshot=account)
    if collateral_snapshot(fetch_lighter_account_by_index(session, account_index)) != collateral_snapshot(account):
        raise VaultRequestInvariantViolation("Lighter account changed during valuation; retry the snapshot")
    observation.amount = equity.collateral
    observation.valid = True
    state.portfolio.lighter_idle_collateral[key] = observation
