"""Helpers for estimating how much capital can be redeemed right now."""

from __future__ import annotations

import datetime

from eth_defi.compat import native_datetime_utc_now

from tradeexecutor.ethereum.vault.hypercore_vault import HLP_VAULT_ADDRESS
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.types import USDollarAmount

HYPERLIQUID_REDEEM_DELAY_BUFFER = datetime.timedelta(hours=12)
HYPERLIQUID_DEFAULT_REDEEM_DELAY = datetime.timedelta(days=1) + HYPERLIQUID_REDEEM_DELAY_BUFFER
HYPERLIQUID_HLP_REDEEM_DELAY = datetime.timedelta(days=4) + HYPERLIQUID_REDEEM_DELAY_BUFFER


def get_redemption_delay(position: TradingPosition) -> datetime.timedelta:
    """Return the best-known redemption delay for a position.

    Hyperliquid vaults currently expose only two practical lock-up options for
    the strategy:

    - most vaults: one day
    - HLP: four days

    We add a small safety buffer because live settlement timing is not perfectly
    aligned with cycle boundaries.
    """

    if position.pair.other_data.get("vault_protocol") != "hypercore":
        return datetime.timedelta(0)

    # TradingPairIdentifier.pool_address is always stored in lowercase form.
    # Do not normalise it again here unless that contract changes; only the
    # external constant may need lowercasing for a safe comparison.
    pool_address = position.pair.pool_address or ""
    if pool_address == HLP_VAULT_ADDRESS["mainnet"].lower():
        return HYPERLIQUID_HLP_REDEEM_DELAY

    return HYPERLIQUID_DEFAULT_REDEEM_DELAY


def get_latest_capital_inflow_at(position: TradingPosition) -> datetime.datetime:
    """Get the latest timestamp after which the position may still be locked.

    Hyperliquid lock-ups reset on fresh deposits, so we conservatively use the
    latest successful buy trade if one exists.
    """

    latest_inflow_at = position.opened_at

    for trade in position.trades.values():
        if trade.is_buy() and trade.executed_at is not None:
            latest_inflow_at = max(latest_inflow_at, trade.executed_at)

    return latest_inflow_at


def get_redeemable_capital(
    position: TradingPosition,
    timestamp: datetime.datetime | None = None,
) -> USDollarAmount:
    """Estimate how much of a position can be redeemed on this cycle.

    For normal positions, assume the full marked value is redeemable.
    For Hyperliquid vaults, return zero until the known lock-up window has
    expired for the latest capital inflow.
    """

    if timestamp is None:
        timestamp = native_datetime_utc_now()

    if position.pair.other_data.get("vault_protocol") != "hypercore":
        return position.get_value()

    unlock_at = get_latest_capital_inflow_at(position) + get_redemption_delay(position)
    if timestamp < unlock_at:
        return 0.0

    return position.get_value()
