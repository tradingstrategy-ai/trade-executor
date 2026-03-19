"""Helpers for estimating how much capital can be redeemed right now."""

from __future__ import annotations

import datetime

from eth_defi.compat import native_datetime_utc_now

from tradeexecutor.ethereum.vault.hypercore_vault import HLP_VAULT_ADDRESS
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.types import Percent, USDollarAmount
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager

#: Safety buffer added on top of Hyperliquid's published redemption windows.
#:
#: We keep a small extra margin because live settlement timing is not perfectly
#: aligned with strategy cycle boundaries.
HYPERLIQUID_REDEEM_DELAY_BUFFER = datetime.timedelta(hours=12)

#: Default Hyperliquid vault redemption delay used for non-HLP Hypercore vaults.
#:
#: Currently modelled as one day plus :py:data:`HYPERLIQUID_REDEEM_DELAY_BUFFER`.
HYPERLIQUID_DEFAULT_REDEEM_DELAY = datetime.timedelta(days=1) + HYPERLIQUID_REDEEM_DELAY_BUFFER

#: HLP-specific Hyperliquid redemption delay.
#:
#: Currently modelled as four days plus :py:data:`HYPERLIQUID_REDEEM_DELAY_BUFFER`.
HYPERLIQUID_HLP_REDEEM_DELAY = datetime.timedelta(days=4) + HYPERLIQUID_REDEEM_DELAY_BUFFER


def get_redemption_delay(position: TradingPosition) -> datetime.timedelta:
    """Return the best-known redemption delay for a position.

    Hyperliquid vaults currently expose only two practical lock-up options for
    the strategy:

    - most vaults: one day
    - HLP: four days

    We add a small safety buffer because live settlement timing is not perfectly
    aligned with cycle boundaries.

    :param position:
        Position whose current redemption delay is being estimated.

    :return:
        Known Hyperliquid redemption delay, or zero for non-Hypercore positions.
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

    :param position:
        Position whose latest capital inflow timestamp is being resolved.

    :return:
        Opening time or the latest executed buy timestamp, whichever is later.
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

    :param position:
        Position whose currently redeemable marked value is being estimated.

    :param timestamp:
        Strategy-cycle timestamp used for the lock-up check. Defaults to the
        current naive UTC time when omitted.

    :return:
        Redeemable marked value in US dollars for the given cycle.
    """

    if timestamp is None:
        timestamp = native_datetime_utc_now()

    if position.pair.other_data.get("vault_protocol") != "hypercore":
        return position.get_value()

    unlock_at = get_latest_capital_inflow_at(position) + get_redemption_delay(position)
    if timestamp < unlock_at:
        return 0.0

    return position.get_value()


def get_redeemable_portfolio_capital(
    position_manager: PositionManager,
) -> USDollarAmount:
    """Get the currently redeemable trading capital for the whole portfolio.

    Hyperliquid vaults can have redemption delays, so not every marked position
    value can necessarily be turned into treasury cash on this cycle.

    :param position_manager:
        Position manager whose current portfolio and cycle timestamp are used
        for the redeemable-capital calculation.

    :return:
        Sum of all currently redeemable open-position values in US dollars.
    """

    portfolio = position_manager.get_current_portfolio()
    return sum(
        get_redeemable_capital(position, timestamp=position_manager.timestamp)
        for position in portfolio.open_positions.values()
    )


def calculate_portfolio_target_value(
    position_manager: PositionManager,
    allocation: Percent,
) -> USDollarAmount:
    """Calculate how much capital Hyper AI may target on this cycle.

    Use total equity and the pending Lagoon redemption queue as the primary
    sizing inputs. If some Hyperliquid vault capital is still inside a known
    redemption lock-up, keep that locked capital as a floor under the target
    invested value so the strategy does not generate impossible sell targets.

    :param position_manager:
        Position manager whose portfolio state, pending redemptions and cycle
        timestamp are used for the calculation.

    :param allocation:
        Target invested share of portfolio equity, expressed as `1.0 = 100%`.

    :return:
        Portfolio target value in US dollars after accounting for pending
        redemptions and temporarily locked Hyperliquid capital.
    """

    portfolio = position_manager.get_current_portfolio()
    total_equity = portfolio.calculate_total_equity()
    pending_redemptions = position_manager.get_pending_redemptions()
    open_position_value = portfolio.get_live_position_equity()
    redeemable_position_value = get_redeemable_portfolio_capital(position_manager)
    locked_position_value = max(open_position_value - redeemable_position_value, 0.0)

    return max(total_equity * allocation - pending_redemptions, locked_position_value, 0.0)
