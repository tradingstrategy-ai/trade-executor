"""Protocol-aware vault deposit availability checks."""

import datetime
from decimal import Decimal

from eth_defi.erc_4626.core import ERC4626Feature
from eth_defi.vault.deposit_redeem import VaultFlowUnavailable

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.redemption import (
    DepositBlockReason,
    DepositCheckResult,
    DepositCheckStage,
)


#: Historical state for these protocols does not yet express whether a new
#: deposit request is accepted. Do not manufacture a closure from its generic
#: ERC-4626 fields. Extend this set only after the historical reader records a
#: protocol-specific admission signal.
BACKTEST_ALWAYS_OPEN_DEPOSIT_FEATURES = {
    ERC4626Feature.lagoon_like,
    ERC4626Feature.morpho_v2_like,
    ERC4626Feature.yearn_v3_like,
}


#: Older universe snapshots predate classified ERC-4626 features. These slugs
#: retain the same conservative historical treatment until their readers offer
#: a dependable protocol-specific deposit-admission field.
BACKTEST_ALWAYS_OPEN_DEPOSIT_PROTOCOLS = {
    "lagoon-finance",
    "morpho",
    "yearn",
}


def _is_backtesting_always_open_protocol(pair: TradingPairIdentifier) -> bool:
    """Does incomplete historical availability make this vault open by default?"""
    features = pair.get_vault_features()
    if features and features & BACKTEST_ALWAYS_OPEN_DEPOSIT_FEATURES:
        return True
    get_vault_protocol = getattr(pair, "get_vault_protocol", None)
    protocol = get_vault_protocol() if callable(get_vault_protocol) else None
    return protocol in BACKTEST_ALWAYS_OPEN_DEPOSIT_PROTOCOLS


def is_backtesting_deposit_open(
    pair: TradingPairIdentifier,
    deposits_open: bool | int | None,
    max_deposit: Decimal | None,
) -> bool:
    """Determine whether a historical vault observation permits a deposit.

    Lagoon, Morpho V2 and Yearn are kept open until their historical readers
    expose protocol-specific admission state. For every other protocol, retain
    the generic historical behaviour: an explicit closed marker or a zero cap
    closes deposits, while missing data is treated as open.
    """
    if _is_backtesting_always_open_protocol(pair):
        return True
    if deposits_open == 0 or deposits_open is False:
        return False
    return max_deposit is None or max_deposit != 0


def check_backtesting_deposit(
    *,
    timestamp: datetime.datetime | None,
    pair: TradingPairIdentifier,
    deposits_open: bool | int | None,
    max_deposit: Decimal | None,
    closed_reason: str | None,
    stage: DepositCheckStage,
) -> DepositCheckResult:
    """Create a diagnostic result from one historical vault-state observation."""
    always_open = _is_backtesting_always_open_protocol(pair)
    is_open = is_backtesting_deposit_open(pair, deposits_open, max_deposit)
    result = DepositCheckResult(
        timestamp=timestamp,
        stage=stage,
        can_deposit=is_open,
        pair_ticker=pair.get_ticker(),
        vault_address=pair.pool_address,
        max_deposit=None if always_open or max_deposit is None else float(max_deposit),
    )
    if is_open:
        return result
    if deposits_open == 0 or deposits_open is False:
        result.reason_code = DepositBlockReason.vault_deposits_closed
        result.message = closed_reason or "Vault deposits closed in historical data"
    else:
        result.reason_code = DepositBlockReason.vault_max_deposit_zero
        result.message = closed_reason or "Vault max deposit was zero in historical data"
    return result


def get_live_max_deposit(
    vault,
    owner: str | None,
) -> Decimal | None:
    """Read owner-scoped live deposit capacity through the vault adapter."""
    if owner is None:
        return None
    deposit_manager = vault.get_deposit_manager()
    if not deposit_manager.has_synchronous_deposit():
        return None
    raw_amount = deposit_manager.fetch_depositable_raw_assets(owner)
    if raw_amount is None:
        return None
    return vault.denomination_token.convert_to_decimals(raw_amount)


def _fetch_live_deposit_closed_reason(vault, deposit_manager, owner: str) -> str | None:
    """Ask the adapter for a global closure before using the generic vault hook."""
    fetch_global_closure = getattr(deposit_manager, "fetch_global_deposit_closure_reason", None)
    if callable(fetch_global_closure):
        return fetch_global_closure(owner)
    return vault.fetch_deposit_closed_reason()


def check_live_deposit(
    *,
    timestamp: datetime.datetime | None,
    pair: TradingPairIdentifier,
    vault,
    owner: str | None,
    stage: DepositCheckStage,
) -> DepositCheckResult:
    """Check live vault admission through its ``DepositManager`` adapter.

    The adapter owns the protocol-specific request gate and owner-scoped limit
    read. This deliberately does not call raw ERC-4626 ``maxDeposit`` because
    several supported protocols use it as a non-capacity signal.
    """
    result = DepositCheckResult(
        timestamp=timestamp,
        stage=stage,
        pair_ticker=pair.get_ticker(),
        vault_address=pair.pool_address,
    )
    deposit_manager = vault.get_deposit_manager()

    if owner is None:
        if not deposit_manager.has_synchronous_deposit():
            result.can_deposit = False
            result.reason_code = DepositBlockReason.deposit_request_unavailable
            result.message = "Cannot resolve owner address for asynchronous vault deposit request check"
        return result

    try:
        can_create_request = deposit_manager.can_create_deposit_request(owner)
    except NotImplementedError:
        can_create_request = None
    except VaultFlowUnavailable as error:
        result.can_deposit = False
        result.reason_code = DepositBlockReason.deposit_request_unavailable
        result.message = str(error)
        return result

    if can_create_request is False:
        result.can_deposit = False
        result.reason_code = DepositBlockReason.deposit_request_unavailable
        result.message = "Vault deposit manager does not currently accept a deposit request"
        return result

    try:
        closed_reason = _fetch_live_deposit_closed_reason(vault, deposit_manager, owner)
    except VaultFlowUnavailable as error:
        result.can_deposit = False
        result.reason_code = DepositBlockReason.deposit_request_unavailable
        result.message = str(error)
        return result

    if closed_reason is not None:
        result.can_deposit = False
        result.reason_code = DepositBlockReason.vault_deposits_closed
        result.message = closed_reason
        return result

    try:
        max_deposit = get_live_max_deposit(vault, owner)
    except VaultFlowUnavailable as error:
        result.can_deposit = False
        result.reason_code = DepositBlockReason.deposit_request_unavailable
        result.message = str(error)
        return result

    result.max_deposit = float(max_deposit) if max_deposit is not None else None
    if max_deposit is not None and max_deposit <= 0:
        result.can_deposit = False
        result.reason_code = DepositBlockReason.vault_max_deposit_zero
        result.message = "Vault reports zero maxDeposit capacity for this owner"
    return result
