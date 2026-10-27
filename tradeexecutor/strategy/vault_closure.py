"""Unified vault closure decisions for strategies, pricing and charts.

``is_vault_closed`` identifies a closed operation, not permanent shutdown.
``can_interact_with_vault`` applies strategy policy to the same assessment. Use ``get_vault_closure``
for the distinction between a known closure, a policy block and unknown state.
Neither a chart nor a strategy should interpret raw ERC-4626 maxima itself.

The historical provider owns as-of selection and original evidence clocks.
Live providers own protocol adapters, account permissions and position lockups.
This module owns historical interpretation and the common decision interface.
See ``docs/vault-closure.md`` for the evidence, limitations and extension contract.

Evidence and earlier fixes:

- https://github.com/tradingstrategy-ai/trade-executor/pull/1622#issuecomment-5236615753
- https://github.com/tradingstrategy-ai/trade-executor/pull/1542
- https://github.com/tradingstrategy-ai/trade-executor/pull/1570
- https://github.com/tradingstrategy-ai/trade-executor/pull/1612
- https://github.com/tradingstrategy-ai/trade-executor/pull/1670
- https://github.com/tradingstrategy-ai/trade-executor/pull/1671
- https://github.com/tradingstrategy-ai/getting-started/pull/40
- https://github.com/tradingstrategy-ai/getting-started/pull/52
- https://github.com/morpho-org/vault-v2#erc-4626-compliance
- https://eips.ethereum.org/EIPS/eip-7540#request-lifecycle
"""

import datetime
import enum
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

import pandas as pd
from eth_defi.erc_4626.core import ERC4626Feature
from eth_defi.vault.deposit_redeem import VaultFlowUnavailable
from tradingstrategy.alternative_data.vault import HYPERCORE_DEPOSIT_STATE_CUTOFF

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.redemption import (
    DepositBlockReason, DepositCheckResult, DepositCheckStage,
    RedemptionBlockReason, RedemptionCheckResult, RedemptionCheckStage,
)


BACKTEST_ALWAYS_OPEN_DEPOSIT_FEATURES = {
    ERC4626Feature.lagoon_like, ERC4626Feature.morpho_v2_like,
    ERC4626Feature.yearn_v3_like,
}
BACKTEST_ALWAYS_OPEN_DEPOSIT_PROTOCOLS = {"lagoon-finance", "yearn"}


class VaultClosureOutcome(enum.Enum):
    """Meaning of the available evidence, independently of execution policy."""

    open = "open"
    closed = "closed"
    unknown = "unknown"
    policy_blocked = "policy_blocked"


@dataclass(frozen=True, slots=True)
class VaultClosure:
    """Evidence and the resulting operation decision, kept deliberately separate."""

    #: Whether the requested operation must be refused under the chosen policy.
    blocked: bool
    #: ``unknown`` is not evidence of either open or closed admission.
    status: VaultClosureOutcome
    #: Historical sample, synthetic schedule, protocol assumption or live adapter.
    source: str
    #: Stable explanation, independent of human-readable message text.
    reason: str | None = None
    message: str | None = None
    #: Reported amount; positive caps do not implement amount-aware sizing here.
    max_amount: Decimal | float | None = None

    def __post_init__(self):
        if not isinstance(self.status, VaultClosureOutcome):
            raise TypeError("Vault closure status must be a VaultClosureOutcome")

    def as_deposit_check(self, vault, timestamp=None, stage=DepositCheckStage.unknown) -> DepositCheckResult:
        """Preserve the existing serialisable deposit-check API."""
        codes = {
            "window_closed": DepositBlockReason.deposit_window_closed,
            "explicit_closed": DepositBlockReason.vault_deposits_closed,
            "capacity_zero": DepositBlockReason.vault_max_deposit_zero,
            "strategy_capacity": DepositBlockReason.vault_max_deposit_zero,
        }
        return DepositCheckResult(
            timestamp=timestamp, stage=stage, can_deposit=not self.blocked,
            pair_ticker=vault.get_ticker() if vault else None,
            vault_address=vault.pool_address if vault else None,
            max_deposit=float(self.max_amount) if self.max_amount is not None else None,
            reason_code=codes.get(self.reason, DepositBlockReason.unknown) if self.blocked else None,
            message=self.message if self.blocked else None,
        )


def _present(value) -> bool:
    return value is not None and not pd.isna(value)


def _is_backtesting_always_open_protocol(pair) -> bool:
    """Compatibility helper for the documented missing-admission-history policy."""
    feature_getter = getattr(pair, "get_vault_features", None)
    protocol_getter = getattr(pair, "get_vault_protocol", None)
    features = feature_getter() if callable(feature_getter) else None
    protocol = protocol_getter() if callable(protocol_getter) else None
    return bool(features and features & BACKTEST_ALWAYS_OPEN_DEPOSIT_FEATURES) or protocol in BACKTEST_ALWAYS_OPEN_DEPOSIT_PROTOCOLS


def is_vault_closed(vault: TradingPairIdentifier, **kwargs) -> bool:
    """Is the requested operation known closed (including an explicit schedule)?

    A policy block or missing evidence does not establish a vault closure.
    Strategies must use :func:`can_interact_with_vault` for execution permission.
    Keyword arguments are the explicit options of :func:`get_vault_closure`;
    unknown arguments fail rather than silently disabling checks. Inspect the
    assessment's source to distinguish assumed schedules from observations.
    """
    return get_vault_closure(vault, **kwargs).status is VaultClosureOutcome.closed


def can_interact_with_vault(vault: TradingPairIdentifier, **kwargs) -> bool:
    """May the strategy perform a deposit or redemption under its chosen policy?

    This is the strategy-facing counterpart of :func:`is_vault_closed`.
    Both consume the same :class:`VaultClosure` assessment, but a strategy can
    refuse an interaction with an open vault (leader-share policy or account
    lockup), or with an unknown vault (missing fresh HyperCore permission).

    Use ``operation="deposit"`` or ``operation="redemption"`` and pass the
    decision's ``pricing_model``, ``timestamp`` and redemption ``position``.
    All keyword-only controls are documented by :func:`get_vault_closure`.
    ``assessment=`` reuses a captured decision without fetching live data again.
    This checks admission only; amount sizing, approvals, funding, transaction
    simulation and execution-time revalidation remain the execution layer's job.
    """
    return not get_vault_closure(vault, **kwargs).blocked


def get_vault_closure(
    vault: TradingPairIdentifier,
    *,
    timestamp: datetime.datetime | None = None,
    operation: Literal["deposit", "redemption"] = "deposit",
    pricing_model=None,
    historical_state: Mapping | None = None,
    window_override=None,
    assessment: VaultClosure | None = None,
    deposit_check: DepositCheckResult | None = None,
    position=None,
    include_strategy_policy: bool = True,
    unknown_is_closed: bool | None = None,
    trust_protocol_state: bool = False,
    apply_hypercore_cutoff: bool = True,
) -> VaultClosure:
    """Interpret one operation using the same rule for execution and reporting.

    :param vault: Engine vault pair identity; never infer protocol from its name.
    :param timestamp: Naive UTC decision time. Providers select evidence at or before it.
    :param operation: Deposit or redemption; these are independent permissions.
    :param pricing_model: Preferred route for strategies/charts. Historical models
        supply fresh state; live models retain owner/position-aware adapters.
    :param historical_state: Already freshness-checked historical observation.
        Direct callers must not pass future, stale or timestamp-refreshed data.
    :param window_override: Explicit synthetic schedule, taking precedence over history.
    :param assessment: Reuse an exact previous result; cannot combine with new options.
    :param deposit_check: Reuse an existing live/alpha-model deposit result.
    :param position: Position context for live redemption lockups and liquidity.
    :param include_strategy_policy: Include known policy blocks, such as HyperCore's
        leader-share amount limit, in the returned boolean. Evidence status is retained.
    :param unknown_is_closed: Override unknown admission policy; ``None`` preserves
        the existing model defaults. Unknown remains labelled unknown either way.
    :param trust_protocol_state: Opt into an independently verified, protocol-specific
        historical admission field for Lagoon/Morpho V2/Yearn. False by default.
    :param apply_hypercore_cutoff: Retain HyperCore's pre-collection assumed-open
        period and later fail-closed unknowns. Disable only for explicitly scoped
        field-classification compatibility callers, not strategy decisions.
    :return: Decision plus evidence status; no transaction is submitted.
    """
    if operation not in ("deposit", "redemption"):
        raise ValueError(f"Unsupported vault operation: {operation}")
    if timestamp is not None and timestamp.tzinfo is not None:
        raise ValueError("Vault closure checks require naive UTC timestamps")
    routes = sum(x is not None for x in (pricing_model, assessment, deposit_check))
    if routes > 1 or (routes and (historical_state is not None or window_override is not None)):
        raise ValueError("Supply exactly one vault closure input route")
    if assessment is not None:
        if not include_strategy_policy or unknown_is_closed is not None or trust_protocol_state or not apply_hypercore_cutoff:
            raise ValueError("Reused assessment cannot be combined with changed policy options")
        return assessment
    if deposit_check is not None:
        if operation != "deposit":
            raise ValueError("A deposit check cannot classify redemptions")
        return _from_deposit_check(deposit_check, include_strategy_policy, unknown_is_closed)
    if pricing_model is not None:
        # Use a real class method, not a fabricated attribute from permissive mocks.
        getter = getattr(type(pricing_model), "get_vault_closure", None)
        if getter is not None:
            return pricing_model.get_vault_closure(
                timestamp, vault, operation=operation, position=position,
                include_strategy_policy=include_strategy_policy,
                unknown_is_closed=unknown_is_closed, trust_protocol_state=trust_protocol_state,
                apply_hypercore_cutoff=apply_hypercore_cutoff,
            )
        if trust_protocol_state or not apply_hypercore_cutoff:
            raise ValueError("Historical-only controls require a historical pricing provider")
        # Native live providers retain their actual owner, exchange permission,
        # guard whitelist and lockup checks. Never replace them with maxDeposit.
        if operation == "deposit":
            result = pricing_model.check_deposit(timestamp, vault)
            return _from_deposit_check(result, include_strategy_policy, unknown_is_closed)
        result = pricing_model.check_redemption(timestamp, vault, position=position)
        reason = result.reason_code.value if result.reason_code else None
        # Account lockup / missing owner / API failure do not prove global closure.
        status = VaultClosureOutcome.open if result.can_redeem else (
            VaultClosureOutcome.unknown if reason in {"safe_address_unavailable", "user_equity_fetch_failed"}
            else VaultClosureOutcome.policy_blocked if reason == "user_lockup_not_expired" else VaultClosureOutcome.closed)
        blocked = not result.can_redeem
        if status is VaultClosureOutcome.unknown and unknown_is_closed is not None:
            blocked = unknown_is_closed
        if status is VaultClosureOutcome.policy_blocked and not include_strategy_policy:
            blocked = False
        return VaultClosure(blocked, status, "live_adapter", reason, result.message, result.max_redemption)

    state = historical_state if historical_state is not None else {}
    is_deposit = operation == "deposit"
    # HYPE++ (Arbitrum 0x75288264fdfea8ce68e6d852696ab1ce2f3e5004): on
    # 2026-10-27 isFunding=false, but today's phase must never be backfilled.
    # NB04/NB05 explicitly supply an assumed schedule. PR #1570 requires this
    # override to win, and PR #1612 permits queue parking only for explicit windows.
    if window_override is not None:
        method = window_override.is_deposit_open if is_deposit else window_override.is_redemption_open
        blocked = timestamp is not None and not method(timestamp)
        return VaultClosure(blocked, VaultClosureOutcome.closed if blocked else VaultClosureOutcome.open, "synthetic_schedule",
                            "window_closed" if blocked else None,
                            f"Vault {operation}s closed by backtest window override" if blocked else None)

    native_getter = getattr(vault, "is_hyperliquid_vault", None)
    native = callable(native_getter) and native_getter()
    if is_deposit and native and apply_hypercore_cutoff:
        # PR #1660: the pre-collection period is an explicit assumption. Do not
        # label it observed open. PR #1670 preserves original permission clocks;
        # freshness must have been checked by the provider before arriving here.
        if timestamp is None or timestamp < HYPERCORE_DEPOSIT_STATE_CUTOFF:
            return VaultClosure(bool(unknown_is_closed), VaultClosureOutcome.unknown, "pre_collection_assumption",
                                "missing_history", "HyperCore admission assumed before reliable collection")

    if is_deposit and not trust_protocol_state and _is_backtesting_always_open_protocol(vault):
        # PR #1622 audited observations dated 2026-08-05 and corrected three bugs:
        # - Lagoon Syntropia Core (Ethereum 0x1b2cb79a4564206f53ba80b4d780f251b4ae6765)
        #   and Boosted (0x8df3deba711ae4a9af16cbca5e4fbb1402f036d5): ERC-7540
        #   maxDeposit counts settled claims, NOT acceptance of new requests.
        #   Rechecked 2026-10-27: zero maxima, unpaused, permissionless, request allowed.
        # - Morpho V2 Api3 dCOMP (0x36cfe1568461e499391ef0a555300f1ae2da2439)
        #   and Alpha Forex (0x153bd1abe60104bd46aa05a27fa12d1346d64a57): maxima
        #   deliberately always zero; actual gates/transaction decide admission.
        # - Yearn Chimi USDC (Base 0xcb2f26898c0893c0bdd5cf76417cf9b2258af0ed)
        #   and Turbo (0x0c352ab624313ae28ed4073dc5d469eb36e164c6): address(0)
        #   returns zero even while a real receiver has positive capacity.
        # Preserve the existing backtest assumption, but expose UNKNOWN, not OPEN.
        # Live checks below use actual request/access/owner-aware protocol adapters.
        return VaultClosure(bool(unknown_is_closed), VaultClosureOutcome.unknown, "protocol_assumption",
                            "unsupported_admission_history", "Generic historical fields do not establish deposit admission")

    flag_name, cap_name = ("deposits_open", "max_deposit") if is_deposit else ("redemption_open", "max_redeem")
    flag, cap = state.get(flag_name), state.get(cap_name)
    flag_known = _present(flag) and flag != -1
    cap = Decimal(str(cap)) if _present(cap) else None
    if not is_deposit and not trust_protocol_state and _is_backtesting_always_open_protocol(vault):
        # ERC-7540 maxRedeem describes claimable requests; Morpho V2 always
        # returns zero, and Yearn's zero-address maxRedeem has no owned shares.
        # None establishes a vault-wide redemption closure. Retain an explicit
        # redemption permission flag if present, but discard this generic cap.
        # Account lockups and actual payout liquidity remain live adapter checks.
        cap = None
    reason = state.get("deposit_closed_reason" if is_deposit else "redemption_closed_reason")
    reason = reason if isinstance(reason, str) and reason else None
    if is_deposit and native and apply_hypercore_cutoff and not flag_known:
        return VaultClosure(True if unknown_is_closed is None else unknown_is_closed, VaultClosureOutcome.unknown, "historical_state",
                            "missing_history", "HyperCore deposit availability is missing, unknown, or stale in historical data")
    if flag_known and flag == 0:
        return VaultClosure(True, VaultClosureOutcome.closed, "historical_state", "explicit_closed",
                            reason or f"Vault {operation}s closed in historical data", cap)
    if cap == 0:
        # HyperCore BitRocket, HYPErQuantum4 and Cold Process were all OPEN on
        # 2026-10-27, but our <5.5% leader-share safety policy set max_deposit=0.
        # PR #1671 explicitly preserves this amount restriction WITHOUT changing
        # exchange permission. Do not claim the exchange itself closes deposits.
        if is_deposit and native:
            return VaultClosure(include_strategy_policy, VaultClosureOutcome.policy_blocked, "historical_state", "strategy_capacity",
                                reason or "Recorded HyperCore deposit amount restricted by trading policy", cap)
        # Other protocols still honour actual zero capacity. Do not universally
        # ignore zeros: HYPE++, Earn ctUSD and Blackhole can have real restrictions.
        # A zero redemption cap is the historical model's operation restriction;
        # it is not proof of permanent shutdown or a new deposit restriction.
        return VaultClosure(True, VaultClosureOutcome.closed, "historical_state", "capacity_zero",
                            reason or f"Historical vault {operation} capacity was zero", cap)
    if not flag_known and cap is None:
        return VaultClosure(bool(unknown_is_closed), VaultClosureOutcome.unknown, "historical_state", "missing_history")
    return VaultClosure(False, VaultClosureOutcome.open, "historical_state", max_amount=cap)


def _from_deposit_check(result, include_strategy_policy, unknown_is_closed):
    """Normalise existing live adapter results without repeating protocol calls."""
    reason = result.reason_code.value if result.reason_code else None
    info = result.used_vault_info
    policy = bool(not result.can_deposit and reason == "vault_max_deposit_zero"
                  and info is not None and info.is_closed is False and info.allow_deposits is True)
    status = VaultClosureOutcome.unknown if reason == "unknown" else VaultClosureOutcome.open if result.can_deposit else VaultClosureOutcome.policy_blocked if policy else VaultClosureOutcome.closed
    blocked = not result.can_deposit
    if policy and not include_strategy_policy:
        blocked = False
    if status is VaultClosureOutcome.unknown and unknown_is_closed is not None:
        blocked = unknown_is_closed
    return VaultClosure(blocked, status, "live_adapter", reason, result.message, result.max_deposit)


def is_backtesting_deposit_open(pair, deposits_open, max_deposit):
    """Compatibility wrapper for callers that already selected their historical scope."""
    return can_interact_with_vault(pair, historical_state={"deposits_open": deposits_open, "max_deposit": max_deposit},
                               apply_hypercore_cutoff=False)


def check_backtesting_deposit(*, timestamp, pair, deposits_open, max_deposit, closed_reason, stage):
    """Compatibility wrapper; admission interpretation lives in get_vault_closure."""
    result = get_vault_closure(pair, timestamp=timestamp,
        historical_state={"deposits_open": deposits_open, "max_deposit": max_deposit, "deposit_closed_reason": closed_reason},
        apply_hypercore_cutoff=False)
    return result.as_deposit_check(pair, timestamp, stage)


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
            # Missing account context is uncertainty, not an observed closed vault.
            result.reason_code = DepositBlockReason.unknown
            result.message = "Cannot resolve owner address for asynchronous vault deposit request check"
        return result

    try:
        can_create_request = deposit_manager.can_create_deposit_request(owner)
    except NotImplementedError:
        can_create_request = None
    except VaultFlowUnavailable as error:
        result.can_deposit = False
        # An unsupported/unavailable adapter flow is not proof of closure.
        result.reason_code = DepositBlockReason.unknown
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
        # An unsupported/unavailable adapter flow is not proof of closure.
        result.reason_code = DepositBlockReason.unknown
        result.message = str(error)
        return result

    if isinstance(closed_reason, str) and closed_reason:
        result.can_deposit = False
        result.reason_code = DepositBlockReason.vault_deposits_closed
        result.message = closed_reason
        return result

    try:
        max_deposit = get_live_max_deposit(vault, owner)
    except VaultFlowUnavailable as error:
        result.can_deposit = False
        # An unsupported/unavailable adapter flow is not proof of closure.
        result.reason_code = DepositBlockReason.unknown
        result.message = str(error)
        return result

    result.max_deposit = float(max_deposit) if max_deposit is not None else None
    if max_deposit is not None and max_deposit <= 0:
        result.can_deposit = False
        result.reason_code = DepositBlockReason.vault_max_deposit_zero
        result.message = "Vault reports zero maxDeposit capacity for this owner"
    return result
