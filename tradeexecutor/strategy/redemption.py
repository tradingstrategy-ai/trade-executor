"""Structured redemption diagnostics for pricing and alpha-model decisions."""

import datetime
import enum
from dataclasses import dataclass
from decimal import Decimal

from dataclasses_json import dataclass_json
from eth_defi.compat import native_datetime_utc_fromtimestamp


class RedemptionCheckStage(enum.Enum):
    """Where the strategy asked whether a position can be redeemed."""

    unknown = "unknown"
    carry_forward = "carry_forward"
    sell_rebalance = "sell_rebalance"


class RedemptionBlockReason(enum.Enum):
    """Stable reason codes for redemption diagnostics."""

    safe_address_unavailable = "safe_address_unavailable"
    user_equity_fetch_failed = "user_equity_fetch_failed"
    user_lockup_not_expired = "user_lockup_not_expired"
    vault_max_withdrawable_zero = "vault_max_withdrawable_zero"


@dataclass_json
@dataclass(slots=True)
class VaultUserEquitySnapshot:
    """Compact vault user equity payload used in diagnostics."""

    #: Vault address used in the user equity lookup.
    vault_address: str | None = None
    #: Current equity reported by Hyperliquid for this user and vault.
    equity: Decimal | None = None
    #: Naive UTC timestamp when the user lockup expires.
    locked_until: datetime.datetime | None = None
    #: Whether Hyperliquid reports the user lockup as already expired.
    is_lockup_expired: bool | None = None


@dataclass_json
@dataclass(slots=True)
class VaultInfoSnapshot:
    """Compact vault info payload used in diagnostics."""

    #: Human-readable vault name from ``vaultDetails``.
    name: str | None = None
    #: Vault address returned by the Hyperliquid API.
    vault_address: str | None = None
    #: Maximum currently withdrawable amount reported by the vault.
    max_withdrawable: Decimal | None = None
    #: Maximum currently distributable amount reported by the vault.
    max_distributable: Decimal | None = None
    #: Whether the vault is permanently closed.
    is_closed: bool | None = None
    #: Whether the leader currently allows deposits.
    allow_deposits: bool | None = None
    #: Relationship type reported by Hyperliquid for the vault.
    relationship_type: str | None = None
    #: Leader share of the vault capital as a float ratio.
    leader_fraction: float | None = None
    #: Number of followers returned in the API payload.
    follower_count: int | None = None
    #: Optional parent vault address for parent-child vault layouts.
    parent: str | None = None


@dataclass_json
@dataclass(slots=True)
class VaultMetadataSnapshot:
    """Vault pair and position metadata used in redemption checks."""

    #: Protocol slug stored on the trading pair metadata.
    vault_protocol: str | None = None
    #: Human-readable vault name stored on the trading pair metadata.
    vault_name: str | None = None
    #: Human-readable reason why deposits are closed, if known.
    deposit_closed_reason: str | None = None
    #: Human-readable reason why redemptions are closed, if known.
    redemption_closed_reason: str | None = None
    #: Next known redemption-open timestamp from the pair metadata.
    redemption_next_open: datetime.datetime | None = None
    #: Whether this pair targets the Hyperliquid testnet API.
    exchange_is_testnet: bool | None = None
    #: Lockup expiry recorded earlier on the open position state.
    position_recorded_lockup_expires_at: datetime.datetime | None = None


@dataclass_json
@dataclass(slots=True)
class RedemptionApiSnapshot:
    """Decision-relevant vault API payloads."""

    #: Compact snapshot of ``vaultDetails``.
    vault_info: VaultInfoSnapshot | None = None
    #: Compact snapshot of ``fetch_user_vault_equity()``.
    user_equity: VaultUserEquitySnapshot | None = None


@dataclass_json
@dataclass(slots=True)
class RedemptionCheckResult:
    """Serialisable result of a redemption check."""

    #: Strategy-cycle timestamp when the check ran.
    timestamp: datetime.datetime | None = None
    #: Where in the alpha-model flow the check was requested.
    stage: RedemptionCheckStage = RedemptionCheckStage.unknown
    #: Final boolean decision used by the caller.
    can_redeem: bool = True
    #: Stable reason code for the decision, when available.
    reason_code: RedemptionBlockReason | None = None
    #: Human-readable explanation of the decision.
    message: str | None = None
    #: Pair ticker shown in logs and diagnostics.
    pair_ticker: str | None = None
    #: Vault address for the checked position.
    vault_address: str | None = None
    #: Safe address used for the lookup, when available.
    safe_address: str | None = None
    #: Configured redeem delay in seconds, if known.
    configured_redeem_delay_seconds: int | None = None
    #: Origin of the configured redeem delay field, if known.
    configured_redeem_delay_source: str | None = None
    #: Lockup expiry recorded earlier on the position state.
    position_recorded_lockup_expires_at: datetime.datetime | None = None
    #: Latest lockup expiry returned by the live user equity lookup.
    user_lockup_expires_at: datetime.datetime | None = None
    #: Raw vault-side max withdrawable amount used in the decision.
    max_withdrawable: Decimal | None = None
    #: Final max redeemable amount exposed to callers.
    max_redemption: Decimal | None = None
    #: Compact raw API payloads used for the decision.
    raw_api_data: RedemptionApiSnapshot | None = None
    #: Pair and position metadata used for the decision.
    used_vault_metadata: VaultMetadataSnapshot | None = None


@dataclass_json
@dataclass(slots=True)
class BlockedRedemptionSummary:
    """Compact blocked-redemption entry persisted in cycle calculations."""

    #: Pair ticker for the blocked position.
    pair_ticker: str | None
    #: Vault address for the blocked position.
    vault_address: str | None
    #: Stage where the block happened.
    stage: str
    #: Stable reason code for the block.
    reason_code: str | None
    #: Human-readable explanation of the block.
    message: str | None
    #: Safe address used for the live lookup, if available.
    safe_address: str | None
    #: Lockup expiry previously recorded on the position state.
    position_recorded_lockup_expires_at: datetime.datetime | None
    #: Lockup expiry returned by the latest user equity lookup.
    user_lockup_expires_at: datetime.datetime | None
    #: Vault max withdrawable amount formatted for compact storage.
    max_withdrawable: str | None
    #: Final max redemption amount formatted for compact storage.
    max_redemption: str | None


@dataclass_json
@dataclass(slots=True)
class RedemptionCycleDiagnostics:
    """Compact per-cycle blocked-redemption diagnostics."""

    #: Strategy cycle number that produced these diagnostics.
    cycle: int
    #: Capital that the allocator considered redeemable this cycle.
    redeemable_capital: float
    #: Capital pinned because blocked positions had to be carried forward.
    locked_capital_carried_forward: float
    #: Count of distinct signals blocked by redemption checks.
    blocked_signal_count: int
    #: Grouped blocked reason counts keyed by reason code value.
    reason_counts: dict[str, int]
    #: Compact blocked-redemption entries for this cycle.
    blocked_redemptions: list[BlockedRedemptionSummary]


def parse_recorded_lockup_expires_at(value: object) -> datetime.datetime | None:
    """Parse a stored ISO timestamp from ``position.other_data`` if present."""
    if value is None:
        return None

    if isinstance(value, datetime.datetime):
        return value

    if isinstance(value, str):
        return datetime.datetime.fromisoformat(value)

    if isinstance(value, (int, float)):
        return native_datetime_utc_fromtimestamp(value)

    raise TypeError(f"Unsupported lockup expiry value: {value!r}")
