"""Typed view of resolved vault economics stored in a pair's existing other_data."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VaultProtocolRules:
    """Protocol defaults and loaded fees; amounts are fractions, delays are seconds.

    Window capability does not establish current availability. Existing ticket
    managers remain authoritative for live claims and request settlement.
    """

    #: Loaded entry charge; None means unknown and never implies zero.
    deposit_fee: float | None = None
    #: Loaded exit charge; kept separate from manager/performance fees.
    redemption_fee: float | None = None
    #: Whether the shared execution path can model the entry fee.
    deposit_fee_supported: bool = False
    #: Whether the shared execution path can model the exit fee.
    redemption_fee_supported: bool = False
    #: Economic terms profile, used separately from execution capability.
    fee_profile: str | None = None
    #: Immediate versus queued deposit lifecycle, including historical event gates.
    deposit_queued: bool = False
    #: Immediate versus queued redemption lifecycle.
    redemption_queued: bool = False
    #: Minimum deposit delay; zero for immediate, None for global timing default.
    deposit_delay_seconds: float | None = None
    #: Minimum redemption delay; zero for immediate, None for global timing default.
    redemption_delay_seconds: float | None = None
    #: Existing next-day UTC settlement hour for Ostium-style requests.
    settlement_hour_utc: int | None = None
    #: Whether deposits also require a historical settlement event.
    deposit_requires_settlement_event: bool = False
    #: Whether closed deposits can represent a temporary protocol window.
    deposit_windowed: bool = False
    #: Whether redemptions are subject to protocol windows.
    redemption_windowed: bool = False
    #: Named source of default rules; individual fee amounts come from metadata.
    provenance: str = "metadata"
