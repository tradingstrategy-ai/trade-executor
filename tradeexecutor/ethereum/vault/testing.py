"""Generic vault testing helpers."""

import datetime

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.pricing_model import PricingModel


PHASE_AWARE_LIVE_E2E_OBSERVATION_KEY = "phase_aware_live_e2e_observations"


def _enum_value(value) -> str | None:
    if value is None:
        return None
    return value.value if hasattr(value, "value") else str(value)


def collect_vault_availability(
    pricing_model: PricingModel,
    ts: datetime.datetime | None,
    pairs: list[TradingPairIdentifier],
) -> dict[str, dict]:
    """Collect vault availability through the generic pricing model path.

    This deliberately does not use raw Web3 reads. Protocol-specific request
    gates must be exposed by the vault's DepositManager implementation and
    reached through ``pricing_model.can_deposit()`` /
    ``pricing_model.check_redemption()``.
    """
    availability = {}
    for pair in pairs:
        redemption = pricing_model.check_redemption(ts, pair)
        availability[pair.pool_address.lower()] = {
            "can_deposit": pricing_model.can_deposit(ts, pair),
            "can_redeem": redemption.can_redeem,
            "redemption_reason": _enum_value(redemption.reason_code),
            "redemption_max": str(redemption.max_redemption) if redemption.max_redemption is not None else None,
        }
    return availability
