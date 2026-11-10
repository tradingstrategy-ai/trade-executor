"""Pure construction-time vault overlays; no acquisition, state ledger or strategy imports."""

import datetime
from dataclasses import asdict

from eth_defi.erc_4626.core import ERC4626Feature

from tradeexecutor.state.vault_protocol import VaultProtocolRules
from tradeexecutor.strategy.vault_protocol import atoma, d2


#: Existing historical minimum delays when a positive protocol estimate is absent.
VAULT_SETTLEMENT_DELAY_FALLBACKS: dict[ERC4626Feature, datetime.timedelta] = {
    ERC4626Feature.atoma_like: datetime.timedelta(seconds=atoma.REDEMPTION_DELAY_SECONDS),
    ERC4626Feature.d2_like: datetime.timedelta(seconds=d2.SETTLEMENT_DELAY_SECONDS),
    ERC4626Feature.plutus_like: datetime.timedelta(days=14),
}
#: Preserve Ostium's next-day UTC backtest approximation; live claims use the adapter.
OSTIUM_BACKTEST_SETTLEMENT_HOUR = 18


def classify_vault_fee_profile(chain_id, address, deposit_fee, redemption_fee):
    """Resolve contractual fee terms without changing a strategy's allowed profiles."""
    if deposit_fee == 0 and redemption_fee == 0:
        return "zero_entry_exit"
    return atoma.fee_profile(chain_id, address, deposit_fee, redemption_fee)


def resolve_vault_protocol_rules(pair) -> VaultProtocolRules:
    """Resolve data once, preserving the existing framework's operation classification."""
    metadata = pair.get_vault_metadata()
    features = pair.get_vault_features() or set()
    deposit_fee = getattr(metadata, "deposit_fee", None)
    redemption_fee = getattr(metadata, "withdrawal_fee", None)
    profile = "zero_entry_exit" if deposit_fee == 0 and redemption_fee == 0 else None
    if ERC4626Feature.atoma_like in features:
        profile = atoma.fee_profile(pair.chain_id, pair.pool_address, deposit_fee, redemption_fee) or profile
    # Exit support is independent of entry completeness. A missing entry fee
    # still makes the whole pair data-only under the explicit override policy.
    supported_exit = redemption_fee == 0 or (
        ERC4626Feature.atoma_like in features
        and atoma.fee_profile(pair.chain_id, pair.pool_address, 0, redemption_fee) == "atoma_standard_exit"
    )
    event_gated = bool(features & {ERC4626Feature.d2_like, ERC4626Feature.lagoon_like})
    queued_deposit = pair.is_async_vault() or event_gated
    queued_redemption = pair.has_delayed_vault_redemption()
    fallback = next((delay.total_seconds() for feature, delay in VAULT_SETTLEMENT_DELAY_FALLBACKS.items()
                     if feature in features), None)
    uses_estimate = fallback is not None or bool(features & {ERC4626Feature.lagoon_like, ERC4626Feature.upshift_like})
    estimate = pair.get_vault_estimated_settlement() if uses_estimate else None
    delay = estimate.total_seconds() if estimate is not None else fallback
    hour = OSTIUM_BACKTEST_SETTLEMENT_HOUR if ERC4626Feature.ostium_like in features and delay is None else None
    return VaultProtocolRules(
        deposit_fee=deposit_fee, redemption_fee=redemption_fee,
        deposit_fee_supported=deposit_fee == 0,
        redemption_fee_supported=supported_exit,
        fee_profile=profile, deposit_queued=queued_deposit, redemption_queued=queued_redemption,
        deposit_delay_seconds=delay if queued_deposit else 0,
        redemption_delay_seconds=delay if queued_redemption else 0,
        settlement_hour_utc=hour, deposit_requires_settlement_event=event_gated,
        deposit_windowed=ERC4626Feature.d2_like in features,
        redemption_windowed=ERC4626Feature.d2_like in features,
        provenance=pair.get_vault_protocol() or "generic_erc4626",
    )


def enrich_vault_pair(pair):
    """Store resolved economics in pair metadata before the translator caches it."""
    if pair.is_vault():
        pair.other_data["vault_protocol_rules"] = asdict(resolve_vault_protocol_rules(pair))
    return pair


def settlement_expectation(pair, timestamp, *, is_buy, overrides=None, default_delay=datetime.timedelta(days=2)):
    """Return (queued, minimum due time) using existing override and protocol precedence."""
    if not pair.is_vault():
        return False, timestamp
    override = (overrides or {}).get(pair.pool_address.lower())
    if override is not None:
        return True, timestamp + override
    rules = pair.get_vault_protocol_rules() or resolve_vault_protocol_rules(pair)
    # Pair estimates are mutable request-time inputs, including values loaded by
    # translation. Refresh timing; persisted requests retain their original due time.
    if pair.other_data.get("vault_estimated_settlement") is not None:
        rules = resolve_vault_protocol_rules(pair)
    queued = rules.deposit_queued if is_buy else rules.redemption_queued
    if not queued:
        return False, timestamp
    delay = rules.deposit_delay_seconds if is_buy else rules.redemption_delay_seconds
    if delay is not None:
        return True, timestamp + datetime.timedelta(seconds=delay)
    if rules.settlement_hour_utc is not None:
        return True, (timestamp + datetime.timedelta(days=1)).replace(hour=rules.settlement_hour_utc, minute=0, second=0, microsecond=0)
    return True, timestamp + default_delay
