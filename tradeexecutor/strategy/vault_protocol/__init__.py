"""Keep strategy admission and simulated execution consistent about vault terms.

Resolve fees and directional settlement once during pair translation. Pricing
and execution also share the timing helper so per-run overrides affect both
entry-horizon checks and the resulting settlement requests.
"""

import datetime
from dataclasses import asdict, replace

from eth_defi.erc_4626.core import ERC4626Feature

from tradingstrategy.vault import VaultProtocolRules
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.vault_protocol import atoma, autopool, d2


#: Preserve historical waits for protocols lacking a positive metadata estimate.
#: These prevent immediate reuse of queued capital; they do not establish live claimability.
VAULT_SETTLEMENT_DELAY_FALLBACKS: dict[ERC4626Feature, datetime.timedelta] = {
    ERC4626Feature.atoma_like: datetime.timedelta(seconds=atoma.REDEMPTION_DELAY_SECONDS),
    ERC4626Feature.d2_like: datetime.timedelta(seconds=d2.SETTLEMENT_DELAY_SECONDS),
    ERC4626Feature.plutus_like: datetime.timedelta(days=14),
}
#: Keep historical Ostium runs on the existing next-day 18:00 UTC approximation
#: when no delay is supplied; changing live intervals must not rewrite that assumption.
OSTIUM_BACKTEST_SETTLEMENT_HOUR = 18


def get_vault_test_trade_time_skip_seconds(pair: TradingPairIdentifier, requested_seconds: int) -> int:
    """Apply the loaded vault holding-interval limit to a disposable-fork test.

    This controls the artificial wait between test entry and exit, not the
    strategy cycle or settlement rules. Preserve explicit shorter waits,
    including zero, and other protocols' existing intervals. A shorter wait
    does not replace the protocol's own oracle freshness checks.

    :param pair: Resolved pair selected by a public test-trade command.
    :param requested_seconds: Existing Anvil time advance requested by the caller.
    :return: Protocol-capped time advance, in seconds.
    """
    if pair.is_vault():
        rules = pair.get_vault_protocol_rules() or resolve_vault_protocol_rules(pair)
        if rules.simulation_max_holding_seconds is not None:
            return min(requested_seconds, rules.simulation_max_holding_seconds)
    return requested_seconds


def classify_vault_fee_profile(
    chain_id: int, address: str, deposit_fee: float | None, redemption_fee: float | None,
) -> str | None:
    """Separate a strategy's acceptable fee terms from the executor's ability to charge them.

    Rates are fractions of deposited or redeemed value. ``None`` means unknown,
    so an incomplete profile cannot be classified as zero-fee.

    :return:
        A profile name for selection filters, or ``None`` when either fee is
        unknown or the combination has no recognised profile.
    """
    if deposit_fee == 0 and redemption_fee == 0:
        return "zero_entry_exit"
    return atoma.fee_profile(chain_id, address, deposit_fee, redemption_fee)


def resolve_vault_protocol_rules(pair: TradingPairIdentifier) -> VaultProtocolRules:
    """Resolve fees and waits centrally so strategies need no protocol-specific branches.

    Published positive settlement estimates take precedence over protocol
    simulation defaults. Deposit and redemption queues are resolved separately:
    a delayed exit must not queue an otherwise immediate deposit.

    Unsupported or missing rates are retained for the universe validator to
    reject. Translation itself must also work for data-only vaults and existing
    strategies that have not opted into strict fee validation.
    """
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
    # Preserve the established timing policy: a reported estimate must not
    # replace schedules for protocol families that never used it.
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
        redemption_delay_is_deposit_lock=(
            atoma.REDEMPTION_DELAY_IS_DEPOSIT_LOCK if ERC4626Feature.atoma_like in features else False
        ),
        simulation_max_holding_seconds=(
            autopool.SIMULATION_CYCLE_SECONDS if ERC4626Feature.autopool_like in features else None
        ),
    )


def enrich_vault_pair(pair: TradingPairIdentifier) -> TradingPairIdentifier:
    """Make one resolved rule set available to both fresh and restored pairs.

    Replace frozen metadata rather than modifying the source DEXPair's shared
    instance. Keep a plain snapshot because State omits the full metadata.
    Non-vault pairs are returned unchanged. The input pair is mutated in place
    so the translation cache and callers retain the same object.
    """
    if pair.is_vault():
        rules = resolve_vault_protocol_rules(pair)
        metadata = pair.get_vault_metadata()
        if metadata is not None:
            pair.other_data["token_metadata"] = replace(metadata, vault_protocol_rules=rules)
        pair.other_data["vault_protocol_rules"] = asdict(rules)
    return pair


def settlement_expectation(
    pair: TradingPairIdentifier,
    timestamp: datetime.datetime,
    *,
    is_buy: bool,
    overrides: dict[str, datetime.timedelta] | None = None,
    default_delay: datetime.timedelta = datetime.timedelta(days=2),
) -> tuple[bool, datetime.datetime]:
    """Use one timing policy for strategy horizon checks and simulated requests.

    ``timestamp`` is the naive UTC request time; ``is_buy`` selects deposits.
    Overrides use lower-case vault addresses and queue both directions, even
    when the protocol would settle that operation immediately. This includes
    zero-delay overrides. Without an override, use the loaded estimate, protocol
    delay or next-day settlement hour, then ``default_delay``.
    The result is a lower bound: historical events and claim resolvers may defer
    settlement. Immediate operations return the request time.
    """
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
