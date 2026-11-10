"""Manual Hyperliquid vault-data overrides.

These are explicit operator decisions, not reconstructed historical permission
observations. Keep closed vault histories available for valuation and preserve
the normal redemption checks for existing holdings.
"""

from tradeexecutor.state.identifier import TradingPairIdentifier


#: All-time deposit closures, keyed by native venue and exact vault address.
#: Growi HF-2 was marked permanently closed by the operator on 10 November
#: 2026. Its recorded permissions contain no observed-open confirmation; early
#: NB04 purchases relied solely on the pre-collection assumed-open policy.
PERMANENT_DEPOSIT_CLOSURES: dict[tuple[int, str], str] = {
    (9999, "0x9b5b1931fa5838a891973f9877c7648c93ee80cc"):
        "Growi HF-2 deposits are permanently closed by manual override (2026-11-10)",
}


def permanent_deposit_closure_reason(pair: TradingPairIdentifier) -> str | None:
    """Return an explicit closure for this native vault identity, if configured."""
    native = getattr(pair, "is_hyperliquid_vault", None)
    if not callable(native) or not native():
        return None
    return PERMANENT_DEPOSIT_CLOSURES.get((getattr(pair, "chain_id", None), pair.pool_address.lower()))
