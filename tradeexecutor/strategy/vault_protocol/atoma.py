"""Atoma's standard public-account economics, shared by all strategy consumers.

Deposits are immediate. Redemptions use the existing epoch resolver; seven
days is the existing backtest estimate, not a guaranteed live claim time.
Fee exemptions for fee-recipient accounts stay with the live deposit manager.
"""

#: Limit the zero-entry/0.5%-exit profile to reviewed deployments: a shared
#: protocol name alone does not establish another vault's fee terms.
STANDARD_FEE_VAULTS = {
    (42161, "0x1c788e14d8e5b446e3f71b5142e2edabcab36da1"),
    (42161, "0xcc56410e1a136af0eceb7241c6ae394f4d8b581c"),
}
#: Retain the existing seven-day backtest wait when metadata supplies no estimate,
#: so a redemption request does not immediately release cash for reinvestment.
REDEMPTION_DELAY_SECONDS = 7 * 86400


def fee_profile(chain_id: int, address: str, deposit_fee: float | None, redemption_fee: float | None) -> str | None:
    """Require both a reviewed identity and matching rates before admitting this profile.

    An address match must not override changed or missing fee data. Rates are
    fractions of deposited or redeemed value. Return ``None`` for unrecognised
    terms so selection cannot treat the identity alone as sufficient evidence.
    """
    if (chain_id, address.lower()) in STANDARD_FEE_VAULTS and deposit_fee == 0 and redemption_fee == 0.005:
        return "atoma_standard_exit"
    return None
