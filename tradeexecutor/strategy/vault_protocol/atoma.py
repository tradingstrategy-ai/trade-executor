"""Atoma's standard public-account economics, shared by all strategy consumers.

Deposits are immediate. Redemptions use the existing epoch resolver; seven
days is the existing backtest estimate, not a guaranteed live claim time.
Fee exemptions for fee-recipient accounts stay with the live deposit manager.
"""

#: Deployments with confirmed zero-entry/0.5%-exit terms; other addresses stay unclassified.
STANDARD_FEE_VAULTS = {
    (42161, "0x1c788e14d8e5b446e3f71b5142e2edabcab36da1"),
    (42161, "0xcc56410e1a136af0eceb7241c6ae394f4d8b581c"),
}
#: Existing conservative redemption estimate when published timing is absent.
REDEMPTION_DELAY_SECONDS = 7 * 86400


def fee_profile(chain_id: int, address: str, deposit_fee: float | None, redemption_fee: float | None) -> str | None:
    """Classify confirmed standard public terms; other fee-bearing deployments return None."""
    if (chain_id, address.lower()) in STANDARD_FEE_VAULTS and deposit_fee == 0 and redemption_fee == 0.005:
        return "atoma_standard_exit"
    return None
