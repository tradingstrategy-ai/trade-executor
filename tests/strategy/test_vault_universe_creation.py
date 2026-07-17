"""Tests for shared vault-universe filtering rules."""

from dataclasses import replace

from tradeexecutor.curator.vault_universe_creation import VaultInfo, filter_vault


def test_filter_vault_excludes_subvaults_before_inclusion_overrides():
    """Exclude explicitly flagged and legacy Yearn Compounder sub-vaults.

    1. Build otherwise eligible vault records, including inclusion overrides.
    2. Run the shared universe filter for both sub-vault representations.
    3. Assert neither non-depositable vault can enter the selected universe.
    """
    # 1. Build records which would otherwise pass every selection criterion.
    base_vault = VaultInfo(
        name="Deposit-ready USDC vault",
        address="0x0000000000000000000000000000000000000001",
        chain_id=1,
        chain_name="Ethereum",
        denomination="USDC",
        age_years=1.0,
        cagr_periods={"1Y": 0.1},
        cagr_all=0.1,
        tvl=1_000_000.0,
        peak_tvl=1_000_000.0,
        risk="Minimal",
        flags=[],
        vault_display_flags=[],
        protocol_slug="yearn",
        deposit_closed_reason=None,
        must_include=True,
        excluded=False,
        excluded_protocol_reason=None,
    )
    flagged_subvault = replace(base_vault, name="Internal strategy vault", flags=["subvault"])
    legacy_yearn_subvault = replace(base_vault, name="Morpho Yearn USDC Compounder")

    filter_kwargs = dict(
        min_tvl=100_000.0,
        min_age=0.5,
        chain_config={1: {"name": "Ethereum"}},
        allowed_denominations={"USDC"},
        excluded_risks=set(),
        excluded_flags=set(),
        require_known_protocol=True,
        hypercore_min_tvl=100_000.0,
    )

    # 2. Run the shared filter for both data representations.
    flagged_result = filter_vault(flagged_subvault, **filter_kwargs)
    legacy_result = filter_vault(legacy_yearn_subvault, **filter_kwargs)

    # 3. Assert the sub-vault guard takes priority over must-include.
    assert flagged_result == (False, "subvault=Internal strategy vault")
    assert legacy_result == (False, "subvault=Morpho Yearn USDC Compounder")
