"""Test trade-ui vault deposit status for xchain-master-vault ERC-4626 pairs.

Before this fix, non-Hyperliquid vault pairs always showed Deposits: Yes
because ``TradingPairIdentifier.can_deposit()`` only checked
``deposit_closed_reason`` for Hyperliquid vaults, and the TUI never
consulted the live pricing model.
"""

import datetime

from eth_defi.erc_4626.core import ERC4626Feature

from tradeexecutor.cli.trade_ui_tui import (
    _format_deposits_open,
    _format_redemptions_open,
    _get_deposit_status,
    _get_redemption_status,
)
from tradeexecutor.state.identifier import (
    AssetIdentifier,
    TradingPairIdentifier,
    TradingPairKind,
)


def _make_erc4626_vault_pair(
    *,
    chain_id: int = 8453,
    vault_address: str = "0x0000000000000000000000000000000000000001",
    vault_name: str = "Test USDC Vault",
    protocol_slug: str = "lagoon-finance",
    deposit_closed_reason: str | None = None,
    redemption_closed_reason: str | None = None,
    vault_features: list[str] | None = None,
    token_metadata: dict | None = None,
    internal_id: int = 1,
) -> TradingPairIdentifier:
    """Create a vault pair resembling xchain-master-vault ERC-4626 vaults."""
    base = AssetIdentifier(
        chain_id=chain_id,
        address=vault_address,
        token_symbol="vUSDC",
        decimals=18,
    )
    quote = AssetIdentifier(
        chain_id=chain_id,
        address="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
        token_symbol="USDC",
        decimals=6,
    )
    return TradingPairIdentifier(
        base=base,
        quote=quote,
        pool_address=vault_address,
        exchange_address="0x0000000000000000000000000000000000000000",
        internal_id=internal_id,
        internal_exchange_id=1,
        fee=0.0,
        kind=TradingPairKind.vault,
        exchange_name=protocol_slug,
        other_data={
            "vault_protocol": protocol_slug,
            "vault_name": vault_name,
            "deposit_closed_reason": deposit_closed_reason,
            "redemption_closed_reason": redemption_closed_reason,
            "vault_features": vault_features,
            "token_metadata": token_metadata,
        },
    )


def test_tui_erc4626_deposit_status_respects_closed_reason() -> None:
    """ERC-4626 vaults with deposit_closed_reason show "No" in the TUI.

    This is the regression test for the core bug: before the fix,
    ``can_deposit()`` always returned True for non-Hyperliquid vaults,
    so the TUI showed Deposits: Yes even for capped vaults.

    1. Create two ERC-4626 vault pairs mimicking the xchain-master-vault
       multi-chain universe: one open on Base, one closed on Ethereum.
    2. Resolve deposit status through the TUI helpers (static fallback path).
    3. Verify the open vault renders "Yes" and the closed vault renders "No".
    """
    # 1. Create two ERC-4626 vault pairs: one open, one closed.
    open_vault = _make_erc4626_vault_pair(
        chain_id=8453, vault_name="Open Base Vault",
        deposit_closed_reason=None, internal_id=1,
    )
    closed_vault = _make_erc4626_vault_pair(
        chain_id=1, vault_name="Capped ETH Vault",
        deposit_closed_reason="Max deposit cap reached", internal_id=2,
    )

    # 2. Resolve deposit status through the TUI helpers (static fallback path).
    open_status = _get_deposit_status(None, open_vault)
    closed_status = _get_deposit_status(None, closed_vault)

    # 3. Verify the open vault renders "Yes" and the closed vault renders "No".
    assert open_status is True
    assert closed_status is False
    assert _format_deposits_open(open_vault, open_status).plain == "Yes"
    assert _format_deposits_open(closed_vault, closed_status).plain == "No"


def test_tui_erc4626_redemption_status_respects_closed_reason() -> None:
    """ERC-4626 vaults with redemption_closed_reason show "No" in the TUI.

    1. Create two ERC-4626 vault pairs: one redeemable, one redemption-closed.
    2. Resolve redemption status through the TUI helpers.
    3. Verify the open vault renders "Yes" and the closed vault renders "No".
    """

    # 1. Create two ERC-4626 vault pairs: one redeemable, one redemption-closed.
    open_vault = _make_erc4626_vault_pair(
        chain_id=8453,
        vault_name="Redeemable Base Vault",
        redemption_closed_reason=None,
        internal_id=1,
    )
    closed_vault = _make_erc4626_vault_pair(
        chain_id=1,
        vault_name="Redemption Closed ETH Vault",
        redemption_closed_reason="Redemptions are closed during the trading epoch",
        internal_id=2,
    )

    # 2. Resolve redemption status through the TUI helpers.
    open_status = _get_redemption_status(None, open_vault)
    closed_status = _get_redemption_status(None, closed_vault)

    # 3. Verify the open vault renders "Yes" and the closed vault renders "No".
    assert open_status is True
    assert closed_status is False
    assert _format_redemptions_open(open_vault, open_status).plain == "Yes"
    assert _format_redemptions_open(closed_vault, closed_status).plain == "No"


def test_tui_redemption_status_uses_live_pricing_model() -> None:
    """Live pricing model redemption status takes priority over metadata.

    1. Create an ERC-4626 vault pair whose metadata says redemptions are open.
    2. Mock a pricing model that reports redemptions closed.
    3. Verify the TUI helper returns the live closed status.
    """

    class PricingModelStub:
        def can_redeem(self, ts, pair):
            return False

    # 1. Create an ERC-4626 vault pair whose metadata says redemptions are open.
    pair = _make_erc4626_vault_pair(redemption_closed_reason=None)

    # 2. Mock a pricing model that reports redemptions closed.
    status = _get_redemption_status(PricingModelStub(), pair, ts=datetime.datetime(2026, 6, 1))

    # 3. The TUI helper returns the live closed status.
    assert status is False
    assert _format_redemptions_open(pair, status).plain == "No"


def test_tui_redemption_status_falls_back_for_supported_vault_providers() -> None:
    """Requested vault providers expose redemption metadata in the TUI.

    1. Create vault pairs for Lagoon, Euler, Morpho, D2, Plutus, Ostium, and Gains.
    2. Resolve redemption status through the metadata fallback path.
    3. Verify closed metadata renders as "No" for each provider.
    """

    # 1. Create vault pairs for the requested providers.
    provider_slugs = [
        "lagoon",
        "euler",
        "morpho",
        "d2",
        "plutus",
        "ostium",
        "gains",
    ]
    pairs = [
        _make_erc4626_vault_pair(
            protocol_slug=slug,
            vault_name=f"{slug} vault",
            redemption_closed_reason=f"{slug} redemptions closed",
            internal_id=idx,
        )
        for idx, slug in enumerate(provider_slugs, start=1)
    ]

    # 2. Resolve redemption status through the metadata fallback path.
    statuses = [_get_redemption_status(None, pair) for pair in pairs]

    # 3. Closed metadata renders as "No" for each provider.
    rendered_statuses = [
        _format_redemptions_open(pair, status).plain
        for pair, status in zip(pairs, statuses)
    ]
    assert statuses == [False] * len(provider_slugs)
    assert rendered_statuses == ["No"] * len(provider_slugs)


def test_tui_erc4626_vault_features_are_normalised_from_remote_metadata() -> None:
    """ERC-4626 vault feature strings resolve to enum values needed by routing.

    This covers the production metadata shape from the xchain-master-vault
    trade-ui failure. The older Ostium async integration tests used a live
    autodetected vault instance, so they carried ``ERC4626Feature`` enum
    objects directly and did not exercise JSON-style remote metadata.

    1. Create an Ostium vault pair whose top-level vault_features are strings.
    2. Create another Ostium vault pair whose features only exist in token_metadata.
    3. Verify both pairs expose enum features for vault routing.
    """
    # 1. Create an Ostium vault pair whose top-level vault_features are strings.
    top_level_features = _make_erc4626_vault_pair(
        chain_id=42161,
        vault_address="0x20d419a8e12c45f88fda7c5760bb6923cee27f98",
        vault_name="Ostium Liquidity Pool Vault",
        protocol_slug="ostium",
        vault_features=["ostium_like"],
    )

    # 2. Create another Ostium vault pair whose features only exist in token_metadata.
    nested_features = _make_erc4626_vault_pair(
        chain_id=42161,
        vault_address="0x20d419a8e12c45f88fda7c5760bb6923cee27f98",
        vault_name="Ostium Liquidity Pool Vault",
        protocol_slug="ostium",
        token_metadata={"features": ["ostium_like"]},
    )

    # 3. Verify both pairs expose enum features for vault routing.
    assert top_level_features.get_vault_features() == {ERC4626Feature.ostium_like}
    assert nested_features.get_vault_features() == {ERC4626Feature.ostium_like}


def test_tui_erc4626_empty_vault_features_are_preserved() -> None:
    """ERC-4626 empty vault feature metadata marks a known synchronous vault.

    Synchronous ERC-4626 queue vaults deliberately carry an empty feature list.
    The empty list must not be collapsed to ``None``, because ``None`` means
    the vault features are unknown and routing may need slow autodetection.

    1. Create a vault pair with an explicit empty top-level vault_features list.
    2. Include async-looking token metadata to prove top-level metadata wins.
    3. Verify the pair exposes an empty feature set.
    """
    # 1. Create a vault pair with an explicit empty top-level vault_features list.
    sync_vault = _make_erc4626_vault_pair(
        chain_id=31337,
        vault_address="0x0000000000000000000000000000000000007540",
        vault_name="Cash Allocation Vault",
        protocol_slug="erc4626",
        vault_features=[],
        token_metadata={"features": ["ostium_like"]},
    )

    # 2. Include async-looking token metadata to prove top-level metadata wins.
    assert sync_vault.other_data["token_metadata"]["features"] == ["ostium_like"]

    # 3. Verify the pair exposes an empty feature set.
    assert sync_vault.get_vault_features() == set()
    assert sync_vault.is_async_vault() is False


def test_tui_erc4626_empty_vault_features_without_protocol_are_conservative() -> None:
    """ERC-4626 empty vault feature metadata without a sync marker is unknown.

    Empty features from remote or legacy metadata are ambiguous. Without an
    explicit generic ERC-4626 protocol marker, treat the vault as async for
    non-routing safety checks so cash is not reused before autodetection can
    classify the vault.

    1. Create a vault pair with an empty top-level vault_features list and no protocol slug.
    2. Verify the empty feature set is preserved for routing.
    3. Verify non-routing async checks treat the unknown vault conservatively.
    """
    # 1. Create a vault pair with an empty top-level vault_features list and no protocol slug.
    unknown_vault = _make_erc4626_vault_pair(
        chain_id=31337,
        vault_address="0x0000000000000000000000000000000000007541",
        vault_name="Unknown Vault",
        protocol_slug=None,
        vault_features=[],
    )

    # 2. Verify the empty feature set is preserved for routing.
    assert unknown_vault.get_vault_features() == set()

    # 3. Verify non-routing async checks treat the unknown vault conservatively.
    assert unknown_vault.is_async_vault() is True
