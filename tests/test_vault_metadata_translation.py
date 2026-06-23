"""Test vault metadata translation into trade-executor pair identifiers."""

import pytest
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import ExchangeType
from tradingstrategy.pair import DEXPair
from tradingstrategy.vault import VaultMetadata

from tradeexecutor.strategy.dex_data_translation import translate_trading_pair


def _make_vault_dex_pair(metadata: VaultMetadata) -> DEXPair:
    """Build a vault DEX pair with attached metadata."""
    return DEXPair(
        pair_id=1,
        chain_id=ChainId.ethereum,
        exchange_id=1,
        address="0x0000000000000000000000000000000000000001",
        token0_address="0x0000000000000000000000000000000000000001",
        token0_symbol="vUSDC",
        token0_decimals=18,
        token1_address="0x0000000000000000000000000000000000000002",
        token1_symbol="USDC",
        token1_decimals=6,
        base_token_symbol="vUSDC",
        quote_token_symbol="USDC",
        dex_type=ExchangeType.erc_4626_vault,
        exchange_slug="morpho",
        exchange_name="morpho",
        fee=0,
        other_data={"token_metadata": metadata},
    )


def test_translate_trading_pair_preserves_vault_display_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Vault display flags are reduced from metadata into pair other data.

    1. Build a vault metadata object and simulate a newer metadata attribute.
    2. Translate a DEX pair carrying the metadata.
    3. Assert the translated pair exposes the flags to trade-ui.
    """

    # 1. Build a vault metadata object and simulate a newer metadata attribute.
    display_flags = [
        {"severity": "yellow", "type": "not_whitelisted", "source": "morpho"},
    ]
    metadata = VaultMetadata(
        vault_name="Morpho USDC",
        protocol_name="Morpho",
        protocol_slug="morpho",
        features=[],
    )
    # Use class-level monkeypatching so this test works both before and after
    # trading-strategy adds the field to the slotted frozen VaultMetadata class.
    monkeypatch.setattr(VaultMetadata, "vault_display_flags", display_flags, raising=False)

    # 2. Translate a DEX pair carrying the metadata.
    pair = translate_trading_pair(_make_vault_dex_pair(metadata))

    # 3. The translated pair exposes the flags to trade-ui.
    assert pair.other_data["vault_display_flags"] == display_flags


def test_translate_trading_pair_preserves_vault_lockup_days() -> None:
    """Vault lockup days are reduced from metadata into pair other data.

    Steps:
    1. Build a vault metadata object with a lockup duration.
    2. Translate a DEX pair carrying the metadata.
    3. Assert the translated pair exposes the lockup days to trade-ui.
    """

    # 1. Build a vault metadata object with a lockup duration.
    metadata = VaultMetadata(
        vault_name="Plutus plHEDGE",
        protocol_name="PlutusDAO",
        protocol_slug="plutus",
        features=[],
        lockup_days=30.0,
    )

    # 2. Translate a DEX pair carrying the metadata.
    pair = translate_trading_pair(_make_vault_dex_pair(metadata))

    # 3. The translated pair exposes the lockup days to trade-ui.
    assert pair.other_data["vault_lockup_days"] == 30.0
