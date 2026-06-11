"""Regression coverage for Hypercore exchange type compatibility.

1. Create a handcrafted Hypercore vault pair
2. Export it through the pair-universe code path used by strategies
3. Verify the pair still round-trips as a vault using the current Trading Strategy exchange type
"""

import pytest
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import ExchangeType
from tradingstrategy.pair import DEXPair

from tradeexecutor.ethereum.vault.hypercore_vault import create_hypercore_vault_pair
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.strategy.dex_data_translation import translate_trading_pair
from tradeexecutor.strategy.trading_strategy_universe import create_pair_universe_from_code


def _make_vault_dex_pair(chain_id: ChainId) -> DEXPair:
    """Build a vault DEX pair whose share/base token has no decimals.

    Mirrors the remote-data shape for a Hypercore-native vault (e.g. HLP):
    token0 is the USDC denomination (6 decimals), token1 is the share token
    with ``None`` decimals because there is no on-chain ERC-20 share token.
    """
    return DEXPair(
        pair_id=269349635,
        chain_id=chain_id,
        exchange_id=1,
        address="0xdfc24b077bc1425ad1dea75bcb6f8158e10df303",
        token0_address="0xb88339cb7199b77e23db6e890353e22632ba630f",
        token0_symbol="USDC",
        token0_decimals=6,
        token1_address="0xdfc24b077bc1425ad1dea75bcb6f8158e10df303",
        token1_symbol="Hyperliqui",
        token1_decimals=None,
        base_token_symbol="Hyperliqui",
        quote_token_symbol="USDC",
        dex_type=ExchangeType.erc_4626_vault,
        exchange_slug="hyperliquid",
        pair_slug="hlp",
        fee=0,
    )


def test_hypercore_pair_translation_uses_current_exchange_type():
    """Verify Hypercore pairs survive translation using the current exchange type.

    1. Build a handcrafted Hypercore vault pair
    2. Feed it through create_pair_universe_from_code()
    3. Translate the exported DEX pair back to a trading pair identifier
    """
    usdc = AssetIdentifier(
        chain_id=999,
        address="0xb88339cb7199b77e23db6e890353e22632ba630f",
        token_symbol="USDC",
        decimals=6,
    )

    # 1. Build a handcrafted Hypercore vault pair.
    pair = create_hypercore_vault_pair(
        quote=usdc,
        vault_address="0x1111111111111111111111111111111111111111",
    )

    # 2. Feed it through create_pair_universe_from_code().
    pair_universe = create_pair_universe_from_code(ChainId.hypercore, [pair])
    dex_pair = pair_universe.get_single()

    # 3. Translate the exported DEX pair back to a trading pair identifier.
    translated_pair = translate_trading_pair(dex_pair)

    assert dex_pair.dex_type == ExchangeType.erc_4626_vault
    assert translated_pair.is_vault() is True
    assert translated_pair.is_hyperliquid_vault() is True


def test_hypercore_vault_missing_share_decimals_defaults_to_18():
    """Hypercore-native vaults with no share-token decimals translate without error.

    The data server emits ``share_token_decimals=null`` for Hypercore-native vaults
    (e.g. HLP) because they have no on-chain ERC-20 share token, so base_token_decimals
    arrives as None. We default only the base/share token to 18 here, never the
    quote/denomination token (defaulting 6-decimal USDC to 18 reverts CCTP transfers),
    and never a non-Hypercore vault whose missing decimals should still surface.

    1. Build a Hypercore vault DEX pair whose share/base token has None decimals.
    2. Translate it and confirm the base defaults to 18 while the quote stays 6.
    3. Confirm the same pair on a non-Hypercore chain still raises on missing decimals.
    """

    # 1. Build a Hypercore vault DEX pair whose share/base token has None decimals.
    hypercore_pair = _make_vault_dex_pair(ChainId.hypercore)
    assert hypercore_pair.base_token_decimals is None

    # 2. Translate it and confirm the base defaults to 18 while the quote stays 6.
    translated = translate_trading_pair(hypercore_pair)
    assert translated.base.decimals == 18
    assert translated.quote.decimals == 6

    # 3. Confirm the same pair on a non-Hypercore chain still raises on missing decimals.
    ethereum_pair = _make_vault_dex_pair(ChainId.ethereum)
    with pytest.raises(AssertionError, match="Base token missing decimals"):
        translate_trading_pair(ethereum_pair)
