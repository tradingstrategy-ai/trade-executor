"""Regression coverage for Hypercore exchange type compatibility.

1. Create a handcrafted Hypercore vault pair
2. Export it through the pair-universe code path used by strategies
3. Verify the pair still round-trips as a vault using the current Trading Strategy exchange type
"""

from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import ExchangeType

from tradeexecutor.ethereum.vault.hypercore_vault import create_hypercore_vault_pair
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.strategy.dex_data_translation import translate_trading_pair
from tradeexecutor.strategy.trading_strategy_universe import create_pair_universe_from_code


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
