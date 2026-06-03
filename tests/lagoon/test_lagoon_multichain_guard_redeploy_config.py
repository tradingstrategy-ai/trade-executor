from pathlib import Path
from unittest.mock import MagicMock

from eth_defi.compat import native_datetime_utc_now

from tradeexecutor.ethereum.lagoon import universe_config as universe_config_module
from tradeexecutor.ethereum.lagoon.universe_config import (
    _collect_erc4626_vault_addresses,
    translate_trading_universe_to_lagoon_config,
)
from tradeexecutor.strategy.strategy_module import read_strategy_module


def test_translate_universe_to_lagoon_config_supports_multichain_guard_redeploy(monkeypatch):
    """Guard-only redeploy generates correct source/satellite configs.

    1. Load a CCTP bridge strategy and create its universe.
    2. Call translate_trading_universe_to_lagoon_config with guard_only=True.
    3. Verify source chain has existing_vault_address, satellite does not.
    """
    strategy_file = Path("strategies/test_only/cctp_bridge_start_test.py")
    module = read_strategy_module(strategy_file)
    universe = module.create_trading_universe(
        ts=native_datetime_utc_now(),
        client=None,
        execution_context=None,
        universe_options=None,
    )

    monkeypatch.setattr(universe_config_module, "_apply_protocol_configs", lambda **kwargs: None)

    existing_safe_address = "0x1000000000000000000000000000000000000001"
    existing_vault_address = "0x2000000000000000000000000000000000000002"
    asset_manager = "0x3000000000000000000000000000000000000003"

    configs = translate_trading_universe_to_lagoon_config(
        universe=universe,
        chain_web3={
            "arbitrum": object(),
            "base": object(),
        },
        asset_manager=asset_manager,
        safe_owners=[asset_manager],
        safe_threshold=1,
        safe_salt_nonce=42,
        any_asset=True,
        guard_only=True,
        existing_safe_address=existing_safe_address,
        existing_vault_address=existing_vault_address,
    )

    assert configs["arbitrum"].satellite_chain is False
    assert configs["arbitrum"].guard_only is True
    assert configs["arbitrum"].existing_safe_address == existing_safe_address
    assert configs["arbitrum"].existing_vault_address == existing_vault_address

    assert configs["base"].satellite_chain is True
    assert configs["base"].guard_only is True
    assert configs["base"].existing_safe_address == existing_safe_address
    assert configs["base"].existing_vault_address is None


def test_collect_erc4626_vault_addresses_from_universe():
    """_collect_erc4626_vault_addresses extracts on-chain vault addresses per chain.

    The guard's anyAsset flag only bypasses token-level checks, not target and
    approval destination whitelisting. Vault pairs in the strategy universe must
    always be collected for guard whitelisting via whitelistERC4626().

    1. Create a mock universe with a vault pair and a non-vault pair.
    2. Verify the vault address is collected for the correct chain.
    3. Verify spot pairs are excluded.
    """
    from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind

    ostium_vault_address = "0x20D419a8e12C45f88fDA7c5760bb6923Cee27F98"

    # 1. Create vault and non-vault pairs
    vault_pair = TradingPairIdentifier(
        base=AssetIdentifier(chain_id=42161, address=ostium_vault_address.lower(), token_symbol="oLP", decimals=18),
        quote=AssetIdentifier(chain_id=42161, address="0xaf88d065e77c8cc2239327c5edb3a432268e5831", token_symbol="USDC", decimals=6),
        pool_address=ostium_vault_address.lower(),
        exchange_address=None,
        internal_id=999,
        internal_exchange_id=999,
        fee=0.0,
        kind=TradingPairKind.vault,
        exchange_name="ostium",
        other_data={"vault_protocol": "ostium", "vault_features": ["ostium_like"]},
    )

    spot_pair = TradingPairIdentifier(
        base=AssetIdentifier(chain_id=42161, address="0x82af49447d8a07e3bd95bd0d56f35241523fbab1", token_symbol="WETH", decimals=18),
        quote=AssetIdentifier(chain_id=42161, address="0xaf88d065e77c8cc2239327c5edb3a432268e5831", token_symbol="USDC", decimals=6),
        pool_address="0xc31e54c7a869b9fcbecc14363cf510d1c41fa443",
        exchange_address="0x1F98431c8aD98523631AE4a59f267346ea31F984",
        internal_id=998,
        internal_exchange_id=998,
        fee=0.0005,
        kind=TradingPairKind.spot_market_hold,
        exchange_name="uniswap-v3",
    )

    mock_universe = MagicMock()
    mock_universe.iterate_pairs.return_value = [vault_pair, spot_pair]

    # 2. Verify vault address is collected
    result = _collect_erc4626_vault_addresses(mock_universe, {42161})
    assert 42161 in result
    assert len(result[42161]) == 1
    assert result[42161][0] == "0x20D419a8e12C45f88fDA7c5760bb6923Cee27F98"


def test_collect_erc4626_excludes_hyperliquid_vaults():
    """_collect_erc4626_vault_addresses excludes Hyperliquid native vaults.

    Hyperliquid vaults live on chain 9999 (Hypercore) and have their own
    whitelisting path via whitelistHypercoreVault. They must not appear in
    the ERC-4626 vault whitelist.

    1. Create a mock universe with a Hyperliquid vault pair.
    2. Verify the vault is NOT collected.
    """
    from tradingstrategy.chain import ChainId
    from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind

    # 1. Hyperliquid vault lives on chain 9999 (Hypercore native)
    hl_vault_pair = TradingPairIdentifier(
        base=AssetIdentifier(chain_id=ChainId.hypercore.value, address="0x8a862fd6c12f9ad34c9c2ff45ab2b6712e8cea27", token_symbol="feUSDC", decimals=6),
        quote=AssetIdentifier(chain_id=ChainId.hypercore.value, address="0xef88d065e77c8cc2239327c5edb3a432268e5831", token_symbol="USDC", decimals=6),
        pool_address="0x8a862fd6c12f9ad34c9c2ff45ab2b6712e8cea27",
        exchange_address=None,
        internal_id=997,
        internal_exchange_id=997,
        fee=0.0,
        kind=TradingPairKind.vault,
        exchange_name="hyperliquid",
    )

    mock_universe = MagicMock()
    mock_universe.iterate_pairs.return_value = [hl_vault_pair]

    # 2. HyperEVM chain (999) is what normalises to — vault should NOT be collected
    result = _collect_erc4626_vault_addresses(mock_universe, {999})
    assert result == {}
