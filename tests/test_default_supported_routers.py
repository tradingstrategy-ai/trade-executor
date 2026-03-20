"""Test default_supported_routers correctly handles Hypercore vault universes.

Validates that ERC-4626 vault routing is NOT created when the universe
contains only Hypercore vaults, even with old cached data that uses
erc_4626_vault exchange type.

1. Build a universe with Hypercore vault pairs using old erc_4626_vault type
2. Verify hypercore_vault routing is created
3. Verify ERC-4626 vault routing is NOT created
4. Test edge case: vault pair missing vault_protocol metadata
"""

import pandas as pd
import pytest
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange, ExchangeUniverse, ExchangeType
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.universe import Universe

from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.strategy.generic.default_protocols import default_supported_routers
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


USDC = AssetIdentifier(chain_id=9999, address="0x02", token_symbol="USDC", decimals=6)


def _make_vault_exchange(exchange_id: int, slug: str, exchange_type=ExchangeType.erc_4626_vault) -> Exchange:
    """Helper to create a vault exchange."""
    return Exchange(
        chain_id=9999,
        chain_slug="hypercore",
        exchange_slug=slug,
        exchange_id=exchange_id,
        address="0x0000000000000000000000000000000000000000",
        exchange_type=exchange_type,
        pair_count=1,
        name=slug,
    )


def _make_vault_pair(pair_id: int, exchange_id: int, slug: str, vault_protocol: str | None = "hypercore") -> dict:
    """Helper to create a vault pair dict."""
    other_data = {}
    if vault_protocol is not None:
        other_data["vault_protocol"] = vault_protocol
    return {
        "pair_id": pair_id,
        "pair_slug": slug,
        "exchange_id": exchange_id,
        "address": f"0x{pair_id:040x}",
        "token0_address": "0x02",
        "token0_symbol": "USDC",
        "token0_decimals": 6,
        "token1_address": f"0x{pair_id + 1000:040x}",
        "token1_symbol": f"V{pair_id}",
        "token1_decimals": 6,
        "dex_type": ExchangeType.erc_4626_vault,
        "base_token_symbol": f"V{pair_id}",
        "quote_token_symbol": "USDC",
        "exchange_slug": slug,
        "exchange_name": slug,
        "fee": 0,
        "chain_id": 9999,
        "buy_volume_all_time": 0,
        "other_data": other_data,
    }


def _build_universe(exchanges: list[Exchange], pairs: list[dict]) -> TradingStrategyUniverse:
    """Build a minimal TradingStrategyUniverse from exchanges and pair dicts."""
    eu = ExchangeUniverse.from_collection(exchanges)
    pairs_df = pd.DataFrame(pairs)
    pair_universe = PandasPairUniverse(pairs_df, exchange_universe=eu)
    data_universe = Universe(
        time_bucket=None,
        chains={ChainId.hypercore},
        exchange_universe=eu,
        pairs=pair_universe,
    )
    return TradingStrategyUniverse(data_universe=data_universe, reserve_assets=[USDC])


def test_hypercore_only_universe_no_erc4626_routing():
    """Verify no ERC-4626 routing when all vaults are Hypercore.

    1. Create a universe with two Hypercore vault exchanges
    2. Both pairs have vault_protocol=hypercore in other_data
    3. Assert only hypercore_vault routing is created, not ERC-4626 vault
    """
    exchanges = [
        _make_vault_exchange(100, "vault-a"),
        _make_vault_exchange(200, "vault-b"),
    ]
    pairs = [
        _make_vault_pair(1, 100, "vault-a", vault_protocol="hypercore"),
        _make_vault_pair(2, 200, "vault-b", vault_protocol="hypercore"),
    ]
    universe = _build_universe(exchanges, pairs)
    routers = default_supported_routers(universe)

    router_names = {r.router_name for r in routers}
    assert "hypercore_vault" in router_names
    assert "vault" not in router_names, "ERC-4626 vault routing should not be created for Hypercore-only universe"


def test_missing_vault_protocol_still_no_erc4626():
    """Verify no ERC-4626 routing even when vault_protocol metadata is missing.

    This simulates old cached data where some vault pairs may not have
    vault_protocol set in other_data. Since the exchange is on the
    Hypercore chain (9999), it should still not create ERC-4626 routing.

    1. Create two vault pairs: one with vault_protocol=hypercore, one without
    2. Assert no ERC-4626 vault routing is created
    """
    exchanges = [
        _make_vault_exchange(100, "vault-a"),
        _make_vault_exchange(200, "vault-b"),
    ]
    pairs = [
        _make_vault_pair(1, 100, "vault-a", vault_protocol="hypercore"),
        _make_vault_pair(2, 200, "vault-b", vault_protocol=None),  # Missing metadata
    ]
    universe = _build_universe(exchanges, pairs)
    routers = default_supported_routers(universe)

    router_names = {r.router_name for r in routers}
    assert "hypercore_vault" in router_names
    assert "vault" not in router_names, (
        "ERC-4626 vault routing must not be created when Hypercore vaults are present — "
        "Hypercore vaults are NOT ERC-4626 contracts and calling ERC-4626 methods on them fails"
    )


def test_mixed_hypercore_and_erc4626_universe():
    """Verify both routing types are created for a mixed vault universe.

    1. Create a Hypercore vault exchange and an ERC-4626 vault exchange
    2. The ERC-4626 exchange is on a different chain (not Hypercore)
    3. Assert both hypercore_vault and vault routing are created
    """
    exchanges = [
        _make_vault_exchange(100, "hypercore-vault"),
        Exchange(
            chain_id=8453, chain_slug="base", exchange_slug="morpho-vault",
            exchange_id=200, address="0x0000000000000000000000000000000000000000",
            exchange_type=ExchangeType.erc_4626_vault, pair_count=1, name="Morpho Vault",
        ),
    ]
    pairs = [
        _make_vault_pair(1, 100, "hypercore-vault", vault_protocol="hypercore"),
        {
            "pair_id": 2, "pair_slug": "morpho-vault", "exchange_id": 200,
            "address": "0x0000000000000000000000000000000000000099",
            "token0_address": "0x02", "token0_symbol": "USDC", "token0_decimals": 6,
            "token1_address": "0x0000000000000000000000000000000000000098",
            "token1_symbol": "MORPHO", "token1_decimals": 6,
            "dex_type": ExchangeType.erc_4626_vault,
            "base_token_symbol": "MORPHO", "quote_token_symbol": "USDC",
            "exchange_slug": "morpho-vault", "exchange_name": "Morpho Vault",
            "fee": 0, "chain_id": 8453, "buy_volume_all_time": 0,
            "other_data": {"vault_protocol": "morpho"},
        },
    ]

    from tradingstrategy.chain import ChainId as TsChainId
    eu = ExchangeUniverse.from_collection(exchanges)
    pairs_df = pd.DataFrame(pairs)
    pair_universe = PandasPairUniverse(pairs_df, exchange_universe=eu)
    data_universe = Universe(
        time_bucket=None,
        chains={TsChainId.hypercore, TsChainId.base},
        exchange_universe=eu,
        pairs=pair_universe,
    )
    universe = TradingStrategyUniverse(data_universe=data_universe, reserve_assets=[USDC])

    routers = default_supported_routers(universe)
    router_names = {r.router_name for r in routers}
    assert "hypercore_vault" in router_names, "Hypercore vault routing should be created"
    assert "vault" in router_names, "ERC-4626 vault routing should be created for Morpho vault"


def test_new_hypercore_vault_exchange_type():
    """Verify correct routing with the new ExchangeType.hypercore_vault.

    1. Create exchanges using the new hypercore_vault exchange type
    2. Assert hypercore_vault routing is created and no ERC-4626 vault routing
    """
    exchanges = [
        _make_vault_exchange(100, "vault-a", exchange_type=ExchangeType.hypercore_vault),
    ]
    pairs = [
        _make_vault_pair(1, 100, "vault-a", vault_protocol="hypercore"),
    ]
    universe = _build_universe(exchanges, pairs)
    routers = default_supported_routers(universe)

    router_names = {r.router_name for r in routers}
    assert "hypercore_vault" in router_names
    assert "vault" not in router_names
