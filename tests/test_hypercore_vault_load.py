"""Test that a single Hypercore vault loads correctly through load_partial_data.

Validates that Hypercore vaults exported from the vault universe have the
current exchange_type representation (erc_4626_vault) and that
default_supported_routers creates hypercore_vault routing, not ERC-4626.

1. Fetch vault universe from the Trading Strategy API
2. Pick one Hypercore vault (chain_id=9999)
3. Check export_as_exchange returns ExchangeType.erc_4626_vault
4. Check export_as_trading_pair returns dex_type=ExchangeType.erc_4626_vault
5. Build a strategy universe from load_partial_data with that vault
6. Verify default_supported_routers returns hypercore_vault, not vault
"""

import datetime
import os
from pathlib import Path

import pytest
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.exchange import ExchangeType
from tradingstrategy.timebucket import TimeBucket

from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.generic.default_protocols import default_supported_routers
from tradeexecutor.strategy.trading_strategy_universe import (
    TradingStrategyUniverse,
    load_partial_data,
    load_vault_universe_with_metadata,
)
from tradeexecutor.strategy.universe_model import UniverseOptions


@pytest.fixture(scope="module")
def client() -> Client:
    api_key = os.environ.get("TRADING_STRATEGY_API_KEY")
    assert api_key, "TRADING_STRATEGY_API_KEY not set"
    return Client.create_live_client(api_key)


@pytest.fixture(scope="module")
def hypercore_vault_universe(client):
    """Load vault universe and pick a single Hypercore vault."""
    vault_universe = client.fetch_vault_universe()
    # Pick first Hypercore vault
    hypercore_vaults = []
    for v in vault_universe.iterate_vaults():
        if v.chain_id == ChainId.hypercore:
            hypercore_vaults.append(v)
            break
    assert len(hypercore_vaults) > 0, "No Hypercore vaults found in the vault universe"
    # Limit to just that one vault
    vault_specs = [(v.chain_id, v.vault_address) for v in hypercore_vaults]
    return vault_universe.limit_to_vaults(vault_specs, check_all_vaults_found=True)


@pytest.mark.timeout(120)
@pytest.mark.skipif(
    os.environ.get("TRADING_STRATEGY_API_KEY") is None,
    reason="Set TRADING_STRATEGY_API_KEY to run this test",
)
def test_hypercore_vault_export_types(hypercore_vault_universe):
    """Verify Hypercore vault exports use the current exchange type encoding.

    1. Get a single Hypercore vault from the universe
    2. Check export_as_exchange returns erc_4626_vault
    3. Check export_as_trading_pair returns erc_4626_vault as dex_type
    """
    # 1. Get the vault
    vault = next(hypercore_vault_universe.iterate_vaults())
    assert vault.chain_id == ChainId.hypercore

    # 2. Check exchange export
    xc_data = vault.export_as_exchange()
    assert xc_data["exchange_type"] == ExchangeType.erc_4626_vault, (
        f"Hypercore vault {vault.name} (protocol_slug={vault.protocol_slug}) "
        f"exports as {xc_data['exchange_type']} instead of erc_4626_vault"
    )

    # 3. Check trading pair export
    pair_data = vault.export_as_trading_pair()
    assert pair_data["dex_type"] == ExchangeType.erc_4626_vault, (
        f"Hypercore vault {vault.name} exports pair dex_type as "
        f"{pair_data['dex_type']} instead of erc_4626_vault"
    )


@pytest.mark.timeout(120)
@pytest.mark.skipif(
    os.environ.get("TRADING_STRATEGY_API_KEY") is None,
    reason="Set TRADING_STRATEGY_API_KEY to run this test",
)
def test_hypercore_vault_routing(client, hypercore_vault_universe):
    """Verify default_supported_routers creates hypercore_vault routing.

    1. Load partial data with the single Hypercore vault
    2. Create strategy universe
    3. Check default_supported_routers returns hypercore_vault, not vault
    """
    from eth_defi.token import USDC_NATIVE_TOKEN
    from eth_defi.compat import native_datetime_utc_now

    # 1. Load partial data
    execution_context = ExecutionContext(mode=ExecutionMode.preflight_check)
    universe_options = UniverseOptions(history_period=datetime.timedelta(days=30))

    dataset = load_partial_data(
        client=client,
        time_bucket=TimeBucket.d1,
        pairs=[],
        execution_context=execution_context,
        universe_options=universe_options,
        liquidity=True,
        liquidity_time_bucket=TimeBucket.d1,
        vaults=hypercore_vault_universe,
        vault_history_source="trading-strategy-website",
        check_all_vaults_found=True,
    )

    # 2. Create strategy universe
    usdc = AssetIdentifier(
        chain_id=999,
        address=USDC_NATIVE_TOKEN[999].lower(),
        token_symbol="USDC",
        decimals=6,
    )
    ts = native_datetime_utc_now()
    strategy_universe = TradingStrategyUniverse.create_from_dataset(
        dataset,
        reserve_asset=usdc,
        forward_fill=True,
        forward_fill_until=ts,
        primary_chain=ChainId.hyperliquid,
    )

    # 3. Check routing
    routers = default_supported_routers(strategy_universe)
    router_names = {r.router_name for r in routers}

    assert "hypercore_vault" in router_names, (
        f"Expected hypercore_vault routing, got: {router_names}. "
        f"Vault exchange types in universe may be wrong."
    )
    assert "vault" not in router_names, (
        f"ERC-4626 vault routing should NOT be created for Hypercore vaults. "
        f"Got routers: {router_names}"
    )
