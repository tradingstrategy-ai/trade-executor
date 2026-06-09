"""Test-only xchain master-vault strategy variant for fork-simulated CCTP flows."""

import importlib.util
from pathlib import Path

from eth_defi.token import USDC_NATIVE_TOKEN
from tradingstrategy.chain import ChainId

from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.strategy.pandas_trader.trading_universe_input import CreateTradingUniverseInput


def _load_base_strategy():
    strategy_path = Path(__file__).resolve().parent.parent / "xchain-master-vault.py"
    spec = importlib.util.spec_from_file_location("xchain_master_vault_base", strategy_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_base = _load_base_strategy()

TEST_SOURCE_VAULTS = [
    (ChainId.arbitrum, "0xbe6a65325a073490d0a2999529633f1ae88bb091"),  # Harvest USDC (instant withdrawal, analyses cleanly)
    (ChainId.base, "0x3094b241aade60f91f1c82b0628a10d9501462f9"),  # maxUSD
    (ChainId.hyperliquid, "0xf9bb65e113418292d1a3555515fbd64637a0be18"),  # Clearstar Yield (Euler, instant withdrawal, deposit liquidity)
]

TEST_SUPPORTING_PAIRS = [
    (ChainId.arbitrum, "uniswap-v3", "WETH", "USDC", 0.0005),
    (ChainId.base, "uniswap-v3", "WETH", "USDC", 0.0005),
]

trading_strategy_engine_version = _base.trading_strategy_engine_version
indicators = _base.indicators
tags = _base.tags
icon = _base.icon


class Parameters(_base.Parameters):
    """Test-only parameter overrides for the fork simulation."""

    id = "xchain-master-vault-test"
    primary_chain_id = ChainId.arbitrum
    supporting_pairs = TEST_SUPPORTING_PAIRS
    source_vaults = TEST_SOURCE_VAULTS
    preferred_stablecoin = AssetIdentifier(
        chain_id=ChainId.arbitrum.value,
        address=USDC_NATIVE_TOKEN[ChainId.arbitrum.value].lower(),
        token_symbol="USDC",
        decimals=6,
    )
    auto_generate_cctp_bridges = True


def create_trading_universe(
    input: CreateTradingUniverseInput,
):
    """Reuse the production universe builder with test-only parameter overrides."""
    if input.parameters is None:
        input = CreateTradingUniverseInput(
            client=input.client,
            timestamp=input.timestamp,
            parameters=Parameters,
            execution_context=input.execution_context,
            execution_model=input.execution_model,
            universe_options=input.universe_options,
        )
    return _base.create_trading_universe(input)


create_indicators = _base.create_indicators
decide_trades = _base.decide_trades
create_charts = _base.create_charts

name = "Xchain master vault strategy test"
short_description = "Fork-simulated xchain master-vault strategy with forward-only CCTP bridge generation"
long_description = """
# Xchain master vault strategy test

Fork-simulated variant of the production xchain master-vault strategy.

- Primary chain: Arbitrum
- Satellite chains: Base, HyperEVM
- Synthetic bridge pairs: forward-only primary-to-satellite CCTP routes
- Vault set: one deterministic instant-withdrawal candidate per tested chain
"""
