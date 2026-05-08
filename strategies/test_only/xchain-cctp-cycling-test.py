"""Deterministic 3-chain position cycling strategy for CCTP integration tests.

Cycles vault positions across Arbitrum (primary), Base, and HyperEVM
over 3 decision cycles to exercise the full automated CCTP rebalance
pipeline: bridge trade injection, multi-phase settlement, and
bridge position accounting.

- Cycle 1: Open vault positions on all 3 chains
- Cycle 2: Close Base vault, increase Arbitrum vault
- Cycle 3: Close all remaining positions
"""

import importlib.util
from pathlib import Path
from decimal import Decimal

from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.pandas_trader.trading_universe_input import CreateTradingUniverseInput


def _load_test_strategy():
    strategy_path = Path(__file__).resolve().parent / "xchain-master-vault-test.py"
    spec = importlib.util.spec_from_file_location("xchain_master_vault_test", strategy_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_base = _load_test_strategy()

trading_strategy_engine_version = _base.trading_strategy_engine_version
indicators = _base.indicators
tags = _base.tags
icon = _base.icon


class Parameters(_base.Parameters):
    """Test-only parameter overrides for the cycling integration test."""

    id = "xchain-cctp-cycling-test"


def create_trading_universe(
    input: CreateTradingUniverseInput,
):
    """Reuse the test universe builder with cycling-test parameter overrides."""
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


def create_indicators(timestamp, parameters, strategy_universe, execution_context):
    return _base.create_indicators(timestamp, parameters, strategy_universe, execution_context)


def decide_trades(input: StrategyInput) -> list[TradeExecution]:
    """Deterministic 3-cycle position cycling across 3 chains.

    The bridge planner injects CCTP bridge trades automatically;
    this strategy never creates bridge trades directly.

    - Cycle 1: open vault positions on all 3 chains (Arb, Base, HyperEVM)
    - Cycle 2: close Base vault, increase Arbitrum vault
    - Cycle 3: close all remaining vault positions
    """
    cycle = input.cycle
    position_manager = input.get_position_manager()
    strategy_universe = input.strategy_universe
    state = input.state

    trades = []

    if cycle == 1:
        # 1. Open vault positions on all 3 chains
        for pair in strategy_universe.iterate_pairs():
            if pair.kind.is_vault():
                trades += position_manager.open_spot(
                    pair,
                    value=Decimal("3"),
                    notes=f"Cycle 1: open vault on chain {pair.chain_id}",
                )

    elif cycle == 2:
        # 2. Close Base vault (chain_id 8453), keep Arb and HyperEVM
        for position in list(state.portfolio.open_positions.values()):
            if position.pair.kind.is_vault() and position.pair.chain_id == 8453:
                trades += position_manager.close_position(
                    position,
                    notes="Cycle 2: close Base vault",
                )

    elif cycle == 3:
        # 3. Close all remaining vault positions
        for position in list(state.portfolio.open_positions.values()):
            if position.pair.kind.is_vault():
                trades += position_manager.close_position(
                    position,
                    notes="Cycle 3: close remaining vaults",
                )

    return trades


name = "Xchain CCTP cycling test"
short_description = "Deterministic 3-chain position cycling for CCTP integration tests"
long_description = """
# Xchain CCTP cycling test

Deterministic 3-cycle strategy that exercises the full CCTP auto-rebalance pipeline:

- Cycle 1: Open vault positions on Arbitrum, Base, and HyperEVM
- Cycle 2: Close Base vault, keep Arbitrum and HyperEVM
- Cycle 3: Close all remaining positions

The bridge planner automatically injects CCTP bridge trades as needed.
"""
