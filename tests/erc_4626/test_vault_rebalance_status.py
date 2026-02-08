"""Test vault rebalance status display.

Tests the print_vault_rebalance_status() function for console output.
"""
import datetime
import os
from decimal import Decimal

import pytest
import pandas as pd
from eth_defi.erc_4626.vault_protocol.ipor.vault import IPORVault

from tradeexecutor.analysis.vault_rebalance import (
    get_vault_rebalance_status,
    print_vault_rebalance_status,
)
from tradeexecutor.ethereum.hot_wallet_sync_model import HotWalletSyncModel
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.strategy.generic.generic_router import GenericRouting
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from eth_defi.compat import native_datetime_utc_now


JSON_RPC_BASE = os.environ.get("JSON_RPC_BASE")
pytestmark = pytest.mark.skipif(not JSON_RPC_BASE, reason="No JSON_RPC_BASE environment variable")


def test_vault_rebalance_status_no_positions(
    strategy_universe: TradingStrategyUniverse,
    sync_model: HotWalletSyncModel,
    base_usdc: AssetIdentifier,
):
    """Test vault rebalance status with no open positions.

    - Vaults from the universe should still be displayed with zero allocations
    - Print output should show zero values
    """

    state = State()
    sync_model.sync_initial(
        state,
        reserve_asset=base_usdc,
        reserve_token_price=1.0,
    )

    # Manually set some cash for testing (sync_initial doesn't query on-chain balance)
    state.portfolio.get_default_reserve_position().quantity = Decimal("1000")

    # Test get_vault_rebalance_status
    df, cash = get_vault_rebalance_status(state, strategy_universe)

    assert isinstance(df, pd.DataFrame)
    assert cash == pytest.approx(1000.0)

    # Should have one vault row (IPOR vault from universe) even with no positions
    vault_rows = df[df["Available"] == True]
    assert len(vault_rows) == 1

    # Vault should be shown with zero allocation
    vault_row = vault_rows.iloc[0]
    assert vault_row["Value USD"] == 0
    assert vault_row["Shares"] == 0
    assert vault_row["Weight %"] == 0
    assert vault_row["Position ID"] is None
    assert "IPOR" in vault_row["Vault"]
    assert vault_row["Protocol"] == "ipor"

    # 1M CAGR column should exist (may be None if not loaded from JSON blob)
    assert "1M CAGR" in df.columns

    # Verify sorting - values should be in descending order
    values = df["Value USD"].tolist()
    assert values == sorted(values, reverse=True)

    # Test print output
    output_lines = []
    df = print_vault_rebalance_status(state, strategy_universe, printer=output_lines.append)
    output = "\n".join(output_lines)

    assert "VAULT REBALANCE STATUS" in output
    assert "IPOR" in output
    assert "$0.00" in output
    assert "0.00%" in output
    assert "1M CAGR" in output  # Column header should be displayed


def test_vault_rebalance_status_with_position(
    vault: IPORVault,
    strategy_universe: TradingStrategyUniverse,
    execution_model,
    routing_model: GenericRouting,
    pricing_model,
    sync_model: HotWalletSyncModel,
    base_usdc: AssetIdentifier,
):
    """Test vault rebalance status with an open vault position.

    - Position should be displayed with correct values
    - Print output should show allocated values
    - Results should be sorted by value (largest first)
    """

    state = State()
    pair = strategy_universe.get_pair_by_smart_contract(vault.address)

    sync_model.sync_initial(
        state,
        reserve_asset=base_usdc,
        reserve_token_price=1.0,
    )

    # Set initial reserve balance (sync_initial doesn't query on-chain balance)
    state.portfolio.get_default_reserve_position().quantity = Decimal("999")

    position_manager = PositionManager(
        native_datetime_utc_now(),
        universe=strategy_universe,
        state=state,
        pricing_model=pricing_model,
        default_slippage_tolerance=0.20,
    )

    # Open a vault position
    trades = position_manager.open_spot(
        pair,
        value=100.0,
    )

    routing_state_details = execution_model.get_routing_state_details()
    routing_state = routing_model.create_routing_state(strategy_universe, routing_state_details)

    execution_model.initialize()

    execution_model.execute_trades(
        native_datetime_utc_now(),
        state,
        trades,
        routing_model,
        routing_state,
        check_balances=True,
    )

    assert trades[0].is_success()

    # Test get_vault_rebalance_status
    df, cash = get_vault_rebalance_status(state, strategy_universe)

    assert isinstance(df, pd.DataFrame)
    assert cash == pytest.approx(899.0)  # 999 - 100

    # Should have the vault with position
    vault_row = df[df["Position ID"].notna()].iloc[0]
    assert vault_row["Available"] == True
    assert vault_row["Value USD"] > 90
    assert vault_row["Value USD"] < 110
    assert float(vault_row["Shares"]) > 0
    assert vault_row["Weight %"] > 0

    # Verify sorting - values should be in descending order
    values = df["Value USD"].tolist()
    assert values == sorted(values, reverse=True)

    # Test print output
    output_lines = []
    df = print_vault_rebalance_status(state, strategy_universe, printer=output_lines.append)
    output = "\n".join(output_lines)

    assert "VAULT REBALANCE STATUS" in output
    assert "Current cash:" in output
    assert "Total vault value:" in output
    assert "Total portfolio value:" in output
    assert "Vault Allocations" in output
    assert "ipor" in output.lower()
    assert "1M CAGR" in output  # Column header should be displayed

    # 1M CAGR column should exist in DataFrame
    assert "1M CAGR" in df.columns
