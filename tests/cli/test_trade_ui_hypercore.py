"""CLI integration test for trade-ui with Hypercore vault strategy.

Validates that trade-ui starts up successfully with the hyper-ai-tui-test
strategy in simulate mode, without ERC-4626 RPC errors or Uniswap
routing failures on HyperEVM.

Environment variables:
    TRADING_STRATEGY_API_KEY: API key for Trading Strategy data
    JSON_RPC_HYPERLIQUID: HyperEVM RPC endpoint

1. Configure CLI with hyper-ai-tui-test strategy, --simulate, and --unit-testing
2. Run trade-ui — unit_testing flag skips the TUI and auto-selects the first pair
3. Verify the command reaches pair selection without ERC-4626 routing errors
"""

import logging
import os
from pathlib import Path

import pytest
from typer.main import get_command

from tradeexecutor.cli.main import app


@pytest.mark.timeout(120)
@pytest.mark.skipif(
    os.environ.get("TRADING_STRATEGY_API_KEY") is None or os.environ.get("JSON_RPC_HYPERLIQUID") is None,
    reason="Set TRADING_STRATEGY_API_KEY and JSON_RPC_HYPERLIQUID environment variables to run this test",
)
def test_trade_ui_hypercore_loads_universe(
    tmp_path: Path,
    mocker,
    caplog,
) -> None:
    """Verify trade-ui loads the Hypercore vault universe without ERC-4626 errors.

    1. Set up environment for hyper-ai-tui-test with --simulate, hot_wallet, and --unit-testing
    2. Run trade-ui — unit_testing auto-selects the first pair, skipping TUI and trade
    3. Verify the command reached pair selection by checking log messages
    4. Verify no ERC-4626 vault adapter was created
    """
    strategy_file = Path(__file__).resolve().parents[2] / "strategies" / "test_only" / "hyper-ai-tui-test.py"
    state_file = tmp_path / "trade-ui-test.json"

    # 1. Set up environment
    environment = {
        "PATH": os.environ["PATH"],
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "STRATEGY_FILE": str(strategy_file),
        "CACHE_PATH": str(tmp_path),
        "JSON_RPC_HYPERLIQUID": os.environ["JSON_RPC_HYPERLIQUID"],
        "PRIVATE_KEY": "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
        "MIN_GAS_BALANCE": "0",
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "info",
        "ASSET_MANAGEMENT_MODE": "hot_wallet",
        "STATE_FILE": str(state_file),
        "SIMULATE": "true",
    }

    cli = get_command(app)
    mocker.patch.dict("os.environ", environment, clear=True)
    caplog.set_level(logging.INFO)

    # 2. Run trade-ui with real data loading
    cli.main(args=["trade-ui"], standalone_mode=False)

    # 3. Verify the command reached pair selection
    messages = [record.getMessage() for record in caplog.records]
    assert any("Unit testing mode: auto-selected pair" in m for m in messages), (
        "trade-ui did not reach pair selection — check for routing errors"
    )
    assert any("skipping actual trade execution" in m for m in messages), (
        "trade-ui did not reach the unit testing skip point"
    )

    # 4. Verify no ERC-4626 vault adapter was created
    assert not any("create_vault_adapter()" in m for m in messages), (
        "ERC-4626 vault adapter should NOT be created for Hypercore-only strategy"
    )
