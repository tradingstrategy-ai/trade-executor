"""CLI integration test for trade-ui with Hypercore vault strategy.

Validates that trade-ui starts up successfully with the hyper-ai-tui-test
strategy in simulate mode, without ERC-4626 RPC errors or Uniswap
routing failures on HyperEVM.

Environment variables:
    TRADING_STRATEGY_API_KEY: API key for Trading Strategy data
    JSON_RPC_HYPERLIQUID: HyperEVM RPC endpoint

1. Configure CLI with hyper-ai-tui-test strategy and --simulate
2. Mock display_pair_selection_ui to capture pairs and return immediately
3. Run trade-ui and verify it reaches pair selection without errors
4. Assert at least one vault pair was passed to the TUI
5. Assert no ERC-4626 vault adapter was created (checked via log)
"""

import logging
import os
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tradeexecutor.cli.main import app
from tradeexecutor.cli.commands import trade_ui as trade_ui_module


@pytest.mark.timeout(120)
@pytest.mark.skipif(
    os.environ.get("TRADING_STRATEGY_API_KEY") is None or os.environ.get("JSON_RPC_HYPERLIQUID") is None,
    reason="Set TRADING_STRATEGY_API_KEY and JSON_RPC_HYPERLIQUID environment variables to run this test",
)
def test_trade_ui_hypercore_starts_without_erc4626_errors(
    tmp_path: Path,
    caplog,
    monkeypatch,
) -> None:
    """Verify trade-ui starts with hyper-ai-tui-test strategy without ERC-4626 errors.

    1. Set up environment for hyper-ai-tui-test with --simulate and hot_wallet mode
    2. Mock display_pair_selection_ui to capture the pairs and raise KeyboardInterrupt
    3. Run trade-ui — it should reach TUI display without RPC retry loops
    4. Assert vault pairs were passed and no ERC-4626 adapter was created
    """
    strategy_file = Path(__file__).resolve().parents[2] / "strategies" / "test_only" / "hyper-ai-tui-test.py"
    state_file = tmp_path / "trade-ui-test.json"

    environment = {
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

    # 2. Mock the TUI to capture pairs and exit cleanly
    captured_pairs = []

    def mock_display(pairs, strategy_universe, **kwargs):
        captured_pairs.extend(pairs)
        raise KeyboardInterrupt("Test complete — TUI reached successfully")

    monkeypatch.setattr(
        trade_ui_module,
        "display_pair_selection_ui",
        mock_display,
    )

    runner = CliRunner()
    caplog.set_level(logging.INFO)

    # 3. Run trade-ui — expect KeyboardInterrupt from mock
    result = runner.invoke(app, ["trade-ui"], env=environment)

    # 4. The command should reach the TUI (KeyboardInterrupt) without hanging on RPC retries
    messages = [record.getMessage() for record in caplog.records]

    # Verify no ERC-4626 vault adapter was created
    assert not any("create_vault_adapter()" in m for m in messages), (
        "ERC-4626 vault adapter should NOT be created for Hypercore-only strategy. "
        "This causes RPC calls to non-existent contracts on HyperEVM."
    )

    # Verify vault pairs were passed to the TUI
    assert len(captured_pairs) > 0, "No pairs were passed to the TUI — trade-ui did not reach pair selection"
    assert any(p.is_vault() for p in captured_pairs), "Expected vault pairs in the TUI"
