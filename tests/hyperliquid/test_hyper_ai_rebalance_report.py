"""Tests for Hyper AI rebalance report diagnostics."""

import importlib.util
from pathlib import Path

import pytest


def _load_hyper_ai_strategy_module():
    strategy_path = Path(__file__).resolve().parents[2] / "strategies" / "hyper-ai.py"
    spec = importlib.util.spec_from_file_location("hyper_ai_strategy_rebalance_report", strategy_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _PlannedTrade:
    """Stand-in for a planned trade before capital allocation."""

    def get_value(self) -> float:
        return 0.0

    def get_planned_value(self) -> float:
        return 9.74323524


def test_hyper_ai_rebalance_volume_uses_planned_trade_value() -> None:
    """Hyper AI report volume must include planned trades before execution.

    1. Load the production Hyper AI strategy module.
    2. Use a planned-trade stand-in whose accounting value is zero, because
       the bug is the report method choice and a full strategy decision setup
       would need a live Hyperliquid universe.
    3. Verify the report volume uses the planned notional.
    """

    # 1. Load the production Hyper AI strategy module.
    module = _load_hyper_ai_strategy_module()

    # 2. Use a planned-trade stand-in whose accounting value is zero.
    trade = _PlannedTrade()
    assert trade.get_value() == 0.0

    # 3. Verify the report volume uses the planned notional.
    assert module.calculate_rebalance_volume([trade]) == pytest.approx(9.74323524)
