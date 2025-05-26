"""Create a strategy with vault yield management.
"""

import os
from pathlib import Path

import pytest

from tradeexecutor.analysis.credit import calculate_yield_metrics, YieldType
from tradeexecutor.backtest.backtest_module import run_backtest_for_module
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.strategy.execution_context import unit_test_execution_context


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


@pytest.fixture()
def strategy_file() -> Path:
    return Path(os.path.join(os.path.dirname(__file__), "../..", "strategies", "test_only", "base-ath-ipor.py"))


def test_backtest_vault_yield_manager(
    persistent_test_client,
    strategy_file,
    logger,
):
    """Test a directional strategy with YieldManager embedded in"""
    client = persistent_test_client

    # Run backtest over 6 months, daily
    result = run_backtest_for_module(
        strategy_file=strategy_file,
        cache_path=client.transport.cache_path,
        execution_context=unit_test_execution_context,
    )

    state = result.state
    assert len(state.portfolio.closed_positions) > 0

    _ = calculate_yield_metrics(
        state,
        yield_type=YieldType.credit
    )

    _ = calculate_yield_metrics(
        state,
        yield_type=YieldType.vault
    )


