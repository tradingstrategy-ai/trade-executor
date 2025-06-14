"""Create a strategy with vault yield management.
"""

import os
from pathlib import Path

import pandas as pd
import pytest
from plotly.graph_objs import Figure

from tradeexecutor.analysis.credit import calculate_yield_metrics, YieldType, display_vault_position_table
from tradeexecutor.analysis.multipair import analyse_multipair
from tradeexecutor.backtest.backtest_module import run_backtest_for_module
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.strategy.execution_context import unit_test_execution_context
from tradeexecutor.strategy.pnl import calculate_pnl
from tradeexecutor.visual.position import calculate_position_timeline, visualise_position


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


@pytest.fixture()
def strategy_file() -> Path:
    return Path(os.path.join(os.path.dirname(__file__), "../..", "strategies", "test_only", "base-ath-ipor.py"))



@pytest.mark.slow_test_group
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

    df = display_vault_position_table(state)
    assert isinstance(df, pd.DataFrame)

    for p in state.portfolio.get_all_positions():
        if p.pair.get_vault_name() == "Autopilot USDC Base":
            autopilot_usdc = p
            break
    else:
        raise ValueError("No Autopilot USDC Base position found in the portfolio.")

    assert autopilot_usdc.pair.get_vault_name() == "Autopilot USDC Base"

    # Test position tracking for a single position
    df = calculate_position_timeline(
        strategy_universe=result.strategy_universe,
        position=autopilot_usdc,
        end_at=result.state.backtest_data.end_at,
    )
    assert isinstance(df, pd.DataFrame)
    fig = visualise_position(autopilot_usdc, df, extended=True)
    assert isinstance(fig, Figure)

    #
    # Compare methods calcuating the profit
    #

    end_at = state.backtest_data.end_at
    profit_usd = autopilot_usdc.get_total_profit_usd()
    profit_pct = autopilot_usdc.get_total_profit_percent(calculation_method="cumulative", end_at=end_at)
    profit_pct_annualised = autopilot_usdc.calculate_total_profit_percent_annualised(end_at=end_at)

    position_timeline_final_row = df.iloc[-1]
    method_2_profit_usd = position_timeline_final_row["pnl"]
    method_2_profit_pct = position_timeline_final_row["pnl_pct"]
    method_2_profit_annualised_pct = position_timeline_final_row["pnl_annualised"]

    profit_data = calculate_pnl(
        autopilot_usdc,
        end_at=end_at,
    )

    assert profit_usd == pytest.approx(method_2_profit_usd)
    assert profit_pct == pytest.approx(method_2_profit_pct)
    assert profit_pct_annualised == pytest.approx(method_2_profit_annualised_pct)

    assert profit_usd == pytest.approx(profit_data.profit_usd)
    assert profit_pct == pytest.approx(profit_data.profit_pct)
    assert profit_pct_annualised == pytest.approx(profit_data.profit_pct_annualised)

    # Check multipair results
    df = analyse_multipair(state)
    assert isinstance(df, pd.DataFrame)
