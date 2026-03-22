"""Regression tests for frozen position NAV accounting."""

import datetime
from unittest.mock import Mock

import pytest

from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.sync import Treasury
from tradeexecutor.statistics.core import calculate_statistics
from tradeexecutor.strategy.execution_context import ExecutionMode


def test_frozen_positions_contribute_to_nav_and_share_price() -> None:
    """Frozen positions count towards NAV and vault share price.

    1. Create a portfolio where deployable equity is lower than total economic value because capital is frozen.
    2. Calculate NAV directly from the portfolio helper.
    3. Calculate portfolio statistics and verify share price follows NAV instead of deployable equity.
    """
    # 1. Create a portfolio where deployable equity is lower than total economic value because capital is frozen.
    portfolio = Portfolio()
    portfolio.open_positions = {}
    portfolio.frozen_positions = {}
    portfolio.closed_positions = {}
    portfolio.get_first_and_last_executed_trade = Mock(return_value=(None, None))
    portfolio.get_position_equity_and_loan_nav = Mock(return_value=1.0)
    portfolio.get_frozen_position_equity = Mock(return_value=7.0)
    portfolio.get_cash = Mock(return_value=0.0)

    treasury = Treasury(share_count=10)

    # 2. Calculate NAV directly from the portfolio helper.
    assert portfolio.get_net_asset_value() == pytest.approx(8.0)

    # 3. Calculate portfolio statistics and verify share price follows NAV instead of deployable equity.
    stats = calculate_statistics(
        clock=datetime.datetime(2026, 3, 22),
        portfolio=portfolio,
        execution_mode=ExecutionMode.backtesting,
        treasury=treasury,
    )

    assert stats.portfolio.total_equity == pytest.approx(1.0)
    assert stats.portfolio.net_asset_value == pytest.approx(8.0)
    assert stats.portfolio.share_price_usd == pytest.approx(0.8)
