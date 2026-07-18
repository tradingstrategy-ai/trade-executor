"""Tests for multipair analysis helpers."""

import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tradeexecutor.analysis.multipair import calculate_pair_annualised_average_yield


def test_pair_annualised_average_yield_is_weighted_by_capital_and_time() -> None:
    """Annualise a pair's yield without compounding short positions.

    1. Create two positions with different principal and holding periods.
    2. Mock their realised profit and duration data.
    3. Check the money-time-weighted annualised yield.
    """
    # 1. Create two positions with different principal and holding periods.
    pair = MagicMock()
    pair.is_cctp_bridge.return_value = False
    first_position = MagicMock(pair=pair)
    first_position.get_value_at_open.return_value = 100.0
    second_position = MagicMock(pair=pair)
    second_position.get_value_at_open.return_value = 200.0
    portfolio = MagicMock()
    portfolio.get_all_positions.return_value = [first_position, second_position]

    # 2. Mock their realised profit and duration data.
    profit_data = [
        SimpleNamespace(profit_usd=10.0, duration=datetime.timedelta(days=10)),
        SimpleNamespace(profit_usd=40.0, duration=datetime.timedelta(days=20)),
    ]
    with patch("tradeexecutor.analysis.multipair.calculate_pnl_generic", side_effect=profit_data):
        yield_percent = calculate_pair_annualised_average_yield(
            pair,
            portfolio,
            datetime.datetime(2026, 1, 1),
        )

    # 3. Check the money-time-weighted annualised yield.
    assert yield_percent == pytest.approx(3.65)
