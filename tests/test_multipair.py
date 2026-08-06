"""Tests for multipair analysis helpers."""

import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tradeexecutor.analysis.multipair import calculate_pair_annualised_average_yield


def _make_position(
    pair,
    opened_at: datetime.datetime,
    closed_at: datetime.datetime,
    trades: list[tuple[datetime.datetime, float, float]],
):
    """Create a completed spot/vault position mock with executed cash flows."""
    position = MagicMock(pair=pair)
    position.opened_at = opened_at
    position.closed_at = closed_at
    position.is_closed.return_value = True
    position.trades = {}
    for index, (executed_at, quantity, value) in enumerate(trades):
        trade = MagicMock()
        trade.executed_at = executed_at
        trade.opened_at = executed_at
        trade.is_success.return_value = True
        trade.get_position_quantity.return_value = quantity
        trade.get_value.return_value = value
        position.trades[index] = trade
    return position


def test_pair_annualised_average_yield_is_weighted_by_capital_and_time() -> None:
    """Annualise a pair's yield without compounding short positions.

    1. Create two positions with different principal and holding periods.
    2. Mock their realised profit and duration data.
    3. Check the money-time-weighted annualised yield.
    """
    # 1. Create two positions with different principal and holding periods.
    pair = MagicMock()
    pair.is_cctp_bridge.return_value = False
    first_position = _make_position(
        pair,
        datetime.datetime(2025, 1, 1),
        datetime.datetime(2025, 1, 11),
        [(datetime.datetime(2025, 1, 1), 100.0, 100.0)],
    )
    second_position = _make_position(
        pair,
        datetime.datetime(2025, 1, 1),
        datetime.datetime(2025, 1, 21),
        [(datetime.datetime(2025, 1, 1), 200.0, 200.0)],
    )
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


def test_pair_annualised_average_yield_includes_position_adjustments() -> None:
    """Include later capital additions in the annualisation denominator."""
    pair = MagicMock()
    pair.is_cctp_bridge.return_value = False
    position = _make_position(
        pair,
        datetime.datetime(2025, 1, 1),
        datetime.datetime(2025, 1, 21),
        [
            (datetime.datetime(2025, 1, 1), 100.0, 100.0),
            (datetime.datetime(2025, 1, 11), 100.0, 100.0),
        ],
    )
    portfolio = MagicMock()
    portfolio.get_all_positions.return_value = [position]

    with patch(
        "tradeexecutor.analysis.multipair.calculate_pnl_generic",
        return_value=SimpleNamespace(profit_usd=20.0),
    ):
        yield_percent = calculate_pair_annualised_average_yield(
            pair,
            portfolio,
            datetime.datetime(2026, 1, 1),
        )

    # $100 is exposed for ten days and $200 for the next ten days.
    assert yield_percent == pytest.approx(20 / 3_000 * 365)
