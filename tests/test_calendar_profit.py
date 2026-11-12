"""Calendar attribution, first-day costs and incomplete evaluation boundaries."""

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from tradeexecutor.visual.calendar import calculate_calendar_profit, visualise_quarterly_profit


def test_first_day_cost_quarter_boundary_and_year_reconciliation():
    index = pd.date_range("2025-12-30", "2026-01-03", inclusive="left", tz="UTC")
    equity = pd.Series([990, 1100, 880, 968], index=index)
    table = calculate_calendar_profit(equity, 1000, index[0], "2026-01-03")
    assert table.loc[(2025, "Q4"), "profit_usd"] == pytest.approx(100)
    assert table.loc[(2026, "Q1"), "profit_usd"] == pytest.approx(-132)
    assert table.loc[(2026, "Q1"), "return"] == pytest.approx(-0.12)
    annual = table.xs("Year", level="period")
    assert annual.profit_usd.sum() == pytest.approx(968 - 1000)
    assert (1 + annual['return']).prod() - 1 == pytest.approx(968 / 1000 - 1)
    assert table.partial.all()
    fig = visualise_quarterly_profit(table)
    assert list(fig.data[0].x) == ["Q1", "Q2", "Q3", "Q4", "Year"]
    assert "N/A" in fig.data[1].text
    assert "$-132<br>-12.00%*" in fig.data[1].text
    assert fig.data[0].zmin == -fig.data[0].zmax


def test_complete_year_compounds_quarters_and_zero_profit_is_observed():
    index = pd.date_range("2024-01-01", "2025-01-01", inclusive="left")
    equity = pd.Series(1000.0, index=index)
    equity.loc["2024-04-01":] = 1100
    equity.loc["2024-07-01":] = 990
    equity.loc["2024-10-01":] = 1200
    table = calculate_calendar_profit(equity, 1000, "2024-01-01", "2025-01-01")
    assert not table.partial.any()
    assert table.loc[(2024, "Q1"), "profit_usd"] == 0
    assert table.loc[(2024, "Q2"), "profit_usd"] == 100
    assert table.loc[(2024, "Year"), "return"] == pytest.approx(0.2)
    quarters = table.drop("Year", level="period")
    assert quarters.profit_usd.sum() == pytest.approx(200)
    assert (1 + quarters['return']).prod() - 1 == pytest.approx(0.2)
    assert table.loc[(2024, "Q1"), "observations"] == 91  # Leap year


@pytest.mark.parametrize("fault", ["gap", "duplicate", "unordered", "nan", "zero", "missing_tail", "outside"])
def test_invalid_coverage_is_not_fabricated_as_zero(fault):
    equity = pd.Series([100.0, 105, 101], index=pd.date_range("2026-01-01", periods=3))
    if fault == "gap":
        equity = equity.drop(equity.index[1])
    elif fault == "duplicate":
        equity.index = pd.DatetimeIndex(["2026-01-01", "2026-01-01", "2026-01-03"])
    elif fault == "unordered":
        equity = equity.iloc[::-1]
    elif fault == "nan":
        equity.iloc[1] = np.nan
    elif fault == "zero":
        equity.iloc[1] = 0
    elif fault == "missing_tail":
        equity = equity.iloc[:-1]
    else:
        equity.index += pd.Timedelta(days=1)
    with pytest.raises(ValueError):
        calculate_calendar_profit(equity, 100, "2026-01-01", "2026-01-04")


def test_utc_conversion_intraday_and_empty():
    index = pd.date_range("2026-03-31 23:00", periods=4, freq="h", tz="Europe/London")
    equity = pd.Series([99, 101, 102, 103], index=index)
    table = calculate_calendar_profit(equity, 100, index[0], index[-1] + pd.Timedelta(hours=1), pd.Timedelta(hours=1))
    assert table.loc[(2026, "Q1"), "profit_usd"] == 1
    assert table.loc[(2026, "Q2"), "profit_usd"] == 2
    empty = calculate_calendar_profit(pd.Series([], index=pd.DatetimeIndex([]), dtype=float), 100, "2026-01-01", "2026-01-02")
    assert empty.empty
    assert visualise_quarterly_profit(empty).layout.annotations[0].text == "No observed valuations"


def test_registered_chart_uses_backtest_capital_and_rejects_live(monkeypatch):
    from tradeexecutor.strategy.chart.standard import calendar
    equity = pd.Series([99.0, 103], index=pd.date_range("2026-01-01", periods=2))
    state = SimpleNamespace(portfolio=SimpleNamespace(get_initial_cash=lambda: 100))
    monkeypatch.setattr(calendar, "calculate_equity_curve", lambda supplied: equity if supplied is state else None)
    input = SimpleNamespace(state=state, live=False, start_at="2026-01-01", end_at="2026-01-03")
    figure, table = calendar.quarterly_profit(input)
    assert table.loc[(2026, "Q1"), "profit_usd"] == 3
    assert table.loc[(2026, "Q1"), "return"] == pytest.approx(0.03)
    assert len(figure.data) == 2
    input.live = True
    with pytest.raises(ValueError, match="fixed-capital"):
        calendar.quarterly_profit(input)
