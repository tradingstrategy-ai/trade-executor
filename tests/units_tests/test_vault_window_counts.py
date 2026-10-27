"""Historical universe charts must not confuse closures with missing evidence."""

import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from tradingstrategy.candle import GroupedCandleUniverse

from tradeexecutor.analysis.vault_windows import calculate_vault_window_counts
from tradeexecutor.backtest.vault_windows import VaultWindowSchedule
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
from tradeexecutor.strategy.chart.definition import ChartInput
from tradeexecutor.strategy.chart.standard.trading_universe import available_trading_pairs
from tradeexecutor.strategy.execution_context import unit_test_execution_context
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorNotFound
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInputIndicators
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


@pytest.fixture
def universe() -> TradingStrategyUniverse:
    """Mock universe metadata only; use real candles, pairs and pricing lookups."""
    pairs = []
    for pair_id in (1, 2, 3):
        asset = AssetIdentifier(1, f"0x{pair_id:040x}", f"V{pair_id}", 18)
        quote = AssetIdentifier(1, f"0x{100:040x}", "USDC", 6)
        pairs.append(TradingPairIdentifier(
            asset, quote, f"0x{pair_id:040x}", f"0x{101:040x}",
            internal_id=pair_id,
            kind=TradingPairKind.vault if pair_id < 3 else TradingPairKind.spot_market_hold,
        ))
    candles = pd.DataFrame([
        dict(pair_id=p, timestamp=ts, open=1., high=1., low=1., close=1., volume=0.)
        for p in (1, 2, 3) for ts in pd.date_range("2026-03-01", periods=7)
    ]).set_index("timestamp", drop=False)
    result = Mock(spec=TradingStrategyUniverse)
    result.iterate_pairs.side_effect = lambda: iter(pairs)
    result.data_universe = SimpleNamespace(candles=GroupedCandleUniverse(candles), pairs=None)
    result.vault_state = None
    result.vault_window_overrides = {}
    return result


def test_historical_directional_closures_and_unknowns(universe: TradingStrategyUniverse):
    """Avoid future samples and distinguish caps, missing state and expired state.

    1. Seed a closed deposit, later open deposit/closed redemption and a zero cap.
    2. Check as-of boundaries, independent directions and expiry after two days.
    3. Exclude a cash vault/non-vault and timestamps outside price history.
    """
    # 1. A future sample must not change the earlier closure.
    universe.vault_state = pd.DataFrame([
        dict(pair_id=1, timestamp=pd.Timestamp("2026-03-02"), deposits_open=False, redemption_open=True, max_deposit=float("nan"), max_redeem=float("nan")),
        dict(pair_id=1, timestamp=pd.Timestamp("2026-03-03"), deposits_open=True, redemption_open=False, max_deposit=float("nan"), max_redeem=float("nan")),
        dict(pair_id=2, timestamp=pd.Timestamp("2026-03-02"), deposits_open=pd.NA, redemption_open=pd.NA, max_deposit=0., max_redeem=float("nan")),
    ])
    # 2. Earlier, boundary and stale states remain distinct.
    counts = calculate_vault_window_counts(universe, pd.date_range("2026-02-28", periods=9))
    assert counts.loc["2026-03-01", "Deposit state unknown"] == 2
    assert counts.loc["2026-03-02", "Deposits closed"] == 2
    assert counts.loc["2026-03-02", "Redemptions closed"] == 0
    assert counts.loc["2026-03-03", "Deposits closed"] == 1
    assert counts.loc["2026-03-03", "Redemptions closed"] == 1
    assert counts.loc["2026-03-05", "Redemptions closed"] == 1
    assert counts.loc["2026-03-06", "Redemptions closed"] == 0
    assert counts.loc["2026-03-06", "Redemption state unknown"] == 2
    # 3. Scope removes the cash vault; a non-vault never increases counts.
    scoped = calculate_vault_window_counts(universe, counts.index, pair_ids={1, 3})
    assert scoped.loc["2026-03-02", "Deposits closed"] == 1
    assert scoped["Vaults with price history"].max() == 1
    assert not scoped.loc[["2026-02-28", "2026-03-08"]].any().any()


def test_permission_and_capacity_clocks(universe: TradingStrategyUniverse):
    """A recent candle must not refresh old permission or capacity evidence.

    1. Attach a fresh state bucket with expired permission and zero-cap clocks.
    2. Verify expired closure evidence becomes unknown, while redemption stays open.
    """
    # 1. Bucket freshness does not authenticate either original evidence clock.
    universe.vault_state = pd.DataFrame([
        dict(pair_id=1, timestamp=pd.Timestamp("2026-03-05"), deposits_open=False,
             redemption_open=True, max_deposit=0., permission_provenance="observed",
             permission_observed_at=pd.Timestamp("2026-03-01"),
             capacity_observed_at=pd.Timestamp("2026-03-01")),
    ])
    # 2. Pricing's independently expired fields cannot supply a closure count.
    counts = calculate_vault_window_counts(universe, pd.DatetimeIndex(["2026-03-05"]), pair_ids={1})
    assert counts.iloc[0].to_dict() == {
        "Vaults with price history": 1, "Deposits closed": 0, "Redemptions closed": 0,
        "Deposit state unknown": 1, "Redemption state unknown": 0,
    }


def test_synthetic_windows_empty_scope_and_chart(universe: TradingStrategyUniverse):
    """Keep synthetic closures labelled and include the count series in charts.

    1. Configure a synthetic window over missing historical state.
    2. Exercise the standard chart using mocked existing indicator series.
    3. Verify empty scopes retain zero columns and invalid clocks fail clearly.
    """
    # 1. An override supplies both open and closed states, not unknowns.
    universe.vault_window_overrides = {1: VaultWindowSchedule(
        cadence=datetime.timedelta(days=3), open_duration=datetime.timedelta(days=1),
        anchor=datetime.datetime(2026, 3, 1),
    )}
    index = pd.date_range("2026-03-01", periods=4)
    # 2. Only indicator access is mocked; rendering and closure analytics are real.
    indicators = Mock(spec=StrategyInputIndicators)
    indicators.strategy_universe = universe
    indicators.resolve_indicator_data.side_effect = IndicatorNotFound("Optional indicator absent")
    indicators.get_indicator_series.return_value = pd.Series(1, index=index)
    figure, counts = available_trading_pairs(
        ChartInput(execution_context=unit_test_execution_context, strategy_input_indicators=indicators),
        vault_pair_ids={1}, with_dataframe=True,
    )
    column = "Deposits closed (includes synthetic windows)"
    assert counts[column].tolist() == [0, 1, 1, 0]
    assert counts["Deposit state unknown"].sum() == 0
    assert column in {trace.name for trace in figure.data}
    assert "Redemptions closed (includes synthetic windows)" in {trace.name for trace in figure.data}
    # 3. No vaults means zero counts, not an error or dropped closure columns.
    assert not calculate_vault_window_counts(universe, index, pair_ids=set()).any().any()
    assert calculate_vault_window_counts(universe, pd.DatetimeIndex([])).empty
    with pytest.raises(ValueError, match="naive UTC"):
        calculate_vault_window_counts(universe, index.tz_localize("UTC"))
