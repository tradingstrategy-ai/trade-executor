"""Tests for live vault-history cutoff handling."""

import datetime
from types import SimpleNamespace

import pandas as pd
import pytest
from tradingstrategy.client import Client
from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.timebucket import TimeBucket

import tradeexecutor.ethereum.vault.checks as vault_checks
import tradeexecutor.strategy.trading_strategy_universe as trading_strategy_universe
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.trading_strategy_universe import (
    _resolve_live_end_timestamps,
    load_partial_data,
)
from tradeexecutor.strategy.universe_model import UniverseOptions


pytestmark = pytest.mark.timeout(300)


def test_resolve_live_end_timestamps_daily_rounding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify live daily datasets keep a floored dataset end and an unfloored vault cutoff.

    1. Freeze the live clock to a specific intraday timestamp.
    2. Resolve live end timestamps for a daily bucket with rounding enabled.
    3. Assert the dataset end is floored to midnight while the vault cutoff keeps the live timestamp.
    """
    fixed_now = datetime.datetime(2026, 4, 11, 17, 9, 41)

    # 1. Freeze the live clock to a specific intraday timestamp.
    monkeypatch.setattr(trading_strategy_universe, "native_datetime_utc_now", lambda: fixed_now)

    # 2. Resolve live end timestamps for a daily bucket with rounding enabled.
    dataset_end_at, vault_history_filter_end_at = _resolve_live_end_timestamps(
        time_bucket=TimeBucket.d1,
        explicit_end_at=None,
        round_start_end=True,
    )

    # 3. Assert the dataset end is floored to midnight while the vault cutoff keeps the live timestamp.
    assert dataset_end_at == datetime.datetime(2026, 4, 11, 0, 0, 0)
    assert vault_history_filter_end_at == fixed_now


def test_resolve_live_end_timestamps_without_rounding() -> None:
    """Verify disabled rounding keeps identical dataset and vault end timestamps.

    1. Create an explicit intraday end timestamp.
    2. Resolve live end timestamps with rounding disabled.
    3. Assert both returned timestamps are unchanged.
    """
    explicit_end_at = datetime.datetime(2026, 4, 11, 17, 9, 41)

    # 1. Create an explicit intraday end timestamp.
    # 2. Resolve live end timestamps with rounding disabled.
    dataset_end_at, vault_history_filter_end_at = _resolve_live_end_timestamps(
        time_bucket=TimeBucket.d1,
        explicit_end_at=explicit_end_at,
        round_start_end=False,
    )

    # 3. Assert both returned timestamps are unchanged.
    assert dataset_end_at == explicit_end_at
    assert vault_history_filter_end_at == explicit_end_at


def test_load_partial_data_uses_unfloored_vault_history_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify live website vault history filtering receives the unfloored cutoff.

    1. Build a stubbed live loader path with a fixed intraday clock and local vault history rows.
    2. Run ``load_partial_data()`` with website vault history enabled and capture the filter cutoff.
    3. Assert the vault-history filter sees the unfloored live timestamp while the dataset end stays floored.
    """
    fixed_now = datetime.datetime(2026, 4, 11, 17, 9, 41)
    captured: dict[str, object] = {}
    transport = SimpleNamespace(
        requests=None,
        get_cached_file_path=lambda filename, cache_path=None: f"/tmp/{filename}",
    )
    client = Client(None, transport)
    client.fetch_exchange_universe = lambda: ExchangeUniverse({})
    client.fetch_vault_price_history = lambda download_root=None: pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-04-11 04:22:29.962000"),
                "chain": 9999,
                "address": "0xabc",
                "share_price": 1.01,
                "total_assets": 1_000.0,
            },
            {
                "timestamp": pd.Timestamp("2026-04-10 23:59:35.537000"),
                "chain": 9999,
                "address": "0xabc",
                "share_price": 1.0,
                "total_assets": 999.0,
            },
        ]
    )

    # 1. Build a stubbed live loader path with a fixed intraday clock and local vault history rows.
    monkeypatch.setattr(trading_strategy_universe, "native_datetime_utc_now", lambda: fixed_now)
    monkeypatch.setattr(
        trading_strategy_universe,
        "load_multiple_vaults",
        lambda vaults, check_all_vaults_found=True: (
            [],
            pd.DataFrame(
                [
                    {
                        "chain_id": 9999,
                        "address": "0xabc",
                        "pair_id": 1,
                    }
                ]
            ),
        ),
    )

    def _capture_filter(
        vault_prices_df: pd.DataFrame,
        vault_pairs_df: pd.DataFrame,
        start_at: datetime.datetime | None = None,
        end_at: datetime.datetime | None = None,
    ) -> pd.DataFrame:
        captured["end_at"] = end_at
        del vault_pairs_df
        if start_at is not None:
            vault_prices_df = vault_prices_df.loc[vault_prices_df["timestamp"] >= pd.Timestamp(start_at)]
        if end_at is not None:
            vault_prices_df = vault_prices_df.loc[vault_prices_df["timestamp"] <= pd.Timestamp(end_at)]
        return vault_prices_df.copy()

    monkeypatch.setattr(trading_strategy_universe, "filter_vault_price_history", _capture_filter)
    monkeypatch.setattr(
        trading_strategy_universe,
        "convert_vault_prices_to_candles",
        lambda df, frequency: (
            pd.DataFrame({"timestamp": [pd.Timestamp("2026-04-11 00:00:00")], "pair_id": [1]}),
            pd.DataFrame(),
        ),
    )
    monkeypatch.setattr(vault_checks, "build_vault_history_diagnostics", lambda *args, **kwargs: None)
    monkeypatch.setattr(vault_checks, "log_vault_history_diagnostics", lambda *args, **kwargs: None)
    monkeypatch.setattr(vault_checks, "log_stale_vault_candle_data", lambda *args, **kwargs: None)

    # 2. Run ``load_partial_data()`` with website vault history enabled and capture the filter cutoff.
    dataset = load_partial_data(
        client=client,
        execution_context=ExecutionContext(ExecutionMode.unit_testing_trading),
        time_bucket=TimeBucket.d1,
        pairs=pd.DataFrame(columns=["dex_type", "exchange_id", "pair_id"]),
        universe_options=UniverseOptions(history_period=datetime.timedelta(days=30)),
        liquidity=False,
        vaults=object(),
        vault_history_source="trading-strategy-website",
    )

    # 3. Assert the vault-history filter sees the unfloored live timestamp while the dataset end stays floored.
    assert captured["end_at"] == fixed_now
    assert dataset.end_at == datetime.datetime(2026, 4, 11, 0, 0, 0)
