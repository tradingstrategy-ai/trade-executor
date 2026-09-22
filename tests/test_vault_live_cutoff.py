"""Tests for live vault-history cutoff handling."""

import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from tradingstrategy.client import Client
from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.vault_data_client import VaultDataClient, VaultDataset

import tradeexecutor.ethereum.vault.checks as vault_checks
import tradeexecutor.strategy.trading_strategy_universe as trading_strategy_universe
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.trading_strategy_universe import (
    _resolve_live_end_timestamps,
    load_partial_data,
)
from tradeexecutor.strategy.universe_model import UniverseOptions


pytestmark = pytest.mark.timeout(300)


def test_verified_snapshot_with_sparse_four_hour_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Read the verified receipt's sparse prices and TVL without midnight leakage.

    1. Download a version-checked private parquet with four-hour observations.
    2. Replace the ordinary shared cache and load through the real vault loader.
    3. Verify daily OHLC/TVL use only pre-slot data and survive missing samples.
    """
    # 1. Only HTTP and metadata discovery are substituted: paid network data
    # would make an exact midnight/version-race regression nondeterministic.
    slot = datetime.datetime(2026, 9, 22)
    address = "0x0000000000000000000000000000000000000001"
    observations = pd.date_range("2026-09-20 01:30", "2026-09-22 05:30", freq="4h")
    # A missed scan leaves an eight-hour gap; hourly row counts are not required.
    observations = observations.delete(8).union(pd.DatetimeIndex([slot]))
    raw = pd.DataFrame({
        "timestamp": observations,
        "chain": 9999,
        "address": address,
        "share_price": [float(i + 1) for i in range(len(observations))],
        "total_assets": [float(1000 + i) for i in range(len(observations))],
    })
    source = tmp_path / "source.parquet"
    raw.to_parquet(source)
    response = Mock(status_code=200, headers={"ETag": '"receipt-version"'})
    response.iter_content.return_value = [source.read_bytes()]
    data_client = VaultDataClient(api_key="test", download_root=tmp_path / "shared", session=Mock())
    data_client.session.get.return_value = response
    data_client.session.head.return_value = Mock(status_code=200, headers={})
    snapshot = data_client.download(
        VaultDataset.vault_prices, expected_etag="receipt-version", destination=tmp_path / "private" / "prices.parquet",
    )

    # 2. The normal cache contains completely different prices. Passing an
    # explicit snapshot must bypass that cache and any additional HTTP request.
    cached = data_client.get_cached_path(VaultDataset.vault_prices)
    cached.parent.mkdir(parents=True, exist_ok=True)
    raw.assign(share_price=999.0, total_assets=999999.0).to_parquet(cached)
    transport = SimpleNamespace(requests=None, get_cached_file_path=lambda filename, cache_path=None: tmp_path / filename)
    client = Client(None, transport)
    client.fetch_exchange_universe = lambda: ExchangeUniverse({})
    monkeypatch.setattr(trading_strategy_universe, "create_vault_data_client", lambda *args: data_client)
    monkeypatch.setattr(trading_strategy_universe, "load_multiple_vaults", lambda *args, **kwargs: (
        [], pd.DataFrame([{"chain_id": 9999, "address": address, "pair_id": 1}]),
    ))
    dataset = load_partial_data(
        client=client,
        execution_context=ExecutionContext(ExecutionMode.unit_testing_trading),
        time_bucket=TimeBucket.d1,
        pairs=pd.DataFrame(columns=["dex_type", "exchange_id", "pair_id"]),
        universe_options=UniverseOptions(
            history_period=datetime.timedelta(days=30),
            end_at=slot - datetime.timedelta(microseconds=1),
            vault_price_snapshot=snapshot,
        ),
        liquidity=True,
        vaults=object(),
        vault_history_source="trading-strategy-website",
    )

    # 3. Aggregate sparse observations directly, without requiring 24 samples
    # or admitting the midnight/post-midnight rows used to prove readiness.
    expected = raw[raw.timestamp < slot].set_index("timestamp")
    prices = dataset.candles.set_index("timestamp")
    tvl = dataset.liquidity.set_index("timestamp")
    pd.testing.assert_series_equal(prices.close, expected.share_price.resample("1d").last(), check_names=False, check_freq=False)
    pd.testing.assert_series_equal(prices.open, expected.share_price.resample("1d").first(), check_names=False, check_freq=False)
    pd.testing.assert_series_equal(tvl.close, expected.total_assets.resample("1d").last(), check_names=False, check_freq=False)
    assert prices.index.max() == pd.Timestamp("2026-09-21")
    assert not prices.forward_filled.any()
    assert not prices.close.isna().any()
    data_client.session.get.assert_called_once()


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



def make_vault_data_client_stub(parquet_path: Path) -> SimpleNamespace:
    """Stand in for the Creem vault dataset client.

    The real client downloads a paid dataset over HTTP, so tests hand the loader
    a local parquet instead.
    """
    return SimpleNamespace(
        download=lambda dataset: parquet_path,
        session=None,
        api_key="test-creem-key",
        get_url=lambda dataset: "https://example.com/vault-prices",
    )


def test_load_partial_data_uses_unfloored_vault_history_cutoff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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
    parquet_path = tmp_path / "vault-prices.parquet"
    pd.DataFrame(
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
    ).to_parquet(parquet_path)

    # 1. Build a stubbed live loader path with a fixed intraday clock and local vault history rows.
    monkeypatch.setattr(
        trading_strategy_universe,
        "create_vault_data_client",
        lambda client, download_root=None: make_vault_data_client_stub(parquet_path),
    )
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

    real_read_vault_price_history_parquet = trading_strategy_universe.read_vault_price_history_parquet

    def _capture_read(
        path,
        vault_pairs_df: pd.DataFrame | None = None,
        start_at: datetime.datetime | None = None,
        end_at: datetime.datetime | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        if vault_pairs_df is not None:
            captured["end_at"] = end_at
        return real_read_vault_price_history_parquet(
            path,
            vault_pairs_df=vault_pairs_df,
            start_at=start_at,
            end_at=end_at,
            columns=columns,
        )

    monkeypatch.setattr(trading_strategy_universe, "read_vault_price_history_parquet", _capture_read)
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


def test_load_partial_data_fast_vault_history_path_filters_parquet(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify website vault history uses the filtered parquet reader when transport exposes it.

    1. Create a local vault history parquet with matching, non-matching and after-cutoff rows.
    2. Run live ``load_partial_data()`` through a transport-level parquet fetch stub.
    3. Assert the fast path gives candle conversion only selected rows and pruned columns.
    """
    fixed_now = datetime.datetime(2026, 4, 11, 17, 9, 41)
    captured: dict[str, pd.DataFrame] = {}
    parquet_path = tmp_path / "vault-price-history.parquet"
    pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-04-10 23:59:35.537000"),
                "chain": 9999,
                "address": "0xABC",
                "share_price": 1.0,
                "total_assets": 999.0,
                "unused_column": "drop-me",
            },
            {
                "timestamp": pd.Timestamp("2026-04-11 04:22:29.962000"),
                "chain": 9999,
                "address": "0xabc",
                "share_price": 1.01,
                "total_assets": 1_000.0,
                "unused_column": "drop-me",
            },
            {
                "timestamp": pd.Timestamp("2026-04-11 04:22:29.962000"),
                "chain": 1,
                "address": "0xabc",
                "share_price": 2.0,
                "total_assets": 2_000.0,
                "unused_column": "drop-me",
            },
            {
                "timestamp": pd.Timestamp("2026-04-11 18:00:00"),
                "chain": 9999,
                "address": "0xabc",
                "share_price": 1.02,
                "total_assets": 1_001.0,
                "unused_column": "drop-me",
            },
        ]
    ).to_parquet(parquet_path)

    transport = SimpleNamespace(
        requests=None,
        get_cached_file_path=lambda filename, cache_path=None: parquet_path,
    )
    client = Client(None, transport)
    client.fetch_exchange_universe = lambda: ExchangeUniverse({})

    # 1. Create a local vault history parquet with matching, non-matching and after-cutoff rows.
    monkeypatch.setattr(
        trading_strategy_universe,
        "create_vault_data_client",
        lambda client, download_root=None: make_vault_data_client_stub(parquet_path),
    )
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

    def _capture_convert(df: pd.DataFrame, frequency: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        captured["df"] = df.copy()
        return (
            pd.DataFrame({"timestamp": [pd.Timestamp("2026-04-11 00:00:00")], "pair_id": [1]}),
            pd.DataFrame(),
        )

    monkeypatch.setattr(trading_strategy_universe, "convert_vault_prices_to_candles", _capture_convert)
    monkeypatch.setattr(vault_checks, "build_vault_history_diagnostics", lambda *args, **kwargs: None)
    monkeypatch.setattr(vault_checks, "log_vault_history_diagnostics", lambda *args, **kwargs: None)
    monkeypatch.setattr(vault_checks, "log_stale_vault_candle_data", lambda *args, **kwargs: None)

    # 2. Run live ``load_partial_data()`` through a transport-level parquet fetch stub.
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

    # 3. Assert the fast path gives candle conversion only selected rows and pruned columns.
    filtered_df = captured["df"]
    assert dataset.end_at == datetime.datetime(2026, 4, 11, 0, 0, 0)
    assert filtered_df["timestamp"].tolist() == [
        pd.Timestamp("2026-04-10 23:59:35.537000"),
        pd.Timestamp("2026-04-11 04:22:29.962000"),
    ]
    assert filtered_df["chain"].tolist() == [9999, 9999]
    assert filtered_df["address"].tolist() == ["0xabc", "0xabc"]
    assert "unused_column" not in filtered_df.columns
