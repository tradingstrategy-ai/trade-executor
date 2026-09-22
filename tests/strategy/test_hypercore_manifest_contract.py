"""Cross-package compatibility checks for the HyperCore scan receipt."""

import datetime
from pathlib import Path

import pandas as pd
from eth_defi.vault.scan_manifest import build_vault_scan_manifest
from tradingstrategy.vault_scan_manifest import validate_vault_scan_manifest
from tradingstrategy.alternative_data.vault import read_vault_price_history_parquet, convert_vault_prices_to_candles
from eth_defi.research.wrangle_vault_prices import forward_fill_vault
from tradeexecutor.strategy.hypercore_data_availability import hypercore_manifest_is_ready


def test_producer_manifest_is_accepted_by_consumer_validator(tmp_path: Path) -> None:
    """Check the producer and consumer keep the v1 wire contract aligned.

    1. Build a producer-shaped manifest with a valid HyperCore receipt.
    2. Pass it through the actual trading-strategy runtime validator.
    3. Verify the fields and null semantics survive the package boundary.
    """

    # 1. Build a producer-shaped manifest with a valid HyperCore receipt.
    parquet_path = tmp_path / "cleaned-vault-prices-1h.parquet"
    pd.DataFrame(
        {
            "chain": [9999],
            "timestamp": [datetime.datetime(2026, 9, 22, 0, 30, tzinfo=datetime.timezone.utc)],
        }
    ).to_parquet(parquet_path)
    state_path = tmp_path / "vault-price-scan-state.json"
    state_path.write_text('{"items": {"9999": "2026-09-22T02:55:00"}}')

    # 2. Pass it through the actual trading-strategy runtime validator.
    manifest = build_vault_scan_manifest(
        cleaned_price_path=parquet_path,
        price_scan_state_path=state_path,
        price_object_key="cleaned-vault-prices-1h.parquet",
        price_etag="etag",
        published_at=datetime.datetime(2026, 9, 22, 3, 20, tzinfo=datetime.timezone.utc),
    )
    validated = validate_vault_scan_manifest(manifest)

    # 3. Verify the fields and null semantics survive the package boundary.
    assert validated["schema_version"] == 1
    assert validated["chains"]["9999"]["last_candle_at"] == "2026-09-22T00:30:00Z"
    assert validated["chains"]["9999"]["last_successful_price_scan_ended_at"] == "2026-09-22T02:55:00Z"


def test_four_hour_observations_survive_producer_and_consumer(tmp_path: Path) -> None:
    """Verify sparse source data stays usable across cleaning and daily resampling.

    1. Clean four-hour observations that stop before the decision midnight.
    2. Publish a post-midnight observation and validate the producer receipt.
    3. Read with a strict cutoff and compare daily price/TVL with sparse input.
    """
    # 1. Exercise the actual producer fill: it must not extend stale data to
    # midnight simply because the scanner has completed a new run.
    slot = datetime.datetime(2026, 9, 22)
    sparse = pd.DataFrame({
        "timestamp": pd.date_range("2026-09-20 01:30", "2026-09-22 01:30", freq="4h"),
        "chain": 9999,
        "address": "0x0000000000000000000000000000000000000001",
        "share_price": [float(i + 1) for i in range(13)],
        "total_assets": [float(1000 + i) for i in range(13)],
    }).set_index("timestamp")
    path = tmp_path / "cleaned.parquet"
    state_path = tmp_path / "scans.json"
    state_path.write_text('{"items":{"9999":"2026-09-22T04:00:00"}}')
    forward_fill_vault(sparse.iloc[:-1]).reset_index().to_parquet(path)
    manifest = build_vault_scan_manifest(path, state_path, "cleaned.parquet", "v1", datetime.datetime(2026, 9, 22, 4))
    assert not hypercore_manifest_is_ready(validate_vault_scan_manifest(manifest), slot)

    # 2. The first real post-midnight observation makes the published file ready.
    forward_fill_vault(sparse).reset_index().to_parquet(path)
    manifest = build_vault_scan_manifest(path, state_path, "cleaned.parquet", "v2", datetime.datetime(2026, 9, 22, 4))
    assert hypercore_manifest_is_ready(validate_vault_scan_manifest(manifest), slot)

    # 3. Prior-day daily closes agree regardless of sparse or hourly-cleaned
    # representation; no observation after the decision cutoff influences them.
    before_slot = read_vault_price_history_parquet(path, end_at=slot - datetime.timedelta(microseconds=1))
    prices, tvl = convert_vault_prices_to_candles(before_slot, "1d")
    raw_prices, raw_tvl = convert_vault_prices_to_candles(sparse[sparse.index < slot].reset_index(), "1d")
    pd.testing.assert_series_equal(prices.close, raw_prices.close)
    pd.testing.assert_series_equal(tvl.close, raw_tvl.close)
    assert prices.timestamp.max() < slot
