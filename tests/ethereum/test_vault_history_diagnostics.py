"""Test vault history startup diagnostics for cache, source and resample freshness."""

import datetime
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from tradeexecutor.ethereum.vault.checks import (
    build_vault_history_diagnostics,
    log_stale_vault_candle_data,
    log_vault_history_diagnostics,
)
from tradingstrategy.alternative_data.vault import convert_vault_prices_to_candles


pytestmark = pytest.mark.timeout(300)


class _MockResponse:
    """Minimal response stub for HEAD metadata tests."""

    def __init__(self, headers: dict[str, str]):
        self.headers = headers

    def raise_for_status(self) -> None:
        """Pretend the response succeeded."""


class _MockSession:
    """Minimal session stub with configurable HEAD behaviour."""

    def __init__(
        self,
        response: _MockResponse | None = None,
        error: Exception | None = None,
    ):
        self.response = response
        self.error = error

    def head(self, url: str, allow_redirects: bool = True, timeout: int = 30) -> _MockResponse:
        """Return the configured response or raise the configured error."""
        del url
        del allow_redirects
        del timeout

        if self.error is not None:
            raise self.error

        assert self.response is not None
        return self.response


def _build_vault_price_history_df(
    address: str,
    timestamps: list[datetime.datetime],
    share_price: float = 1.02,
    total_assets: float = 100_000.0,
) -> pd.DataFrame:
    """Build a minimal vault history DataFrame for diagnostics tests."""
    rows = []
    for timestamp in timestamps:
        rows.append(
            {
                "timestamp": pd.Timestamp(timestamp),
                "chain": 8453,
                "address": address,
                "share_price": share_price,
                "total_assets": total_assets,
            }
        )
    return pd.DataFrame(rows)


def test_build_vault_history_diagnostics_includes_cache_and_remote_metadata(
    tmp_path: Path,
) -> None:
    """Verify diagnostics capture cache mtime, remote headers and freshness timestamps.

    1. Create a temporary cached parquet placeholder with a fixed mtime.
    2. Build diagnostics with mocked remote HEAD metadata and explicit timestamps.
    3. Assert cache, remote and parquet freshness fields are populated correctly.
    """
    # 1. Create a temporary cached parquet placeholder with a fixed mtime.
    cache_path = tmp_path / "vault-price-history.parquet"
    cache_path.write_bytes(b"vault-history")
    cache_mtime = datetime.datetime(2026, 4, 11, 10, 0, 0)
    cache_path.touch()
    os_mtime = datetime.datetime(2026, 4, 11, 10, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
    cache_path.chmod(0o644)
    os.utime(cache_path, (os_mtime, os_mtime))

    now = datetime.datetime(2026, 4, 11, 14, 0, 0)
    raw_df = _build_vault_price_history_df(
        "0x1234000000000000000000000000000000000000",
        [datetime.datetime(2026, 4, 11, 13, 0, 0)],
    )
    filtered_df = raw_df.copy()
    resampled_df = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp(datetime.datetime(2026, 4, 11, 0, 0, 0))],
            "pair_id": [1],
        }
    )
    session = _MockSession(
        response=_MockResponse(
            {
                "Last-Modified": "Fri, 11 Apr 2026 12:30:00 GMT",
                "ETag": '"abc123"',
                "Content-Length": "12345",
            }
        )
    )

    # 2. Build diagnostics with mocked remote HEAD metadata and explicit timestamps.
    diagnostics = build_vault_history_diagnostics(
        raw_vault_price_df=raw_df,
        filtered_vault_price_df=filtered_df,
        resampled_vault_candle_df=resampled_df,
        cache_path=cache_path,
        http_session=session,
        now=now,
    )

    # 3. Assert cache, remote and parquet freshness fields are populated correctly.
    assert diagnostics.cache_path == cache_path
    assert diagnostics.local_cache_mtime == cache_mtime
    assert diagnostics.local_cache_age == datetime.timedelta(hours=4)
    assert diagnostics.remote_last_modified == datetime.datetime(2026, 4, 11, 12, 30, 0)
    assert diagnostics.remote_last_modified_age == datetime.timedelta(hours=1, minutes=30)
    assert diagnostics.remote_etag == '"abc123"'
    assert diagnostics.remote_content_length == 12345
    assert diagnostics.parquet_max_timestamp == datetime.datetime(2026, 4, 11, 13, 0, 0)
    assert diagnostics.filtered_max_timestamp == datetime.datetime(2026, 4, 11, 13, 0, 0)
    assert diagnostics.resampled_max_timestamp == datetime.datetime(2026, 4, 11, 0, 0, 0)
    assert diagnostics.parquet_data_age == datetime.timedelta(hours=1)


def test_build_vault_history_diagnostics_detects_daily_resample_floor_gap() -> None:
    """Verify diagnostics separate fresh parquet data from older daily candle timestamps.

    1. Build raw vault history whose latest source sample is still under 24 hours old.
    2. Convert the same history to daily candles so the visible candle is floored to midnight.
    3. Assert source data is still fresh while the resampled candle looks older than 24 hours.
    """
    # 1. Build raw vault history whose latest source sample is still under 24 hours old.
    address = "0xabcd000000000000000000000000000000000000"
    now = datetime.datetime(2026, 4, 11, 13, 59, 0)
    raw_df = _build_vault_price_history_df(
        address,
        [datetime.datetime(2026, 4, 10, 23, 59, 0)],
    )

    # 2. Convert the same history to daily candles so the visible candle is floored to midnight.
    resampled_df, _ = convert_vault_prices_to_candles(raw_df.copy(), "1d")

    # 3. Assert source data is still fresh while the resampled candle looks older than 24 hours.
    diagnostics = build_vault_history_diagnostics(
        raw_vault_price_df=raw_df,
        filtered_vault_price_df=raw_df.copy(),
        resampled_vault_candle_df=resampled_df,
        cache_path=None,
        http_session=_MockSession(response=_MockResponse({})),
        now=now,
    )

    assert diagnostics.parquet_data_age == datetime.timedelta(hours=14)
    assert diagnostics.resampled_data_age == datetime.timedelta(days=1, hours=13, minutes=59)
    assert diagnostics.filtered_to_resampled_delta == datetime.timedelta(hours=23, minutes=59)
    assert diagnostics.expected_daily_flooring_reason is not None


def test_log_vault_history_diagnostics_warns_when_source_data_is_stale(caplog: pytest.LogCaptureFixture) -> None:
    """Verify stale parquet data escalates the summary log level to warning.

    1. Build diagnostics with a parquet max timestamp older than the 24-hour tolerance.
    2. Emit the startup summary log.
    3. Assert the summary is logged at warning level and mentions the parquet age.
    """
    # 1. Build diagnostics with a parquet max timestamp older than the 24-hour tolerance.
    diagnostics = build_vault_history_diagnostics(
        raw_vault_price_df=_build_vault_price_history_df(
            "0xdead000000000000000000000000000000000000",
            [datetime.datetime(2026, 4, 8, 12, 0, 0)],
        ),
        filtered_vault_price_df=_build_vault_price_history_df(
            "0xdead000000000000000000000000000000000000",
            [datetime.datetime(2026, 4, 8, 12, 0, 0)],
        ),
        resampled_vault_candle_df=pd.DataFrame(
            {
                "timestamp": [pd.Timestamp(datetime.datetime(2026, 4, 8, 0, 0, 0))],
                "pair_id": [1],
            }
        ),
        cache_path=None,
        http_session=_MockSession(response=_MockResponse({})),
        now=datetime.datetime(2026, 4, 11, 13, 59, 0),
    )

    # 2. Emit the startup summary log.
    with caplog.at_level("INFO"):
        log_vault_history_diagnostics(diagnostics)

    # 3. Assert the summary is logged at warning level and mentions the parquet age.
    summary_record = next(record for record in caplog.records if "Vault history freshness summary" in record.message)
    assert summary_record.levelname == "WARNING"
    assert "parquet_data_age=3 days, 1:59:00" in summary_record.message


def test_log_vault_history_diagnostics_warns_when_remote_head_fails(caplog: pytest.LogCaptureFixture) -> None:
    """Verify remote HEAD failure is recorded without aborting startup.

    1. Build diagnostics with a mocked HEAD failure but otherwise fresh parquet data.
    2. Emit the startup summary log.
    3. Assert the summary is logged at warning level and includes the HEAD error.
    """
    # 1. Build diagnostics with a mocked HEAD failure but otherwise fresh parquet data.
    diagnostics = build_vault_history_diagnostics(
        raw_vault_price_df=_build_vault_price_history_df(
            "0xbeef000000000000000000000000000000000000",
            [datetime.datetime(2026, 4, 11, 12, 0, 0)],
        ),
        filtered_vault_price_df=_build_vault_price_history_df(
            "0xbeef000000000000000000000000000000000000",
            [datetime.datetime(2026, 4, 11, 12, 0, 0)],
        ),
        resampled_vault_candle_df=pd.DataFrame(
            {
                "timestamp": [pd.Timestamp(datetime.datetime(2026, 4, 11, 0, 0, 0))],
                "pair_id": [1],
            }
        ),
        cache_path=None,
        http_session=_MockSession(error=requests.RequestException("boom")),
        now=datetime.datetime(2026, 4, 11, 13, 59, 0),
    )

    # 2. Emit the startup summary log.
    with caplog.at_level("INFO"):
        log_vault_history_diagnostics(diagnostics)

    # 3. Assert the summary is logged at warning level and includes the HEAD error.
    summary_record = next(record for record in caplog.records if "Vault history freshness summary" in record.message)
    assert summary_record.levelname == "WARNING"
    assert "RequestException: boom" in summary_record.message


def test_log_vault_history_diagnostics_does_not_warn_for_expected_d1_floor(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify expected 1d floor artefacts stay informational instead of warning.

    1. Build diagnostics where the parquet is still fresh but the 1d candle looks stale.
    2. Emit the startup summary log.
    3. Assert the summary carries the daily-floor explanation and no suspicious-values warning is emitted.
    """
    # 1. Build diagnostics where the parquet is still fresh but the 1d candle looks stale.
    diagnostics = build_vault_history_diagnostics(
        raw_vault_price_df=_build_vault_price_history_df(
            "0xcafe000000000000000000000000000000000000",
            [datetime.datetime(2026, 4, 10, 22, 7, 51)],
        ),
        filtered_vault_price_df=_build_vault_price_history_df(
            "0xcafe000000000000000000000000000000000000",
            [datetime.datetime(2026, 4, 10, 22, 7, 49)],
        ),
        resampled_vault_candle_df=pd.DataFrame(
            {
                "timestamp": [pd.Timestamp(datetime.datetime(2026, 4, 10, 0, 0, 0))],
                "pair_id": [1],
            }
        ),
        cache_path=None,
        http_session=_MockSession(
            response=_MockResponse(
                {
                    "Last-Modified": "Fri, 10 Apr 2026 22:19:05 GMT",
                }
            )
        ),
        vault_history_filter_end_at=datetime.datetime(2026, 4, 11, 16, 43, 33),
        now=datetime.datetime(2026, 4, 11, 16, 43, 33),
    )

    # 2. Emit the startup summary log.
    with caplog.at_level("INFO"):
        log_vault_history_diagnostics(diagnostics)

    # 3. Assert the summary carries the explanation and no suspicious-values warning is emitted.
    summary_record = next(record for record in caplog.records if "Vault history freshness summary" in record.message)
    assert summary_record.levelname == "INFO"
    assert "expected_daily_flooring_reason=Expected 1d floor artefact" in summary_record.message
    assert not any("Vault history freshness has suspicious values" in record.message for record in caplog.records)


def test_log_vault_history_diagnostics_warns_for_unexpected_parquet_to_filtered_gap(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify fresh source rows lost before filtering still trigger a warning.

    1. Build diagnostics where the source parquet is fresh but the filtered max timestamp trails unexpectedly.
    2. Emit the startup summary log.
    3. Assert the suspicious-values warning highlights the parquet-to-filter gap.
    """
    # 1. Build diagnostics where the source parquet is fresh but the filtered max timestamp trails unexpectedly.
    diagnostics = build_vault_history_diagnostics(
        raw_vault_price_df=_build_vault_price_history_df(
            "0xface000000000000000000000000000000000000",
            [datetime.datetime(2026, 4, 11, 12, 0, 0)],
        ),
        filtered_vault_price_df=_build_vault_price_history_df(
            "0xface000000000000000000000000000000000000",
            [datetime.datetime(2026, 4, 11, 9, 30, 0)],
        ),
        resampled_vault_candle_df=pd.DataFrame(
            {
                "timestamp": [pd.Timestamp(datetime.datetime(2026, 4, 11, 9, 30, 0))],
                "pair_id": [1],
            }
        ),
        cache_path=None,
        http_session=_MockSession(response=_MockResponse({})),
        vault_history_filter_end_at=datetime.datetime(2026, 4, 11, 13, 0, 0),
        now=datetime.datetime(2026, 4, 11, 13, 30, 0),
    )

    # 2. Emit the startup summary log.
    with caplog.at_level("INFO"):
        log_vault_history_diagnostics(diagnostics)

    # 3. Assert the suspicious-values warning highlights the parquet-to-filter gap.
    warning_record = next(record for record in caplog.records if "Vault history freshness has suspicious values" in record.message)
    assert warning_record.levelname == "WARNING"
    assert "parquet_to_filtered_delta" in warning_record.message
    assert "Selected vault history trails the freshest parquet row" in warning_record.message


def test_vault_history_logging_keeps_summary_before_per_vault_stale_entries(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify the startup summary is logged before the aggregated stale-vault warning.

    1. Build fresh summary diagnostics and a stale per-vault candle snapshot.
    2. Emit the summary log and then the aggregated stale-vault warning.
    3. Assert the summary message appears before the stale-vault warning and the stale warning is at warning level.
    """
    # 1. Build fresh summary diagnostics and a stale per-vault candle snapshot.
    diagnostics = build_vault_history_diagnostics(
        raw_vault_price_df=_build_vault_price_history_df(
            "0xcafe000000000000000000000000000000000000",
            [datetime.datetime(2026, 4, 10, 23, 59, 0)],
        ),
        filtered_vault_price_df=_build_vault_price_history_df(
            "0xcafe000000000000000000000000000000000000",
            [datetime.datetime(2026, 4, 10, 23, 59, 0)],
        ),
        resampled_vault_candle_df=pd.DataFrame(
            {
                "timestamp": [pd.Timestamp(datetime.datetime(2026, 4, 10, 0, 0, 0))],
                "pair_id": [272079929],
            }
        ),
        cache_path=None,
        http_session=_MockSession(response=_MockResponse({})),
        now=datetime.datetime(2026, 4, 11, 13, 59, 0),
    )
    vault_candle_df = pd.DataFrame(
        {
            "pair_id": [272079929],
            "timestamp": [pd.Timestamp(datetime.datetime(2026, 4, 10, 0, 0, 0))],
        }
    )
    vault_pairs_df = pd.DataFrame(
        {
            "pair_id": [272079929],
            "exchange_name": ["MirVault"],
            "address": ["0x10aa8b767d0742de206bfafe36b8556634379c39"],
            "token_metadata": [SimpleNamespace(tvl=80_107.73)],
        }
    )

    # 2. Emit the summary log and then the aggregated stale-vault warning.
    with caplog.at_level("INFO"):
        log_vault_history_diagnostics(diagnostics)
        log_stale_vault_candle_data(
            vault_candle_df=vault_candle_df,
            vault_pairs_df=vault_pairs_df,
            now=datetime.datetime(2026, 4, 11, 13, 59, 0),
        )

    # 3. Assert the summary message appears before the stale-vault warning and the stale warning is at warning level.
    relevant_records = [
        record
        for record in caplog.records
        if "Vault history freshness summary" in record.message or "Vault candle data is stale" in record.message
    ]
    assert "Vault history freshness summary" in relevant_records[0].message
    assert relevant_records[1].levelname == "WARNING"
    assert "Vault candle data is stale (>24h after 1d resampling floor) for 1 vault(s)" in relevant_records[1].message
    assert "\nvault" in relevant_records[1].message
    assert "MirVault" in relevant_records[1].message
