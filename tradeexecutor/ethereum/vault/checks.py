"""Vault data quality checks.

Guards against making allocation decisions on stale vault data.
The framework forward-fill pipeline (see ``create_from_dataset()``)
extends every pair's candle and TVL data to the decision timestamp,
which prevents indicator crashes but also masks stale data. A vault
whose real data stopped days ago will have forward-filled rows that
look valid — ``tvl()`` sees the last real TVL repeated, ``age()``
keeps growing, and ``age_ramp_weight()`` increases. Without an
explicit staleness check, ``decide_trades()`` would allocate to
vaults based on synthetic forward-filled data.

This module provides ``check_stale_vault_data()`` which inspects the
``forward_filled`` column in the candle universe to find the last
*real* data timestamp per vault pair and raises if any vault exceeds
a staleness tolerance.

It also provides ``get_vault_data_freshness()`` which returns a
diagnostic DataFrame of freshness information for all vault pairs,
covering both candle and TVL data. This is intended for operator
dashboards and strategy chart registries.
"""

import datetime
import logging
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd
import requests
from tabulate import tabulate

from eth_defi.compat import native_datetime_utc_fromtimestamp, native_datetime_utc_now
from tradingstrategy.alternative_data.vault import CLEANED_VAULT_PRICE_PARQUET_URL

from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.execution_context import ExecutionMode
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse

logger = logging.getLogger(__name__)


DEFAULT_VAULT_HISTORY_STALE_TOLERANCE = datetime.timedelta(hours=24)


class StaleVaultData(Exception):
    """Raised when one or more vaults have data older than the tolerance."""
    pass


@dataclass(slots=True)
class VaultHistoryDiagnostics:
    """Summarise vault history freshness across cache, source, filter and resample stages."""

    cache_path: Path | None
    local_cache_mtime: datetime.datetime | None
    local_cache_age: datetime.timedelta | None
    remote_last_modified: datetime.datetime | None
    remote_last_modified_age: datetime.timedelta | None
    remote_etag: str | None
    remote_content_length: int | None
    remote_head_error: str | None
    parquet_max_timestamp: datetime.datetime | None
    parquet_data_age: datetime.timedelta | None
    filtered_max_timestamp: datetime.datetime | None
    filtered_data_age: datetime.timedelta | None
    resampled_max_timestamp: datetime.datetime | None
    resampled_data_age: datetime.timedelta | None
    parquet_to_filtered_delta: datetime.timedelta | None
    filtered_to_resampled_delta: datetime.timedelta | None
    vault_history_filter_end_at: datetime.datetime | None
    expected_daily_flooring_reason: str | None


def _coerce_naive_utc_datetime(value: pd.Timestamp | datetime.datetime | None) -> datetime.datetime | None:
    """Convert pandas and aware datetimes to naive UTC datetimes."""
    if value is None:
        return None

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        value = value.to_pydatetime()

    if value.tzinfo is not None:
        value = value.astimezone(datetime.timezone.utc).replace(tzinfo=None)

    return value


def _get_max_timestamp(df: pd.DataFrame | None) -> datetime.datetime | None:
    """Get the maximum timestamp from a DataFrame if available."""
    if df is None or len(df) == 0:
        return None

    if "timestamp" in df.columns:
        return _coerce_naive_utc_datetime(pd.Timestamp(df["timestamp"].max()))

    if df.index.name == "timestamp" and len(df.index) > 0:
        return _coerce_naive_utc_datetime(pd.Timestamp(df.index.max()))

    return None


def _calculate_age(
    timestamp: datetime.datetime | None,
    reference_now: datetime.datetime,
) -> datetime.timedelta | None:
    """Calculate age from a timestamp to the reference time."""
    if timestamp is None:
        return None
    return reference_now - timestamp


def _calculate_delta(
    older: datetime.datetime | None,
    newer: datetime.datetime | None,
) -> datetime.timedelta | None:
    """Calculate the gap between two timestamps when both are present."""
    if older is None or newer is None:
        return None
    return older - newer


def _parse_last_modified_header(header_value: str | None) -> datetime.datetime | None:
    """Parse an HTTP Last-Modified header to a naive UTC datetime."""
    if not header_value:
        return None

    parsed = parsedate_to_datetime(header_value)
    return _coerce_naive_utc_datetime(parsed)


def _calculate_expected_daily_flooring_reason(
    parquet_data_age: datetime.timedelta | None,
    filtered_data_age: datetime.timedelta | None,
    filtered_max_timestamp: datetime.datetime | None,
    resampled_max_timestamp: datetime.datetime | None,
    vault_history_filter_end_at: datetime.datetime | None,
    stale_tolerance: datetime.timedelta = DEFAULT_VAULT_HISTORY_STALE_TOLERANCE,
) -> str | None:
    """Explain when a stale-looking resampled timestamp is an expected daily floor artefact."""
    if (
        parquet_data_age is None
        or filtered_data_age is None
        or parquet_data_age > stale_tolerance
        or filtered_data_age > stale_tolerance
        or filtered_max_timestamp is None
        or resampled_max_timestamp is None
        or filtered_max_timestamp <= resampled_max_timestamp
    ):
        return None

    filtered_floor = filtered_max_timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    if resampled_max_timestamp != filtered_floor:
        return None

    if vault_history_filter_end_at is not None:
        return (
            "Expected 1d floor artefact: vault history kept intraday source data until "
            f"{vault_history_filter_end_at}, but the resampled candle timestamp is floored to 00:00 UTC"
        )

    return "Expected 1d floor artefact: the resampled candle timestamp is floored to 00:00 UTC"


def _fetch_remote_vault_history_metadata(
    http_session: requests.Session | None,
    remote_url: str,
) -> tuple[datetime.datetime | None, str | None, int | None, str | None]:
    """Fetch remote vault parquet metadata with a lightweight HEAD request."""
    if http_session is None:
        return None, None, None, "No HTTP session available for remote vault metadata HEAD request"

    try:
        response = http_session.head(remote_url, allow_redirects=True, timeout=30)
        response.raise_for_status()
        last_modified = _parse_last_modified_header(response.headers.get("Last-Modified"))
        etag = response.headers.get("ETag")
        content_length_header = response.headers.get("Content-Length")
        content_length = int(content_length_header) if content_length_header is not None else None
        return last_modified, etag, content_length, None
    except Exception as exc:
        return None, None, None, f"{exc.__class__.__name__}: {exc}"


def build_vault_history_diagnostics(
    raw_vault_price_df: pd.DataFrame,
    filtered_vault_price_df: pd.DataFrame,
    resampled_vault_candle_df: pd.DataFrame | None,
    cache_path: Path | None,
    http_session: requests.Session | None,
    vault_history_filter_end_at: datetime.datetime | None = None,
    remote_url: str = CLEANED_VAULT_PRICE_PARQUET_URL,
    now: datetime.datetime | None = None,
) -> VaultHistoryDiagnostics:
    """Build startup diagnostics for vault history freshness."""
    reference_now = now or native_datetime_utc_now()

    local_cache_mtime = None
    local_cache_age = None
    if cache_path is not None and cache_path.exists():
        local_cache_mtime = native_datetime_utc_fromtimestamp(cache_path.stat().st_mtime)
        local_cache_age = reference_now - local_cache_mtime

    remote_last_modified, remote_etag, remote_content_length, remote_head_error = _fetch_remote_vault_history_metadata(
        http_session=http_session,
        remote_url=remote_url,
    )

    parquet_max_timestamp = _get_max_timestamp(raw_vault_price_df)
    filtered_max_timestamp = _get_max_timestamp(filtered_vault_price_df)
    resampled_max_timestamp = _get_max_timestamp(resampled_vault_candle_df)
    parquet_data_age = _calculate_age(parquet_max_timestamp, reference_now)
    filtered_data_age = _calculate_age(filtered_max_timestamp, reference_now)

    return VaultHistoryDiagnostics(
        cache_path=cache_path,
        local_cache_mtime=local_cache_mtime,
        local_cache_age=local_cache_age,
        remote_last_modified=remote_last_modified,
        remote_last_modified_age=_calculate_age(remote_last_modified, reference_now),
        remote_etag=remote_etag,
        remote_content_length=remote_content_length,
        remote_head_error=remote_head_error,
        parquet_max_timestamp=parquet_max_timestamp,
        parquet_data_age=parquet_data_age,
        filtered_max_timestamp=filtered_max_timestamp,
        filtered_data_age=filtered_data_age,
        resampled_max_timestamp=resampled_max_timestamp,
        resampled_data_age=_calculate_age(resampled_max_timestamp, reference_now),
        parquet_to_filtered_delta=_calculate_delta(parquet_max_timestamp, filtered_max_timestamp),
        filtered_to_resampled_delta=_calculate_delta(filtered_max_timestamp, resampled_max_timestamp),
        vault_history_filter_end_at=vault_history_filter_end_at,
        expected_daily_flooring_reason=_calculate_expected_daily_flooring_reason(
            parquet_data_age=parquet_data_age,
            filtered_data_age=filtered_data_age,
            filtered_max_timestamp=filtered_max_timestamp,
            resampled_max_timestamp=resampled_max_timestamp,
            vault_history_filter_end_at=vault_history_filter_end_at,
        ),
    )


def log_vault_history_diagnostics(
    diagnostics: VaultHistoryDiagnostics,
    stale_tolerance: datetime.timedelta = DEFAULT_VAULT_HISTORY_STALE_TOLERANCE,
) -> None:
    """Log a one-line startup summary for vault history freshness."""
    level = logging.INFO
    if diagnostics.remote_head_error:
        level = logging.WARNING
    elif diagnostics.parquet_data_age is not None and diagnostics.parquet_data_age > stale_tolerance:
        level = logging.WARNING

    logger.log(
        level,
        "Vault history freshness summary: cache_path=%s, local_cache_mtime=%s, local_cache_age=%s, "
        "remote_last_modified=%s, remote_last_modified_age=%s, remote_etag=%s, remote_content_length=%s, "
        "parquet_max_timestamp=%s, parquet_data_age=%s, filtered_max_timestamp=%s, filtered_data_age=%s, "
        "resampled_max_timestamp=%s, resampled_data_age=%s, parquet_to_filtered_delta=%s, "
        "filtered_to_resampled_delta=%s, vault_history_filter_end_at=%s, expected_daily_flooring_reason=%s, "
        "remote_head_error=%s. When using 1d resampling, the latest candle is floored to 00:00 UTC, "
        "so a candle timestamp can still reflect materially newer source data.",
        diagnostics.cache_path,
        diagnostics.local_cache_mtime,
        diagnostics.local_cache_age,
        diagnostics.remote_last_modified,
        diagnostics.remote_last_modified_age,
        diagnostics.remote_etag,
        diagnostics.remote_content_length,
        diagnostics.parquet_max_timestamp,
        diagnostics.parquet_data_age,
        diagnostics.filtered_max_timestamp,
        diagnostics.filtered_data_age,
        diagnostics.resampled_max_timestamp,
        diagnostics.resampled_data_age,
        diagnostics.parquet_to_filtered_delta,
        diagnostics.filtered_to_resampled_delta,
        diagnostics.vault_history_filter_end_at,
        diagnostics.expected_daily_flooring_reason,
        diagnostics.remote_head_error,
    )

    warning_rows = _build_vault_history_warning_rows(
        diagnostics=diagnostics,
        stale_tolerance=stale_tolerance,
    )
    if warning_rows:
        warning_table = tabulate(
            pd.DataFrame(warning_rows),
            headers="keys",
            tablefmt="plain",
            showindex=False,
        )
        logger.warning(
            "Vault history freshness has suspicious values:\n%s",
            warning_table,
        )


def _build_vault_history_warning_rows(
    diagnostics: VaultHistoryDiagnostics,
    stale_tolerance: datetime.timedelta,
) -> list[dict[str, object]]:
    """Build warning rows for suspicious vault-history freshness values."""
    warning_rows: list[dict[str, object]] = []

    if diagnostics.remote_head_error:
        warning_rows.append(
            {
                "field": "remote_head_error",
                "value": diagnostics.remote_head_error,
                "reason": "Remote metadata check failed",
            }
        )

    if diagnostics.local_cache_age is not None and diagnostics.local_cache_age > stale_tolerance:
        warning_rows.append(
            {
                "field": "local_cache_age",
                "value": diagnostics.local_cache_age,
                "reason": "Local cached parquet is older than the warning tolerance",
            }
        )

    if diagnostics.remote_last_modified_age is not None and diagnostics.remote_last_modified_age > stale_tolerance:
        warning_rows.append(
            {
                "field": "remote_last_modified_age",
                "value": diagnostics.remote_last_modified_age,
                "reason": "Remote parquet object itself looks old",
            }
        )

    if diagnostics.parquet_data_age is not None and diagnostics.parquet_data_age > stale_tolerance:
        warning_rows.append(
            {
                "field": "parquet_data_age",
                "value": diagnostics.parquet_data_age,
                "reason": "Source parquet data is older than the warning tolerance",
            }
        )

    if diagnostics.filtered_data_age is not None and diagnostics.filtered_data_age > stale_tolerance:
        warning_rows.append(
            {
                "field": "filtered_data_age",
                "value": diagnostics.filtered_data_age,
                "reason": "Filtered vault history is older than the warning tolerance",
            }
        )

    if (
        diagnostics.parquet_to_filtered_delta is not None
        and diagnostics.parquet_to_filtered_delta > datetime.timedelta(hours=1)
        and diagnostics.parquet_data_age is not None
        and diagnostics.parquet_data_age <= stale_tolerance
        and diagnostics.filtered_data_age is not None
        and diagnostics.filtered_data_age <= stale_tolerance
        and diagnostics.expected_daily_flooring_reason is None
    ):
        warning_rows.append(
            {
                "field": "parquet_to_filtered_delta",
                "value": diagnostics.parquet_to_filtered_delta,
                "reason": "Selected vault history trails the freshest parquet row even though the source parquet is still fresh",
            }
        )

    if (
        diagnostics.resampled_data_age is not None
        and diagnostics.resampled_data_age > stale_tolerance
        and diagnostics.parquet_data_age is not None
        and diagnostics.parquet_data_age <= stale_tolerance
        and diagnostics.expected_daily_flooring_reason is None
    ):
        warning_rows.append(
            {
                "field": "resampled_data_age",
                "value": diagnostics.resampled_data_age,
                "reason": "Looks stale after 1d resampling floor, but the source parquet is fresher",
            }
        )

    if (
        diagnostics.filtered_to_resampled_delta is not None
        and diagnostics.filtered_to_resampled_delta > datetime.timedelta(hours=12)
        and diagnostics.parquet_data_age is not None
        and diagnostics.parquet_data_age <= stale_tolerance
        and diagnostics.expected_daily_flooring_reason is None
    ):
        warning_rows.append(
            {
                "field": "filtered_to_resampled_delta",
                "value": diagnostics.filtered_to_resampled_delta,
                "reason": "Large gap caused by 1d candle flooring to 00:00 UTC",
            }
        )

    return warning_rows


def log_stale_vault_candle_data(
    vault_candle_df: pd.DataFrame | None,
    vault_pairs_df: pd.DataFrame,
    source_vault_price_df: pd.DataFrame | None = None,
    now: datetime.datetime | None = None,
    stale_tolerance: datetime.timedelta = datetime.timedelta(hours=24),
) -> None:
    """Log stale vault candle data as a single warning for live startup diagnostics."""
    if vault_candle_df is None or len(vault_candle_df) == 0:
        return

    reference_now = pd.Timestamp(now or native_datetime_utc_now())
    stale_entries = []
    checked_vault_count = 0
    source_max_timestamp_by_pair_id: dict[int, pd.Timestamp] = {}

    if (
        source_vault_price_df is not None
        and len(source_vault_price_df) > 0
        and "timestamp" in source_vault_price_df.columns
        and "address" in source_vault_price_df.columns
        and "pair_id" in vault_pairs_df.columns
        and "address" in vault_pairs_df.columns
    ):
        source_with_pairs = source_vault_price_df.copy()
        source_with_pairs["address"] = source_with_pairs["address"].astype(str).str.lower()
        source_with_pairs["timestamp"] = pd.to_datetime(source_with_pairs["timestamp"])

        pair_id_column = "pair_id" if "pair_id" in source_with_pairs.columns else None

        if pair_id_column is None:
            merge_columns = ["address"]
            pair_lookup = vault_pairs_df[["pair_id", "address"]].copy()

            if "chain" in source_with_pairs.columns and "chain_id" in vault_pairs_df.columns:
                merge_columns = ["chain", "address"]
                pair_lookup = vault_pairs_df[["pair_id", "chain_id", "address"]].copy()
                pair_lookup = pair_lookup.rename(columns={"chain_id": "chain"})
                pair_lookup["chain"] = pd.to_numeric(pair_lookup["chain"], errors="coerce")
                source_with_pairs["chain"] = pd.to_numeric(source_with_pairs["chain"], errors="coerce")

            pair_lookup["address"] = pair_lookup["address"].astype(str).str.lower()
            source_with_pairs = source_with_pairs.merge(
                pair_lookup,
                on=merge_columns,
                how="inner",
                suffixes=("", "_lookup"),
            )

            if "pair_id" in source_with_pairs.columns:
                pair_id_column = "pair_id"
            elif "pair_id_lookup" in source_with_pairs.columns:
                pair_id_column = "pair_id_lookup"

        if len(source_with_pairs) > 0 and pair_id_column is not None:
            source_max_timestamp_by_pair_id = (
                source_with_pairs.groupby(pair_id_column)["timestamp"].max().to_dict()
            )

    for pair_id in vault_candle_df["pair_id"].unique():
        pair_candles = vault_candle_df[vault_candle_df["pair_id"] == pair_id]
        if len(pair_candles) == 0:
            continue

        checked_vault_count += 1
        last_ts = pair_candles["timestamp"].max()
        last_source_ts = source_max_timestamp_by_pair_id.get(pair_id)
        reference_ts = last_source_ts if last_source_ts is not None else last_ts
        age = reference_now - reference_ts
        if age <= pd.Timedelta(stale_tolerance):
            continue

        vault_row = vault_pairs_df[vault_pairs_df["pair_id"] == pair_id]
        vault_name = vault_row["exchange_name"].iloc[0] if len(vault_row) > 0 else "unknown"
        vault_address = vault_row["address"].iloc[0] if len(vault_row) > 0 else "unknown"
        vault_tvl = None
        if len(vault_row) > 0:
            metadata = vault_row["token_metadata"].iloc[0]
            if metadata is not None:
                vault_tvl = metadata.tvl

        stale_entries.append(
            {
                "vault": vault_name,
                "address": vault_address,
                "tvl": vault_tvl,
                "pair_id": pair_id,
                "last_candle": last_ts,
                "last_source": last_source_ts,
                "age": age,
            }
        )

    if stale_entries:
        up_to_date_vault_count = checked_vault_count - len(stale_entries)
        stale_entries_df = pd.DataFrame(stale_entries)
        stale_entries_table = tabulate(
            stale_entries_df,
            headers="keys",
            tablefmt="plain",
            showindex=False,
        )
        logger.warning(
            "Vault candle data is stale (>24h after source-history check) for %d vault(s); %d vault(s) are up to date:\n%s",
            len(stale_entries),
            up_to_date_vault_count,
            stale_entries_table,
        )


def check_stale_vault_data(
    strategy_universe: TradingStrategyUniverse,
    decision_timestamp: datetime.datetime,
    execution_mode: ExecutionMode,
    tolerance: datetime.timedelta = datetime.timedelta(hours=36),
    min_tvl: USDollarAmount = 5_000,
) -> None:
    """Check that vault candle data is fresh enough for allocation decisions.

    Iterates over all vault pairs in the universe, finds the last
    non-forward-filled candle timestamp for each, and raises
    :class:`StaleVaultData` if any vault's real data is older than
    ``tolerance`` relative to ``decision_timestamp``.

    Vaults whose latest TVL is below ``min_tvl`` are silently skipped,
    because low-TVL vaults often have sporadic data and would otherwise
    cause false-positive staleness failures.

    Call this at the start of ``decide_trades()`` to bail out early
    rather than allocating based on synthetic forward-filled data.

    Silently returns for non-live execution modes (backtesting,
    simulation) where data is always complete by construction.

    :param strategy_universe:
        The trading universe with forward-filled candle data.
    :param decision_timestamp:
        The current decision cycle timestamp (naive UTC).
    :param execution_mode:
        The current execution mode. The check only runs for live
        trading modes; backtesting and simulation are skipped.
    :param tolerance:
        Maximum allowed age of real (non-forward-filled) candle data.
        Defaults to 36 hours, which accommodates the 24h parquet cache
        plus normal pipeline lag.
    :param min_tvl:
        Minimum TVL in USD for a vault to be checked. Vaults below
        this threshold are skipped. Defaults to 5000 USD.
    :raises StaleVaultData:
        If any vault pair has stale data. The exception message lists
        every stale vault with its name, address, last real timestamp,
        and data age.
    """
    if not execution_mode.is_live_trading():
        return

    candles = strategy_universe.data_universe.candles
    if candles is None:
        return

    liquidity = strategy_universe.data_universe.liquidity
    has_forward_filled_column = "forward_filled" in candles.df.columns

    stale_vaults = []
    now = pd.Timestamp(decision_timestamp)

    for pair in strategy_universe.iterate_pairs():
        if not pair.is_vault():
            continue

        # Look up the latest TVL from liquidity data.
        # Skip vaults below the min_tvl threshold — they often have
        # sporadic data and would cause false-positive staleness failures.
        latest_tvl = None
        if liquidity is not None:
            try:
                pair_liquidity = liquidity.get_samples_by_pair(pair.internal_id)
                if len(pair_liquidity) > 0:
                    latest_tvl = float(pair_liquidity["close"].iloc[-1])
                    if latest_tvl < min_tvl:
                        continue
            except KeyError:
                pass

        try:
            pair_candles = candles.get_samples_by_pair(pair.internal_id)
        except KeyError:
            continue

        if len(pair_candles) == 0:
            continue

        # Find the last real (non-forward-filled) timestamp
        if has_forward_filled_column and "forward_filled" in pair_candles.columns:
            real_candles = pair_candles[pair_candles["forward_filled"] != True]
            if len(real_candles) == 0:
                # All data is forward-filled — vault has no real data at all
                stale_vaults.append((pair, None, None, latest_tvl))
                continue
            last_real_ts = real_candles["timestamp"].max()
        else:
            # No forward_filled column — use last timestamp as-is
            last_real_ts = pair_candles["timestamp"].max()

        age = now - pd.Timestamp(last_real_ts)
        if age > pd.Timedelta(tolerance):
            stale_vaults.append((pair, last_real_ts, age, latest_tvl))

    if stale_vaults:
        lines = []
        for pair, last_ts, age, pair_tvl in stale_vaults:
            tvl_str = f", TVL ${pair_tvl:,.0f}" if pair_tvl is not None else ""
            if last_ts is None:
                lines.append(f"  - {pair.get_ticker()} ({pair.pool_address}): no real data (all forward-filled){tvl_str}")
            else:
                lines.append(f"  - {pair.get_ticker()} ({pair.pool_address}): last real data {last_ts}, age {age}{tvl_str}")
        detail = "\n".join(lines)
        raise StaleVaultData(
            f"{len(stale_vaults)} vault(s) have stale candle data (tolerance {tolerance}, min TVL ${min_tvl:,.0f}):\n{detail}"
        )


def _count_trailing_forward_filled(pair_data: pd.DataFrame) -> int:
    """Count contiguous forward-filled rows at the tail of a per-pair DataFrame.

    Only counts the trailing stale stretch, not forward-filled rows
    scattered throughout the series (which may be legitimate gap fills).

    Returns 0 if no ``forward_filled`` column exists or the tail is real data.
    Returns the total row count if every row is forward-filled.
    """
    if "forward_filled" not in pair_data.columns:
        return 0

    ff_col = pair_data["forward_filled"]
    count = 0
    for val in reversed(ff_col.values):
        if val is True or val == True:  # noqa: E712 — deliberate identity check for np.bool_
            count += 1
        else:
            break
    return count


def _get_last_real_timestamp(
    pair_data: pd.DataFrame,
    has_forward_filled_column: bool,
) -> pd.Timestamp | None:
    """Find the last non-forward-filled timestamp in a per-pair DataFrame.

    Returns ``None`` when all rows are forward-filled or the column is absent
    and the DataFrame is empty.
    """
    if has_forward_filled_column and "forward_filled" in pair_data.columns:
        real_rows = pair_data[pair_data["forward_filled"] != True]
        if len(real_rows) == 0:
            return None
        return pd.Timestamp(real_rows["timestamp"].max())
    # No forward_filled column — treat all data as real
    if len(pair_data) == 0:
        return None
    return pd.Timestamp(pair_data["timestamp"].max())


def _resolve_reference_timestamp(
    strategy_universe: TradingStrategyUniverse,
) -> pd.Timestamp | None:
    """Derive the reference timestamp from the universe data.

    Uses the global maximum candle timestamp across all pairs as the
    reference point. This matches the ``forward_fill_until`` value that
    ``create_from_dataset()`` uses when constructing the universe.

    Returns ``None`` when no candle data is available.
    """
    candles = strategy_universe.data_universe.candles
    if candles is None or len(candles.df) == 0:
        return None
    return pd.Timestamp(candles.df["timestamp"].max())


def get_vault_data_freshness(
    strategy_universe: TradingStrategyUniverse,
    reference_timestamp: datetime.datetime | None = None,
) -> pd.DataFrame:
    """Build a diagnostic table of data freshness for all vault pairs.

    For each vault pair in the universe, reports the last real (non-forward-filled)
    timestamp for both candle and TVL data and how old that data is relative to
    ``reference_timestamp``. This allows operators to see at a glance which vaults
    have stale data — whether or not the staleness is masked by framework
    forward-fill.

    Data age is computed as ``reference_timestamp - last_real_timestamp``, so it
    correctly reflects staleness even when the ``forward_filled`` column is absent
    (e.g. when forward-fill was skipped or the universe was built without it).

    Unlike :func:`check_stale_vault_data`, this function:

    - Always runs (no execution-mode or tolerance filtering)
    - Returns a DataFrame instead of raising
    - Reports both candle and TVL freshness separately
    - Includes all vault pairs regardless of TVL

    :param strategy_universe:
        The trading universe with (possibly forward-filled) candle and liquidity data.
    :param reference_timestamp:
        The point in time to measure freshness against. Defaults to the global
        maximum candle timestamp across all pairs (which matches the
        ``forward_fill_until`` value used by ``create_from_dataset()``).
    :return:
        A DataFrame sorted by worst staleness descending (stalest first),
        with ``NaT`` (no real data at all) appearing at the top.
    """
    candles = strategy_universe.data_universe.candles
    liquidity = strategy_universe.data_universe.liquidity

    has_candle_ff_col = candles is not None and "forward_filled" in candles.df.columns
    has_liquidity_ff_col = liquidity is not None and "forward_filled" in liquidity.df.columns

    if reference_timestamp is not None:
        ref_ts = pd.Timestamp(reference_timestamp)
    else:
        ref_ts = _resolve_reference_timestamp(strategy_universe)

    rows = []

    for pair in strategy_universe.iterate_pairs():
        if not pair.is_vault():
            continue

        vault_name = pair.get_vault_name() or pair.get_ticker()

        # -- Candle freshness --
        last_real_candle = None
        latest_candle = None
        candle_data_age = None
        trailing_stale_candles = 0

        if candles is not None:
            try:
                pair_candles = candles.get_samples_by_pair(pair.internal_id)
                if len(pair_candles) > 0:
                    latest_candle = pd.Timestamp(pair_candles["timestamp"].max())
                    last_real_candle = _get_last_real_timestamp(pair_candles, has_candle_ff_col)
                    trailing_stale_candles = _count_trailing_forward_filled(pair_candles)
                    if last_real_candle is not None and ref_ts is not None:
                        candle_data_age = ref_ts - last_real_candle
            except KeyError:
                pass

        # -- TVL freshness --
        last_real_tvl = None
        latest_tvl_ts = None
        tvl_data_age = None
        latest_tvl_usd = None

        if liquidity is not None:
            try:
                pair_liquidity = liquidity.get_samples_by_pair(pair.internal_id)
                if len(pair_liquidity) > 0:
                    latest_tvl_ts = pd.Timestamp(pair_liquidity["timestamp"].max())
                    latest_tvl_usd = float(pair_liquidity["close"].iloc[-1])
                    last_real_tvl = _get_last_real_timestamp(pair_liquidity, has_liquidity_ff_col)
                    if last_real_tvl is not None and ref_ts is not None:
                        tvl_data_age = ref_ts - last_real_tvl
            except KeyError:
                pass

        rows.append({
            "Vault": vault_name,
            "Address": pair.pool_address,
            "Last real candle": last_real_candle,
            "Latest candle": latest_candle,
            "Candle data age": candle_data_age,
            "Trailing stale candles": trailing_stale_candles,
            "Last real TVL": last_real_tvl,
            "Latest TVL timestamp": latest_tvl_ts,
            "TVL data age": tvl_data_age,
            "Latest TVL (USD)": latest_tvl_usd,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Sort by worst staleness across both candle and TVL; NaT sorts to top
    df["_sort_key"] = df[["Candle data age", "TVL data age"]].max(axis=1)
    df = df.sort_values("_sort_key", ascending=False, na_position="first")
    df = df.drop(columns=["_sort_key"])

    diagnostics = getattr(strategy_universe, "vault_history_diagnostics", None)
    if diagnostics is not None:
        df["Vault history cache path"] = str(diagnostics.cache_path) if diagnostics.cache_path is not None else None
        df["Vault history cache mtime"] = diagnostics.local_cache_mtime
        df["Vault history cache age"] = diagnostics.local_cache_age
        df["Vault history remote last modified"] = diagnostics.remote_last_modified
        df["Vault history remote last modified age"] = diagnostics.remote_last_modified_age
        df["Vault history remote ETag"] = diagnostics.remote_etag
        df["Vault history remote Content-Length"] = diagnostics.remote_content_length
        df["Vault history remote HEAD error"] = diagnostics.remote_head_error
        df["Vault history parquet max timestamp"] = diagnostics.parquet_max_timestamp
        df["Vault history parquet data age"] = diagnostics.parquet_data_age
        df["Vault history filtered max timestamp"] = diagnostics.filtered_max_timestamp
        df["Vault history filtered data age"] = diagnostics.filtered_data_age
        df["Vault history resampled max timestamp"] = diagnostics.resampled_max_timestamp
        df["Vault history resampled data age"] = diagnostics.resampled_data_age
        df["Vault history parquet->filtered delta"] = diagnostics.parquet_to_filtered_delta
        df["Vault history filtered->resampled delta"] = diagnostics.filtered_to_resampled_delta
        df["Vault history filter end timestamp"] = diagnostics.vault_history_filter_end_at
        df["Vault history expected daily flooring reason"] = diagnostics.expected_daily_flooring_reason

    df = df.reset_index(drop=True)
    return df
