"""Vault deposit and redemption window diagnostics.

- Summarise the per-cycle deposit and redemption availability a backtest uses
  for vault gating, from the same inputs ``BacktestPricing`` consults:
  synthetic :py:class:`tradeexecutor.backtest.vault_windows.VaultWindowSchedule`
  overrides first, then the historical ``vault_state`` frame

- Surface data-quality issues: vaults with no usable state samples,
  long closed or unknown runs, and stale sample ages

See :py:func:`calculate_vault_window_diagnostics` for the entry point.
"""

import datetime
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


#: Cycle status: the window is known open.
STATUS_OPEN = "open"

#: Cycle status: the window is known closed (flag false or zero cap).
STATUS_CLOSED = "closed"

#: Cycle status: no fresh vault-state sample; BacktestPricing allows the trade.
STATUS_UNKNOWN_ALLOWED = "unknown_allowed"


@dataclass(slots=True, frozen=True)
class VaultWindowDiagnostics:
    """Per-cycle vault deposit/redemption availability summary.

    Output of :py:func:`calculate_vault_window_diagnostics`.
    """

    #: One row per vault pair: open/closed/unknown day counts, longest runs,
    #: first/last open dates, vault-state sample coverage and closed reasons.
    summary_df: pd.DataFrame

    #: Longest closed and unknown runs per vault and window kind, longest first.
    gap_df: pd.DataFrame

    #: Status day totals grouped by window kind, status and data source.
    status_counts_df: pd.DataFrame

    #: Cycle timestamps the statuses were evaluated on.
    cycle_index: pd.DatetimeIndex

    #: Maximum vault-state sample age accepted for a cycle.
    availability_tolerance: pd.Timedelta

    def get_description(self) -> str:
        """One-line description of the diagnostics run for notebook output."""
        return (
            f"Vault window diagnostics over {len(self.cycle_index)} daily cycles "
            f"({self.cycle_index[0].date()} to {self.cycle_index[-1].date()}); "
            f"availability tolerance {self.availability_tolerance}. "
            f"Unknown means allowed by BacktestPricing."
        )


def _is_zero_cap(value) -> bool:
    if value is None or pd.isna(value):
        return False
    return float(value) == 0.0


def _is_false(value) -> bool:
    if value is None or pd.isna(value):
        return False
    return bool(value) is False


def _clean_reason(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    value = str(value).strip()
    return value or None


def _run_summary(statuses: list[str], timestamps: pd.DatetimeIndex, target_statuses: set[str]) -> tuple[int, pd.Timestamp | None, pd.Timestamp | None]:
    """Find the longest consecutive run of the target statuses."""
    best_length = 0
    best_start = None
    best_end = None
    current_length = 0
    current_start = None
    current_end = None

    for ts, status in zip(timestamps, statuses):
        if status in target_statuses:
            if current_length == 0:
                current_start = ts
            current_length += 1
            current_end = ts
        else:
            if current_length > best_length:
                best_length = current_length
                best_start = current_start
                best_end = current_end
            current_length = 0
            current_start = None
            current_end = None

    if current_length > best_length:
        best_length = current_length
        best_start = current_start
        best_end = current_end

    return best_length, best_start, best_end


def _format_ts(ts) -> str:
    return "" if ts is None or pd.isna(ts) else pd.Timestamp(ts).strftime("%Y-%m-%d")


def calculate_vault_window_diagnostics(
    strategy_universe: "TradingStrategyUniverse",
    start: datetime.datetime,
    end: datetime.datetime,
    freq: pd.Timedelta | pd.offsets.BaseOffset,
    availability_tolerance: pd.Timedelta = pd.Timedelta(2, unit="D"),
) -> VaultWindowDiagnostics:
    """Summarise per-cycle vault deposit and redemption availability.

    Replays the window gating inputs the backtest uses for every vault pair
    and decision cycle:

    - A synthetic ``vault_window_overrides`` schedule wins when present
      (source ``override``)

    - Otherwise the latest ``vault_state`` sample within
      ``availability_tolerance`` decides: an explicit false flag or zero
      deposit/redeem cap is ``closed``, all-NA fields are ``unknown_allowed``,
      anything else is ``open`` (source ``vault_state``)

    - A vault with no state data at all is always ``unknown_allowed``
      (source ``no_state_allowed``), which ``BacktestPricing`` permits

    Example:

    .. code-block:: python

        from tradeexecutor.analysis.vault_windows import calculate_vault_window_diagnostics

        diagnostics = calculate_vault_window_diagnostics(
            strategy_universe,
            start=Parameters.backtest_start,
            end=Parameters.backtest_end,
            freq=Parameters.candle_time_bucket.to_frequency(),
        )
        print(diagnostics.get_description())
        display(diagnostics.status_counts_df)
        display(diagnostics.summary_df)
        display(diagnostics.gap_df.head(30))

    :param strategy_universe:
        Universe with vault pairs, ``vault_state`` and optional
        ``vault_window_overrides``.

    :param start:
        First decision cycle timestamp, usually the backtest start.

    :param end:
        Backtest end timestamp, exclusive: the last evaluated cycle is
        one ``freq`` before it.

    :param freq:
        Decision cycle frequency, e.g. ``TimeBucket.d1.to_frequency()``.

    :param availability_tolerance:
        Maximum age of a vault-state sample still considered fresh for a cycle.

    :return:
        The three diagnostics tables and the evaluated cycle index.
    """
    cycle_index = pd.date_range(start=start, end=end - freq, freq=freq)
    vault_state_df = strategy_universe.vault_state
    vault_window_overrides = getattr(strategy_universe, "vault_window_overrides", {}) or {}
    vault_pairs = [pair for pair in strategy_universe.iterate_pairs() if pair.kind.is_vault()]

    vault_state_by_pair = {}
    if vault_state_df is not None and len(vault_state_df) > 0:
        for pair_id, group in vault_state_df.groupby("pair_id", sort=False):
            group = group.sort_values("timestamp").reset_index(drop=True)
            vault_state_by_pair[int(pair_id)] = group

    summary_rows = []
    gap_rows = []
    status_rows = []

    for pair in vault_pairs:
        pair_id = pair.internal_id
        override = vault_window_overrides.get(pair_id)
        state_group = vault_state_by_pair.get(pair_id)
        source = "override" if override is not None else ("vault_state" if state_group is not None else "no_state_allowed")

        deposit_statuses = []
        redemption_statuses = []
        state_sample_ages = []
        deposit_reasons = []
        redemption_reasons = []

        if state_group is not None:
            state_timestamps = pd.to_datetime(state_group["timestamp"])
            state_sample_count = len(state_group)
            state_first_sample = state_timestamps.min()
            state_last_sample = state_timestamps.max()
            state_gap_days = state_timestamps.sort_values().diff().dt.total_seconds().div(86400).max()
        else:
            state_sample_count = 0
            state_first_sample = pd.NaT
            state_last_sample = pd.NaT
            state_gap_days = np.nan

        for ts in cycle_index:
            if override is not None:
                deposit_statuses.append(STATUS_OPEN if override.is_deposit_open(ts.to_pydatetime()) else STATUS_CLOSED)
                redemption_statuses.append(STATUS_OPEN if override.is_redemption_open(ts.to_pydatetime()) else STATUS_CLOSED)
                state_sample_ages.append(np.nan)
                continue

            sample = None
            sample_age_days = np.nan
            if state_group is not None:
                idx = state_group["timestamp"].searchsorted(ts, side="right") - 1
                if idx >= 0:
                    candidate = state_group.iloc[int(idx)]
                    sample_age = ts - pd.Timestamp(candidate["timestamp"])
                    if sample_age <= availability_tolerance:
                        sample = candidate
                        sample_age_days = sample_age / np.timedelta64(1, "D")
            state_sample_ages.append(sample_age_days)

            if sample is None:
                deposit_statuses.append(STATUS_UNKNOWN_ALLOWED)
                redemption_statuses.append(STATUS_UNKNOWN_ALLOWED)
                continue

            deposits_open = sample.get("deposits_open")
            max_deposit = sample.get("max_deposit")
            deposit_reason = _clean_reason(sample.get("deposit_closed_reason"))
            if _is_false(deposits_open) or _is_zero_cap(max_deposit):
                deposit_statuses.append(STATUS_CLOSED)
                if deposit_reason:
                    deposit_reasons.append(deposit_reason)
            elif pd.isna(deposits_open) and pd.isna(max_deposit):
                deposit_statuses.append(STATUS_UNKNOWN_ALLOWED)
            else:
                deposit_statuses.append(STATUS_OPEN)

            redemption_open = sample.get("redemption_open")
            max_redeem = sample.get("max_redeem")
            redemption_reason = _clean_reason(sample.get("redemption_closed_reason"))
            if _is_false(redemption_open) or _is_zero_cap(max_redeem):
                redemption_statuses.append(STATUS_CLOSED)
                if redemption_reason:
                    redemption_reasons.append(redemption_reason)
            elif pd.isna(redemption_open) and pd.isna(max_redeem):
                redemption_statuses.append(STATUS_UNKNOWN_ALLOWED)
            else:
                redemption_statuses.append(STATUS_OPEN)

        deposit_counts = pd.Series(deposit_statuses).value_counts().to_dict()
        redemption_counts = pd.Series(redemption_statuses).value_counts().to_dict()

        deposit_closed_run = _run_summary(deposit_statuses, cycle_index, {STATUS_CLOSED})
        redemption_closed_run = _run_summary(redemption_statuses, cycle_index, {STATUS_CLOSED})
        deposit_unknown_run = _run_summary(deposit_statuses, cycle_index, {STATUS_UNKNOWN_ALLOWED})
        redemption_unknown_run = _run_summary(redemption_statuses, cycle_index, {STATUS_UNKNOWN_ALLOWED})

        deposit_open_dates = [ts for ts, status in zip(cycle_index, deposit_statuses) if status == STATUS_OPEN]
        redemption_open_dates = [ts for ts, status in zip(cycle_index, redemption_statuses) if status == STATUS_OPEN]

        summary_rows.append({
            "Pair id": pair_id,
            "Vault": pair.get_vault_name() or pair.get_ticker(),
            "Ticker": pair.get_ticker(),
            "Protocol": pair.get_vault_protocol(),
            "Chain": pair.chain_id,
            "Source": source,
            "Cycles": len(cycle_index),
            "Deposit open days": deposit_counts.get(STATUS_OPEN, 0),
            "Deposit closed days": deposit_counts.get(STATUS_CLOSED, 0),
            "Deposit unknown days": deposit_counts.get(STATUS_UNKNOWN_ALLOWED, 0),
            "Deposit open %": deposit_counts.get(STATUS_OPEN, 0) / len(cycle_index),
            "Longest deposit closed gap": deposit_closed_run[0],
            "Longest deposit unknown gap": deposit_unknown_run[0],
            "First deposit open": _format_ts(deposit_open_dates[0] if deposit_open_dates else None),
            "Last deposit open": _format_ts(deposit_open_dates[-1] if deposit_open_dates else None),
            "Redemption open days": redemption_counts.get(STATUS_OPEN, 0),
            "Redemption closed days": redemption_counts.get(STATUS_CLOSED, 0),
            "Redemption unknown days": redemption_counts.get(STATUS_UNKNOWN_ALLOWED, 0),
            "Redemption open %": redemption_counts.get(STATUS_OPEN, 0) / len(cycle_index),
            "Longest redemption closed gap": redemption_closed_run[0],
            "Longest redemption unknown gap": redemption_unknown_run[0],
            "First redemption open": _format_ts(redemption_open_dates[0] if redemption_open_dates else None),
            "Last redemption open": _format_ts(redemption_open_dates[-1] if redemption_open_dates else None),
            "Vault-state samples": state_sample_count,
            "Vault-state first": _format_ts(state_first_sample),
            "Vault-state last": _format_ts(state_last_sample),
            "Longest vault-state sample gap days": state_gap_days,
            "Max sample age days used": np.nanmax(state_sample_ages) if not np.all(pd.isna(state_sample_ages)) else np.nan,
            "Deposit closed reasons": "; ".join(sorted(set(deposit_reasons)))[:160],
            "Redemption closed reasons": "; ".join(sorted(set(redemption_reasons)))[:160],
        })

        for kind, statuses, closed_run, unknown_run in [
            ("deposit", deposit_statuses, deposit_closed_run, deposit_unknown_run),
            ("redemption", redemption_statuses, redemption_closed_run, redemption_unknown_run),
        ]:
            if closed_run[0] > 0:
                gap_rows.append({
                    "Kind": kind,
                    "Status": STATUS_CLOSED,
                    "Pair id": pair_id,
                    "Ticker": pair.get_ticker(),
                    "Protocol": pair.get_vault_protocol(),
                    "Chain": pair.chain_id,
                    "Source": source,
                    "Days": closed_run[0],
                    "Start": _format_ts(closed_run[1]),
                    "End": _format_ts(closed_run[2]),
                })
            if unknown_run[0] > 0:
                gap_rows.append({
                    "Kind": kind,
                    "Status": STATUS_UNKNOWN_ALLOWED,
                    "Pair id": pair_id,
                    "Ticker": pair.get_ticker(),
                    "Protocol": pair.get_vault_protocol(),
                    "Chain": pair.chain_id,
                    "Source": source,
                    "Days": unknown_run[0],
                    "Start": _format_ts(unknown_run[1]),
                    "End": _format_ts(unknown_run[2]),
                })

        status_rows.extend(
            {"Kind": "deposit", "Status": status, "Days": days, "Pair id": pair_id, "Ticker": pair.get_ticker(), "Source": source}
            for status, days in deposit_counts.items()
        )
        status_rows.extend(
            {"Kind": "redemption", "Status": status, "Days": days, "Pair id": pair_id, "Ticker": pair.get_ticker(), "Source": source}
            for status, days in redemption_counts.items()
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["Source", "Longest deposit closed gap", "Longest deposit unknown gap", "Deposit closed days"],
        ascending=[True, False, False, False],
    )
    gap_df = pd.DataFrame(gap_rows).sort_values(["Days", "Status", "Kind"], ascending=[False, True, True])
    status_counts_df = (
        pd.DataFrame(status_rows)
        .groupby(["Kind", "Status", "Source"], as_index=False)
        .agg(Vaults=("Pair id", "nunique"), Days=("Days", "sum"))
        .sort_values(["Kind", "Source", "Status"])
    )

    return VaultWindowDiagnostics(
        summary_df=summary_df,
        gap_df=gap_df,
        status_counts_df=status_counts_df,
        cycle_index=cycle_index,
        availability_tolerance=availability_tolerance,
    )
