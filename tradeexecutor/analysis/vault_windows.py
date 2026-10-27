"""Vault deposit and redemption window diagnostics.

- Summarise the per-cycle deposit and redemption availability a backtest uses
  for vault gating, from the same inputs ``BacktestPricing`` consults:
  synthetic :py:class:`tradeexecutor.backtest.vault_windows.VaultWindowSchedule`
  overrides first, then the historical ``vault_state`` frame

- Surface data-quality issues: vaults with no usable state samples,
  long closed or unknown runs, and stale sample ages

Use :py:func:`calculate_vault_window_diagnostics` for per-vault summaries and
:py:func:`calculate_vault_window_counts` for trading-universe histories.
"""

import datetime
from collections.abc import Collection
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tradeexecutor.backtest.backtest_pricing import BacktestPricing
from tradeexecutor.strategy.vault_closure import (
    VaultClosureOutcome, get_vault_closure, is_vault_closed, can_interact_with_vault,
)

if TYPE_CHECKING:
    from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


#: Cycle status: the window is known open.
STATUS_OPEN = "open"

#: Cycle status: the shared rule identifies an operation closure.
STATUS_CLOSED = "closed"

#: Cycle status: evidence is missing/unsuitable and the chosen policy allows the trade.
STATUS_UNKNOWN_ALLOWED = "unknown_allowed"
STATUS_UNKNOWN_BLOCKED = "unknown_blocked"
STATUS_POLICY_BLOCKED = "policy_blocked"


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
            f"Vault window diagnostics over {len(self.cycle_index)} decision cycles "
            f"({self.cycle_index[0].date()} to {self.cycle_index[-1].date()}); "
            f"availability tolerance {self.availability_tolerance}. "
            f"Unknown is reported separately from admission policy."
        )


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


def calculate_vault_window_counts(
    strategy_universe: "TradingStrategyUniverse",
    cycle_index: pd.DatetimeIndex,
    pair_ids: Collection[int] | None = None,
    availability_tolerance: pd.Timedelta = pd.Timedelta(days=2),
) -> pd.DataFrame:
    """Count historical vault closures separately for deposits and redemptions.

    Counts cover vaults within their candle-history bounds, excluding non-vault
    pairs. Pass investable ``pair_ids`` to exclude cash-management vaults too.
    Price coverage is not proof that the strategy's admission criteria pass.

    Uses the pricing model's as-of state lookup, including its permission and
    capacity evidence clocks and staleness tolerance. There is no backfill from
    future state. The unified vault-closure rule distinguishes known closures, strategy-policy
    blocks and unknown evidence. The total blocked counts follow the exact
    interaction policy used by the backtest; unknown need not mean allowed. Synthetic window overrides take precedence and are labelled in the
    returned column names. These are window closures, not permanent shutdowns.

    :param cycle_index: Naive UTC decision timestamps to evaluate.
    :param pair_ids: Optional vault scope, e.g. the universe minus queue venues.
    :param availability_tolerance: Maximum historical state age, matching the run.
    :return: Integer counts indexed by the supplied timestamps, including zeros.
    """
    if cycle_index.tz is not None:
        raise ValueError("Vault window counts require naive UTC timestamps")
    if availability_tolerance < pd.Timedelta(0):
        raise ValueError("Availability tolerance must not be negative")

    pairs = [
        pair for pair in strategy_universe.iterate_pairs()
        if pair.kind.is_vault() and (pair_ids is None or pair.internal_id in pair_ids)
    ]
    overrides = strategy_universe.vault_window_overrides or {}
    suffix = " (includes synthetic windows)" if any(p.internal_id in overrides for p in pairs) else ""
    columns = ["Vaults with price history", "Deposits closed" + suffix,
               "Redemptions closed" + suffix, "Deposit state unknown", "Redemption state unknown",
               "Deposit policy blocks", "Redemption policy blocks", "Deposits blocked by rules", "Redemptions blocked by rules"]
    values = np.zeros((len(cycle_index), len(columns)), dtype="int64")
    counts = pd.DataFrame(values, index=cycle_index, columns=columns, copy=False)
    if not pairs or cycle_index.empty:
        return counts
    pricing = BacktestPricing(strategy_universe, routing_model=None, data_delay_tolerance=availability_tolerance)
    for pair in pairs:
        candles = strategy_universe.data_universe.candles.get_candles_by_pair(pair.internal_id)
        if candles is None or candles.empty:
            continue
        timestamps = candles.index.get_level_values("timestamp")
        covered_rows = np.flatnonzero((cycle_index >= timestamps.min()) & (cycle_index <= timestamps.max()))
        values[covered_rows, 0] += 1
        for row in covered_rows:
            ts = cycle_index[row]
            for operation, closed, unknown, policy, blocked in (
                ("deposit", 1, 3, 5, 7),
                ("redemption", 2, 4, 6, 8),
            ):
                assessment = get_vault_closure(pair, timestamp=ts, pricing_model=pricing, operation=operation)
                # Only presentation belongs here. The shared module owns all
                # protocol exceptions, policy blocks and unknown admission rules.
                values[row, closed] += int(is_vault_closed(assessment))
                values[row, unknown] += int(assessment.status is VaultClosureOutcome.unknown)
                values[row, policy] += int(assessment.status is VaultClosureOutcome.policy_blocked)
                values[row, blocked] += int(not can_interact_with_vault(assessment))

    return counts


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

    - Otherwise :func:`tradeexecutor.strategy.vault_closure.get_vault_closure`
      interprets fresh historical evidence, including protocol exceptions and
      strategy-policy blocks. Unknown admission is reported separately.

    - Missing state is unknown; the shared rule decides whether interaction is allowed
      (source ``no_state``); post-cutoff HyperCore unknowns are refused.

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
        the final timestamp strictly before it on the requested cycle grid.

    :param freq:
        Decision cycle frequency, e.g. ``TimeBucket.d1.to_frequency()``.

    :param availability_tolerance:
        Maximum age of a vault-state sample still considered fresh for a cycle.

    :return:
        The three diagnostics tables and the evaluated cycle index. Historical
        column names ending in "days" (and "Days") count decision cycles; they
        represent days only for daily input. Source labels describe the input
        table/override, not observed permission. Empty vault scopes return empty
        tables; an empty decision calendar raises ``ValueError``.
    """
    if start >= end:
        raise ValueError("Vault diagnostics require at least one decision cycle")
    cycle_index = pd.date_range(start=start, end=end, freq=freq, inclusive="left")
    if cycle_index.empty:
        raise ValueError("Vault diagnostics require at least one decision cycle")
    if cycle_index.tz is not None or availability_tolerance < pd.Timedelta(0):
        raise ValueError("Use naive UTC timestamps and a non-negative availability tolerance")
    vault_state_df = strategy_universe.vault_state
    vault_window_overrides = getattr(strategy_universe, "vault_window_overrides", {}) or {}
    vault_pairs = [pair for pair in strategy_universe.iterate_pairs() if pair.kind.is_vault()]

    vault_state_by_pair = {}
    if vault_state_df is not None and len(vault_state_df) > 0:
        for pair_id, group in vault_state_df.groupby("pair_id", sort=False):
            group = group.sort_values("timestamp").reset_index(drop=True)
            vault_state_by_pair[int(pair_id)] = group

    pricing = BacktestPricing(strategy_universe, routing_model=None, data_delay_tolerance=availability_tolerance)
    summary_rows = []
    gap_rows = []
    status_rows = []

    for pair in vault_pairs:
        pair_id = pair.internal_id
        override = vault_window_overrides.get(pair_id)
        state_group = vault_state_by_pair.get(pair_id)
        source = "override" if override is not None else ("vault_state" if state_group is not None else "no_state")

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
            # Evidence lookup and interpretation are shared with actual decisions;
            # do not reconstruct flags, clocks, caps or protocol rules here.
            sample = pricing._lookup_vault_state(ts, pair)
            # Retain sample-age diagnostics only; this does not classify a vault.
            sample_age_days = np.nan
            if sample is not None and state_group is not None and override is None:
                idx = state_group["timestamp"].searchsorted(ts, side="right") - 1
                if idx >= 0:
                    sample_age_days = (ts - pd.Timestamp(state_group.iloc[int(idx)]["timestamp"])) / np.timedelta64(1, "D")
            state_sample_ages.append(sample_age_days)
            for operation, statuses, reasons in (
                ("deposit", deposit_statuses, deposit_reasons),
                ("redemption", redemption_statuses, redemption_reasons),
            ):
                assessment = get_vault_closure(pair, timestamp=ts, pricing_model=pricing, operation=operation)
                if is_vault_closed(assessment):
                    status = STATUS_CLOSED
                elif assessment.status is VaultClosureOutcome.policy_blocked:
                    status = STATUS_POLICY_BLOCKED
                elif assessment.status is VaultClosureOutcome.unknown:
                    status = STATUS_UNKNOWN_ALLOWED if can_interact_with_vault(assessment) else STATUS_UNKNOWN_BLOCKED
                else:
                    status = STATUS_OPEN
                statuses.append(status)
                if assessment.status is VaultClosureOutcome.closed and assessment.message:
                    reasons.append(assessment.message)

        deposit_counts = pd.Series(deposit_statuses).value_counts().to_dict()
        redemption_counts = pd.Series(redemption_statuses).value_counts().to_dict()

        deposit_closed_run = _run_summary(deposit_statuses, cycle_index, {STATUS_CLOSED})
        redemption_closed_run = _run_summary(redemption_statuses, cycle_index, {STATUS_CLOSED})
        deposit_unknown_run = _run_summary(deposit_statuses, cycle_index, {STATUS_UNKNOWN_ALLOWED, STATUS_UNKNOWN_BLOCKED})
        redemption_unknown_run = _run_summary(redemption_statuses, cycle_index, {STATUS_UNKNOWN_ALLOWED, STATUS_UNKNOWN_BLOCKED})

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
            "Deposit unknown days": deposit_counts.get(STATUS_UNKNOWN_ALLOWED, 0) + deposit_counts.get(STATUS_UNKNOWN_BLOCKED, 0),
            "Deposit policy-blocked days": deposit_counts.get(STATUS_POLICY_BLOCKED, 0),
            "Deposit open %": deposit_counts.get(STATUS_OPEN, 0) / len(cycle_index),
            "Longest deposit closed gap": deposit_closed_run[0],
            "Longest deposit unknown gap": deposit_unknown_run[0],
            "First deposit open": _format_ts(deposit_open_dates[0] if deposit_open_dates else None),
            "Last deposit open": _format_ts(deposit_open_dates[-1] if deposit_open_dates else None),
            "Redemption open days": redemption_counts.get(STATUS_OPEN, 0),
            "Redemption closed days": redemption_counts.get(STATUS_CLOSED, 0),
            "Redemption unknown days": redemption_counts.get(STATUS_UNKNOWN_ALLOWED, 0) + redemption_counts.get(STATUS_UNKNOWN_BLOCKED, 0),
            "Redemption policy-blocked days": redemption_counts.get(STATUS_POLICY_BLOCKED, 0),
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
                    "Status": "unknown",
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

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["Source", "Longest deposit closed gap", "Longest deposit unknown gap", "Deposit closed days"],
            ascending=[True, False, False, False],
        )
    gap_df = pd.DataFrame(gap_rows, columns=["Kind", "Status", "Pair id", "Ticker", "Protocol", "Chain", "Source", "Days", "Start", "End"]).sort_values(["Days", "Status", "Kind"], ascending=[False, True, True])
    status_counts_df = (
        pd.DataFrame(status_rows, columns=["Kind", "Status", "Days", "Pair id", "Ticker", "Source"])
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
