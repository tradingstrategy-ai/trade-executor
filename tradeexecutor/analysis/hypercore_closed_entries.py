"""Analyse HyperCore entries skipped by historical deposit closures."""

from __future__ import annotations

import pandas as pd

from tradingstrategy.alternative_data.vault import HYPERCORE_DEPOSIT_STATE_CUTOFF

from tradeexecutor.state.state import State


HYPERCORE_CLOSED_ENTRY_COLUMNS = [
    "Vault name",
    "Skipped entries",
    "Accepted entries",
    "First skip",
    "Last skipped",
]

HYPERCORE_CLOSED_ENTRY_SUMMARY_COLUMNS = ["Metric", "Value"]


def _read_attr_or_key(value: object, key: str, default=None):
    """Read a field from either a mapping or a dataclass-like object."""
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _normalise_timestamp(value: object) -> pd.Timestamp | None:
    """Normalise persisted cycle timestamps."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return pd.Timestamp(value, unit="s")
    return pd.Timestamp(value)


def _iter_entry_events(state: State):
    """Yield v8 entry-gate events persisted in cycle calculations."""
    if state is None or state.visualisation is None:
        return
    for cycle_timestamp, calculations in state.visualisation.calculations.items():
        for event in calculations.get("hypercore_closed_entry_events", []):
            timestamp = _normalise_timestamp(
                _read_attr_or_key(event, "timestamp") or cycle_timestamp
            )
            yield timestamp, event


def _event_status(event: object) -> str:
    """Return the compact event status."""
    return str(_read_attr_or_key(event, "status", "")).lower()


def _is_explicit_closed_event(event: object) -> bool:
    """Check that an event represents an archived explicit deposit closure."""
    reason = _read_attr_or_key(event, "reason_code")
    return _event_status(event) == "skipped" and reason == "vault_deposits_closed"


def analyse_hypercore_closed_entry_events(state: State) -> pd.DataFrame:
    """Build the vault table for entries skipped by explicit historical closures."""
    totals: dict[str, dict[str, object]] = {}

    for timestamp, event in _iter_entry_events(state):
        address = str(_read_attr_or_key(event, "vault_address", ""))
        name = (
            _read_attr_or_key(event, "vault_name")
            or _read_attr_or_key(event, "pair_ticker")
            or address
            or "Unknown vault"
        )
        key = address or str(name)
        row = totals.setdefault(
            key,
            {
                "Vault name": str(name),
                "Skipped entries": 0,
                "Accepted entries": 0,
                "First skip": None,
                "Last skipped": None,
            },
        )

        if _event_status(event) == "accepted":
            row["Accepted entries"] += 1
        elif _is_explicit_closed_event(event):
            row["Skipped entries"] += 1
            if timestamp is not None:
                first_skip = row["First skip"]
                last_skipped = row["Last skipped"]
                row["First skip"] = timestamp if first_skip is None else min(first_skip, timestamp)
                row["Last skipped"] = timestamp if last_skipped is None else max(last_skipped, timestamp)

    rows = [row for row in totals.values() if row["Skipped entries"]]
    if not rows:
        return pd.DataFrame(columns=HYPERCORE_CLOSED_ENTRY_COLUMNS)

    return pd.DataFrame(rows, columns=HYPERCORE_CLOSED_ENTRY_COLUMNS).sort_values(
        ["Skipped entries", "Vault name"],
        ascending=[False, True],
    ).reset_index(drop=True)


def _is_closed_value(value: object) -> bool:
    """Normalise a historical deposits-open cell to an explicit closed flag."""
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"false", "0", "closed"}
    return value is False or value == 0


def count_hypercore_close_periods(strategy_universe) -> tuple[int, int]:
    """Count closed HyperCore vaults and distinct false-state periods after the cutoff."""
    state = getattr(strategy_universe, "vault_state", None)
    if state is None or state.empty or "deposits_open" not in state.columns:
        return 0, 0

    cutoff = pd.Timestamp(HYPERCORE_DEPOSIT_STATE_CUTOFF)
    closed_vaults: set[str] = set()
    periods = 0

    for pair_id, group in state.groupby("pair_id", sort=False):
        try:
            pair = strategy_universe.get_pair_by_id(int(pair_id))
        except (KeyError, TypeError, ValueError):
            continue
        if pair is None or not pair.is_hyperliquid_vault():
            continue

        group = group.sort_values("timestamp")
        was_closed = False
        address = str(pair.pool_address)
        for _, row in group.iterrows():
            timestamp = pd.Timestamp(row["timestamp"])
            if timestamp < cutoff:
                continue
            is_closed = _is_closed_value(row["deposits_open"])
            if is_closed:
                closed_vaults.add(address)
                if not was_closed:
                    periods += 1
            was_closed = is_closed

    return len(closed_vaults), periods


def analyse_hypercore_closed_entry_summary(
    state: State,
    strategy_universe=None,
) -> pd.DataFrame:
    """Build the summary table for HyperCore closed-entry diagnostics."""
    events = analyse_hypercore_closed_entry_events(state)
    event_count = int(events["Skipped entries"].sum()) if not events.empty else 0
    fallback_vaults = len(events)
    fallback_periods = fallback_vaults

    # Backtest states may be rendered without the original StrategyInputIndicators object.
    # v8 persists these source-frame totals with each cycle so the summary remains accurate
    # after serialisation.
    persisted_summary = None
    if state is not None and state.visualisation is not None:
        for calculations in reversed(list(state.visualisation.calculations.values())):
            persisted_summary = calculations.get("hypercore_closed_entry_summary")
            if persisted_summary:
                break

    if persisted_summary:
        closed_vaults = int(persisted_summary.get("total_closed_vaults", 0))
        close_periods = int(persisted_summary.get("total_close_periods", 0))
    else:
        closed_vaults, close_periods = count_hypercore_close_periods(strategy_universe)
    if closed_vaults == 0 and fallback_vaults:
        closed_vaults = fallback_vaults
    if close_periods == 0 and fallback_periods:
        close_periods = fallback_periods

    return pd.DataFrame(
        [
            {
                "Metric": "Proper closed data starting date",
                "Value": HYPERCORE_DEPOSIT_STATE_CUTOFF.date().isoformat(),
            },
            {"Metric": "Total closed vaults", "Value": closed_vaults},
            {"Metric": "Total close periods", "Value": close_periods},
            {"Metric": "Total missed entries", "Value": event_count},
        ],
        columns=HYPERCORE_CLOSED_ENTRY_SUMMARY_COLUMNS,
    )
