"""Explain historical HyperCore deposit gates in the standard chart registry.

Hyper-AI v8 saves candidate-entry events in ``state.visualisation.calculations``.
The chart registry calls these helpers for notebook and web tables, including
after a state JSON round trip. Accepted entries mean candidates that passed the
deposit gate, not filled trades. Explicit closures are counted separately from
missing or stale availability data so the table does not mislabel data gaps.

The source-universe summary counts observed closed-state runs, not skipped
decisions. A saved summary supports rendering without downloading the original
universe; when neither is available, those totals are reported as unavailable.
"""

import pandas as pd

from tradingstrategy.alternative_data.vault import HYPERCORE_DEPOSIT_STATE_CUTOFF

from tradeexecutor.state.state import State
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


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
    """Build the chart registry's per-vault explanation of skipped candidates.

    Only vaults with an explicit closure skip appear. Counts describe selection
    attempts across decisions, not unique positions or execution outcomes.

    :param state: Live or deserialised backtest state containing v8 gate events.
    :return: One row per skipped vault, or an empty table with stable columns.
    """
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


def count_hypercore_close_periods(strategy_universe: TradingStrategyUniverse | None) -> tuple[int, int]:
    """Count observed closure runs for v8's saved source-universe summary.

    Consecutive false samples form one period; an open or unknown sample ends
    that run. This measures the supplied history, not continuous wall-clock
    closure duration or the number of rejected strategy entries.

    :param strategy_universe: Universe whose point-in-time vault state is counted.
    :return: Distinct closed vaults and observed closed-state periods.
    """
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
    strategy_universe: TradingStrategyUniverse | None = None,
) -> pd.DataFrame:
    """Build the chart registry's closure summary without inventing missing data.

    Prefer v8's persisted source totals so a saved backtest renders without its
    original universe. Candidate skips alone cannot reveal the number of source
    closure periods and must not be used as a substitute for those totals.

    :param state: State containing gate events and optionally a saved summary.
    :param strategy_universe: Optional original universe when no summary is saved.
    :return: Stable metric/value table; unavailable source counts are labelled.
    """
    events = analyse_hypercore_closed_entry_events(state)
    event_count = int(events["Skipped entries"].sum()) if not events.empty else 0

    # Backtest states may be rendered without the original universe.
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
    elif getattr(strategy_universe, "vault_state", None) is not None:
        closed_vaults, close_periods = count_hypercore_close_periods(strategy_universe)
    else:
        closed_vaults = close_periods = "Unavailable"

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
