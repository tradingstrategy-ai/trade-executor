"""Analyse missed vault deposit and redemption opportunities."""

import pandas as pd
from pandas.io.formats.style import Styler

from tradeexecutor.state.state import State


MISSED_VAULT_EVENT_COLUMNS = [
    "Vault name",
    "Missed deposit count",
    "Missed deposit US dollar",
    "Missed redemption count",
    "Missed redemption US dollar",
]


MISSED_VAULT_EVENT_TIMELINE_COLUMNS = [
    "Timestamp",
    "Event type",
    "Vault name",
    "Missed event count",
    "Missed US dollar",
]


def _read_attr_or_key(value: object, key: str, default=None):
    """Read a field from either a dataclass-like object or a plain dict."""
    if value is None:
        return default

    if isinstance(value, dict):
        return value.get(key, default)

    return getattr(value, key, default)


def _normalise_event_type(value: object) -> str | None:
    """Normalise missed event type values."""
    if value is None:
        return None
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        value = enum_value
    return str(value).lower()


def _normalise_timestamp(value: object) -> pd.Timestamp | None:
    """Normalise persisted calculation timestamps for timeline charts."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return pd.Timestamp(value, unit="s")
    return pd.Timestamp(value)


def _iter_persisted_missed_vault_events(state: State):
    """Yield missed vault event rows persisted in visualisation calculations."""
    for calculations in state.visualisation.calculations.values():
        events = calculations.get("missed_vault_events", [])
        for event in events:
            yield event


def _iter_persisted_missed_vault_events_with_timestamp(state: State):
    """Yield missed vault event rows and the cycle timestamp that stored them."""
    for timestamp, calculations in state.visualisation.calculations.items():
        events = calculations.get("missed_vault_events", [])
        for event in events:
            event_timestamp = (
                _read_attr_or_key(event, "timestamp")
                or _read_attr_or_key(event, "cycle")
                or timestamp
            )
            yield _normalise_timestamp(event_timestamp), event


def _iter_latest_missed_vault_events(state: State):
    """Yield missed vault event rows from the latest in-memory alpha model."""
    alpha_model = state.visualisation.discardable_data.get("alpha_model")
    if alpha_model is None:
        return

    get_missed_vault_events = getattr(alpha_model, "get_missed_vault_events", None)
    if get_missed_vault_events is not None:
        yield from get_missed_vault_events()


def _get_vault_name(event: object) -> str:
    """Get the best available vault label for a missed event."""
    return (
        _read_attr_or_key(event, "vault_name")
        or _read_attr_or_key(event, "pair_ticker")
        or _read_attr_or_key(event, "vault_address")
        or "Unknown vault"
    )


def analyse_missed_vault_deposit_redemption_events(
    state: State,
) -> pd.DataFrame:
    """Create a table of missed vault deposit and redemption events."""
    totals: dict[str, dict[str, float | int]] = {}

    events = list(_iter_persisted_missed_vault_events(state))
    if not events:
        events = list(_iter_latest_missed_vault_events(state))

    for event in events:
        event_type = _normalise_event_type(_read_attr_or_key(event, "event_type"))
        if event_type not in {"deposit", "redemption"}:
            continue

        vault_name = _get_vault_name(event)
        missed_usd = float(_read_attr_or_key(event, "missed_usd", 0.0) or 0.0)
        row = totals.setdefault(
            vault_name,
            {
                "Missed deposit count": 0,
                "Missed deposit US dollar": 0.0,
                "Missed redemption count": 0,
                "Missed redemption US dollar": 0.0,
            },
        )

        if event_type == "deposit":
            row["Missed deposit count"] += 1
            row["Missed deposit US dollar"] += missed_usd
        elif event_type == "redemption":
            row["Missed redemption count"] += 1
            row["Missed redemption US dollar"] += missed_usd

    if not totals:
        return pd.DataFrame(columns=MISSED_VAULT_EVENT_COLUMNS)

    df = pd.DataFrame(
        [
            {"Vault name": vault_name, **row}
            for vault_name, row in totals.items()
        ],
        columns=MISSED_VAULT_EVENT_COLUMNS,
    )
    return df.sort_values(
        [
            "Missed redemption US dollar",
            "Missed deposit US dollar",
            "Vault name",
        ],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def format_missed_vault_deposit_redemption_events(df: pd.DataFrame) -> Styler:
    """Format the missed vault event table for notebook and web output."""
    return df.style.format(
        {
            "Missed deposit count": "{:,.0f}",
            "Missed deposit US dollar": "${:,.2f}",
            "Missed redemption count": "{:,.0f}",
            "Missed redemption US dollar": "${:,.2f}",
        },
    )


def analyse_missed_vault_deposit_redemption_event_timeline(
    state: State,
) -> pd.DataFrame:
    """Create a timeline table of missed vault deposit and redemption events."""
    rows = []

    for timestamp, event in _iter_persisted_missed_vault_events_with_timestamp(state):
        event_type = _normalise_event_type(_read_attr_or_key(event, "event_type"))
        if event_type not in {"deposit", "redemption"}:
            continue

        rows.append(
            {
                "Timestamp": timestamp,
                "Event type": event_type,
                "Vault name": _get_vault_name(event),
                "Missed event count": 1,
                "Missed US dollar": float(_read_attr_or_key(event, "missed_usd", 0.0) or 0.0),
            }
        )

    if not rows:
        return pd.DataFrame(columns=MISSED_VAULT_EVENT_TIMELINE_COLUMNS)

    df = pd.DataFrame(rows, columns=MISSED_VAULT_EVENT_TIMELINE_COLUMNS)
    return df.groupby(
        [
            "Timestamp",
            "Event type",
            "Vault name",
        ],
        as_index=False,
    ).agg(
        {
            "Missed event count": "sum",
            "Missed US dollar": "sum",
        },
    ).sort_values(
        [
            "Timestamp",
            "Event type",
            "Vault name",
        ],
    ).reset_index(drop=True)
