"""Analyse missed vault deposit and redemption opportunities."""

import pandas as pd

from tradeexecutor.state.state import State


MISSED_VAULT_EVENT_COLUMNS = [
    "Vault name",
    "Missed deposit count",
    "Missed deposit US dollar",
    "Missed redemption count",
    "Missed redemption US dollar",
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


def _iter_persisted_missed_vault_events(state: State):
    """Yield missed vault event rows persisted in visualisation calculations."""
    for calculations in state.visualisation.calculations.values():
        events = calculations.get("missed_vault_events", [])
        for event in events:
            yield event


def _iter_latest_missed_vault_events(state: State):
    """Yield missed vault event rows from the latest in-memory alpha model."""
    alpha_model = state.visualisation.discardable_data.get("alpha_model")
    if alpha_model is None:
        return

    get_missed_vault_events = getattr(alpha_model, "get_missed_vault_events", None)
    if get_missed_vault_events is not None:
        yield from get_missed_vault_events()


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

        vault_name = (
            _read_attr_or_key(event, "vault_name")
            or _read_attr_or_key(event, "pair_ticker")
            or _read_attr_or_key(event, "vault_address")
            or "Unknown vault"
        )
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
