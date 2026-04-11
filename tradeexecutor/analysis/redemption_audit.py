"""Read-only helpers for auditing redemption diagnostics from state."""

import datetime
from dataclasses import dataclass

from tradeexecutor.state.state import State
from tradeexecutor.strategy.redemption import parse_recorded_lockup_expires_at


@dataclass(frozen=True, slots=True)
class RedemptionAuditRow:
    """One blocked-redemption audit row derived from saved state."""

    #: Pair ticker for the blocked signal.
    pair_ticker: str | None
    #: Vault address for the blocked signal.
    vault_address: str | None
    #: Latest recorded redemption-check stage.
    stage: str | None
    #: Latest recorded redemption-check reason code.
    reason_code: str | None
    #: Human-readable explanation from the latest recorded result.
    message: str | None
    #: Safe address used in the latest recorded live lookup, if any.
    safe_address: str | None
    #: Lockup expiry stored on the position state.
    position_recorded_lockup_expires_at: datetime.datetime | None
    #: Lockup expiry returned by the latest live user equity lookup, if any.
    user_lockup_expires_at: datetime.datetime | None
    #: Whether the recorded position lockup expiry is already in the past.
    recorded_lockup_expired: bool


def _read_attr_or_key(value: object, key: str, default=None):
    """Read a field from either a dataclass-like object or a plain dict."""
    if value is None:
        return default

    if isinstance(value, dict):
        return value.get(key, default)

    return getattr(value, key, default)


def _extract_stage_value(value: object) -> str | None:
    """Normalise enum-like or plain-string values."""
    if value is None:
        return None

    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return enum_value

    return str(value)


def _extract_latest_result(signal: object) -> object | None:
    """Get the latest redemption result from a serialised or live signal."""
    results = _read_attr_or_key(signal, "redemption_check_results", [])
    if not results:
        return None
    return results[-1]


def _extract_pair_ticker(pair: object) -> str | None:
    """Resolve a readable ticker from a live or serialised pair object."""
    ticker = _read_attr_or_key(pair, "ticker")
    if ticker:
        return ticker

    base = _read_attr_or_key(pair, "base")
    return _read_attr_or_key(base, "token_symbol")


def audit_redemption_state(
    state: State,
    *,
    now: datetime.datetime,
) -> tuple[list[RedemptionAuditRow], int]:
    """Inspect the latest alpha-model snapshot for blocked redemptions.

    1. Read the latest ``discardable_data["alpha_model"]`` snapshot.
    2. Find signals still marked with ``cannot_redeem``.
    3. Compare their stored position lockup expiry against ``now``.
    """
    alpha_model = state.visualisation.discardable_data.get("alpha_model")
    if alpha_model is None:
        return [], 0

    signals = _read_attr_or_key(alpha_model, "signals", {})
    if not signals:
        return [], 0

    positions_by_pair_id = {
        position.pair.internal_id: position
        for position in state.portfolio.open_positions.values()
    }

    rows = []
    for signal in signals.values():
        flags = _read_attr_or_key(signal, "flags", [])
        flag_values = {getattr(flag, "value", flag) for flag in flags}
        if "cannot_redeem" not in flag_values:
            continue

        pair = _read_attr_or_key(signal, "pair")
        pair_internal_id = _read_attr_or_key(pair, "internal_id")
        position = positions_by_pair_id.get(pair_internal_id)

        latest_result = _extract_latest_result(signal)
        position_recorded_lockup_expires_at = None
        if position is not None:
            position_recorded_lockup_expires_at = parse_recorded_lockup_expires_at(
                position.other_data.get("vault_lockup_expires_at"),
            )
        elif latest_result is not None:
            position_recorded_lockup_expires_at = parse_recorded_lockup_expires_at(
                _read_attr_or_key(
                    latest_result,
                    "position_recorded_lockup_expires_at",
                )
            )

        recorded_lockup_expired = (
            position_recorded_lockup_expires_at is not None
            and position_recorded_lockup_expires_at <= now
        )

        rows.append(
            RedemptionAuditRow(
                pair_ticker=_extract_pair_ticker(pair),
                vault_address=_read_attr_or_key(pair, "pool_address"),
                stage=_extract_stage_value(_read_attr_or_key(latest_result, "stage")),
                reason_code=_extract_stage_value(_read_attr_or_key(latest_result, "reason_code")),
                message=_read_attr_or_key(latest_result, "message"),
                safe_address=_read_attr_or_key(latest_result, "safe_address"),
                position_recorded_lockup_expires_at=position_recorded_lockup_expires_at,
                user_lockup_expires_at=parse_recorded_lockup_expires_at(
                    _read_attr_or_key(latest_result, "user_lockup_expires_at"),
                ),
                recorded_lockup_expired=recorded_lockup_expired,
            )
        )

    mismatch_count = sum(1 for row in rows if row.recorded_lockup_expired)
    return rows, mismatch_count
