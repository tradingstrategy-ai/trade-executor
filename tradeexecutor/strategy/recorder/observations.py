"""Small explicit decision observations."""

from typing import Any

from tradeexecutor.strategy.recorder.serialisation import to_json_value


def observation(
    sequence: int,
    kind: str,
    name: str,
    value: Any,
    *,
    pair_key: str | None = None,
    arguments: dict[str, Any] | None = None,
    observed_at: Any = None,
    source_at: Any = None,
    state_refs: list[dict[str, Any]] | None = None,
    provenance: Any = None,
    error: Any = None,
) -> dict[str, Any]:
    """Create one JSON-safe observation stored in a decision row."""

    return {
        "sequence": sequence,
        "kind": kind,
        "name": name,
        "pair_key": pair_key,
        "observed_at": to_json_value(observed_at),
        "source_at": to_json_value(source_at),
        "arguments": to_json_value(arguments or {}),
        "value": to_json_value(value),
        "state_refs": to_json_value(state_refs or []),
        "provenance": to_json_value(provenance),
        "error": to_json_value(error),
    }
