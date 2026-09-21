"""Schema for small, explicit observations made during one strategy decision.

An observation is not an automatic trace or an API-response archive. Strategy
code calls :meth:`DecisionRecorder.record
<tradeexecutor.strategy.recorder.recorder.DecisionRecorder.record>` for the
numbers, ranked candidates, admission checks, and allocations that explain a
decision. This module converts that call to the JSON object embedded in the
``decisions.observations`` column.

Every observation has a monotonically increasing ``sequence``, a caller-defined
``kind`` and ``name``, an ``observed_at`` wall-clock timestamp, and a JSON-safe
``value``. Optional ``arguments``, ``pair_key``, ``source_at``, ``state_refs``,
``provenance``, and ``error`` add context without duplicating the state file.
Values are encoded by :func:`to_json_value`, which preserves supported numeric,
datetime, enum, and domain-object values deterministically.

This module is an internal schema boundary. Strategy authors call
``StrategyInput.recorder.record()``; :class:`DecisionRecorder` calls
:func:`observation` to build the stored representation.
"""

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
    """Create one JSON-safe observation stored in a decision row.

    Called by :meth:`DecisionRecorder.record
    <tradeexecutor.strategy.recorder.recorder.DecisionRecorder.record>` after
    it assigns the per-decision sequence number and observation time. Keeping
    this conversion separate gives every strategy-authored observation one
    documented schema before :class:`RecorderStorage` writes the decision row.

    :param sequence:
        Zero-based order within the active ``decide_trades()`` invocation.
    :param kind:
        Stable caller-defined category, such as ``signal`` or ``allocation``.
    :param name:
        Stable caller-defined observation name within ``kind``.
    :param value:
        Decision-relevant value to serialise.
    :param pair_key:
        Optional stable pair or vault identifier for a pair-specific value.
    :param arguments:
        Inputs used to derive ``value`` when they are not already in the input
        manifest.
    :param observed_at:
        Wall-clock timestamp at which strategy code observed the value.
    :param source_at:
        Timestamp of the underlying source data, when different from
        ``observed_at``.
    :param state_refs:
        Small state-file identifiers needed to correlate the observation with
        existing state, rather than a copy of state.
    :param provenance:
        Optional caller-defined calculation or data-source metadata.
    :param error:
        Optional serialisable error detail associated with this observation.
    :return:
        JSON-ready observation dictionary for ``decisions.observations``.
    """
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
