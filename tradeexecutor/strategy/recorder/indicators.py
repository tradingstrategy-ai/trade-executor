"""Indicator-record schema for a decision-input recorder.

The recorder retains enough information to identify every calculated indicator
without copying its potentially large result series. Each ``indicator`` object
contains its definition, parameters, pair key, source, dependency order,
function hashes where source is available, cache metadata, and a fingerprint of
the result. The fingerprint stores its hash, length, shape, non-null count,
schema, first/last index, and temporal range. It intentionally omits raw
indicator values; constructed-universe frames are captured separately.

:meth:`DecisionRecorder.begin
<tradeexecutor.strategy.recorder.recorder.DecisionRecorder.begin>` calls
:func:`capture_indicators` once per live decision after the runner has calculated
the current indicator set. Researchers use these records to prove which
indicator implementation and result range informed a decision, even though the
recorder is not a replay engine.
"""

import hashlib
import inspect
from collections.abc import Callable
from typing import Any

import pandas as pd

from tradeexecutor.strategy.recorder.serialisation import canonical_json, encode_frame_chunks, to_json_value


def _fingerprint(data: pd.DataFrame | pd.Series) -> dict[str, Any]:
    """Return a result fingerprint without storing raw indicator values.

    :func:`capture_indicators` calls this for each current ``IndicatorResult``.
    A fingerprint is needed because recording complete indicator series every
    cycle would largely duplicate the captured universe data and rapidly grow
    the recorder file.

    The returned JSON object has ``sha256``, row ``length``, ``shape``,
    ``non_null_count``, frame ``schema``, first/last source index, and the
    earliest/latest datetime index values when the result has one.

    :param data:
        Calculated indicator result supplied by ``StrategyInputIndicators``.
    :return:
        JSON-ready structural and content fingerprint for later comparison.
    """
    schema, chunks = encode_frame_chunks(data)
    value = {"schema": schema, "chunks": [chunk["payload"] for chunk in chunks]}
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    index = data.index
    first_index = index[0] if len(index) else None
    last_index = index[-1] if len(index) else None
    start_at = first_index
    end_at = last_index
    if len(index):
        if isinstance(index, pd.DatetimeIndex):
            start_at = index.min()
            end_at = index.max()
        elif isinstance(index, pd.MultiIndex):
            for level in range(index.nlevels):
                values = index.get_level_values(level)
                if isinstance(values, pd.DatetimeIndex):
                    start_at = values.min()
                    end_at = values.max()
                    break
    return {
        "sha256": digest,
        "length": len(data),
        "non_null_count": int(data.notna().to_numpy().sum()),
        "shape": list(data.shape),
        "schema": schema,
        "first_index": to_json_value(first_index),
        "last_index": to_json_value(last_index),
        "start_at": to_json_value(start_at),
        "end_at": to_json_value(end_at),
    }


def capture_indicators(
    indicators: Any,
    put_object: Callable[[str, dict[str, Any]], dict[str, str]],
    *,
    inputs: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Store definition and fingerprint objects for all calculated indicators.

    :meth:`DecisionRecorder.begin
    <tradeexecutor.strategy.recorder.recorder.DecisionRecorder.begin>` calls
    this before the strategy performs its decision calculations. The returned
    references become ``input_manifest.indicators`` for later comparison with
    an equivalent backtest or another live cycle.

    :param indicators:
        Current :class:`StrategyInputIndicators
        <tradeexecutor.strategy.pandas_trader.strategy_input.StrategyInputIndicators>`.
    :param put_object:
        Content-addressed object writer supplied by :class:`RecorderStorage`.
    :param inputs:
        Object references on which each indicator depends, currently the
        decision's constructed-universe object.
    :return:
        ``{"object": <sha256>}`` references in indicator-result-map order.
    """
    refs = []
    for key, result in indicators.indicator_results.items():
        definition = key.definition
        function_hash = None
        source_hash = None
        if definition.func is not None:
            try:
                function_hash = definition.get_function_body_hash()
            except Exception:
                function_hash = None
            try:
                source_hash = hashlib.sha256(inspect.getsource(definition.func).encode("utf-8")).hexdigest()
            except (OSError, TypeError):
                source_hash = None
        payload = {
            "name": definition.name,
            "pair_key": None if key.pair is None else str(getattr(key.pair, "pool_address", getattr(key.pair, "internal_id", None))).lower(),
            "parameters": to_json_value(dict(definition.parameters)),
            "source": definition.source.value,
            "dependency_order": definition.dependency_order,
            "variations": definition.variations,
            "function": {"module": getattr(definition.func, "__module__", ""), "qualname": getattr(definition.func, "__qualname__", ""), "source_hash": source_hash, "framework_body_hash": function_hash},
            "universe_key": result.universe_key,
            "cached": result.cached,
            "result": _fingerprint(result.data),
            "inputs": list(inputs or []),
            "dependencies": [],
            "dependencies_complete": False,
        }
        refs.append(put_object("indicator", payload))
    return refs
