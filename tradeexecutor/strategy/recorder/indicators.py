"""Indicator definitions and result fingerprints."""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable
from typing import Any

import pandas as pd

from tradeexecutor.strategy.recorder.serialisation import canonical_json, encode_frame_chunks, to_json_value


def _fingerprint(data: pd.DataFrame | pd.Series) -> dict[str, Any]:
    """Return reproducibility metadata without storing indicator values."""

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
    """Store definitions and fingerprints for every calculated indicator."""

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
