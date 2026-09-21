"""Stable JSON schema used by the strategy-input recorder.

The state serialiser remains authoritative for executor state files. This module
only provides exact, deterministic JSON values for recorder inputs and pandas
data. It never reconstructs domain objects or replaces state serialisation.

Scalar values retain otherwise lossy types with explicit ``$type`` tags:
``decimal``, ``timestamp_ns``, ``datetime``, ``date``, ``timedelta_ns``,
``missing``, ``numpy_scalar``, ``tuple``, ``set``, and ``mapping``. The encoder
sorts values whose native order is unstable, and :func:`canonical_json` sorts
object keys, so content hashes are repeatable. Data frames and series use a
separate schema plus row chunks; callers must not pass them to
:func:`to_json_value` directly.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime
import hashlib
import json
import math
from collections.abc import Mapping
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import pandas as pd

from tradeexecutor.monkeypatch.dataclasses_json import patch_dataclasses_json


def _domain_to_dict(value: Any) -> Any:
    """Use an existing dataclasses-json codec without mutating the source object."""
    patch_dataclasses_json()
    candidate = copy.copy(value)
    # TradingPairIdentifier's custom encoder deletes transient keys in-place.
    # Never let that mutation reach the live universe.
    if hasattr(candidate, "other_data"):
        try:
            candidate.other_data = copy.deepcopy(dict(candidate.other_data or {}))
        except Exception:
            candidate.other_data = dict(candidate.other_data or {})
    try:
        return candidate.to_dict(encode_json=False)
    except TypeError:
        return candidate.to_dict()


def to_json_value(value: Any) -> Any:
    """Convert a supported Python value into deterministic, exact JSON data.

    Raises :class:`TypeError` for frames, arrays, and unsupported object types
    rather than silently recording an incomplete decision input.
    """
    if isinstance(value, Enum):
        return {"$type": "enum", "class": f"{value.__class__.__module__}.{value.__class__.__qualname__}", "value": to_json_value(value.value)}
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if value is pd.NaT:
        return {"$type": "missing", "value": "nat"}
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return {"$type": "missing", "value": "nat"}
        ts = value.tz_convert("UTC").tz_localize(None) if value.tzinfo is not None else value
        return {"$type": "timestamp_ns", "value": str(int(ts.value))}
    if isinstance(value, datetime.datetime):
        if value.tzinfo is not None:
            value = value.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        return {"$type": "datetime", "value": value.isoformat()}
    if isinstance(value, datetime.date):
        return {"$type": "date", "value": value.isoformat()}
    if isinstance(value, (datetime.timedelta, pd.Timedelta)):
        return {
            "$type": "timedelta_ns",
            "value": str(int(value.total_seconds() * 1_000_000_000)),
            "python_type": "pandas" if isinstance(value, pd.Timedelta) else "datetime",
        }
    if isinstance(value, Decimal):
        return {"$type": "decimal", "value": str(value)}
    if value is pd.NA:
        return {"$type": "missing", "value": "pd_na"}
    if isinstance(value, float):
        if math.isnan(value):
            return {"$type": "missing", "value": "nan"}
        if math.isinf(value):
            return {"$type": "missing", "value": "pos_inf" if value > 0 else "neg_inf"}
        return value
    if isinstance(value, np.generic):
        if np.issubdtype(value.dtype, np.datetime64):
            return to_json_value(pd.Timestamp(value))
        return {"$type": "numpy_scalar", "dtype": str(value.dtype), "value": to_json_value(value.item())}
    if isinstance(value, (pd.DataFrame, pd.Series, np.ndarray)):
        raise TypeError("Use encode_frame() for pandas and NumPy containers")
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        items = list(value.items())
        if all(isinstance(k, str) and k != "$type" for k, _ in items):
            return {k: to_json_value(v) for k, v in items}
        encoded_items = [[to_json_value(k), to_json_value(v)] for k, v in items]
        encoded_items.sort(key=lambda item: canonical_json(item[0]))
        return {"$type": "mapping", "items": encoded_items}
    if isinstance(value, tuple):
        return {"$type": "tuple", "items": [to_json_value(v) for v in value]}
    if isinstance(value, (set, frozenset)):
        items = [to_json_value(v) for v in value]
        items.sort(key=lambda v: canonical_json(v))
        return {"$type": "set", "items": items}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return to_json_value(_domain_to_dict(value))
    if dataclasses.is_dataclass(value):
        return to_json_value({field.name: getattr(value, field.name) for field in dataclasses.fields(value)})
    if isinstance(value, list):
        return [to_json_value(v) for v in value]
    raise TypeError(f"Unsupported recorder value: {type(value)!r}")


def canonical_json(value: Any) -> str:
    """Return canonical JSON suitable for hashes and DuckDB ``JSON`` columns."""
    return json.dumps(
        to_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_hash(kind: str, schema_version: int, payload: Any) -> str:
    """Return the content address for one versioned recorder object."""

    envelope = {"kind": kind, "schema_version": schema_version, "payload": payload}
    return hashlib.sha256(canonical_json(envelope).encode("utf-8")).hexdigest()


def decode_json_value(value: Any) -> Any:
    """Decode recorder type tags without importing arbitrary application classes.

    This is an inspection helper only. It cannot recreate domain objects or a
    strategy decision from a recorder database.
    """
    if isinstance(value, list):
        return [decode_json_value(v) for v in value]
    if not isinstance(value, dict):
        return value
    tag = value.get("$type")
    if tag == "decimal":
        return Decimal(value["value"])
    if tag == "timestamp_ns":
        return pd.Timestamp(int(value["value"]), unit="ns")
    if tag == "datetime":
        return datetime.datetime.fromisoformat(value["value"])
    if tag == "date":
        return datetime.date.fromisoformat(value["value"])
    if tag == "timedelta_ns":
        return pd.Timedelta(int(value["value"]), unit="ns") if value.get("python_type") == "pandas" else datetime.timedelta(seconds=int(value["value"]) / 1_000_000_000)
    if tag == "missing":
        return {"nan": float("nan"), "pos_inf": float("inf"), "neg_inf": float("-inf"), "pd_na": pd.NA, "nat": pd.NaT}[value["value"]]
    if tag == "tuple":
        return tuple(decode_json_value(v) for v in value["items"])
    if tag == "set":
        return set(decode_json_value(v) for v in value["items"])
    if tag == "mapping":
        return {decode_json_value(k): decode_json_value(v) for k, v in value["items"]}
    if tag == "numpy_scalar":
        return np.dtype(value["dtype"]).type(decode_json_value(value["value"]))
    return {k: decode_json_value(v) for k, v in value.items()}


def frame_schema(frame: pd.DataFrame | pd.Series) -> dict[str, Any]:
    """Describe a pandas frame's container, labels, index, and dtypes.

    Values are written by :func:`encode_frame_chunks`; this schema is held once
    in the parent universe or indicator fingerprint object.
    """

    def dtype_schema(dtype: Any) -> dict[str, Any]:
        descriptor = {"name": str(dtype), "kind": getattr(dtype, "kind", None)}
        if isinstance(dtype, pd.CategoricalDtype):
            descriptor["categories"] = [to_json_value(value) for value in dtype.categories.tolist()]
            descriptor["ordered"] = dtype.ordered
        return descriptor

    def axis_schema(axis: pd.Index) -> dict[str, Any]:
        descriptor = {
            "class": type(axis).__name__,
            "names": [to_json_value(value) for value in axis.names],
            "dtype": str(axis.dtype),
        }
        if isinstance(axis, pd.RangeIndex):
            descriptor["range"] = {
                "start": axis.start,
                "stop": axis.stop,
                "step": axis.step,
            }
        elif isinstance(axis, pd.MultiIndex):
            descriptor["levels"] = [
                [to_json_value(value) for value in level.tolist()]
                for level in axis.levels
            ]
            descriptor["codes"] = [
                [to_json_value(value) for value in code.tolist()]
                for code in axis.codes
            ]
        elif isinstance(axis, pd.CategoricalIndex):
            descriptor["categories"] = [to_json_value(value) for value in axis.categories.tolist()]
            descriptor["ordered"] = axis.ordered
        if isinstance(axis, pd.DatetimeIndex):
            descriptor["timezone"] = str(axis.tz) if axis.tz is not None else None
        return descriptor

    is_series = isinstance(frame, pd.Series)
    columns = [frame.name] if is_series else list(frame.columns)
    return {
        "container": "series" if is_series else "dataframe",
        "name": to_json_value(frame.name) if is_series else None,
        "columns": [to_json_value(c) for c in columns],
        "column_dtypes": [dtype_schema(frame.dtype)] if is_series else [dtype_schema(v) for v in frame.dtypes],
        "index": axis_schema(frame.index),
        "column_index": axis_schema(frame.columns) if not is_series else None,
    }


def encode_frame_chunks(
    frame: pd.DataFrame | pd.Series,
    chunk_size: int = 2_048,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Encode a frame into a schema and placement-independent row chunks.

    :param frame:
        Data frame or series to capture.
    :param chunk_size:
        Maximum number of rows in each returned chunk.
    :return:
        Schema and chunk dictionaries. Each chunk has ``start``, ``count``, and
        a JSON-ready ``payload`` containing index values and row values.
    """
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    schema = frame_schema(frame)
    chunks = []
    for start in range(0, len(frame), chunk_size):
        part = frame.iloc[start:start + chunk_size]
        payload = {
            "column_dtypes": [{"name": str(v), "kind": getattr(v, "kind", None)} for v in part.dtypes],
            "index_dtypes": [str(part.index.dtype)],
            "index_values": [[to_json_value(v)] for v in part.index.tolist()],
            "rows": [[to_json_value(v) for v in row] for row in part.itertuples(index=False, name=None)],
        }
        chunks.append({"start": start, "count": len(part), "payload": payload})
    return schema, chunks
