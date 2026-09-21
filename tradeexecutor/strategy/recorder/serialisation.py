"""Create stable JSON values for live strategy-input recording.

Capture modules call these helpers before handing decision inputs to
:class:`~tradeexecutor.strategy.recorder.storage.RecorderStorage`. The separate
encoding is needed because ordinary JSON loses types, pandas metadata, and
stable ordering, which would make two equivalent live inputs hash differently.
The state serialiser remains authoritative for executor state files; this
module records only data that is absent from state and never reconstructs
application domain objects.

Scalar values retain otherwise lossy types with explicit ``$type`` tags:
``decimal``, ``timestamp_ns``, ``datetime``, ``date``, ``timedelta_ns``,
``missing``, ``numpy_scalar``, ``tuple``, ``set``, and ``mapping``. The encoder
sorts values whose native order is unstable, and :func:`canonical_json` sorts
object keys, so content hashes are repeatable. Data frames and series use a
separate schema plus row chunks; callers must not pass them to
:func:`to_json_value` directly.
"""

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
    """Safely reuse a domain object's existing JSON codec.

    :func:`to_json_value` calls this helper for supported domain objects. A
    shallow copy and isolated ``other_data`` mapping are necessary because
    some codecs mutate transient fields; recording must never change the live
    universe that ``decide_trades()`` is reading.

    :param value:
        Domain object exposing the project's ``to_dict()`` convention.
    :return:
        Plain Python data ready for recursive recorder encoding.
    """
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

    Universe, indicator, observation, and storage capture paths call this at
    the boundary between live Python values and the recorder schema. Explicit
    type tags preserve decision-relevant values that native JSON would coerce
    or reject, while deterministic container ordering supports content hashes.

    Raises :class:`TypeError` for frames, arrays, and unsupported object types
    rather than silently recording an incomplete decision input.

    :param value:
        Scalar, container, dataclass, enum, or supported domain object to
        encode.
    :return:
        A value accepted by the standard JSON encoder without type loss.
    :raises TypeError:
        If the value needs frame chunking or has no explicit recorder codec.
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
        raise TypeError("Use encode_frame_chunks() for pandas and NumPy containers")
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
    """Return byte-stable JSON for hashing and DuckDB ``JSON`` columns.

    :class:`RecorderStorage` uses this for every persisted payload, and the
    encoder uses it to sort mappings and sets that have no stable native order.
    Stable output makes content-addressed objects deduplicate across decisions.

    :param value:
        Supported value or already JSON-ready recorder payload.
    :return:
        Compact JSON text with sorted keys and no non-standard numbers.
    """
    return json.dumps(
        to_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_hash(kind: str, schema_version: int, payload: Any) -> str:
    """Return the storage identity of one versioned recorder object.

    :meth:`RecorderStorage.put_object` calls this before insertion. Including
    the object kind and schema version prevents equal-looking payloads with
    different meanings or decoding contracts from sharing an identity.

    :param kind:
        Semantic object category stored alongside the payload.
    :param schema_version:
        Version of the payload's decoding contract.
    :param payload:
        Recorder value to address.
    :return:
        Lowercase SHA-256 hexadecimal digest of the canonical envelope.
    """
    envelope = {"kind": kind, "schema_version": schema_version, "payload": payload}
    return hashlib.sha256(canonical_json(envelope).encode("utf-8")).hexdigest()


def decode_json_value(value: Any) -> Any:
    """Decode recorder type tags without importing arbitrary application classes.

    Tests and analyst inspection tools call this after reading DuckDB ``JSON``
    values. It intentionally cannot recreate domain objects or a strategy
    decision: avoiding arbitrary imports keeps offline inspection predictable
    and separates recording from future reconstruction work.

    :param value:
        Parsed JSON value read from a recorder column.
    :return:
        Nested Python values with built-in recorder type tags decoded.
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

    :func:`encode_frame_chunks` calls this once per captured frame. Keeping
    structural metadata in the parent object lets row chunks deduplicate while
    retaining enough information for an analyst to interpret their values.

    :param frame:
        Dataframe or series whose structure will accompany captured row chunks.
    :return:
        JSON-ready container, label, index, and dtype description.
    """

    def dtype_schema(dtype: Any) -> dict[str, Any]:
        """Describe one dtype without depending on pandas' Python objects.

        ``frame_schema()`` calls this for every value column so inspection can
        distinguish categories and ordered categoricals from plain strings.

        :param dtype:
            Pandas or NumPy dtype to describe.
        :return:
            JSON-ready dtype name, kind, and categorical metadata.
        """
        descriptor = {"name": str(dtype), "kind": getattr(dtype, "kind", None)}
        if isinstance(dtype, pd.CategoricalDtype):
            descriptor["categories"] = [to_json_value(value) for value in dtype.categories.tolist()]
            descriptor["ordered"] = dtype.ordered
        return descriptor

    def axis_schema(axis: pd.Index) -> dict[str, Any]:
        """Describe one pandas axis, including specialised index metadata.

        ``frame_schema()`` calls this for row and column indexes. Recording the
        index form explains how captured row values were aligned at decision
        time, which a values-only dump could not establish.

        :param axis:
            Pandas row or column index to describe.
        :return:
            JSON-ready index type, names, dtype, and specialised metadata.
        """
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

    Universe and indicator capture functions call this for their pandas data.
    Chunking allows unchanged portions to share content hashes across decisions
    and prevents one large frame from becoming an indivisible DuckDB object.

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
