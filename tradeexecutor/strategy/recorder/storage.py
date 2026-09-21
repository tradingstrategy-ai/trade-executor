"""DuckDB schema and writer for live strategy-decision records.

The recorder database retains decision history in three version-1 tables. Run
and object rows are append-only; a decision row is created as ``started`` and
updated once to its terminal lifecycle state:

``runs``
    One row per executor process. :meth:`RecorderStorage.start_run` creates it
    when :class:`~tradeexecutor.strategy.recorder.recorder.DecisionRecorder`
    is constructed. ``metadata`` records the loaded strategy source reference,
    strategy file, executor revision, package versions, and Python version.
    ``run_id`` is the logical parent of its decisions.

``objects``
    A content-addressed object store shared by all runs and decisions in this
    database. :meth:`RecorderStorage.put_object` serialises a JSON-ready
    payload, hashes its kind, schema version, and canonical payload, then
    inserts it once. It holds relatively large or reusable decision inputs:
    strategy source, parameters, execution context, constructed universe and
    frame chunks, indicator definitions, and model projections.

``decisions``
    One row per ``decide_trades()`` invocation. :meth:`start_decision` writes
    the immutable input manifest and ``started`` lifecycle row before strategy
    calculations execute. The manifest contains references to ``objects``;
    these references, and ``run_id``, are logical relationships rather than
    database foreign-key constraints. :meth:`finish_decision` updates that
    same row with its terminal status, explicit observations, returned trade
    IDs, and (for failures) a concise error summary.

``serialisation.py`` is responsible for converting Python values into stable,
exact JSON. This module persists those JSON values, validates the version-1
table shape when a file is reopened, and checkpoints completed decisions so
they are inspectable from another DuckDB connection.
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

import duckdb
from eth_defi.compat import native_datetime_utc_now

from tradeexecutor.strategy.recorder.serialisation import canonical_json, content_hash

def validate_recorder_strategy_id(strategy_id: str) -> str:
    """Validate the executor ID before using it in a recorder filename.

    :param strategy_id:
        Executor identifier selected by live CLI bootstrap.
    :return:
        The unchanged validated identifier.
    :raises ValueError:
        If the identifier is empty or cannot form a single safe filename.
    """
    if not strategy_id or strategy_id in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", strategy_id):
        raise ValueError(f"Unsafe recorder strategy ID: {strategy_id!r}")
    return strategy_id


class RecorderStorage:
    """Single-process writer for the versioned recorder DuckDB schema.

    One instance owns one read-write DuckDB attachment. The higher-level
    :class:`DecisionRecorder` serialises lifecycle calls; this class provides
    content-addressed object writes, decision row transitions, checkpoints, and
    idempotent close behaviour.
    """

    def __init__(self, path: Path) -> None:
        """Open ``path``, create the version-1 schema, and validate it."""

        self.path = Path(path)
        if self.path.name in {"", ".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", self.path.name):
            raise ValueError(f"Unsafe recorder filename: {self.path.name!r}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = duckdb.connect(":memory:")
        self._closed = False
        escaped = str(self.path.absolute()).replace("'", "''")
        self.connection.execute(f"ATTACH '{escaped}' AS recorder (STORAGE_VERSION 'v1.2.0')")
        self.connection.execute("SET force_compression='zstd'")
        self._create_schema()
        self._validate_schema()

    def _create_schema(self) -> None:
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS recorder.runs (
                run_id UUID PRIMARY KEY,
                strategy_id VARCHAR NOT NULL,
                started_at TIMESTAMP NOT NULL,
                format_version INTEGER NOT NULL CHECK (format_version = 1),
                metadata JSON NOT NULL
            )
        """)

        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS recorder.objects (
                content_hash VARCHAR PRIMARY KEY CHECK (length(content_hash) = 64),
                kind VARCHAR NOT NULL,
                schema_version INTEGER NOT NULL CHECK (schema_version = 1),
                payload JSON NOT NULL
            )
        """)
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS recorder.decisions (
                invocation_id UUID PRIMARY KEY,
                run_id UUID NOT NULL,
                cycle BIGINT NOT NULL,
                decision_at TIMESTAMP_NS NOT NULL,
                started_at TIMESTAMP NOT NULL,
                finished_at TIMESTAMP,
                state_path VARCHAR NOT NULL,
                status VARCHAR NOT NULL CHECK (status IN ('started', 'completed', 'failed')),
                input_manifest JSON NOT NULL,
                observations JSON NOT NULL DEFAULT '[]',
                output_trade_ids JSON NOT NULL DEFAULT '[]',
                error JSON,
                CHECK ((status = 'started' AND finished_at IS NULL) OR (status IN ('completed', 'failed') AND finished_at IS NOT NULL))
            )
        """)

    def _validate_schema(self) -> None:
        expected = {
            "runs": {"run_id", "strategy_id", "started_at", "format_version", "metadata"},
            "objects": {"content_hash", "kind", "schema_version", "payload"},
            "decisions": {
                "invocation_id", "run_id", "cycle", "decision_at", "started_at",
                "finished_at", "state_path", "status", "input_manifest",
                "observations", "output_trade_ids", "error",
            },
        }
        for table, expected_columns in expected.items():
            columns = {
                row[1]
                for row in self.connection.execute(
                    f"SELECT * FROM pragma_table_info('recorder.{table}')"
                ).fetchall()
            }
            if columns != expected_columns:
                raise RuntimeError(
                    f"Unsupported strategy recorder schema for {table}: "
                    f"expected {sorted(expected_columns)}, got {sorted(columns)}"
                )
        versions = self.connection.execute(
            "SELECT DISTINCT format_version FROM recorder.runs"
        ).fetchall()
        if any(row[0] != 1 for row in versions):
            raise RuntimeError(f"Unsupported strategy recorder format versions: {versions}")
        object_versions = self.connection.execute(
            "SELECT DISTINCT schema_version FROM recorder.objects"
        ).fetchall()
        if any(row[0] != 1 for row in object_versions):
            raise RuntimeError(f"Unsupported recorder object schema versions: {object_versions}")

    def put_object(
        self,
        kind: str,
        payload: dict[str, Any],
        schema_version: int = 1,
    ) -> dict[str, str]:
        """Store one content-addressed JSON object and return its reference."""

        digest = content_hash(kind, schema_version, payload)
        self.connection.execute(
            "INSERT OR IGNORE INTO recorder.objects VALUES (?, ?, ?, ?::JSON)",
            [digest, kind, schema_version, canonical_json(payload)],
        )
        return {"object": digest}

    def start_run(self, strategy_id: str, metadata: dict[str, Any]) -> UUID:
        """Create metadata for one executor process."""

        run_id = uuid4()
        now = native_datetime_utc_now()
        self.connection.execute(
            "INSERT INTO recorder.runs VALUES (?, ?, ?, ?, ?::JSON)",
            [str(run_id), strategy_id, now, 1, canonical_json(metadata)],
        )
        self.connection.commit()
        return run_id

    def start_decision(
        self,
        run_id: UUID,
        cycle: int,
        decision_at: datetime.datetime,
        state_path: str,
        manifest: dict[str, Any],
    ) -> UUID:
        """Persist the active decision manifest before strategy code executes."""

        if self.connection.execute("SELECT 1 FROM recorder.runs WHERE run_id = ?", [str(run_id)]).fetchone() is None:
            raise ValueError(f"Unknown recorder run ID: {run_id}")
        object_refs = set()
        stack = [manifest]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                if isinstance(value.get("object"), str) and len(value["object"]) == 64:
                    object_refs.add(value["object"])
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
        if object_refs:
            placeholders = ",".join("?" for _ in object_refs)
            found = {
                row[0]
                for row in self.connection.execute(
                    f"SELECT content_hash FROM recorder.objects WHERE content_hash IN ({placeholders})",
                    list(object_refs),
                ).fetchall()
            }
            missing = object_refs - found
            if missing:
                raise ValueError(f"Decision manifest references missing recorder objects: {sorted(missing)}")
        invocation_id = uuid4()
        now = native_datetime_utc_now()
        self.connection.execute(
            "INSERT INTO recorder.decisions(invocation_id, run_id, cycle, decision_at, started_at, state_path, status, input_manifest) VALUES (?, ?, ?, ?::TIMESTAMP_NS, ?, ?, 'started', ?::JSON)",
            [str(invocation_id), str(run_id), cycle, decision_at.isoformat(), now, state_path, canonical_json(manifest)],
        )
        self.connection.commit()
        return invocation_id

    def finish_decision(
        self,
        invocation_id: UUID,
        status: Literal["completed", "failed"],
        observations: list[dict[str, Any]],
        output_trade_ids: list[int],
        error: dict[str, str] | None = None,
    ) -> None:
        """Write a terminal status and the strategy's explicit observations."""

        now = native_datetime_utc_now()
        self.connection.execute(
            "UPDATE recorder.decisions SET finished_at=?, status=?, observations=?::JSON, output_trade_ids=?::JSON, error=?::JSON WHERE invocation_id=?",
            [now, status, canonical_json(observations), canonical_json(output_trade_ids), canonical_json(error) if error is not None else "null", str(invocation_id)],
        )
        self.connection.commit()

    def checkpoint(self) -> None:
        """Flush committed recorder data to its DuckDB file."""

        self.connection.execute("CHECKPOINT")

    def close(self) -> None:
        """Checkpoint and close this writer exactly once.

        This method is safe when both the normal decision lifecycle and an
        execution-loop ``finally`` path attempt to close the recorder.
        """

        if self._closed:
            return
        try:
            self.checkpoint()
        finally:
            self.connection.close()
            self._closed = True
