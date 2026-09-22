# Plan: record strategy decision inputs in DuckDB

## Objective

Capture the exact non-state inputs supplied to each trading decision, especially
the constructed universe, so a live decision can be investigated later. Retain
indicator fingerprints and source inputs for future analysis, but do not build
reconstruction or replay tooling in this change. Implement this in
`tradeexecutor/strategy/recorder/` and enable it in `hyper-ai-v8.py`.

This work spans two repositories:

- Framework: `/Users/moo/code/trade-executor`.
- Strategy: `/Users/moo/code/strategies/strategy/hyper-ai-v8.py`, in the separate
  strategies repository. It is not under trade-executor's `strategies/` directory.

Record decision-relevant numbers, flags and calculation results derived from
HyperCore and other APIs. Full API response bodies are explicitly out of scope.

This document is an implementation plan, not an implementation. Use Zstandard
(Zstd) compression. Keep the implementation small: one embedded database,
explicit serialisers and direct recording calls inside `decide_trades()`.
No recording service, generic object-graph serialiser, migrations, or rollback
framework.

## Scope decisions after review

- Let recording failures crash the executor through its existing error path.
  No degraded mode or recorder-specific recovery machinery.
- Build recording directly into v8's `decide_trades()` using the framework
  recorder API. No observer hooks, model wrappers or monkey-patching.
- Live trading only. Backtests and notebooks do not initialise or write a
  recorder, even when the strategy uses the recording decorator.
- No new state archives, immutable revisions, retention or pruning machinery.
- Defer indicator reconstruction and full decision replay tooling.
- Defer additional storage-safety work such as snapshot/copy tooling and
  cross-process access support. Retain basic database transactions and Zstd.

## Existing integration points

- `tradeexecutor/strategy/pandas_trader/runner.py:on_clock()` constructs
  `StrategyInput` after treasury synchronisation and live indicator calculation,
  then calls `self.decide_trades(input)`. Make the configured framework recorder
  available on `StrategyInput`; v8 starts capture as the first operation inside
  `decide_trades()`, before filtering or mutation. Include early returns and
  exceptions after the input manifest has been committed. Failures before this
  boundary remain in existing executor logs without a recorded invocation.
- `calculate_live_indicators()` currently uses `MemoryIndicatorStorage`; its
  latest result is exposed through `RunState.latest_indicators`. This is not a
  durable history of the indicator inputs.
- `tradeexecutor/cli/commands/start.py` resolves executor ID and state path.
  `JSONFileStore.path` identifies the authoritative state file; the execution
  loop owns persistence and the recorder lifecycle.
- The normal state owns portfolio, trades, balances, valuations and persisted
  visualisation calculations. The recorder must reference these, not copy them.
- v8's current optional JSON manifest captures filtered candidates and outputs,
  not the full constructed universe or raw input data. Its `replay` mode only
  supplies an address allow-list. It does not replay exact inputs.
- HyperCore pricing makes live requests during the callback. A snapshot taken
  only at callback entry cannot capture the values consumed from those requests.

## Capture contract

Inventory every `StrategyInput` field and the nested universe fields during
implementation. Maintain a coverage table in the recorder documentation: each
field must be captured, referenced from state, fingerprinted as a derived value,
or explicitly identified as an executable handle. Newly encountered data fields
must cause a descriptive unsupported-field error, not disappear silently.

| Input | Recording |
|---|---|
| Cycle and timestamp | Cycle number, exact logical decision time, wall-clock capture/start/end times, unique invocation ID and process/run ID; use naive UTC |
| Parameters | Entire resolved parameter mapping, including effective defaults and overrides |
| Execution context | Mode, engine version and decision-relevant serialisable flags |
| Constructed universe | Exact membership, stable chain/address keys and internal-ID mapping, pairs/assets/exchanges, reserves, metadata, vault state, candle and liquidity history, lending/stop-loss data if present, universe options, routing-related data, freshness diagnostics and filtering results |
| Source data | Actual in-memory frames presented to indicators and decision logic, with their schema, index, order, precision, nulls and timestamps preserved; upstream source identifiers/checksums and observation timestamps where available |
| Calculated indicators | Definition, function/source hash, parameters, dependencies, pair/universe identity, result hash, length, shape, dtypes, index metadata, start/end dates, and input snapshot references; no duplicate full indicator series |
| State | Authoritative state path, cycle/time and existing position/trade IDs where relevant; no state snapshots, new revisions or portfolio/trade/valuation JSON copied into DuckDB |
| Pricing | Configuration and decision-relevant cache values at entry; ordered records of consumed numbers, flags and pricing/calculation results during the callback; exclude full API response bodies |
| Routing and Web3 | Implementation identity, public chain/block identifiers and serialisable decision configuration; never serialise provider sessions, signers or connections |
| `other_data` | Serialisable decision inputs not already represented in state, with explicit references for shared state objects |
| Decision diagnostics | Candidate rejection reasons, consumed indicator values and selected IDs where not already persisted in state; retain existing state-owned diagnostics in state |

Universe capture is performed before strategy filtering: v8's post-gate candidate
list is insufficient. Capture the actual constructed frames, including their
lookback history; do not fetch replacement history from today's API later.
Preserve exactly what was supplied, even if it contains a data-quality defect.

“Everything” means data used by the decision. Functions, network clients,
threads, locks and wallet secrets are not serialisable market inputs. Store
implementation/version identities for executable handles and capture their
decision-relevant observations instead. Exclude credentials, signing keys,
authentication headers and secrets embedded in endpoint URLs explicitly.

## Storage and file naming

The CLI-resolved executor/strategy ID is the filename authority, not the display
name or v8's descriptive `Parameters.id`:

```text
state/hyper-ai.json
state/hyper-ai-record.duckdb
```

Compute `state_path.parent / f"{strategy_id}-record.duckdb"` centrally after the
state path is resolved. Reuse `validate_executor_id()` from
`tradeexecutor/cli/bootstrap.py` to validate the filename component. Honour
custom state directories. That validator only checks non-empty/no spaces, so
also require a single safe filename component (ASCII letters, digits, `_`, `-`
and `.`, excluding `.` and `..`); reject slashes and backslashes. Log the
resolved recorder path on startup. The file
and its DuckDB WAL belong on the existing persistent state volume.

Use DuckDB native `JSON` columns for serialisable Python payloads and a few typed
columns for cycle IDs, timestamps, hashes and lookup keys. Do not pickle inputs
or hide JSON inside compressed BLOBs. Encode DataFrames/Series as typed JSON
chunks with explicit schema and index information; do not use lossy
`default=str`, floating-point conversions of Decimal, or JSON object keys that
erase index types. Canonical serialization must preserve nanosecond timestamps,
Decimal values, enums, tuples, sets, missing values and non-finite numbers using
small explicit tagged representations where JSON alone is insufficient.

Use three tables initially:

| Table | Contents |
|---|---|
| `runs` | Run ID, strategy ID, code/build identities, dependency versions and recorder format version; deduplicated source/configuration references |
| `decisions` | Invocation ID, run ID, cycle/logical/wall timestamps, authoritative state path, status, JSON input manifest, completion/error metadata and output trade-ID references |
| `objects` | SHA-256 content hash, kind, schema version, JSON payload; reusable universe/frame chunks, configuration and external observation objects |

The decision manifest references immutable objects by hash. Reuse unchanged
metadata and frame chunks across cycles. Split large histories along fixed
pair/time boundaries so one new observation does not duplicate years of data.
Preserve row order within chunks; keep frame placement in the per-decision
universe manifest, outside chunk payloads and their hashes. Historical corrections
create new hashes while earlier decisions keep their original objects. Avoid
per-row insertion: serialize/hash in bounded chunks and batch writes.

## Concrete database schema (format version 1)

Use these three tables, not a separate table per Python type. All SQL timestamps
are naive UTC. Use `TIMESTAMP_NS` for the pandas logical timestamp and
`TIMESTAMP` for Python wall-clock datetimes. IDs are recorder-generated UUIDs;
cycle numbers are not unique across restarts. SQL `NULL` means an optional
column is absent, whereas JSON `null` is a value inside a payload.

```sql
CREATE TABLE runs (
    run_id UUID PRIMARY KEY,
    strategy_id VARCHAR NOT NULL,
    started_at TIMESTAMP NOT NULL,
    format_version INTEGER NOT NULL CHECK (format_version = 1),
    metadata JSON NOT NULL
);

CREATE TABLE objects (
    content_hash VARCHAR PRIMARY KEY CHECK (length(content_hash) = 64),
    kind VARCHAR NOT NULL,
    schema_version INTEGER NOT NULL CHECK (schema_version = 1),
    payload JSON NOT NULL
);

CREATE TABLE decisions (
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
    CHECK ((status = 'started' AND finished_at IS NULL)
        OR (status IN ('completed', 'failed') AND finished_at IS NOT NULL))
);
```

Insert the run once. Insert new objects and the `started` decision in one
transaction. Finalisation updates only observations, output IDs, status, error
and finish time; never mutate the input manifest or existing objects. A callback
error has `{type, message}` in `error` (sanitised, no provider response bodies).
Successful decisions have SQL `NULL` error; unrecordable failures can leave
`started`. Trade IDs are integers owned by state, not copies of trade objects.

`content_hash` is lowercase SHA-256 over canonical UTF-8 JSON for
`{kind, schema_version, payload}`, not just the payload. Canonicalise string-key
ordering and separators, reject bare NaN/Infinity, and preserve array order.
Hash before insertion; do not hash DuckDB's rendered JSON. Insert only missing
hashes. JSON object references use `{"object": "<64-character hash>"}` and are
validated by the writer before commit and by readback tests: DuckDB foreign keys
cannot enforce references embedded inside JSON. The recorder opens its file as
a Zstd-enabled attached database, where DuckDB does not allow cross-database
foreign-key declarations; therefore `decisions.run_id -> runs.run_id` is a
logical foreign key checked by `start_decision()` and readback tests. No
additional indexes or migration system initially; fail clearly on an unsupported
format version.

### JSON payload contracts

The following are recorder-owned data-transfer schemas, not new fields on the
existing universe/indicator classes. Use small typed dataclasses with
`@dataclass_json` for these envelopes and explicit capture functions. `Value`
means a JSON scalar/list/string-key mapping or one of the exact-value tags below;
`ObjectRef` is the hash reference above. Required keys are always present;
optional values are `null`, collections are empty lists/maps, not omitted.

| Payload | Required fields and types |
|---|---|
| `runs.metadata` | `strategy_file: str`, `strategy_source: ObjectRef`, `executor_revision: str or null`, `executor_dirty: bool or null`, `packages: map[str, str]`, `python_version: str` |
| `decisions.input_manifest` | `parameters: ObjectRef`, `execution_context: ObjectRef`, `universe: ObjectRef`, `indicators: list[ObjectRef]`, `pricing: ObjectRef or null`, `routing_model: ObjectRef or null`, `routing_state: ObjectRef or null`, `web3: ObjectRef or null`, `other_data: Value`, `state_refs: list[Value]` |
| `source` object | `path: str`, `text: str` (strategy source only; exclude secret files) |
| `parameters` object | `values: map[str, Value]` (resolved `StrategyParameters` items, not its class members) |
| `execution_context` object | `mode: str`, `engine_version: str or null`, `parameters: ObjectRef or null`, `flags: map[str, Value]`, `handles: map[str, str]` (class/function identities only for non-data handles) |
| `universe` object | `metadata: Value`, `pairs: list[PairRecord]`, `reserves: list[Value]`, `frames: map[str, FrameManifest]` |
| `frame_chunk` object | `column_dtypes: list[Value]`, `index_dtypes: list[Value]`, `index_values: list[list[Value]]`, `rows: list[list[Value]]` |
| `indicator` object | `name: str`, `pair_key: str or null`, `parameters: Value`, `source: str`, `dependency_order: int`, `variations: bool`, `function: {module: str, qualname: str, source_hash: str or null, framework_body_hash: str or null}`, `universe_key: str`, `cached: bool`, `result: ResultFingerprint`, `inputs: list[ObjectRef]`, `dependencies: list[ObjectRef]`, `dependencies_complete: bool` |
| Pricing/routing/Web3 configuration object | `implementation: str`, `values: map[str, Value]`, `handles: map[str, str]`; capture only explicit decision-relevant fields, no sessions, keys or URLs containing credentials |
| `decisions.observations` | Ordered list of `Observation` below; small consumed values only, large shared inputs use object references |

`PairRecord = {pair_key: str, internal_id: int or null, identifier: Value,
extra_metadata: Value}`. The stable pair key is
`"<chain_id>:<lowercase pool_address>"`; retain the original identifier fields
and cache mapping too. `identifier` uses existing `TradingPairIdentifier`
serialisation where lossless. `extra_metadata` preserves decision-relevant
fields its serializer drops. Reserves use `AssetIdentifier` serialisation.
Do not assume pair IDs or vault names are stable identifiers across runs.

`universe.metadata` explicitly records non-frame fields from both objects:

- `TradingStrategyUniverse`: `options`, `primary_chain`, `required_history_period`,
  `price_data_delay_tolerance`, `other_data`, `vault_history_diagnostics`,
  `vault_window_overrides`, `ignore_routing`, `backtest_stop_loss_time_bucket`
  and `pair_cache` membership/metadata. `reserve_assets` comes from its parent.
- Nested `data_universe`: `time_bucket`, `chains`, `exchange_universe`, the
  legacy `exchanges`, `forward_filled`, `vault_specs`, `start_hint`, `end_hint`.
  Capture serialisable specification data, not live vault clients.

Do not confuse these with loader `Dataset` fields that are not propagated to
the constructed universe. Capture available build/filter diagnostics explicitly
when constructing the universe: capture at `decide_trades()` is after loader
filters, but before callback selection gates. Do not claim to include vaults
already excluded by dataset construction. Do not call `asdict()` on the whole
universe. Frame keys are source paths, e.g.
`data_universe.pairs.df`, `data_universe.candles.df`,
`data_universe.liquidity.df`, `vault_state`; include lending/stop-loss frames
when present, including `data_universe.resampled_liquidity` and
`data_universe.lending_candles` data, and document absent optional fields in
metadata. Preserve cached
pair objects already carrying modified metadata without warming/mutating the
universe merely to capture it.

`FrameSchema = {container: "dataframe" or "series", name: Value,
columns: list[Value], column_dtypes: list[Value], index: Value,
column_index: Value}`. Axis descriptors include class, names, dtype(s), and
RangeIndex start/stop/step or MultiIndex levels/codes where needed; dtype
descriptors include categorical categories/order and nullable dtype identity.
Rows and columns are positional, so duplicate labels and non-string labels
survive.

`FrameManifest = {schema: FrameSchema, row_count: int,
chunks: list[FrameChunkRef]}` and
`FrameChunkRef = {object: str, row_positions: list[int]}`. The `object` field
references a `frame_chunk` hash; positions belong to this particular input
frame, not to the reusable chunk. Each reference has one position per chunk
row. Positions across references must cover `0..row_count-1` exactly once,
including for interleaved pairs or duplicate index labels. An empty frame has
its full schema, zero row count and no chunks.

Only chunk-local dtype descriptors, actual index labels and row values enter
the chunk payload/hash. Keep frame keys, absolute row positions, full-frame
RangeIndex bounds, MultiIndex levels/codes, row counts and capture times in the
frame manifest, not in chunk payloads. Encode index labels directly rather
than with full-frame-dependent level codes. Align pair/time partitions to fixed
calendar boundaries, not to the start of a rolling window. A window shift may
change boundary chunks and the universe manifest, but must reuse interior
chunks whose values, labels and dtypes are unchanged. Genuine label/dtype changes
appropriately produce new hashes. Apply the normal object-envelope hashing
rule unchanged; no special fields need to be excluded at hash time.

`ResultFingerprint = {sha256: str, length: int, non_null_count: int,
shape: list[int], schema: FrameSchema, first_index: Value, last_index: Value,
start_at: Value, end_at: Value}`. Hash the actual result values using the same
canonical frame/value rules, without persisting the full derived series.
For DataFrames non-null count is across cells; bounds refer to the time index
level when present (otherwise `null`). Empty results have null bounds.
Indicator object hashes include metadata; `result.sha256` identifies the data
alone. Set-valued inclusion cells use deterministically ordered set tags.

`dependencies` contains only known indicator object references;
`dependencies_complete` states whether this is the entire dependency set:

- `[]` with `true`: verified to have no indicator dependencies.
- A non-empty list with `true`: the complete known dependency set.
- `[]` with `false`: dependency information unavailable.
- A non-empty list with `false`: only a partial dependency set is known.

Do not infer completeness from an empty list or `dependency_order` alone.
Unknown provenance is not a recording failure; record `false` without inventing
references or adding dependency-tracing hooks. Direct source-frame references
remain in `inputs` independently of indicator-to-indicator dependencies.

`Observation = {sequence: int, kind: str, name: str, pair_key: str or null,
observed_at: Value, source_at: Value, arguments: Value, value: Value,
state_refs: list[Value], provenance: Value, error: Value}`. Sequence starts at
zero per invocation. Kinds initially are `indicator`, `pricing`, `admission`,
`calculation`. Nullable metadata stays null if the source did not supply it;
never invent observation/block times or cache provenance. State references
identify existing position/trade IDs or state field paths; they are correlation
references, not recoverable historical snapshots. Observations made during a
callback are held locally and written on finalisation; a process kill can lose
them while the committed entry manifest survives.

### Existing JSON framework and exact-value exceptions

Reuse the project's `dataclasses_json` framework and
`tradeexecutor.monkeypatch.dataclasses_json.patch_dataclasses_json()` for known
domain objects and the recorder envelope dataclasses. Use their existing
`to_dict()`/`from_dict()` or `to_json()`/`from_json()` contracts where applicable;
store JSON objects, not double-encoded JSON strings. Do not introduce Pydantic,
a separate JSON Schema framework or a generic Python object loader.

Source evidence and boundaries:

- `state/identifier.py`: `AssetIdentifier` and `TradingPairIdentifier` already
  use `@dataclass_json`. The latter's `_reduce_other_data` field encoder drops
  exactly `token_metadata`, `token_risk_data`, and `lagoon_compat_check_data`:
  explicitly capture relevant omitted decision inputs alongside the normal
  representation rather than assuming `to_dict()` is a full snapshot. It also
  deletes keys in-place, and the installed `dataclasses_json.core._asdict()`
  passes fields with custom encoders directly to them. Serialise a detached
  pair/metadata copy, never the live object; test that recording leaves its
  nested `other_data` unchanged. Do not deep-copy network/client handles.
- `strategy/trade_pricing.py`: `TradePricing` also uses `@dataclass_json`.
  Reuse field conventions for relevant projections, but reference state-owned
  pricing results instead of copying them and do not capture unused fields.
- `state/state.py:State.to_json_safe()` uses the patched encoder and validator.
  Reuse conventions, not the full state serialisation operation or its payload.
- The patched encoder emits datetimes/timedeltas as numeric seconds, Decimal
  as strings and enums as values. Those typed object contracts are useful but
  do not by themselves preserve nanosecond frames, arbitrary mapping key types,
  null distinctions or container types required by fingerprints.

For frames, untyped parameters/diagnostics and fields where the domain codec
loses fidelity, use recorder-local `dataclasses_json.config` field codecs and
the following small exact-value representation. Do not change the global state
encoder or pass raw pandas frames to it. Decode known types explicitly, never
import arbitrary classes named by recorded data.

| Python value | Recorder JSON representation |
|---|---|
| `None`, bool, string, integer, finite Python float | Native JSON; retain integer precision and finite float round-trip (including negative zero) |
| `Decimal` | `{"$type":"decimal","value":"1.2300"}` |
| `pd.Timestamp` | `{"$type":"timestamp_ns","value":"1789948800000000001"}` (integer epoch nanoseconds as a string, naive UTC) |
| Python datetime | `{"$type":"datetime","value":"2026-09-21T00:00:00.000001"}` (naive UTC) |
| Timedelta | `{"$type":"timedelta_ns","value":"1000000000","python_type":"pandas"}`; distinguish Python and pandas types |
| NaN, +/-Infinity, `pd.NA`, `pd.NaT` | `{"$type":"missing","value":"nan"}` with values `nan`, `pos_inf`, `neg_inf`, `pd_na`, `nat`; never merge with JSON null |
| Enum | `{"$type":"enum","class":"module.QualifiedName","value":Value}`; known-type allow-list for decoding |
| Tuple or set | `{"$type":"tuple","items":[...]}` or `{"$type":"set","items":[...]}`; sort set items by canonical encoded bytes |
| Non-string-key map or map containing reserved `$type` | `{"$type":"mapping","items":[[key,value],...]}` preserving key types and iteration order |
| NumPy scalar outside a typed frame | `{"$type":"numpy_scalar","dtype":"float32","value":Value}`; frame dtype describes homogeneous cell types |

Keep ordinary string-key maps queryable as JSON objects; require an explicit
mapping tag where iteration order affects the decision. Add codecs only for
types encountered in the coverage inventory; unsupported values fail loudly.
Canonical hashes operate on the explicitly encoded representation, not on
`hash()`, `repr()` of arbitrary objects, or lossy state JSON. Test domain-object
round-trips separately from exact frame/value round-trips.

### Source-to-schema checks and readback example

The `StrategyInput` fields in `pandas_trader/strategy_input.py` map as follows:
`cycle`/`timestamp` to typed decision columns; `state` to state path and IDs;
`strategy_universe`, `parameters`, `indicators`, `execution_context`,
`pricing_model`, `routing_model`, `routing_state`, `web3`, `other_data` to the
manifest fields above. The proposed recorder field is an executable handle,
not recursively recorded data.

`StrategyParameters` is a mapping, not a `dataclass_json` object: use
`dict(parameters)` or `iterate_parameters()`. Preserve effective framework
keys such as `grid_search` as well as strategy keys; do not silently filter
them. `execution_context.flags` and `.handles` are recorder projections, not
attributes on `ExecutionContext`: project `grid_search`, `optimiser`, `jupyter`,
`progress_bars`, `force_visualisation` into flags, and the identity of
`timed_task_context_manager` into handles. Its `parameters` mapping uses the
same object reference when identical, or a separate reference in the context
payload when different; never silently lose a differing context configuration.

`IndicatorKey.pair` can be null for universe-level indicators.
`IndicatorDefinition` supplies name, func, parameters, source, dependency_order
and variations. `IndicatorResult` supplies universe_key, indicator_key, data
and cached. Dependency references are recorder metadata, not an existing
`IndicatorResult.dependencies` attribute: derive known dependencies from the
calculation definitions/resolution information. Set `dependencies_complete`
to `false` when dynamic provenance is missing or only partially known, rather
than treating it as a verified absence of dependencies.

`IndicatorDefinition.get_function_body_hash()` already supplies the framework's
bytecode identity. Reuse it as named provenance when applicable; do not label
it a SHA-256 source hash without checking its algorithm. The recorded strategy
source object supplies the separate SHA-256 content identity.

The happy-path integration test must use these column/key names and check both
existing domain codec reuse and exact-value readback. For example, after the
executor shuts down:

```sql
SELECT d.cycle, d.status, json_extract(p.payload, '$.values') AS parameters
FROM decisions d
JOIN objects p
  ON p.content_hash = json_extract_string(d.input_manifest, '$.parameters.object')
ORDER BY d.started_at;
```

## Zstd compression

Use native DuckDB storage compression while keeping the file directly queryable
as `{strategy-id}-record.duckdb`. Create/attach it with a storage version that
supports Zstd, and set `force_compression='zstd'` on the dedicated recorder
connection. Checkpoint after a committed cycle, outside active transactions.

Verified locally with DuckDB 1.5.4: a JSON payload table using
`ATTACH ... (STORAGE_VERSION 'v1.2.0')`, `SET force_compression='zstd'` and
`CHECKPOINT` reports `ZSTD` payload segments in `pragma_storage_info`. The same
test without the storage-version option reported uncompressed segments. Constant
validity segments may remain `Constant`; not every segment needs Zstd.

Add a realistic persisted-data compression check using the minimum supported
DuckDB version. Report actual database/WAL size and compression codecs. Do not
claim compression based solely on the setting, and do not wrap the active
database in a `.zst` archive.

References: [DuckDB storage versions](https://duckdb.org/docs/current/internals/storage),
[Zstd storage introduction](https://www.duckdb.org/2025/02/05/announcing-duckdb-120).
DuckDB is already a declared dependency (`>=1.4.2,<2.0.0`).

## State references without duplicate state

Keep existing state persistence unchanged. Record the authoritative state path,
cycle/time and existing position/trade IDs for correlation. Do not call `sync()`
just for the recorder, copy state into DuckDB, or add state-store revision APIs,
archives, compression or retention policies.

The latest state file is overwritten by normal persistence. These references
are not an immutable pre-decision snapshot; this version does not promise exact
historical state recovery or full decision replay.

## Indicator fingerprints

For every indicator result, include the series length (including nulls), non-null
count, shape, data/index dtypes, index names and timezone convention, first/last
index entries and minimum/maximum time. Define empty-series bounds as null.
Record the actual result the decision could access and the decision timestamp
that constrains access.

Hash a canonical, versioned representation of values, index, order, null masks
and schema. Use SHA-256 rather than Python's process-dependent `hash()`.
Fingerprint set-valued inclusion indicators deterministically. Preserve sequence
order where it is semantically meaningful. Record exact selected values and
their source timestamps when accessed by decision code; these small observation
records are useful for explaining a decision without storing derived series.

Link each indicator to captured input chunks, its resolved parameters,
dependency fingerprints and calculation-code identity. Record strategy module
source once by content hash, including dirty local source, plus executor and
dependency revisions/package versions. A Git commit alone does not identify a
dirty checkout. Retain these identifiers and captured source data for future
analysis. Hashes alone do not recreate code or indicators.

Indicator recomputation, callable-source fallback machinery, source loading and
replay verification are deferred. This change records the fingerprints and
metadata, not a reconstruction engine.

## Runtime observations and durability

Capture the immutable input manifest in a committed transaction at the start of
v8's `decide_trades()`, before its decision logic. Give every invocation a new ID even if cycle and
timestamp repeat after a restart. Use statuses `started`, `completed`, `failed`;
a crash leaving `started` is visible evidence of an interrupted decision.

These statuses cover attempts after the input manifest commit. Treasury-sync
failures, indicator-calculation failures, runner assertions before `StrategyInput`
construction, and capture failures before commit remain visible through existing
executor error reporting, not a fabricated complete input record. Callback early
returns are recorded as `completed` with no planned trades. A failed recorder
commit propagates and crashes the executor before decision logic or before
planned trades are returned, as appropriate.

Use direct recorder calls inside `decide_trades()` and its calculation helpers
at the places that consume indicator values, pricing results and other decision
inputs. A small framework context manager can commit entry/completion records
and handle early returns and exceptions, but is invoked explicitly inside
`decide_trades()`, not installed as a runner or model hook.
Record decision-relevant arguments, consumed values or error, sequence number,
logical time and observation time, with public block/cache provenance where
already available. Project only relevant fields from existing results; never
persist full responses or provider caches. If a calculation helper hides needed
numbers, pass the recorder explicitly to that helper or expose its diagnostics
to the caller. Do not introduce a generic observation mechanism or repeat API
calls to obtain them.

For v8, capture the indicator values, consumed pricing results and
`check_deposit()` outcomes (`can_deposit`, `reason_code`) actually used, not a
wish-list of all fields on framework/API objects. If called pricing/calculation
helpers use maximum withdrawable/depositable amounts, lock-up expiry, leader
fraction, equity or share prices internally, capture those consumed values
directly in those helpers with the explicit recorder API; otherwise omit them.
Do not confuse the backtest-only redemption-accounting helper with live logic.
Capture
calculation inputs/results such as inverse-volatility weights, rank and momentum
gates, pool limits, target allocations, cash available for buys, pending
settlement capital, thresholds and rounding where these are not already in state.
Reference state-owned balances or trade fields instead of copying them.

Do not store unrelated vault details, follower lists, API portfolio histories or
other unused response fields. Market history that is actually part of the
constructed universe remains required. Preserve repeated calls when consumed
values differ and include cache-hit provenance. Do not re-query a service to
manufacture a snapshot: capture values from the result used in the calculation.

`decide_trades()` finalises observations and commits completion before returning
planned trades. On an exception, retain committed inputs and attempt to save
collected observations and mark the invocation failed, then propagate the error.
If storage itself has failed, a `started` row or existing executor logs may be
all that survives; do not promise an error record on a broken database.

Let recording failures crash the executor. This can stop trigger checks and
other executor work until normal operator recovery/restart; that is the accepted
failure policy. No silent fallback, independent trigger loop or special recovery
mode is required. Recording cannot undo already-broadcast transactions.

Use one connection owned by the executor process, closed on normal shutdown.
Separate schema initialisation from per-cycle transactions. Additional storage
safety, live-copy tooling and concurrent research access are out of scope.
No background service, unbounded queue or automatic deletion.

## Module layout

```text
tradeexecutor/strategy/recorder/
    __init__.py       # small public recorder API
    recorder.py       # lifecycle context and direct recording API
    storage.py        # DuckDB schema, path, transactions, deduplication, compression
    serialisation.py  # canonical JSON and stable fingerprints
    universe.py       # constructed universe capture
    indicators.py     # definitions, fingerprints and consumed-value metadata
    observations.py   # serialisation of explicitly supplied decision diagnostics
```

Keep classes and interfaces minimal. Reuse existing serializers only where they
round-trip accurately. Do not recursively serialise arbitrary runtime objects.
The recorder retains no wallet credentials or network connections.

## Framework and v8 integration

1. Apply `@record_decision` to v8's decision callback. Live bootstrap detects
   the decorator marker and enables recording automatically. Strategy parameters
   remain reserved for strategy logic; no recorder flag is added to them.
2. Wire the recorder from resolved CLI ID and state path through the execution
   loop to an optional recorder field on `StrategyInput`. Initialise only for
   enabled live trading. In backtests/notebooks leave it absent regardless of
   the decorator; create no database and require no recording destination.
3. The decorator wraps v8's `decide_trades()` in the recording lifecycle
   if the recorder is present. Capture inputs immediately, record consumed
   values/calculations directly as they are used, and commit completion before
   returning. When absent, use the same decision logic without recording.
   Do not add runner, pricing-model, routing or indicator-access observer hooks.
4. Update `/Users/moo/code/strategies/strategy/hyper-ai-v8.py` to use the framework
   recorder. Remove its separate decision-file
   writer and `HYPER_AI_V8_MANIFEST_DIR`; log the framework decision ID instead.
   Keep independent universe allow-list behaviour if still used by research,
   and document that it is distinct from recorded-input reconstruction.
5. Audit v8 diagnostics against state ownership. Keep persisted closed-entry
   tables/unallocatable signals in the state; store references only. Capture
   transient ranks, gate outcomes and calculations that currently disappear.
6. Update operator docs with file location, enabling, live-only scope, accepted
   crash behaviour and simple JSON SQL query examples. State explicitly that
   historical state archives and replay/reconstruction tooling are not included.

## Implementation and verification order

1. Complete the field/ownership inventory and identify direct recording sites
   inside `decide_trades()` and its calculation helpers.
2. Implement canonical serialization, storage and representative universe
   round-trip. Validate native JSON queries and actual Zstd storage.
3. Add indicator fingerprints, lengths, time bounds and source-data references.
4. Pass the configured recorder on `StrategyInput`, integrate direct recording
   into v8's `decide_trades()`, and retire its separate JSON writer.
5. Test a representative live-mode v8 decision using deterministic local external
   responses. Verify that consumed numbers/flags and calculation results are
   recorded, while full API bodies and irrelevant fields are absent. Include
   early returns and a raised exception. Test restart with the same logical
   timestamp; both invocations must survive.
6. Compare recording on/off on the same deterministic live-mode decision:
   selected assets, planned trades and sizing must be identical. Verify that
   a recording error propagates rather than returning trades for execution.
   Check non-live mode creates no recorder/files even with the flag enabled;
   do not add backtest recording support or require a backtest run.
7. Reopen the database and query stored JSON, universe chunks and fingerprints.
   Verify exact serialisation round-trips without building a replay engine or
   retaining new state revisions. Cover the complete happy path through the
   Typer integration test below, not just isolated recorder unit tests.
8. Measure capture latency, peak memory, initial/history growth and database/WAL
   size on the actual HyperCore universe. Validate unchanged chunks are reused
   and one changed historical row invalidates only the affected chunk/manifest.
   Shift a rolling frame's start so retained rows move to new absolute positions:
   unchanged interior chunk hashes must survive, and both manifests must restore
   their own exact row/index order. Check empty and interleaved frames too.

Test dependency metadata readback for all four list/completeness combinations
above, so unavailable or partial provenance cannot be mistaken for a complete
dependency set.

Keep focused tests grouped around: exact round-trip and deduplication; lifecycle
and recording failure; indicator fingerprints; state ownership; live v8 parity.
Include timestamps, Decimal precision, NaN/NaT, index order, same-name different
vaults, no candidates, serialisation errors and corrupted/missing references.

## Typer integration test: one-second live cycles

Add one happy-path integration test in
`tests/cli/test_strategy_input_recorder.py`. Invoke the real Typer `start`
command with `typer.testing.CliRunner`, exercising CLI configuration, state-path
resolution, the execution loop, `StrategyInput`, direct recording inside
`decide_trades()`, and actual DuckDB persistence. Do not call the recorder alone
or mock its serialisation, hashing or database writes.

- Use `CYCLE_DURATION=1s` (`CycleDuration.cycle_1s`), `MAX_CYCLES=3`,
  `UNIT_TESTING=true` and an isolated `tmp_path` state/cache directory. Exercise
  the live/unit-testing-trading path, not backtesting. Ensure this live test mode
  can enable the recorder while backtesting remains disabled. Bound the test
  with a timeout; assert three completed cycles rather than exact wall-clock
  spacing, which can vary with scheduling and capture time.
- Use a small fixture strategy with recording enabled and direct recorder calls
  in `decide_trades()`, following the same framework API as v8. Supply a
  deterministic local vault universe, indicators and decision observations.
  Mock external data/RPC responses and use a safe local execution fixture so
  there are no live funds, network dependencies or production credentials.
  Keep the real CLI, loop, recorder and state persistence in the test path.
- Populate every applicable capture-contract category with known values:
  resolved parameters/context; pair/asset/reserve metadata and ID mappings;
  candle, TVL and vault-closure history; indicator definitions, hashes, lengths
  and date bounds; state references; serialisable pricing/routing configuration;
  `other_data`; and consumed prices, admission flags, ranks, weights and sizing
  diagnostics. Include a filtered-out vault to prove capture occurs before
  selection. Document genuinely absent optional inputs instead of pretending
  the fixture covers them.
- Include representative Decimal values, nanosecond timestamps, nulls and frame
  ordering. Reuse some data across cycles and change a known input in a later
  cycle, so the test checks both deduplication and preservation of earlier data.
  Include a successful no-trade cycle and a cycle producing locally handled
  planned trades, to check completion and state-owned trade-ID references.
- Require a successful CLI exit and normal shutdown. Assert the expected
  `{strategy-id}-record.duckdb` exists beside the configured state file, and
  normal state persistence still works without new state archives.
- After the executor closes its connection, open a fresh DuckDB connection.
  Inspect `runs`, `decisions` and `objects` with SQL: all three decisions are
  completed with distinct invocation IDs, the expected cycle/time metadata,
  and resolvable object hashes. Query fields through native JSON extraction;
  merely checking that the file or rows exist is insufficient.
- Read back and decode the stored payloads with the framework serialisation
  helpers. Compare them with the fixture's known universe, frames, parameters,
  fingerprints and consumed values, including types, precision, nulls, ordering
  and per-cycle changes. Check state/trade references against the normal state
  file, and that full API bodies, secret sentinel fields and duplicate state
  snapshots were not recorded. This is data readback, not strategy replay or
  indicator reconstruction.
- Include a sufficiently sized, realistic JSON history payload to check actual
  Zstd payload segments after checkpoint using `pragma_storage_info`; tiny or
  constant metadata segments need not report Zstd.

Keep this as one end-to-end happy-path test, with focused failure tests separate.
No new inspection CLI, live database-copy mechanism or backtest recorder is
needed: SQL plus payload decoding demonstrates that recorded data is inspectable
and readable after shutdown.

## Acceptance criteria

- An enabled v8 live run automatically records to the required filename beside
  the authoritative state file, including empty and failed callback attempts
  after the input manifest commit. Pre-boundary failures retain existing error
  reporting without claiming a complete recorded input.
- The constructed universe and all non-state data consumed by the decision are
  recoverable, with a documented coverage map and no silent omissions.
- Every calculated indicator has a stable hash, length and time bounds, plus
  source-data/code references for future analysis. Reconstruction is deferred.
- Rolling-window placement changes do not invalidate otherwise unchanged
  interior chunks; each frame manifest retains its own exact ordering.
- Indicator dependency metadata distinguishes verified-empty, complete,
  unavailable and partial dependency sets explicitly.
- Existing state-owned data is referenced, not copied into DuckDB. No new state
  archives or revisions are created. Normal state persistence is unchanged.
- Python payloads are queryable DuckDB JSON; representative large payload
  segments demonstrably use Zstd after checkpoint.
- Live observations contain the decision-relevant values actually consumed,
  including calculation inputs/results absent from state. Full API response
  bodies and unused response fields are not recorded.
- Successful recording does not change strategy decisions. Recording failures
  propagate and may crash the executor before new trades are handed to execution.
- The storage and serialisation machinery lives in the framework recorder
  package; v8 invokes it directly inside `decide_trades()` and its calculation
  helpers. No observer hooks are introduced.
- Recording is live-only; backtests/notebooks create no recorder or files.
- A Typer `start` integration test completes three one-second live test cycles
  and verifies the recording happy path end to end, including SQL inspection
  and exact payload readback from a newly opened DuckDB connection.
