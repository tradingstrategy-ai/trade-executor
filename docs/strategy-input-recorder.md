# Live strategy-input recorder

The strategy-input recorder preserves the inputs and explicit calculations of a
live v0.5 pandas-runner strategy decision. It is intended for investigating why
a live decision differed from a backtest; it is not a state archive or a replay
engine.

## Enable it

Set this boolean strategy parameter:

```python
class Parameters:
    record_strategy_inputs = True
```

The executor uses one DuckDB file per executor ID. Its directory is the
configured state file's directory and its name comes from `EXECUTOR_ID`:

```text
STATE_FILE=state/custom-state.json
EXECUTOR_ID=hyper-ai-v8

state/hyper-ai-v8-record.duckdb
```

The state file's stem does not affect the recorder name. Each executor process
appends one `runs` row to that file. A live run requires a persistent state
path; backtests and notebooks never create a recorder file.

## Record a decision

The live pandas runner constructs `input.recorder` when enabled. Strategy code
uses `@record_decision` to place the lifecycle around the complete callback:

```python
from tradeexecutor.strategy.recorder import record_decision


@record_decision
def decide_trades(input: StrategyInput) -> list[TradeExecution]:
    return _decide_trades(input)
```

Record decision-relevant intermediate values inside `_decide_trades()` or its
helpers:

```python
if input.recorder is not None:
    input.recorder.record("signal", "ranked_vaults", ranked_vaults)
    input.recorder.record("allocation", "target_weights", target_weights)
```

The decorator calls `begin()` before strategy calculations, `finish()` with the
returned trades, and `fail()` before re-raising a callback exception. When
`input.recorder` is absent, including backtests, it invokes the callback without
recording. Recorder errors are intentionally fatal; the executor also closes
the recorder when a strategy run exits through an exception.

## Stored data

The file has three native DuckDB tables:

| Table | Contents |
| --- | --- |
| `runs` | Strategy source reference, strategy and executor identities, Python and package versions |
| `objects` | Content-addressed JSON objects: source, parameters, execution context, constructed universe, universe frames, and indicator definitions/fingerprints |
| `decisions` | Cycle, timestamps, state path, input manifest, lifecycle status, observations, output trade IDs, and any exception summary |

The universe manifest contains the constructed pair and vault metadata plus
content-addressed chunks of decision-time universe frames. Indicator records
contain the definition, parameters, function hashes where available, result
hash, length, shape, non-null count, and first/last and time-range values.

The normal state file remains authoritative for portfolio and trade history. A
recorder decision keeps only the state file path plus open-position, pair, and
trade IDs needed to correlate it.

The recorder uses native DuckDB `JSON` columns and requests Zstandard
compression. Objects with identical content are stored once and referred to by
their SHA-256 hash.

It deliberately excludes credentials, RPC/provider handles, full API responses,
and state-file copies. It does not reconstruct or replay a decision.

## Operational constraints

The database retains decision history and grows as live decisions add new
universe or indicator content. Monitor it as part of the executor's normal
disk-space operations. DuckDB has a single writer, so do not run overlapping
executors with the same `EXECUTOR_ID` and recorder file.

## Inspect a completed run

Open the file with DuckDB after the executor has checkpointed or stopped:

```sql
SELECT cycle, decision_at, status, observations, error
FROM decisions
ORDER BY started_at;

SELECT kind, count(*) AS object_count
FROM objects
GROUP BY kind
ORDER BY kind;

SELECT
    d.cycle,
    json_pretty(json_extract(p.payload, '$.values')) AS parameters
FROM decisions AS d
JOIN objects AS p
    ON p.content_hash = json_extract_string(
        d.input_manifest,
        '$.parameters.object'
    )
ORDER BY d.started_at;
```

After a checkpoint, `pragma_storage_info('objects')` exposes the persisted
compression codecs.
