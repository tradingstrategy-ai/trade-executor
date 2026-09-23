# HyperCore data-availability trigger

Use `STRATEGY_CYCLE_TRIGGER=hypercore_data_available` for Hyper-AI's two-day
calendar-aligned decisions. Trade decisions wait for published data instead of
letting process restarts or trade execution duration move the decision clock.
The strategy must load vault prices with the framework's `UniverseOptions` so
it consumes the verified private snapshot and respects its cutoff.

## What readiness means

From the logical midnight slot, the executor polls the authenticated, uncached
vault scan JSON manifest on a 15-minute grid, for at most eight hours. These
decision polls do not calculate indicators or download parquet until the data
is ready. HyperCore's scan completion and cleaned-file maximum timestamp must
both reach the slot.

HyperCore observations can be four hours apart. This gate establishes that the
published chain history crosses midnight; it does **not** guarantee an hourly
observation or a fresh row for every vault. The strategy's per-vault freshness
and eligibility rules remain necessary. Empty resampling buckets carry the
last close; a synthetic gap is not evidence that a new observation arrived.

After readiness, the executor downloads the price file into a process-private
directory and verifies the response ETag against the receipt. One publication
race gets an immediate JSON recheck, then polling resumes on the original grid.
The universe excludes all observations at or after the logical decision slot
(`end_at = slot - 1 microsecond`), including a partially formed current day.
The receipt metadata is included in decision recorder inputs before strategy
execution, when the strategy enables the recorder decorator.

Each JSON request has a five-minute budget capped by the remaining window.
Late JSON responses are rejected. Socket inactivity timeouts can delay error
delivery while a read is blocked; this is not a hard process-kill deadline.
A matching parquet transfer that started before the deadline may finish after
it. Authentication, missing endpoint and invalid schema errors fail visibly;
transient JSON/network failures retry without a stale-data fallback.
Verified parquet transport failures currently fail the process; they are not
classified as retryable JSON failures. Restart within the same unexpired
pending slot to try again. Do not catch all runtime errors as retryable:
missing/weak ETags and other deployment defects must fail visibly.

## Startup and shutdown

Start-up fetches the latest manifest once, downloads its ETag-verified price
file, and builds a universe immediately, even when the next decision slot is
in the future. This surfaces manifest, transfer and universe-construction
failures on restart. The start-up universe may include partial current-day
data; it is used for accounting, charts and position valuation, **not** for a
trade decision. Strategy indicators and decision logic are still calculated
only after a slot-ready snapshot is downloaded and its universe rebuilt.

Starting during an open window joins it; starting after an unclaimed window
builds the current universe and schedules the next slot. Background valuation
jobs start after the normal warm-up and accounting checks, at their configured
cadence. The first and subsequent decisions perform one readiness probe per
scheduled job. If data is not ready, the job schedules the next quarter-hour
probe and returns, leaving the single worker free for valuation, Lagoon
NAV/settlement and position-trigger jobs between probes. Existing universe
data is retained until a new snapshot is ready. HTTP requests and a ready
snapshot's download still occupy that worker while in flight.
No concurrent state-mutating workers are introduced.
`TRADE_IMMEDIATELY` and `PRELOAD_WEBHOOK_DATA` cannot bypass the gate.

The pending logical slot is saved before polling. Restarting an unexpired slot
resumes it; an expired pending slot with no trades is logged and automatically
replaced by the next future slot on the strategy's cycle grid. A slot that
already created trades must be reconciled instead of replayed. Completion and
clearing the pending slot are persisted together after a successful tick.
There is still a crash window between trade persistence and that completion
write. If a pending slot contains completed trades, restart refuses it, even
after ordinary transaction repair. There is
currently no supported command to mark such a slot completed. Do not blindly
clear the flag or automatically infer completion: some intended trades may
never have executed. A reviewed explicit recovery procedure is needed.

The completed-trade recovery gap remains an unresolved deployment blocker for
this trigger, not a guarantee provided by this runbook.

There is no long start-up polling wait to interrupt. An in-flight HTTP request
still relies on its network timeout. Process-private price files are removed
when the live loop exits. This is not an archive of historical parquet inputs.

## Restart after a data timeout

The eight-hour timeout still fails the process visibly. Restarting requires no
state-editing command when the missed decision created no trades: start-up logs
the skipped slot and saves the next future slot before building its universe.
It does not mark the missed decision as executed or catch up missed trades.
This uses the configured cycle duration rather than a hard-coded two-day
recovery policy.
Trade-bearing pending decisions still require reconciliation as described above.

## Deployment dependencies

Deploy the eth-defi manifest publisher and the frontend's authenticated
`/vaults/datasets/download/vault-scan-manifest` endpoint before enabling this
trigger. The executor also needs the matching trading-strategy client and a
valid `VAULT_PRO_API_KEY`. A branch passing unit tests does not demonstrate that
the production endpoint is deployed or that its receipts are fresh.

Coverage is in `tests/strategy/test_hypercore_data_availability.py` (polling,
deadlines, restart selection, trade replay protection and publication races)
and `test_hypercore_manifest_contract.py` (actual producer/consumer and sparse
data). Full manifest-triggered `start` lifecycle coverage is still missing.
