# HyperCore data-availability trigger

Use `STRATEGY_CYCLE_TRIGGER=hypercore_data_available` for Hyper-AI's two-day
calendar-aligned decisions. This mode waits for published data instead of
letting process restarts or trade execution duration move the decision clock.
The strategy must load vault prices with the framework's `UniverseOptions` so
it consumes the verified private snapshot and respects its cutoff.

## What readiness means

From the logical midnight slot, the executor polls the authenticated, uncached
vault scan JSON manifest on a 15-minute grid, for at most eight hours. It does
not calculate indicators or download parquet on these polls. HyperCore's scan
completion and cleaned-file maximum timestamp must both reach the slot.

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

The first slot is gated **before universe warm-up**. Starting during an open
window joins it; starting after an unclaimed window waits for the next slot.
Consequently, startup accounting checks, chart setup and background valuation
jobs do not start until this initial wait finishes. This is intentional in the
current implementation, not continuous valuation coverage during startup.
Subsequent readiness waits also occupy the scheduler's single worker, delaying
valuation, Lagoon NAV/settlement and position-trigger jobs for the wait's
duration. This differs from keeping background services running normally.
Resolving it safely needs a separate readiness scheduling design; simply adding
concurrent workers would permit unsafe concurrent state mutations.
`TRADE_IMMEDIATELY` and `PRELOAD_WEBHOOK_DATA` cannot bypass the gate.

The pending logical slot is saved before polling. Restarting an unexpired slot
resumes it; an expired pending slot requires operator recovery. A slot that
already created trades must be reconciled instead of replayed. Completion and
clearing the pending slot are persisted together after a successful tick.
There is still a crash window between trade persistence and that completion
write. If a pending slot contains completed trades, both restart and the
abandonment command refuse it, even after ordinary transaction repair. There is
currently no supported command to mark such a slot completed. Do not blindly
clear the flag or automatically infer completion: some intended trades may
never have executed. A reviewed explicit recovery procedure is needed.

The background-service pause and completed-trade recovery gap are unresolved
deployment blockers for this trigger, not guarantees provided by this runbook.

Shutdown interrupts polling waits through an event; an in-flight HTTP request
still relies on its network timeout. Process-private price files are removed
when the live loop exits. This is not an archive of historical parquet inputs.

## Recovering an expired slot

Stop the executor and use the deployment's Compose maintenance shell, as
described in [the Docker runbook](docker.md). Confirm the authoritative state
path, then run:

```shell
poetry run trade-executor abandon-hypercore-slot --state-file state/hyper-ai.json
```

This local-only command backs up the JSON state, writes an audit message and
sets the next future **two-day** midnight slot. It does not mark a missed cycle
as executed and does not replay trades. It refuses an open readiness window,
a slot with trades or outstanding transaction repair. The command is specific
to Hyper-AI's two-day schedule; do not use it for another cycle duration.

## Deployment dependencies

Deploy the eth-defi manifest publisher and the frontend's authenticated
`/vaults/datasets/download/vault-scan-manifest` endpoint before enabling this
trigger. The executor also needs the matching trading-strategy client and a
valid `VAULT_PRO_API_KEY`. A branch passing unit tests does not demonstrate that
the production endpoint is deployed or that its receipts are fresh.

Coverage is in `tests/strategy/test_hypercore_data_availability.py` (polling,
deadlines and publication races), `test_hypercore_manifest_contract.py` (actual
producer/consumer and sparse data), and
`tests/cli/test_cli_abandon_hypercore_slot.py` (real Typer recovery entry point).
