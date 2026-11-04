# Ride out a lagging RPC read in Lagoon settlement preflight

Date: 4 November 2026. Status: settlement change implemented; alert, release and
executor upgrades pending. This plan authorises no release, production restart or
transaction.

## Objective

A settlement simulation that fails because an RPC node has not yet seen the NAV
post must not stop the executor. Retry it with backoff, then fail as today. Make sure a
stopped executor reaches an operator.

## Source baseline

Executor master `3a4f794280a93f054824070de19e5b5f1c7c2dd9`, pinning eth-defi
`7d4c1cea9d73332897fd9d08f77cb04dc4b259b5`. Newest release tag v1473. #1699 is
merged but untagged.

## Incidents

**lighter-ai, 2026-11-01 12:00 UTC, v1460.** Lighter `/api/v1/account` returned 503
for about 20 seconds. The session gave up after 5 retries and the valuation refresh
stopped the loop. #1699 retries for 10 minutes and needs no further change here.

**hyper-ai, 2026-11-03 16:00 UTC, v1473.** NAV tx `0x12cab872…5803` succeeded at
block 50289249. Thirty blocks later `_preflight_lagoon_settlement()` reverted with
`0x87d895da`, `NewTotalAssetsMissing()`. Guarded modules re-raise unknown selectors,
so the read-only simulation stopped the loop.

The stale-read explanation is inferred: on-chain `newTotalAssets` is still
49266430820, the vault emitted no later log through block 50289999, and replaying
the call on latest state succeeds. No provider response was logged, so the serving
node is unknown. `_preflight_lagoon_settlement()` already pins its provider on Anvil
for exactly this reason, but not in production. A revert is a valid RPC response,
so `FallbackProvider` does not fail over on it.

## Change

In `tradeexecutor/ethereum/lagoon/vault.py`, extend the `except` in
`_preflight_lagoon_settlement()`.

Add two selectors beside the existing ones, derived with `Web3.keccak`:
`NewTotalAssetsMissing()` and `WrongNewTotalAssets()`. The second is what a
lagging node returns when it still holds an older pending NAV, which hyper-ai
will have on restart.

When a guarded module returns either selector, retry the same `eth_call` with the
policy of the executor's Lighter session from #1699: exponential backoff of 0, 1,
2, 4 ... seconds, at most 120 seconds per sleep, for 10 minutes after the first
failure. `LoggingRetry` itself cannot do this, because a revert is not an HTTP
failure. Log each retry with the active provider name from
`web3.get_active_call_provider()`, so the next occurrence identifies the node.
If a retry passes, continue to settlement. If a retry returns any other result, such
as a GuardV0 limit once the node has caught up, the existing classification handles
it, so the retry wraps the `eth_call` rather than living inside that `except`. If all
retries fail, the last exception reaches the caller unchanged: the executor stops as
it does today, with 10 minutes of evidence in the log. Both selectors can be genuine,
for example after a direct Safe settlement, so exhaustion is not treated as success.

Unrelated selectors and modules without settlement safety keep their current paths.

Do not pin the call to the NAV block. The public HyperEVM RPC ignored the block
argument of `eth_getStorageAt` in testing, so block pinning needs per-provider
verification that this change does not require.

## Alert

Both crashes set `exception` and `executor_running: false` in `/status`, and
`scripts/check-webhook-status.sh` already turns that into `unhealthy`. Nothing
acted on it for 70 hours.

Before the release, check whether the crash reached Discord. `die()` logs at ERROR
and the Discord handler accepts TRADE and above, and hyper-ai's log shows that
handler active, so the notification probably went out and was missed. If so, the
fix is where the channel is watched, not more monitoring. If not, add an external
check of `executor_running` and `exception` on each public executor hostname; it is
stateless, so an uptime monitor can do it. Clear up why `master-vault`, `vega` and
`base-ath` also report `unhealthy` first.

## Tests

One unit test module, `tests/lagoon/test_lagoon_preflight_stale_read.py`, with a fake provider raising the production
`ExtraValueError({'code': 3, 'message': 'execution reverted', 'data': '0x87d895da'})`
and the clock and sleep patched out:

- stale once, then success: settlement proceeds;
- `WrongNewTotalAssets()`: same;
- every retry fails: the exception propagates after the full backoff schedule;
- stale, then a GuardV0 window limit: the existing manual settlement deferral applies;
- an unrelated selector on a guarded module: raises after one attempt;
- a module without settlement safety: existing empty-queue warning path unchanged.

Run with the pinned submodule and the worktree `PYTHONPATH` rule.

## Sequence

1. Resolve the alert question above.
2. One `fix:` PR from a master worktree. Add a line to the GuardV0 table in
   `.claude/docs/lagoon-treasury-settlement.md` and to `CHANGELOG.md`.
3. Merge, run `scripts/release.sh`, verify the image revision.
4. Upgrade hyper-ai per the strategies `AGENTS.md` procedure. Its first sync will
   post a new NAV over the pending 49266430820; watch that settlement in the log.
5. Upgrade lighter-ai separately. v1460 to this release crosses the Lighter
   sub-account and flow-neutral profit changes, which need their own check against
   its live state.

## Not done, and why

- Deferring a failed valuation refresh instead of stopping. #1699 covers the
  observed outage; a longer one stops decision ticks too, so a refresh-only
  deferral would protect little and needs a failure classifier, a streak counter
  and two-pass bookkeeping in `live_positions()`.
- `restart: on-failure`. The process stays in web-server wait mode and never exits.
- More RPC providers. A revert does not trigger failover.

Restarting either executor on its current image before this release is a separate
decision.
