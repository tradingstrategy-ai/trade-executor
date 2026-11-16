# T3tris async deposit and redemption checks

## Goal

Use the existing Lagoon-style framework for T3tris closed-mode deposits and
redemptions in backtests and live checks. Both operations are queued; pending
redemption proceeds cannot fund another purchase. Implementation and validation
are complete; the dependency revisions are published in supporting PRs.

Reuse the current eth-defi manager, executor routing, tickets, State persistence,
settlement retry, pricing checks and accounting. No new framework, adapter,
router, state model, configuration option or fallback is needed. The dependency
loader must expose the producer's existing settlement estimate and event fields.

## Existing support and the actual gaps

Initial source inspection on 15 November 2026 found the following gaps, now
addressed by the implementation described below:

| Area | Existing support / missing wiring |
| --- | --- |
| Dependency | [eth-defi PR #1784](https://github.com/tradingstrategy-ai/web3-ethereum-defi/pull/1784) adds T3tris deposit settlement events and estimated settlement intervals. The pinned dependency already has the T3tris request/claim manager and Guard support, but predates this timing change. |
| Pair classification | `tradeexecutor/state/identifier.py` omits `t3tris_like` from `ASYNC_VAULT_FEATURES`. The delayed-redemption set derives from it. T3tris detection does not necessarily include the generic ERC-7540 feature. |
| Timing rules | `tradeexecutor/strategy/vault_protocol/__init__.py` does not opt T3tris into settlement estimates or historical deposit-event gating. |
| Backtest event gate | `BacktestExecution._requires_historical_deposit_settlement()` duplicates the Lagoon/D2 feature selection instead of consuming the existing resolved capability. |
| Live checks | `check_live_deposit()` already calls `can_create_deposit_request(owner)`. `VaultPricing` already checks `can_create_redemption_request(owner)` and avoids claim-only ERC-4626 maxima for async managers. |
| Live execution | `VaultRouting` and `settlement_retry` already select manager capabilities, persist/reconstruct tickets and complete claims. Upstream `setup_guard()` already resolves T3tris descriptors and requires Guard internal version 5 or newer. |

## Minimal implementation

1. **Use the existing upstream support.** Select a merged eth-defi revision
   containing #1784 and update the existing submodule gitlink if needed. Verify
   the merged revision rather than relying on the pre-merge PR-head SHA. No new
   dependency or copied protocol code is required.
2. **Classify T3tris as queued.** Add `ERC4626Feature.t3tris_like` to
   `ASYNC_VAULT_FEATURES`. The derived delayed-redemption classification then
   covers live allocation and backtest exits without another feature set.
3. **Extend the existing overlay dispatcher.** Add T3tris to `event_gated` and
   `uses_estimate` in `resolve_vault_protocol_rules()`. Reuse the current
   `VaultProtocolRules` fields and pair snapshot. A new `t3tris.py` overlay module
   is unnecessary because there are no additional defaults or manual corrections.
4. **Use that rule in the existing backtest gate.** Keep the non-vault early
   return and read
   `resolve_vault_protocol_rules(pair).deposit_requires_settlement_event` in
   `_requires_historical_deposit_settlement()`. This removes the duplicate
   protocol list without adding a helper or registry. Resolve this flag afresh:
   an old saved rule may default it to false, while current Lagoon/D2 behaviour
   gates deposits by features. Leave saved request timing and economic terms alone.
5. **Keep live execution on its current path.** Verify T3tris reaches the public
   deposit/redemption pricing checks and shared request/claim lifecycle. Change
   live code only if a concrete failure is demonstrated; do not add T3tris
   branches to pricing, routing, settlement, deployment callers or accounting.
6. **Update the existing documentation briefly.** Add T3tris to the lifecycle
   code/test map and overlay timing description. No separate lifecycle guide.

The initial estimate covered the dependency gitlink and the three executor
files above, plus focused tests and short documentation edits. A discovered
adapter defect belongs upstream; it is not a reason to duplicate its behaviour
inside the executor.

## Preserve the existing behaviour

- Apply Lagoon's timing precedence: explicit per-vault override, positive
  `estimated_settlement`, then the existing global two-day default. The single
  estimate applies to both directions. A zero override remains queued until a
  later resolver cycle; missing/non-positive metadata is not instant settlement.
- Only deposits require historical settlement events. The upstream marker
  contains `DepositsSettled`, not `RedemptionsSettled`. Preserve the existing
  request/due-time boundary checks and two-day no-observation grace fallback.
  Do not redesign event selection or infer redemption events from deposit markers.
- Live claimability remains manager-owned. The estimate is a weak average,
  never permission to claim. Reuse the adapter's refusals for open mode,
  whitelists, disabled deposits and competing requests. Synchronous open-mode
  support, partial settlement, cancellation and refunds are not being added.
- Preserve historical admission and universe membership. Do not add an
  always-open T3tris exception, new selection filter or historical mode tracker.
  Check classification consumers in `PositionManager`, CCTP planning,
  phase-aware parking, the test-trade CLI and charts. Expected changes are
  delayed funding, async handling and refusal of T3tris as a synchronous parking
  reserve; no ranked-investment universe exclusion is intended.
- Keep the existing custody-owner resolution, pending-value accounting,
  pre-broadcast persistence and startup/cycle settlement sweep ordering.

## Proportionate validation

Extend existing coverage instead of building another test harness:

1. Add T3tris-only metadata to the existing timing/rule tests, proving both
   directions queue and consume the published estimate. Include metadata
   translation/snapshot assertions in the existing overlay test fixture.
   Extend the existing sell-proceeds funding parameterisation with T3tris.
2. Add T3tris to the existing historical-event and no-observation-grace test
   parameterisations in `tests/backtest/test_backtest_async_vault.py`. Ensure a
   T3tris lifecycle runs without a delay override so forced async execution
   cannot hide missing classification. Check the normal deposit/redemption cash
   and share outcomes there. Retain Lagoon/D2 controls, including an old rule
   snapshot whose event flag is missing or false, to protect the small gate refactor.
3. Follow `tests/erc_4626/test_vault_async_lagoon_erc_7540.py` for at most two
   T3tris fork tests: one guarded deposit/redemption round trip and one request
   refusal. Exercise public pricing checks, pending-state reload and a repeated
   settlement sweep within those scenarios. Reuse the upstream First USDC
   fixture evidence and `T3trisDepositManager.force_settle()` on Anvil. Use a
   fixed Arbitrum block with real curator liquidity and a fresh Anvil for
   deployment-heavy tests. Do not recreate Guard negative-call tests, adapter
   status tests or the shared accounting/recovery suite already covered upstream
   and in `tests/ethereum/test_vault_settlement_lifecycle.py`.

Run affected tests only, with the repository's Poetry, secrets include, worktree
`PYTHONPATH`, parallelism and timeout conventions. Synthetic metadata and event
fixtures establish executor wiring independently of production data publication.
Reuse the fork's real-provider connection for a minimal read-only compatibility
check; current live admission needs the intended custody owner. Record unavailable
owner/provider evidence as a gap, not a reason to add a diagnostic command or
block the other regression checks. No real-fund transaction is needed.

## Prerequisites and completion

Upstream migration `079-t3tris-settlement-estimate.py`, scanner backfill and
normal data publication populate production estimates/history. They are data
operations owned by eth-defi, not additional executor implementation steps.
Missing data continues through the existing fallbacks.

For validation, the dependency submodules were initialised and the parent's
user-prepared `.local-test.env` was copied without modification. Tests used the
parent Poetry environment with the executor and dependency worktrees first on
`PYTHONPATH`. The guarded fork verified custody admission at the fixed upstream
Arbitrum block; this does not establish current production-owner admission.

Complete when T3tris-only features select the shared async lifecycle in both
modes, backtests follow Lagoon's deposit-event policy and delay redemption cash,
and the guarded fork verifies the existing live checks and request/claim flow.
Existing Lagoon/D2 and synchronous controls must retain their timing rules.
Restoring the shared metadata loader makes published estimates available to
Lagoon, Upshift, D2, Plutus and Atoma where supplied, and historical deposit
events available to Lagoon/D2. Their real-data backtest timing and results can
therefore change even though investment membership and timing policy are unchanged.

## Review record

Claude CLI reviewed the earlier, longer plan on 15 November 2026 with
`claude-opus-5-5`; its terminal result confirmed successful completion. Raw review:
`/tmp/t3tris-opus55-plan-review.jsonl`. The primary agent addressed its concrete
snapshot-compatibility, timing-boundary and classification-consumer concerns.
The adapter already refuses open mode and checks competing requests separately
by direction, so speculative mode-history and cross-direction mechanisms were
not adopted. The subsequent code and cleanup reviews are recorded below.

The subsequent overengineering check used the repository's sanity-check-plan
skill and rechecked the current rule, pricing and test paths:

| Finding | Simplification |
| --- | --- |
| The live section could suggest reimplementing working support | Made the existing manager, pricing, Guard and settlement paths explicit; no live changes without a demonstrated gap. |
| Validation repeated generic and upstream guarantees | Reuse existing parameterisations and suites; cap new fork coverage at a round trip and a refusal. |
| Operational and speculative work obscured the small change | Separate upstream data prerequisites; remove broad protocol audits, hypothetical features and unrelated PR tasks. |
| Duplicate event selection could drift | Consume the existing resolved capability directly, retaining the old-snapshot control without a compatibility subsystem. |

## Implementation findings and validation

Implementation kept the existing lifecycle and revealed two additional wiring
gaps beyond the initial three-file estimate:

- The pinned trading-strategy loader dropped both `estimated_settlement` and
  `vault_settlement_at`. Its existing upstream PR #247 already supplies these
  fields. A local merge retains them alongside the newer admission-history
  implementation, and the executor now includes settlement markers in its
  existing parquet column projection. Loader and metadata round-trip tests pass.
- Reconstructing a mined T3tris deposit reran request admission and failed
  because the deposit was already pending. The manager now supplies
  `create_deposit_request_for_analysis()`, mirroring the existing redemption
  analysis hook, and shared routing uses it. Initial request preflights remain
  intact. This adds no alternative routing or settlement mechanism.

Validation on 16 November 2026 passed: 30 async backtest cases, 8 overlay/loader cases, 2 directional
timing cases, 26 shared settlement cases, both T3tris fork scenarios and the
existing Lagoon fork lifecycle. The fork round trip includes real Safe/Guard
execution, pending State reload, actual curator liquidity and idempotent claims.
Production owner-specific admission and upstream data publication remain
separate operational evidence. Dependency changes were initially local; their
publication is recorded below.

The dependency metadata test and T3tris mined-request parser regression also
passed, bringing the focused total to 71. The final bounded Claude CLI
`claude-opus-5-5` code review completed without confirmed blocking findings
(`/tmp/t3tris-opus55-code-review.jsonl`). Its compatibility question was checked
against all 18 deposit-request implementations: each supports the analysis hook's
keywords. The documentation placement nit was corrected.

Local dependency commits are `78529d853e081d47f5f9a21c5c75166c2c65b45b`
(trading-strategy, merging the existing settlement metadata work) and
`d226068671398249998dcccd6dd15e1c78666c1f` (eth-defi, mined deposit analysis
plus the T3tris integration documentation correction).

## PR cleanup review

On 16 November 2026, Claude CLI `claude-opus-5-5` independently inspected the
root patch, untracked tests and committed dependency changes. The completed
review (`/tmp/t3tris-cleanup-review.jsonl`) found no blocking code defects or
unapproved investment-universe exclusions. It identified an undisclosed effect
on other protocols: restored metadata loading can change their backtest timing.
The changelog, overlay documentation and this plan now explain that effect.

Cleanup also corrected the parser-hook comment, the funding-test comment and
the upstream README's outdated integration status. The T3tris funding scenario
was retained because it checks that pending exit proceeds cannot fund buys.
Saved pre-feature T3tris rules are not migrated: rebuild the universe to apply
the new timing policy. Existing pending requests retain their recorded terms.
The base analysis hook still delegates to request creation; adapters whose
admission changes after submission must override it, as T3tris does.

Writing was checked for factual accuracy, UK English and sentence-case headings.
Cleanup changed documentation and comments only, so the previous 71 test passes
remain applicable; tests were not repeated. Root and dependency diff checks pass.
At cleanup completion, the root patch was uncommitted and the dependency commits
were local. Following the user's request to open a PR, both dependency revisions
were published on 16 November 2026:

- [eth-defi #1786](https://github.com/tradingstrategy-ai/web3-ethereum-defi/pull/1786)
  contains the mined-deposit analysis hook and documentation correction.
- [trading-strategy #268](https://github.com/tradingstrategy-ai/trading-strategy/pull/268)
  integrates #247's settlement metadata work with current permission handling.

The executor pins these published branch revisions. Update the pins to merged
revisions before merging the executor PR if the dependency PRs are squash-merged.
