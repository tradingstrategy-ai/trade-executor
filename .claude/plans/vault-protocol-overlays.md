# Vault protocol overlays and strategy conversion

Status: implemented locally on `feat/vault-pair-overlays`; focused verification
completed. Remote CI and a full NB07 financial backtest have not run.
Updated 9 November 2026 UTC with the user's decisions. The preceding version
received a completed Claude CLI Opus 5.5 review; this revision incorporates that
review where applicable and supersedes its Avantis, early-history and legacy-state
scope. A subsequent clean-context subagent review of this revision completed;
two findings were verified and corrected in the plan as recorded below.

## Scope and branch

Implement documented protocol overlays in trade-executor, retrofit their resolved
rules onto `TradingPairIdentifier` at construction, reuse shared lifecycle logic,
and rewrite the multichain strategy to consume the fields. Add an offline
integration test with a hardcoded universe. Implementation was subsequently
authorised and completed as recorded below.

- Branch: `feat/vault-pair-overlays`.
- Worktree: `/home/mikko/code/trade-executor-vault-pair-overlays`.
- Base: fetched `origin/master` at
  `7e2d7277b67a1b1932ceccde3d04e630fc004e05`, including native funding PR #1715.
- Pinned trading-strategy: `f80f684758413d27b5350a57718d3ca594a91574`.
- Pinned eth-defi: `ffedca4edb506da17c68afce72eff48b1d821321`.
- Destination: `~/code/strategies/strategy/master-vault-multichain.py`, draft
  [strategies #83](https://github.com/tradingstrategy-ai/strategies/pull/83).
- Source: NB07, [vault-notebooks #10](https://github.com/tradingstrategy-ai/vault-notebooks/pull/10).

The canonical checkout and its dependencies remain untouched. Its source-only
`.local-test.env` wrapper was copied to the worktree; external secrets were not
read or copied. Before implementation tests, set up isolated pinned dependencies,
reuse the existing Poetry environment, put the worktree first on `PYTHONPATH`,
and verify actual import paths. No test environment setup is needed for planning.

## Decisions from the user

1. Exclude Avantis/Veranta from the strategy universe for now. Do not build its
   overlay, fee handling, manager-fee investigation or dedicated execution tests.
   Keep a small selection assertion that the excluded candidate cannot re-enter
   through a protocol/curator inclusion branch. This does not remove or change
   its existing upstream adapter or support in unrelated strategies.
2. Use actual HYPE++ price-parquet availability and settlement data. Supply
   missing early window history in `d2.py` as a clearly documented historical
   assumption, with provenance and bounded dates; no strategy-local schedule.
3. Backtest universe construction asserts on missing required fee data by
   default. Provide the explicit strategy parameter
   `allow_missing_vault_fee_data = False` to override the refusal when set true.
4. Defer old-state migrations, legacy metadata reconstruction and dedicated
   legacy State serialisation work. Preserve compatibility for existing strategy
   code, particularly Hyper-AI, and use normal executor CI as the regression gate.

## Existing mechanisms to extend

Read the executor's shared vault lifecycle, phase-aware, vault closure, state
persistence, strategy parameter and backtest-universe-preservation guides.

- `strategy/dex_data_translation.py:translate_trading_pair` constructs pairs,
  attaches `VaultMetadata`, filters `other_data`, then caches the result. Enrich
  after that metadata/filtering step and before caching. Cover lazy universe
  accessors too; changing only `create_from_dataset()` misses those paths.
- `state/identifier.py` already exposes async deposit classification, delayed
  redemption classification and a metadata settlement estimate.
- `backtest/backtest_execution.py` already separates operation direction,
  historical D2/Lagoon settlement events, estimates, overrides and fallbacks.
  Retain its two-day no-observation grace and event gate after minimum due time.
- Existing simulation defaults: global two-day delay, D2/Plutus fourteen-day
  fallback, Atoma seven-day redemption fallback, Ostium next day at 18:00 UTC.
  These are existing labelled approximations, not new protocol observations.
- `vault_closure`, pricing, phase-aware allocation and YieldManager already own
  admission and parking. Existing managers/tickets, resolver, pending trades,
  retries, funding gates and State own execution and accounting.
- `backtest/vault_fees.py` computes externalised manager fees and consumes fixed
  exit charges through `get_sell_tax()`. The translator's buy-tax condition loses
  sell-only fees. Preserve unrelated DEX tax semantics when correcting vault fees.

## Pair fields and protocol overlays

Add a small serialisable `VaultProtocolRules` class in
`tradeexecutor/state/vault_protocol.py`, an optional `vault_protocol_rules` field
on `TradingPairIdentifier`, and documented public accessors. Extend existing
State models only; introduce no ledger, sidecar or migration framework.
Document every field, constant and enum with Sphinx `#:` comments, including
units, rationale, unknown semantics and provenance.

| Proposed fields | Meaning |
| --- | --- |
| `deposit_fee`, `redemption_fee` | Fractional charges; `None` is unknown and zero is explicit. |
| `deposit_fee_supported`, `redemption_fee_supported` | Direction-specific fee mechanism support, separate from investor admission. |
| `deposit_timing`, `redemption_timing` | Separate immediate/delayed/next-day-at-hour/unknown classifications. |
| `deposit_delay_seconds`, `redemption_delay_seconds` | Minimum estimates, zero only for an explicitly immediate operation. |
| `settlement_hour_utc` | Existing Ostium next-day hour when applicable. |
| `deposit_requires_settlement_event` | Existing D2/Lagoon event gate, not a promised completion time. |
| `deposit_windowed`, `redemption_windowed` | Capabilities; never interpreted as current openness. |
| `fee_profile` | Resolved contractual fee terms, distinct from mechanical support; used by declarative strategy fee-selection policy. Unknown is not a zero-fee profile. |
| `provenance` | Simple source label: observed metadata, named protocol assumption or explicit override. No additional observation timestamp model. |

Reuse existing manager-fee/internalisation fields, dated vault state, next-open
metadata and tickets. Add fee-basis information only where operation alone does
not specify it. Avoid duplicate authoritative fields; existing getters/tax slots
are compatibility projections from resolved rules.

Create `tradeexecutor/strategy/vault_protocol/` with a dispatcher, documented
`d2.py` and `atoma.py`, and small Lagoon/Gains helpers only where current rules
need moving. Overlays are pure functions of loaded data and framework defaults;
no RPC, file access, environment reads or strategy imports. Dispatch by canonical
slug/features; refuse conflicting classification rather than using iteration
order. Keep deployment/version-specific facts in the relevant adapter.

Apply enrichment once during translation; repeated translation/cache access is
idempotent and fetch-order independent. Avoid mutating shared token metadata in
ways that alter another pair's fee treatment. Preserve existing pair equality
and hashing (base, quote and fee), constructor defaults and old public getter
signatures. An existing strategy that does not use the new fields continues to
use its existing framework behaviour; do not require a new argument or accessor
in Hyper-AI. Do not add lazy old-State backfilling, migration passes or a legacy
serialisation test matrix. New fields must still work in normal current-State
serialisation, and new pending requests use existing trade records to retain
request-time inputs. Live ticket/receipt evidence remains authoritative.

## HYPE++ evidence and bounded early-history fallback

Inspected the existing local parquet on 9 November 2026, without downloading or
modifying it: `/srv/vaults/vaults/downloads/vault-prices.parquet`, 318,836,841 bytes.
Filter: chain 42161, address `0x75288264fdfea8ce68e6d852696ab1ce2f3e5004`.

- 873 observations spanning `2024-11-20 09:20:49` to `2026-11-08 19:01:12` UTC.
- Deposit flags: 147 true, 492 false, 234 unknown. Redemption flags: 152 true,
  487 false, 234 unknown.
- Both flags first become known at `2026-02-11 02:32:56`; unknown observations
  span `2024-11-20 09:20:49` to `2026-02-10 08:37:00` UTC.
- 60 non-null settlement observations, ranging from `2024-11-30 04:05:23`
  to `2026-10-12 00:46:48`. A settlement event does not prove a deposit window.

Inspection receipt is local research evidence at
`~/code/vault-notebooks/data/vault-of-vault-cleanup/strategy-conversion-clean/hype-window-inspection.json`;
it is not a runtime dependency or fixture path in the strategy.

Implementation decisions:

1. Reuse the existing decision-bucket state model and as-of pricing. For D2
   availability, place an observation into the next decision boundary with the
   existing timestamp column (`ceil` instead of `floor`); exact-boundary samples
   stay on that boundary. Keep the last observation for each bucket. For the
   daily strategy, a sample at `2026-02-11 02:32:56` becomes available at
   `2026-02-12 00:00:00`. Do not add observation/effective timestamp fields,
   retain intraday transitions, or build a new freshness/provenance subsystem.
   Keep settlement-event handling and unrelated protocols unchanged. Do not
   infer openness from price returns or from the occurrence of a settlement.
2. Put the early HYPE++ history fallback in `d2.py`, scoped by chain and identity
   inside the protocol overlay. The strategy sees only resolved pair fields and
   the shared window API. The user explicitly authorises this engine-owned
   historical assumption; document it separately from observed protocol data.
3. For the early missing period, retain the previously declared research
   approximation: fourteen-day cadence, two-day open duration, anchor
   `2025-03-30 00:00:00` UTC. Encode it using the existing `VaultWindowSchedule`
   machinery with effective range
   `[2024-11-20 09:20:49, 2026-02-12 00:00:00)` UTC. These constants describe
   assumed history, not proven openings, and must be documented in `d2.py`.
4. Precedence: explicit backtest override; valid observed as-of availability;
   bounded early fallback only where observations are missing. Do not pass the
   fallback as today's unconditional `vault_window_overrides`, whose precedence
   would overwrite actual observations. Extend the existing resolver/pricing
   path to consult a protocol history fallback after observed evidence instead.
5. Never overwrite a known closed observation. Keep actual settlement events
   and existing request/settlement rules separate from window assumptions.
   After the early range, stale/missing evidence remains unknown; do not extend
   the historical assumption into live trading or later gaps. Before the first
   price observation, do not manufacture price coverage or tradability.
6. Known window closure may park capital. Capacity/private/permanent restrictions
   may not. Missing evidence outside the authorised fallback suppresses a new
   deposit with an unknown diagnostic and releases unsupported park earmarks.
   Tests cover the boundary where observations replace the fallback, not just
   fabricated all-open/all-closed series.

Refresh the inventory when implementing if loaded parquet data changes; preserve
newly available observations and record the actual boundary used. Keep any
handwritten early-history assumptions and changes visible in module documentation.
The fallback is not fitted to improve returns. Early price sparsity and assumed
windows remain economic limitations even when framework tests pass.

## Missing-fee policy and fixed-fee execution

Add shared validation using the existing fee-completeness rules, invoked by the
converted strategy during backtest universe construction after source membership
and resolved fees are available, before indicators or trading start.

- Default `Parameters.allow_missing_vault_fee_data = False`: assert with the
  affected vault identities and missing field names. Check every selected vault,
  including cash support and diagnostic-only selected pairs. Do not validate
  thousands of unrelated metadata candidates excluded by selection policy.
- An explicit true parameter allows construction to continue while incomplete
  vaults remain data-only through `mark_missing_fee_vaults_ignored`. It does not
  invent zero fees, permit their trading or bypass unrelated admission checks.
  This preserves the existing diagnostic treatment as the override behaviour.
- Known internalised manager fees and explicitly zero manager fees follow the
  existing completeness contract; do not require irrelevant fields merely
  because they are null. Entry/exit fees still require known values.
- Implement this as an explicit policy/helper call from the converted backtest
  universe, not a new unconditional assertion in the global loader/translator.
  Existing Hyper-AI and other strategies must not acquire a new failure just
  because they do not declare this parameter. Live admission remains unchanged.
- Unknown fee data fails by default; a known unsupported nonzero fee is a
  separate actionable support error, not a missing-data override case.

For this scope, add the required Atoma nonzero-exit support and preserve existing
supported mechanisms elsewhere. Avantis/Veranta is explicitly excluded before
fee validation and cannot re-enter via other selection branches. No fee-related
expansion to other protocols is implicit; inventory collateral changes separately.

Mechanical fee support does not grant membership. Preserve NB07's economic
selection policy: known zero entry and exit fees qualify; the two intended
Atoma identities qualify only with zero entry and 0.005 exit fees. All other
known nonzero entry/exit fees remain excluded, even when the engine can execute
them. Put deployment identity facts and validation of the standard Atoma terms
in `atoma.py`, exposing a resolved fee profile. The strategy selects allowed
profiles declaratively rather than branching on a vault address/protocol in
`decide_trades()`. Do not label every vault with a supported 0.005 fee as that
profile, and do not silently accept changed Atoma fee terms. Preserve candidates
with missing fees until strict universe validation can assert (or the explicit
override can mark them data-only); unknown must not become a zero-fee profile or
silently vanish through fee filtering. A known mechanically unsupported fee
raises only if the candidate otherwise passes the intended fee-selection policy;
known policy-excluded candidates do not cause an unrelated universe failure.

Fix sell-only fee transport and resolve rates as finite fractions, e.g. `0.005`.
Preserve Atoma's immediate deposits, delayed redemption and adapter-owned live
fee-recipient exemptions. Fixed entry/exit charges stay distinct from externalised
manager fees; ranking, pricing and funding forecasts use the same fee convention.

Extend `backtest/vault_fees.py:calculate_vault_fees`, or its factored pure core,
to support fixed-only/internalised-fee vaults as well as externalised manager
fees. Quote/forecast calls are pure; only existing execution/settlement wallet
and ledger updates debit the calculated net amount. Pricing must not also apply
a projected tax after consuming that result. Audit sync/async and legacy fee
branches for one economic charge; do not build migration tooling for old records.
Test the whole quote-to-settlement path, not only the calculator.

Use reviewed operation-specific fee basis and rounding: redeem fixes shares,
withdraw fixes assets, deposit and mint likewise differ. The hardcoded $1,000
redemption at 0.005 producing $995/$5 below is explicitly fee-on-total. It is a
fixture expectation, not a blanket formula for every ERC-4626 method. Live
execution continues to use actual manager previews/receipts without a second tax.

## Shared directional settlement and allocation

Extract existing timing into a public helper such as
`get_vault_settlement_expectation(pair, operation, timestamp, context)`. Read pair
rules, the existing configured overrides and as-of event state; return whether
that operation queues, its minimum due time, provenance and remaining event gate.

Preserve exact precedence: explicit override; positive legacy metadata estimate
only for current eligible families (Lagoon, Upshift, D2, Plutus, Atoma); protocol
fallback; Ostium next-day 18:00 UTC; configured global fallback. Overrides retain
both-directions-queue semantics, including zero-delay resolver-cycle behaviour.
Without an override, classify direction first. A redemption delay never queues
an immediate deposit. Preserve Gains treatment even with positive metadata.

Expose the helper through `PricingModel`, already available to `decide_trades()`.
Bind the same execution timing policy/overrides through the runner's existing
execution/pricing setup, beside fee-pricing binding. GenericPricing delegates
correctly; cover all factory/direct construction paths. The strategy and simulator
must not see different overrides. Audit either-direction `_is_async_vault` callers
where operation actually matters.

Keep D2/Lagoon historical events, minimum delay, no-observation grace, tickets,
claimability, retries and idempotent reconciliation in the existing resolver and
managers. A due-time estimate does not authorise live settlement. Use deposit
expectations for end-of-backtest entry checks, keeping delayed withdrawals
separate from immediate deposits.

Use pair window capability plus closure reason/next-open evidence for phase-aware
parking, never an address branch in decisions. Release park earmarks if eligibility
fails or evidence ceases to support a future opening. Apply the $10,000 historical
TVL rule to directional buys, top-ups, promotions and YieldManager parking buys;
below-floor exits/releases remain possible. The cash venue is not exempt: failed
parking leaves reserve cash liquid. Reuse existing admission/size-risk inputs.

## Strategy rewrite

Rewrite `~/code/strategies/strategy/master-vault-multichain.py` on its conversion
branch during implementation, coordinating with draft PR #83.

- Retain dynamic data-based selection; explicitly exclude Avantis/Veranta for now.
  Preserve private restrictions, native tags, required in-scope identities and
  Lighter LLP exclusion. Missing fees now assert by default as specified above.
- Keep daily decisions, Monday ranking, 90-day score/downside gates, holding and
  concentration rules, $100,000 capital and $3,000 bridge buffer.
- Replace the three unconditional capability stubs with pair accessors and shared
  fee, timing and closure APIs. Derive windowed pairs from loaded overlay fields.
- All external data is loaded through `create_trading_universe()` and supported
  loaders. No replay modes, research-file imports, input hashes or protocol
  patches in the strategy. All constants receive Sphinx `#:` comments.
- Compare membership/assumptions with source evidence; 420 is a reference count,
  not a substitute for checking actual changed policy and input data.
- Keep live refusal until appropriate existing route/manager evidence supports
  lifting it; this engineering test alone is not a deployment authorisation.

## Integration tests and compatibility gate

Add `tests/backtest/test_vault_protocol_overlays.py`, reusing existing builders
or a small fixture helper in `tradeexecutor/testing/`. Hardcode raw metadata,
candles and dated window/event observations for ordinary ERC-4626, Lagoon, Gains,
D2, Atoma and the synchronous cash venue. Include an excluded Avantis selection
row only to prove the explicit exclusion; no Avantis execution feature work.
Use the real parser, dataset construction, lazy translation, pricing, alpha/yield
models and settlement resolver; never prepopulate or mock the new rules.

Use a small happy-path integration and refusal/regression case:

1. Verify fields/provenance, fraction units, two-direction timing and idempotent
   cache translation. Compare old/new due times across existing families,
   metadata estimates, overrides and Ostium/event/grace rules. A mechanically
   supported ordinary protocol/curator candidate with a 0.005 exit fee must still
   be excluded; both intended standard-fee Atoma identities remain eligible.
2. Verify observed HYPE++ availability replaces the early fallback at the
   decision-bucket boundary. One real-parser regression is sufficient: a sample
   during 11 February is unavailable at that day's midnight and available on
   12 February. No intraday transition or separate timestamp-expiry test matrix.
   Atoma deposits remain immediate and redemptions delayed; Lagoon/D2 retain
   event gates. Capacity/private closure does not park.
3. Exercise close/park/open/promote/request/settle using the normal runner. Check
   $1,000 gross redemption at 0.005 gives $995 proceeds/$5 fee once, fee-reduced
   funding and pending proceeds remaining unavailable. Include immediate deposits
   near the backtest end and whole-wallet conservation.
4. Incomplete selected fees assert before trading by default. With the explicit
   true override, construction succeeds and those vaults remain data-only. A
   known unsupported fee still refuses. Excluded Avantis does not trigger missing
   fee validation. Test below-floor promotion closing its earmark and below-floor
   cash venue leaving liquid reserve; existing holdings can still exit.
5. Test the actual converted strategy in its own repository against the new engine.
   The executor integration uses a minimal consumer strategy and never imports a
   sibling repository or `tests/` subtree. Follow numbered test-docstring steps,
   matching body comments and typed fixtures per repository instructions.

No legacy-state migration or old-JSON matrix is part of acceptance. Keep existing
strategy imports, optional constructor defaults and public method calls compatible.
Use existing Hyper-AI, vault-window, phase-aware, externalised-fee, yield, Atoma
and native-funding regression cases, then require the normal trade-executor CI
checks to pass before merging implementation. Existing CI is the cross-strategy
compatibility gate; do not claim it has passed during planning or disable tests
to avoid a regression. No live funds, service restart or dependency switch in the
shared checkout is required for this validation.

Use the shared Poetry environment, source-only wrapper and worktree-first
`PYTHONPATH`, with verified dependency imports. Follow focused-test parallelism
rules and a five-minute per-command limit; longer research coordinators use the
existing supervised deadline system. Mocks replace external acquisition/transport
only. No tests are run as part of this plan update.

The source evaluation `[2025-04-01, 2026-10-21)` UTC and warm-up
`[2024-06-01, 2025-04-01)` UTC are past provenance, not scheduled backtests.
Training/calibration/holdout: N/A. No economic metrics are claimed. A later full
historical run requires its own authorised execution scope and input/engine receipt.

## Implementation sequence and completion gates

1. Document pair fields and D2/Atoma overlays; wire construction without changing
   public behaviour for other strategies.
2. Shared fixed-fee handling and strict default fee validation with explicit override.
3. Shared directional timing, observed-window loading and bounded early fallback.
4. All-allocation-path eligibility and the hardcoded-universe integration.
5. Rewrite strategy, run focused checks and normal CI, inspect membership/assumptions.
6. Update API/lifecycle docs and conversion checklist; record pins and evidence.

Completion requires removal of unconditional capability stubs through real shared
implementations, no duplicate fee debit, no missing-fee trading disguised as zero,
no synthetic early fallback overwriting observed/live availability, and no crashes
in existing strategies under normal CI. Avantis and legacy-state migration are
explicitly deferred; they do not block this scope.

## Review and revision record

The previous version had one completed read-only Claude CLI review using
`claude-opus-5-5`, with terminal completed status and nonempty output. Retained
findings: fee basis/units and one-debit ownership; exact settlement precedence;
shared override visibility; capacity-aware parking and release of blocked earmarks;
pair identity compatibility. The current user decisions supersede its Avantis
and legacy-state work and its prohibition on an engine-owned early-history fallback.

Primary review applied `sanity-check-plan` to this revision, including the actual
HYPE++ parquet inspection and conflict removal. No second Claude review or tests
were run. Original review evidence remains under
`~/code/vault-notebooks/data/agent-reviews/vault-pair-overlays-plan-20261109.env`
and `skill_output_1794249033_3319828.md` in that directory. This plan is not a runtime
dependency and has not been published as part of this update.


### Clean-context subagent review

A fresh subagent reviewed the implementation proposal without the conversation
history or prior review conclusions. Its input copy omitted this document's
review section. Read-only code inspection found two substantive issues:

- HIGH: pinned upstream conversion floors non-HyperCore state to daily buckets
  and selects the last intraday row. Following the user's simplification request,
  resolve this with next-decision-bucket availability and one parser regression;
  no extra timestamp fields or intraday state machinery.
- MEDIUM: mechanical fee capability could broaden the source membership rule.
  Corrected the plan to retain zero-fee selection plus the intended Atoma terms,
  represented by overlay-resolved fee profiles, and added a supported-but-excluded
  ordinary 0.005-fee candidate to acceptance.

The primary verified both findings against pinned upstream code and the source
NB07 selector before editing. No further blocking findings or new user decisions
were identified by that focused review. No implementation or tests were run;
these corrections have not received another review pass.


### Timestamp simplification

The user requested avoiding observation-timestamp overengineering. The plan now
uses the existing single timestamp and decision buckets, with a narrow D2
floor-to-ceil conversion and one boundary test. The early fallback ends at the
first observed daily decision bucket (12 February 2026), not at an intraday
sample time. No implementation or additional review was performed.


## Implementation record

Implemented the pair rules/overlays, shared settlement API and pricing binding,
fixed-only fee calculation, strict missing-fee policy, window fallback/observed
precedence, D2 next-bucket conversion, entry-policy callbacks and strategy rewrite.
The final code uses explicit queued-operation booleans rather than another timing
enum hierarchy. Existing managers, event gates, tickets and State remain in use.

- trading-strategy dependency commit: `43966eb2d965a5882c663dcf59ebf486e76f6073`,
  local branch `feat/d2-window-buckets`; executor gitlink updated together.
- 105 focused executor/strategy checks passed in the dedicated executor Poetry
  environment with worktree/dependency imports verified. These include the new
  hardcoded-universe actual Atoma redemption test and all existing fee/window/
  async/phase-aware cases selected for regression.
- The existing simulated cross-chain HyperCore/CCTP fork lifecycle also passed.
- An independent read-only review found unknown-state forwarding, explicit-capacity
  precedence, nullable reasons, unknown-redemption compatibility and fee-mode
  ranking issues. These were corrected, with public pricing regression checks.
- No full NB07 historical backtest or economic parity claim. Normal remote CI
  remains the pre-merge compatibility gate and has not run for these local changes.

The canonical executor/dependencies were not switched or edited. The isolated
worktree uses the existing full executor environment, not a new package install.
Source-only credentials wrapper and local environment symlink are not versioned.
