# Reliable multichain account checks and bounded corrections

Approved on 5 November 2026. Implementation record follows below.

## Objective and scope
Make check-accounts and correct-accounts trustworthy for the existing Arbitrum Lagoon hub and Ethereum satellite deployment. Reuse state, balance readers, deployment metadata and simulation harness. Manual recovery is acceptable: refuse ambiguous operations rather than infer economic outcomes from wallet balances. Preserve historical NB05 universe and existing specialised HyperCore recovery. No production transactions, new recovery ledger, dependency upgrade, general token-discovery service or automatic satellite cleanup.

Baseline findings before implementation: check-accounts forces one RPC and does not attach web3config. correct-accounts already selects primary chain and attaches web3config through its execution factory; its existing two-chain test is a hot-wallet/token test, not complete Lagoon coverage. The shared reader falls back to hub RPC. Pool holdings are omitted from generic checking; existing refresh_lighter_idle_collateral already validates account ownership, holdings and stable API snapshot. Pending vault escrow and allocated bridge capital already have correct wallet projections. Signed negative USD differences are incorrectly labelled low value. Correction has reserve-only skip paths and one hub transfer builder. The old failed-mint test changes balances without a real message and expects a recoverable transfer to disappear from NAV; do not treat it as real lifecycle evidence.

## Implementation contract

The implementation breakdown below is the authoritative gap-to-fix mapping.
Keep the command flow small:

1. Resolve public custody metadata and required RPCs without a signer. Support
   the existing shared-Safe topology and explicit Anvil configuration.
2. Inspect settled cash, ERC-20 holdings and the public pool account. Preserve
   existing quantity/relative tolerances; compare USD differences by magnitude.
   Compare raw pool shares exactly and idle collateral within one USDC raw unit.
3. Report pending IDs/statuses and explicitly unchecked assets. Generic correction
   refuses ambiguous operations; existing startup retries, settlement and manual
   recovery remain responsible for resolving them.
4. Stage ordinary corrections in existing state, use the existing confirmation,
   recheck balances and investor flows, then save once. Preserve specialised
   HyperCore and verified exchange-transfer checkpoints.

Explain the changed money/state boundaries in line comments. Keep the numbered
pytest steps and labelled simulation components required by repository guidance.
The runbook and readiness notes record the delivered command and recovery limits.

Leave existing missing-bridge-position bookkeeping and Lagoon's dedicated
settlement-discovery cursor in place. Keep the command-specific pending gate
outside the shared startup accounting gate so recovery can still proceed.

## Fixed-block integration acceptance
Extend the existing persistent dual-fork harness and MultiVaultStartDriver; move reusable setup to tradeexecutor.testing if needed, never import helpers from tests. Pin Arbitrum 520900000 and Ethereum 25598869 explicitly in these tests, avoiding a moving midnight default. Deploy real Lagoon/Safe/Guard contracts and use actual native USDC, Atoma, Upshift and CCTP contracts. Reuse FakeLighterCore and forged Circle attestations; label them as simulation. Reuse snapshot/reset support with matching state and sequencer snapshots, or fresh existing fixture instances; do not build a new harness. All retained fixes keep public-command integration coverage. The following are coverage groups, not five separate deployment fixtures: A, D and E share the settled scenario; B shares one refusal deployment; C uses the bridge/venue restart scenario plus the existing queued Upshift test.

A. Settled inspection and correction: both chains configured; bootstrap pool collateral, three held venue positions, allocated versus free satellite USDC and independent observed-assets reconciliation. Check succeeds without private signer and without changing state. Inject attributable hub/satellite cash discrepancies and an untracked known ERC-20 holding in the small universe, verify signed deficit/excess reporting, lazy-price reconstruction, applied correction and idempotent rerun. Include large negative deficit, non-dust zero-price holding and a reserve-only final state. Assert chain/owner/block columns.

B. Incomplete observation/refusal: remove satellite RPC, supply wrongly mapped chain, conflict satellite Safe metadata, fail a required RPC, alter fake pool holdings/collateral/owner, invalidate saved pool observation and omit account metadata. Each command refuses/informs unclean, original state bytes and transaction counts remain unchanged. Verify single-chain-only configuration never reads satellite assets on hub. Verify that the existing internal `block_identifier` passed by `correct-accounts` applies only to the hub; the satellite uses its own safe block. This adds no historical-block CLI option.

C. Pending lifecycle and restart: real CCTP burn with destination receive delayed, then real receive with bookkeeping completion interrupted; pending Atoma/Upshift request/claim, and pending Lighter operation. Check exits nonzero with original IDs and correction refuses. Resume through existing recovery and verify no duplicate burn/mint/claim or cash credit and conserved cash plus claims. Read actual pending entitlement rather than using the settled-only observed_assets helper. No fabricated zero-destination-balance proof of loss. The interruption is a simulation-only monkeypatch in `tradeexecutor.testing.multi_vault_manual` that exits before booking success; production recovery uses the existing `ExecutionModel.sync_state_before_broadcast()` callback and state store.

D. Mutation boundaries: multichain transfer-away and process-redemption refuse before signing; a fresh balance read failure after staging corrections leaves source state unchanged; retry persists exactly once. Queue an investor deposit and verify generic correction refuses until existing treasury processing books it; its event cursor remains unchanged by generic corrections. Snapshot state, holdings and wallet nonces for refusal assertions.

E. Coverage boundary: a universe with at least 20 pairs still checks tracked holdings and configured cash and explicitly reports its discovery limit. Do not assert detection of an untracked token outside the selected set. Verify the shared-Safe supported topology, avoiding a different-Safe feature.

Keep the three deployed command scenarios and reuse existing protocol lifecycle
regressions. Test changed command behaviour locally; avoid another complete
protocol-by-failure matrix. The existing mixed-strategy lifecycle tests remain
regression evidence, and the queued Upshift case reuses its own deployment.
Every retained boundary above still has its fixed-block command assertions.

For subsequent changes, run the affected command scenarios and relevant shared
reader/accounting regressions. Broaden to protocol lifecycle tests when routing,
settlement or persistence changes. The full verification table below records this
implementation's checks; it is not a requirement to rerun all 81 tests for every
documentation edit or deployment inspection. Follow the repository test environment,
parallelism and timeout instructions. Keep missing provider access explicit.

## Deferred issues
No exhaustive scanning of hundreds of tokens, different-Safe topology, cross-chain atomic snapshot coordinator, automatic entitlement recovery, general satellite token disposal, new persistence/checkpoint layer, arbitrary historical balance API or universal transaction replay. Provide explicit limits/refusals and existing manual recovery instructions. No changes to the actual NB05 allocation policy.

## Completion and review

Completion requires no false clean result for material or incomplete observations,
no erased entitlement, and no authoritative state mutation from failed ordinary
correction. The verification record below distinguishes passing coverage from
external validation gaps.

Claude CLI `claude-fable-5-1` completed the approved plan review with a successful
terminal result (`end_turn`, `completed`, `is_error=false`). Raw review:
`/tmp/multichain-account-plan-fable51.jsonl`. Its accepted corrections are included
in the implementation contract, gap table and coverage groups above.

## Simplification review

Checked against the implemented command helpers, state/store callbacks, shared
vault lifecycle and fork driver. The scope needs no additional runtime framework.
Manual recovery remains sufficient for ambiguous exceptional operations.

| Finding | Evidence and decision |
| --- | --- |
| Repeated requirements obscured current work | The implementation instructions, reviewer bullets and gap table repeated the same rules. Consolidated the contract and kept the detailed gap table and verification evidence. Labelled the old defect description as baseline history. |
| Validation wording could expand into duplicate lifecycle matrices | Three command scenarios already share the existing driver; queued Upshift uses its existing test. Keep those assertions and the original regression evidence. Subsequent changes run affected checks, with broader lifecycle coverage only when the changed behaviour warrants it. |
| Large-universe fixture used brittle source rewriting | The implementation follow-up replaces function-name string replacement with `strategies/test_only/multi_vault_account_checks.py`, a static inspection strategy calling the existing `expand_account_check_universe`. The test copies this fixture using its existing driver. No parser, configuration knob or fixture framework was added. |
| Repeated validation has distinct safety purposes | Early custody checks precede side effects; universe-aware checks cover newly discovered assets. Final pool/investor reads detect changes during confirmation. Keep these boundaries rather than caching an earlier clean result. |
| A copied state may look expensive | The existing pool validator changes validity and valuation fields. A copy isolates read-only inspection. Manual commands do not justify a new snapshot schema or a separate pure-validation framework without measured cost. |
| Recovery fixes are necessary within the bounded scope | The real interruption exposed lost expired IDs and an uncheckpointed receive. Existing `expired_positions` and the pre-broadcast callback fix them directly. Retain these fixes and hard-exit evidence; do not add a recovery registry or replay engine. |

This review changes this plan only. It adds no implementation work to the
completed deployment scope and does not rerun tests. Historical results below
remain the results of the implementation run.

### Independent Claude overengineering review

Claude CLI `claude-fable-5-1` reviewed the current plan with tools disabled and
completed successfully (`end_turn`, `completed`, `is_error=false`). Verdict:
**no blocking overengineering found**. Raw review:
`/tmp/claude-multichain-overengineering-review.jsonl`.

| Reviewer finding | Checked outcome |
| --- | --- |
| Home-block handling could imply a speculative CLI option | Clarified group B: this is the existing internal `block_identifier` argument, not a new historical-balance feature. |
| Crash hook could imply a test-only production branch | Clarified group C: the monkeypatch lives in the simulation harness; production uses the existing checkpoint callback. No new production test branch is needed. |
| Pool refusal matrix could eventually be reduced | Optional only if maintenance cost appears. Keep the currently passing per-boundary command assertions required by the user. Missing RPC configuration and a configured provider failing during observation exercise different paths. |
| A/D idempotency checks might duplicate deployment work | A, D and E already share the same settled scenario and deployment. No extra fixture or deployment can be removed on this basis. |
| Source rewriting, copied state and early validation could be tidied | Source rewriting was optional maintenance at review time and is replaced by the static fixture in the implementation follow-up. Shared-validator mutation and distinct early/final safety boundaries justify the copy and validation passes; avoid refactoring them without a demonstrated benefit. |

The review applied two wording clarifications. The subsequent implementation
follow-up also completes the static-fixture cleanup. Runtime accounting behaviour,
acceptance coverage and historical test results remain as recorded below.

## Implementation breakdown

| Gap and possible failure | Implemented boundary | Integration evidence / manual boundary |
| --- | --- | --- |
| Inspection selected a single RPC before reading the strategy | Select the strategy's primary chain, attach the existing RPC configuration and load public pool metadata without a signer | Deployed dual-fork `check-accounts` with `PRIVATE_KEY` removed |
| A missing satellite provider could read an unrelated hub token or zero balance | Validate custody RPCs and actual chain IDs; never fall back to the hub for a satellite asset | Missing Ethereum RPC and Ethereum mapped to Arbitrum both refuse; real independent chain balances remain unchanged |
| A different satellite Safe would make balance comparisons meaningless | Refuse conflicting public deployment ownership | Conflicting satellite metadata refuses; different-Safe routing remains deferred |
| Negative USD discrepancies were always below the positive low-value threshold | Compare absolute magnitude and preserve the signed event/report delta; unknown or zero prices cannot certify dust | Material reserve deficit, surplus and zero-priced Atoma share discrepancy; arithmetic regression covers both signs and missing price |
| A portfolio with no open positions could skip reserve correction | Include configured reserves regardless of trading positions | Reserve-only state is corrected against real Safe USDC; known sub-dust bridge cash is retained separately for exact NAV assertions |
| One hub block number could be used on the satellite | Preserve each chain's observation block and timestamp; display chain and owner | Corrections use the hub block explicitly while Ethereum remains at its own much lower block height |
| Pool shares and idle collateral were omitted | Validate ownership, complete raw-share map and stable public observations on a state copy | Missing metadata, invalid saved observation, API failure, changed owner, unexpected shares, negative collateral and one-raw-unit rounding boundary |
| Pending operations could make locked/escrowed capital look missing | Report pending trade IDs/statuses; exclude affected cash and shares; refuse generic correction before effects | Real CCTP burn/receive interruption and queued venue exits; resume through existing recovery |
| Expiring an unfunded CCTP batch prevented restart | Preserve its records in `expired_positions`, and include that existing archive in integrity checks | Real burn interruption exposed this defect; focused lifecycle regression verifies archived IDs and integrity |
| A startup receive could mint before its signed identity was saved | Call the existing execution-model checkpoint before broadcasting the retry receive; use robust receipt visibility | The simulation exits before booking success without saving in its interruption hook; restart must use the saved signed receive and real mined receipt |
| Known untracked vault shares lacked a pricing model for reconstruction | Initialise existing routing/pricing only when reconstruction needs a quote | Remove the ledger entry for a real Upshift holding, reconstruct it, compare independent assets and preserve allocated bridge capital |
| Large universes could omit zero free satellite cash | Keep bridge cash in the finite checked set at zero expected balance; retain bounded token discovery | Twenty-pair inspection includes tracked holdings and configured cash, while untracked dummy tokens are not queried |
| Multichain token disposal or investor processing could act through the hub signer | Refuse `--transfer-away` and `--process-redemption`; use dedicated operator workflows | Refusals preserve both chain heights and state bytes |
| Failed final validation previously left ordinary changes saved | Stage in memory, confirm through the existing mechanism, validate, then save once | Inject final observation failure; unchanged authoritative file, successful retry and idempotent repeat. Existing exchange-transfer checkpoints remain supported |
| Unbooked investor flows could be mistaken for unexplained cash | Reuse Lagoon settlement discovery and queue reads; refuse until `lagoon-settle` reconciles them | Queued subscription refuses correction; settlement clears it and preserves the dedicated discovery cursor |

No automatic pool write-off, bridge recovery inside account commands, satellite disposal, cross-chain
snapshot coordinator or new persistence system was added. Manual recovery remains
the boundary for ambiguous economic outcomes.

The original fork timestamps are Arbitrum block `520900000` at
`2026-11-03 14:31:18` UTC and Ethereum block `25598869` at
`2026-07-23 23:59:59` UTC. The shared fixture records these before advancing both
fork clocks to the same simulated UTC; executor freshness and the fake sequencer
then follow that clock. These are fixed historical contract snapshots, not a
claim of simultaneous historical cross-chain state.

## Verification record

Tests ran with `.local-test.env`, the installed Poetry environment and explicit
worktree/dependency `PYTHONPATH`. Long fork cases used bounded extended deadlines;
the dual-fork scenarios ran in parallel with separate deployments and state.

| Focused module / scenario | Result |
| --- | --- |
| `tests/lagoon/test_multichain_account_commands.py` | 3 passed; settled and pending scenarios also passed a final rerun with zero-free-cash and explicit unchecked-asset assertions |
| `tests/lagoon/test_multi_vault_start_e2e.py` | 2 passed |
| `tests/erc_4626/test_upshift_cli.py::test_upshift_pending_resume` | 1 passed, including refusal with the real queued request and its ticket-unparsed snapshot |
| `tests/ethereum/test_correct_accounts_cctp_bridge.py` | 2 passed |
| `tests/ethereum/test_correct_accounts_cctp_bridge_failures.py` | 1 passed for genuine settled cash correction; fabricated failed-mint write-off removed |
| `tests/ethereum/test_cctp_in_transit_lifecycle.py` | 4 passed, including expired IDs and the receive checkpoint |
| `tests/ethereum/test_vault_settlement_lifecycle.py` | 26 passed |
| `tests/lagoon/test_lighter_idle_collateral.py` | 7 passed |
| `tests/ethereum/test_cctp_bridge_cash_aware.py` | 23 passed; zero available cash remains an explicit custody row |
| `tests/units_tests/test_account_correction_confirmation.py` | 5 passed |
| `tests/cli/test_cli_correct_account_price_missing.py` | 1 passed |
| `tests/hyperliquid/test_correct_accounts_hypercore.py` | 1 passed |
| `tests/hyperliquid/test_correct_accounts_vault_dust_writeoff.py` | 4 passed |
| `tests/lagoon/test_lagoon_lighter_sub_account_corrections_e2e.py` | 1 passed, preserving verified transfer recovery and requiring unchanged ordinary state on failed final validation |

External validation gaps: the legacy single-chain low-value test could not
construct its market-data universe because `tradingstrategy.ai/api/tvl` returned
repeated HTTP 500 responses; its second case is already skipped. Three Derive
credential-dependent cases were skipped. These are not passing results. Fixed
Arbitrum/Ethereum archive access, the new low-value arithmetic checks, and the
single-chain Lagoon/Upshift and simulated exchange-account checks succeeded.

`git diff --check` and targeted undefined-name/unused-import checks passed for
the new helpers and tests. No production state, live deployment or remote branch
was changed. The earlier Claude plan review's terminal result was rechecked:
`claude-fable-5-1`, `end_turn`, `completed`, `is_error=false`.

The strengthened receive-crash scenario also passed separately (327 seconds):
the process exited before booking success, with no test-side executor save;
startup recovered the original pre-broadcast-checkpointed receive, independent
NAV agreed, and the queued venue exits completed. Across the focused modules
above, 81 distinct tests passed. The external API failure and four skips remain
explicitly excluded from that count.

## Implementation follow-up

The reviewed plan is implemented in the worktree. The large-universe scenario
now copies the static `strategies/test_only/multi_vault_account_checks.py`
inspection fixture instead of rewriting strategy function names. It re-exports
the base strategy settings and indicator hook, uses the existing universe
expander, and places no orders on its synthetic untracked tokens.

The affected fixed-block public-command scenario,
`tests/lagoon/test_multichain_account_commands.py::test_settled_accounts_and_atomic_correction`,
passed again in 322.54 seconds with this fixture. Normal strategy loading,
targeted undefined-name/unused-import checks and `git diff --check` also passed.
This reruns an existing case and does not increase the 81 distinct-test count.
The previously recorded external API and credential validation gaps remain.

## PR cleanup review

Claude CLI `claude-opus-5-5` completed a read-only implementation review and
identified four compatibility regressions. Its verdict was not ready pending
fixes; that initial result is not a final approval. Raw review:
`/tmp/claude-opus55-multichain-pr-review.jsonl`.

| Finding | Correction and focused evidence |
| --- | --- |
| Historical failed or reclaimed trades stayed pending forever | Treat confirmed failures as terminal, matching existing slot semantics. The shared reclaimed-deposit regression verifies that cash is restored and no assets remain excluded as pending. |
| Normalised single-chain deployment reports lack a Safe field | Validate per-chain Safe metadata for coordinated multichain deployments. Single-chain custody still comes from the actual vault. Public Upshift commands now exercise a normalised report without `safe_address`; unknown or missing multichain metadata still refuses. |
| Investor queues blocked specialised HyperCore recovery | Exclude incident, withdrawal and completed-top-up recovery from the ordinary investor gate. Real-Lagoon interrupted-deposit and completed-top-up tests simulate a queued investor observation while retaining actual fork custody and receipt validation. |
| Moving exchange equity and virtual reserves prevented corrections | Preserve the existing API-only route for exchange positions, while checking actual Lagoon reserves and reserve-only custody. Final validation checks ERC-20/pool balances; exchange equity records the successful sync snapshot. The public Lighter regression includes drifting subsequent equity and a non-custodial virtual reserve. |

The reviewer also recorded that it
ran a disallowed no-op Python process while checking tool access. It reported no
tests or file edits; this process deviation does not count as validation.

Remaining scope limits: the receive checkpoint test covers startup retry, whose
callback is wired by `start`. The initial inline CCTP receive and some manual
recovery callers still have older checkpoint/wait behaviour; a crash there can
require receipt-led manual recovery. No general receive/replay refactor is claimed.
The runbook's new settled `check-accounts` step has syntax validation, but the
shared deployment fixture exits before that final shell step; command coverage
comes from the public-command tests. Block-height display assertions are distinct
from inspection of the reader's per-chain block selection.

### Final cleanup outcome

The focused Opus 5.5 follow-up verified the terminal-failure, single-chain
metadata and HyperCore recovery fixes. It then identified a swallowed-read gap
in the exchange snapshot handling. The correction command now records failed
reader calls locally, refuses them even when the shared daemon sync catches the
error, and refuses a missing reader outright. Shared sync and client factories
remain unchanged. Successful equity snapshots still tolerate later price movement.
Multichain metadata must also cover every required custody chain.

Final completed reviews found no blocking findings in these remaining fixes:

- `/tmp/claude-opus55-multichain-pr-final.jsonl`: failed-read handling and
  metadata checks, with final log/current-test assertions left for verification.
- `/tmp/claude-opus55-multichain-reader-closure.jsonl`: missing-reader guard
  and both public-command failure assertions, with the integration rerun pending.

Both used `claude-opus-5-5` and ended with `end_turn`, `completed`,
`is_error=false`. The first was a repository inspection; the final reader-guard
closure reviewed supplied code excerpts with tools disabled. The pending test
evidence was subsequently verified by the primary agent:

| Cleanup check | Final result |
| --- | --- |
| Settled accounting and real bridge interruption scenarios | 2 passed in the dual-fork module rerun; the third scenario's missing-file harness setup failed before the CLI and was corrected separately |
| Complete custody refusals, including missing artefact, missing chain and unknown slug | 1 passed in 253.93s: `/tmp/account-cleanup-custody-verified-tests.log` |
| Exchange drift, virtual reserve, swallowed transport failure and absent reader | 1 passed in 44.12s: `/tmp/account-cleanup-exchange-complete-tests.log` |
| Single-chain legacy metadata and queued Upshift resume | 1 passed: `/tmp/account-cleanup-upshift-final-tests.log` |
| HyperCore completed top-up with queued investor observation | 2 passed: `/tmp/account-cleanup-topup-tests.log` |
| HyperCore interrupted deposit with queued investor observation | 1 passed: `/tmp/account-cleanup-incident-tests.log` |
| Shared vault settlement regressions | 26 passed; the precisely affected reclaim case also passed after its assertions were relocated |
| Two-chain token accounting | 2 passed: `/tmp/account-cleanup-twochain-verified-tests.log` |
| Arithmetic and confirmation | 5 passed: `/tmp/account-pr-cleanup-tests.log` |

The earlier setup errors and successful `SystemExit(0)` caught incorrectly by
the new legacy-metadata test are superseded by the specific passing logs above.
The expanded missing-file test captures the deployment-derived environment before
removing its artefact, so both real CLI entry points reach the intended refusal.
No broader suite rerun or new live provider claim is implied by these results.

Targeted static checks, normal helper/strategy loading, shell syntax,
documentation links and `git diff --check` passed. Changed narrative text was
reviewed for UK English, sentence-case headings and scope accuracy. Changelog and
agent guidance were updated. The local PR description uses Why, Lessons learnt
and Summary; no PR was created or pushed.

Existing limits remain: unavailable historical market-data and external account
credentials, manual recovery for older inline/manual CCTP crash windows, and
non-atomic cross-chain observations. Completed HyperCore top-up recovery retains
its existing treatment of unrelated corrections; operators must inspect preview
and custody evidence if an externally settled but unbooked investor flow coexists.
