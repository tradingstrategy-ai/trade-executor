# Reliable multichain account checks and bounded corrections

Approved on 5 November 2026. Implemented in [PR #1706](https://github.com/tradingstrategy-ai/trade-executor/pull/1706).

## Objective and scope
Make check-accounts and correct-accounts trustworthy for the existing Arbitrum Lagoon hub and Ethereum satellite deployment. Reuse state, balance readers, deployment metadata and simulation harness. Manual recovery is acceptable: refuse ambiguous operations rather than infer economic outcomes from wallet balances. Preserve historical NB05 universe and existing specialised HyperCore recovery. No production transactions, new recovery ledger, dependency upgrade, general token-discovery service or automatic satellite cleanup.

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
| `tests/erc_4626/test_atoma_async_cli.py::test_atoma_start_blackbox` | 1 passed; pending and claimable requests report incomplete, then inspection passes after claim |
| `tests/erc_4626/test_vault_async_cli_commands.py::test_ostium_v15_check_accounts_cli` | 1 passed; pending inspection names the trade and preserves state; the previous clean-result expectation reproduced locally before correction |
| `tests/lagoon/test_lighter_move_funds.py::test_cli_lighter_move_funds_records_safe_and_lighter_custody` | 1 passed with normal deployment metadata and the existing settlement cursor acknowledging its directly booked initial subscription |
| `tests/mainnet_fork/test_hyper_ai_completed_top_up.py::test_correct_accounts_completed_top_up` | 1 passed; unrelated historical investor logs are mocked to avoid public-RPC range limits, while deposit receipts remain real |

External validation gaps: the legacy single-chain low-value test could not
construct its market-data universe because `tradingstrategy.ai/api/tvl` returned
repeated HTTP 500 responses; its second case is already skipped. Three Derive
cases carry an existing module-wide skip pending test-account/session-key setup
for the Derive v3 migration. These are not passing results. Fixed
Arbitrum/Ethereum archive access, the new low-value arithmetic checks, and the
single-chain Lagoon/Upshift and simulated exchange-account checks succeeded.

Additional compatibility checks passed for single-chain legacy deployment
metadata, reclaimed vault trades, queued-investor observations during specialised
HyperCore recovery, and exchange equity drift, missing readers and transport
failures. Relevant logs are `/tmp/account-cleanup-upshift-final-tests.log`,
`/tmp/account-cleanup-reclaim-tests.log`, `/tmp/account-cleanup-topup-tests.log`,
`/tmp/account-cleanup-incident-tests.log` and
`/tmp/account-cleanup-exchange-complete-tests.log`.

After rebasing onto PR #1705's Lighter SDK retry changes, all three multichain
command scenarios passed again in 348.52 seconds and its three SDK regressions
passed. Logs: `/tmp/account-pr-rebase-multichain-tests.log` and
`/tmp/account-pr-rebase-sdk-tests.log`. Repeated runs do not increase the original
81 distinct-test count. Targeted static checks, shell syntax, documentation links
and `git diff --check` also passed.

After removing the redundant reader conditional, the public exchange-account
correction test passed again in 44.88 seconds:
`/tmp/account-simplification-exchange-tests.log`.

## Framework reuse and simplification review

| Boundary | Existing mechanism and decision |
| --- | --- |
| Ordinary corrections | Existing balance-update events and state store; stage in memory and save after validation. No additional journal or transaction framework. |
| Pending operations | Existing trade statuses, tickets and settlement/retry paths. Commands refuse ambiguous corrections; they do not recover funds themselves. |
| Pool inspection | Run the existing NAV validator on a copied state because it changes prices and validity. A separate validation framework would duplicate ownership, holdings and stable-observation rules. |
| Early and final validation | Preflight checks recorded custody before side effects; universe-aware checks include configured assets; final reads catch changes during confirmation. These checks serve different purposes. |
| Exchange read failures | The shared sync deliberately skips some non-Lighter API failures and exposes no strict mode. A local reader wrapper lets this command refuse stale equity without changing daemon policy. Removed the redundant conditional after the missing-reader guard. |
| Integration setup | Reuse the existing driver and runbook. Three command scenarios own matching forks, executor state and fake sequencer; queued Upshift uses its existing test. The large-universe fixture calls the existing universe builder rather than rewriting strategy source. |

The review history has been condensed here; the implementation contract and
acceptance boundaries are unchanged. Comments explain money, escrow and restart
invariants. No extra lifecycle matrix, routing abstraction or recovery framework
is needed for this scope.

## Independent reviews

Claude CLI completed the plan and overengineering reviews with
`claude-fable-5-1`; both ended successfully. Raw reviews:
`/tmp/multichain-account-plan-fable51.jsonl` and
`/tmp/claude-multichain-overengineering-review.jsonl`.

The `claude-opus-5-5` implementation review found compatibility regressions in
terminal-failure classification, legacy metadata, specialised HyperCore recovery
and exchange snapshots. These were corrected and covered by the focused checks
above. Follow-up reviews found no blocking findings in the remaining metadata
and failed/missing-reader fixes. Raw reviews:
`/tmp/claude-opus55-multichain-pr-review.jsonl`,
`/tmp/claude-opus55-multichain-pr-followup.jsonl`,
`/tmp/claude-opus55-multichain-pr-final.jsonl` and
`/tmp/claude-opus55-multichain-reader-closure.jsonl`.
All completed with `end_turn`, `completed`, `is_error=false`; the final
reader-guard review used supplied excerpts with tools disabled. Earlier reviewers
ran disallowed no-op processes; those processes provide no test evidence.

## Remaining limits

The receive-crash test covers startup retry with the existing pre-broadcast
callback. Initial inline CCTP receives and some manual recovery callers retain
older checkpoint/wait behaviour; a crash there can require receipt-led manual
recovery. Completed HyperCore top-up recovery retains its existing treatment of
unrelated corrections: inspect preview and custody evidence if an externally
settled but unbooked investor flow coexists.

The runbook's settled account-check step has syntax validation. The shared
fixture exits before that final shell step, so command coverage comes from the
public-command tests. Cross-chain observations are not atomic. External market
data, Derive account setup and live exchange-provider coverage remain validation
gaps; simulated sequencer results do not establish live-provider compatibility.
