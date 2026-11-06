# Lix HyperSync event reads

Investigation and simplification date: 2026-11-06. Status: implemented locally;
validation and code review results are recorded below.

## Scope and evidence

Prefer HyperSync for Lix's Lighter withdrawal payout checks and Lagoon event
reads, passing credentials from Typer. Retain `eth_getLogs` when HyperSync is
unavailable, fails or cannot cover the requested blocks. Keep existing accounting,
withdrawal baselines and executor state unchanged.

Lix is an Ethereum Lagoon vault with a Lighter master and four sub-accounts.
Its strategy selects custody movements; the executor and eth-defi own event
reads. No changes to `strategy/lix.py` or its `Parameters` are needed.

Lix was inspected from the fetched strategies `origin/master`; the local
`feat/hyper-ai-v13` checkout and its uncommitted work were left untouched.
Investigation started with eth-defi pinned at
`6087f8d91fdec5d774d37a0f569fa09a7f96c388` and its submodule uninitialised.
Sources were read in `~/code/trade-executor/deps/web3-ethereum-defi`; the Lighter
payout and generic settlement reader files matched the original pin. The pinned
submodules were then initialised locally for implementation, leaving the
canonical checkout unchanged.

## Event-read map before conversion

Paths beginning `eth_defi/` belong to the dependency.

| Operation | Reader and callers | Baseline behaviour |
| --- | --- | --- |
| Lighter payout proof | `eth_defi/lighter/lagoon.py:fetch_lighter_claimed_usdc()` through `is_lighter_withdrawal_paid()` and `wait_for_lighter_withdrawal()` | Direct RPC in 250-block chunks, capped at RPC head. Filters ZkLighter address, `WithdrawPending` and owner; checks USDC asset index and converts raw amounts using token decimals. |
| Runtime custody withdrawals | `tradeexecutor/ethereum/lighter/lighter_routing.py` | Uses the above waits and relayer-payout verification. |
| Interrupted-transfer reconciliation | `ethereum/lighter/transfer_verification.py:get_lighter_relayer_payout()` through `verify_lighter_transfer()` and `reconcile_verified_lighter_transfers()` | Reached by routing settlement, `check-accounts`, `correct-accounts` and `repair`. |
| Manual custody operations | `cli/commands/lighter_move_funds.py`, `lagoon_lighter_test_trade.py` | Direct withdrawal waits and relayer-race checks use the same payout reader. |
| Lagoon settlement replay and unconfirmed-settlement probe | `ethereum/lagoon/vault.py:_fetch_lagoon_settlements()` → `eth_defi/erc_4626/vault_protocol/lagoon/settlement.py:fetch_lagoon_settlements()` → `eth_defi/erc_4626/settlement_events.py` | Already selects HyperSync or RPC using environment variables internally; disables HyperSync on Anvil. HyperSync errors currently propagate. |
| Manual Lagoon pending NAV inspection | `cli/commands/lagoon_manual_settle.py:_fetch_pending_new_total_assets()` | Uses the generic settlement reader when the contract lacks `newTotalAssets()`. |
| Shared Lighter pool withdrawal proof | `ethereum/lighter/lighter_pool_routing.py` → `is_lighter_withdrawal_paid()` | Not Lix's exchange-account strategy, but changing the shared reader affects it. Recovery constructs a router separately in `ethereum/vault/settlement_retry.py`. |

Lighter equity/history use HTTP APIs; pending balances use contract calls and
transaction checks use receipts. Leave these unchanged. Enzyme, CCTP and the
ERC-4626 claim-recovery `get_logs()` in `settlement_retry.py` are outside scope.

## CLI configuration and propagation

Reuse `shared_options.hypersync_api_key` (`--hypersync-api-key`,
`HYPERSYNC_API_KEY`) and broaden its guard-only help text. Parse the existing
`USE_HYPERSYNC` setting at Typer entry points as a shared
`--use-hypersync/--no-use-hypersync` option. Its help must explicitly scope the
switch to Lighter payout and Lagoon settlement reads; existing guard/CCTP
consumers retain their current behaviour.

Resolve the enabled boolean at the command boundary and pass the effective key
explicitly through existing functions. Disabled use or simulation resolves to
`None`, keeping internal plumbing to one optional credential. No key means RPC.
Never write overrides to `os.environ`, persist credentials or read them from
strategies/routers.
HyperSync access must not require Lighter signing keys.

| Entry points | Propagation to cover |
| --- | --- |
| `start`, `perform-test-trade`, `lagoon-settle` | `create_execution_and_sync_model()` → sync model and execution model → `EthereumPairConfigurator` → Lighter adapter/router. |
| `check-accounts`, `correct-accounts`, `repair` | Sync construction plus reconciliation → verification → payout reader. Cover both sync-construction sites in `correct-accounts`. |
| `lighter-move-funds`, `lagoon-lighter-test-trade` | Direct helper calls, payout checks and manually constructed Lagoon sync models. |
| `lagoon-manual-settle` | Inspection/proposal helpers → pending-NAV event read. |
| Existing shared-bootstrap users | Audit `close-position`, `close-all`, `retry`, `check-position-triggers`, `check-wallet`, `console`, `trade-ui`, `check-universe`, `init`, `reset`, `reset-deposits`, `blacklist` and `reclaim-satellites`; forward settings wherever Lagoon sync or Lighter execution can occur. Preserve existing deployment/report key consumers. |
| Shared pool routing | Both adapter construction and the separately constructed recovery router. |

Use ordinary optional keyword parameters on existing constructors/helpers.
Search all call sites after editing and update strict test doubles. Do not
introduce a configuration object or make unrelated constructors mandatory-keyword
APIs just to enforce this audit.

## Eth-defi changes

### Lighter payout reader

In `eth_defi/lighter/lagoon.py`, add optional `hypersync_api_key` and
`use_hypersync` keywords to the payout reader, boolean check and withdrawal wait.
Keep existing callers working without a key. Check `is_anvil(web3)` before
creating a HyperSync client even when the fork reports Ethereum's chain ID.

Implement the following bounded read inside the existing payout reader:

1. Snapshot the requested end, capped at RPC head. Return zero for an empty range.
2. Use existing HyperSync server/session helpers, deriving the chain from the
   custody Web3. Query the same contract, event and owner. Use verified indexer
   height/stream coverage to determine the indexed prefix; never infer coverage
   from the last matching event. HyperSync's end block is exclusive.
3. Fetch that prefix with HyperSync and the remaining tail with existing chunked
   RPC. Keep ranges disjoint, normalise log data/topics before decoding, and
   preserve asset-index filtering, decimal conversion and withdrawal baselines.
4. If HyperSync fails or its coverage cannot be established, discard its partial
   output and read the whole bounded range with RPC. Never add a full RPC retry
   to a partially accumulated amount. If RPC also fails, propagate the failure.

Retain the small prefix/tail split: withdrawal checks read through chain tip,
so falling back for the entire range whenever the indexer lags could leave the
main Lix path permanently on RPC. Do not add cursors, incremental scan caches,
new finality policies or cross-provider consistency machinery. Preserve existing
safe-block and replay behaviour; RPC results retain their existing node-trust
assumptions, including the risk of silently incorrect empty results.

Keep a finite timeout for the whole HyperSync attempt using the existing async
session machinery, cancelling/discarding a timed-out stream before fallback.
Expose `hypersync_timeout` only on these low-level reader/wait functions; no CLI
or constructor propagation for timeout tuning. The wait passes its remaining
polling budget, capped by the reader timeout. Preserve existing RPC timeouts
and polling semantics; do not promise a strict wall-clock withdrawal deadline
while an RPC request is in flight.

### Lagoon readers

In `eth_defi/erc_4626/settlement_events.py`, add an optional explicit key and
forward it through the existing single-address, multi-address and HyperSync
functions. A supplied key overrides environment lookup; omitted/`None` keeps
legacy library behaviour. No sentinel is needed: executor callers always pass
an explicit enabled boolean and select false when they have no key.
Forward the key through `fetch_lagoon_settlements()` in
`eth_defi/erc_4626/vault_protocol/lagoon/settlement.py`.

Before a HyperSync settlement read, check that the requested end is indexed and
verify stream completion. Raise on incomplete coverage rather than returning
partial/false-empty settlement evidence. Reuse existing Web3-log conversion.

Keep automatic error fallback in the two executor consumers:
`LagoonVaultSyncModel._fetch_lagoon_settlements()` and the manual pending-NAV
inspection. Retry the full range once with `use_hypersync=False` on indexer
failure or lag. No generic `fallback_to_rpc` option or multi-address fallback
policy is needed. This preserves existing library callers' error behaviour.
Use RPC directly for the small Lagoon unconfirmed-settlement probe.

This deliberately differs from Lighter: historical Lagoon replay can fall back
as one operation, while Lighter needs the prefix/tail split to benefit from
HyperSync on live withdrawal checks. Do not build a new general event-reader
framework to combine these two cases.

Update `eth_defi/lighter/README-lighter-guard.md` and the existing executor
Lighter guide with the key, scoped opt-out and fallback behaviour. Extend shared
session/flow helpers only if the pinned SDK requires it to expose coverage.
No CCTP reader or guard-policy changes.

## Implementation and validation

1. Implement eth-defi reader changes first, then update the executor dependency
   revision and wire the CLI paths above. The existing dependency already requests
   the HyperSync extra; no new package or container work is planned. A deliverable
   pin must eventually be reachable from its remote; publish only when authorised.
2. Add a compact set of dependency tests covering normal HyperSync/RPC equivalence
   and filters, indexed-prefix/RPC-tail boundaries, empty results, indexer failure
   after partial output/timeout, and both providers failing. Reuse existing
   Lighter and Lagoon test fixtures; group assertions rather than creating many
   near-identical tests. Check Anvil never creates an indexer client with a key set.
3. Extend existing executor tests for CLI environment values, explicit key override
   and forced RPC. Cover shared bootstrap, read-only reconciliation and direct
   manual commands. Use the existing relayer-payout and Lagoon tests to verify
   accounting behaviour remains unchanged; update mocks for added keywords.
4. Run relevant individual tests from `tests/lagoon/test_lighter_relayer_withdrawal.py`,
   `test_lighter_move_funds.py`, `test_lagoon_lighter_cli_options.py`,
   `test_lagoon_manual_settle.py` and
   `tests/exchange_account/test_lighter_cli_accounts.py`. Select cases after
   inspecting fixtures; do not run every module just because it is listed.
5. Make one read-only live comparison over fixed historical ranges containing
   known Lighter payouts and Lagoon settlements. Call the HyperSync backend
   without fallback, or assert it actually supplied the indexed data; a correct
   RPC fallback must not count as successful HyperSync validation. Record range,
   source and amounts without credentials. Test fallback separately and include
   a near-tip coverage check. No transactions or state mutation.

Use an environment that imports this worktree and its changed dependency.
Because the dependency revision changes, follow repository instructions for an
isolated Poetry environment. Source `.local-test.env`, prepend
`PYTHONPATH="$(pwd):$(pwd)/deps/web3-ethereum-defi:$PYTHONPATH"`, and verify the
import origins of both `tradeexecutor` and `eth_defi` before accepting results.
Use `poetry run` and the repository's focused-test/timeouts rules; never edit
`.local-test.env`. Follow eth-defi's own test instructions for dependency checks.

Acceptance: HyperSync demonstrably serves indexed Lighter/Lagoon data; the CLI
key reaches runtime, manual and recovery reads; indexer lag/failure falls back
without gaps or double-counting; simulations stay on RPC; failed reads preserve
existing accounting refusal/deferral. No new persistence, state schema,
strategy policy or production deployment changes.

## Review and simplification record

The earlier revision was reviewed on 2026-11-06 with Claude CLI model
`claude-opus-5-5`; its terminal result was successful. It was a bounded plan-text
review, not a code audit. Retained its useful findings: correct dependency import
origins, proof HyperSync actually ran, bounded indexer attempts, fork detection,
log normalisation and caller coverage. This simplified revision has been checked
directly against repository instructions and reader code, not re-reviewed by
Claude.

| Complexity found | Evidence and simplification |
| --- | --- |
| Extending the opt-out into guards and CCTP | Those scans already have separate consumers and are not the Lix payout path. Scope the switch's help to Lighter/Lagoon and leave those APIs alone. |
| Credential omission sentinel and mandatory constructor keywords | Existing `use_hypersync=False` already prevents environment selection. Resolve the effective boolean at the CLI; use normal optional parameters and audit callers. |
| Generic automatic fallback switch | Two executor consumers need Lagoon fallback. Handle it there, preserving library error behaviour without another public policy setting. |
| New finality and RPC endpoint/hash checks | The conversion does not replace RPC trust or reorg handling. Retain current provider and safe-block rules; document their limits. |
| Timeout settings across application plumbing | Only the low-level wait knows its remaining budget. Keep the timeout there; no new CLI or routing configuration. |
| Release-image work and repeated verification lists | HyperSync is already a declared dependency extra. Use focused existing tests and one explicit live comparison. |

The plan review itself did not run tests or provider requests. Implementation
subsequently verified SDK height/progress and cancellation behaviour as below.

## Implementation validation

Local checks on 2026-11-06 use an isolated virtual environment sharing the
parent environment's installed packages read-only. Explicit `PYTHONPATH` and
import-origin checks select this worktree and both initialised local dependencies.
The parent environment and canonical source checkout were not modified.

| Check | Result |
| --- | --- |
| Dependency Lighter payout module | 10 passed, including actual timeout cancellation, partial-stream fallback, Anvil bypass and two live RPC/HyperSync reads. |
| Dependency Lagoon settlement module | 4 passed, including truncated/lagging stream refusal and a 1,000-block real HyperSync/RPC comparison. |
| New registered CLI and executor fallback tests | 8 passed, including RPC-only unconfirmed probing, manual NAV fallback and separate pool recovery construction. |
| Existing relayer-payout verification | 4 passed. |
| Manual Lagoon inspection and Safe proposal | 3 passed. |
| Manual Lighter custody command on Ethereum fork | 1 passed. |
| Account check/correction/repair on Ethereum fork | 3 passed. |
| Real start and cash-management pipeline on Ethereum fork | 1 passed; spies verify the credential reaches sync/execution models and the actual Lighter router, is never persisted and never creates an indexer on Anvil. |
| Existing HyperCore dust/cleanup compatibility | 3 passed. |

Live payout comparison: abandoned Lix Safe
`0xc9129A57299b0Db10b752E27507569D3F37dBBA4`, Ethereum blocks
26,348,457–26,348,510; direct HyperSync returned 10 USDC and covered the full
exclusive range, matching RPC without using fallback. Lagoon comparison:
Hub Capital vault `0xca790385506b790554571cbc9da73f0130cdcfd5`, blocks
24,371,920–24,372,919; event identities and naive UTC timestamps matched.
The comparison exposed RPC's hex `blockTimestamp` field, now normalised by the
shared settlement converter before accounting consumes it.

Near-tip check: blocks 26,355,452–26,355,462, same Lix Safe; direct indexed
coverage ended at exclusive block 26,355,463 and matched RPC's zero payout.
The indexer was current during this observation; deterministic tests establish
the non-empty RPC-tail behaviour when it lags. These are manual live-provider
checks against Ethereum RPC and the Ethereum HyperSync endpoint, not CI results.
No real transactions or production state mutations were performed.

After adopting the conservative `get_height() - 1` boundary, another live
comparison of blocks 26,355,535–26,355,545 covered the entire requested prefix
and matched RPC's zero payout. The indexer had advanced past the RPC snapshot;
the deterministic prefix/tail tests cover an unindexed suffix.

The first fork checks failed because the fresh Trading Strategy checkout lacked
its nested chain-data submodule. Initialising that dependency resolved the
failures; no runtime workaround was added.

The first independent Opus 5.5 code review completed successfully and its
substantiated findings were addressed:

- Preserve native SDK retry defaults and per-receive limits for existing
  unkeyed library callers. Whole-read timeouts are optional and set only by the
  executor. Completeness checks are enabled for settlement reads without changing
  unrelated vault-flow readers' default behaviour.
- Use a conservative exclusive archive-height boundary in both readers.
- Forward credentials through real `vault-test-trade` setup and existing
  Hyperliquid cleanup/dust helpers as well as the original Lix call map.
- Verify actual runtime, reconciliation and manual-command forwarding with
  spies on existing fork tests, plus the separate recovery/probe unit cases.
- Remove two unused routing-state parameters and correct API documentation.

The focused Opus 5.5 follow-up completed with no blocking or high-confidence
correctness findings. Both reviews have successful terminal result events.
The final legacy-client assertions and near-tip comparison subsequently passed.

Eth-defi revision: `4fce376e82e665806274203f2d327f8dc8a87cda`, the squash merge of
`feat/lix-hypersync` (reviewed head `570a01aed401ae1b14643d7bc9b75bb7f898c655`).
The executor gitlink now selects the merged revision from
[eth-defi PR #1734](https://github.com/tradingstrategy-ai/web3-ethereum-defi/pull/1734).
The unpublished dependency commits were rebased onto current master before PR
creation; the reviewed reader implementations are unchanged. Nothing was deployed.

Validation comprises 37 passing focused cases plus the recorded manual
near-tip comparison. `git diff --check` passes for both repositories. No source
formatter was run, following the user's repository instructions. Narrative text
was checked for UK English, sentence-case headings and factual accuracy.

Remaining provider assumptions: indexed progress verifies range completion,
not independent correctness of the provider's logs. Existing safe-block/reorg
rules and RPC trust remain unchanged. Legacy manual NAV inspection can fall
back to a long RPC scan when the indexer lags; it refuses provider errors rather
than advancing accounting from incomplete evidence.


## Cleanup for PR

Completed the repository's `clean-up-for-pr` workflow on 2026-11-06. The event
map is explicitly labelled as the pre-conversion baseline, old workaround
instructions are scoped to versions without automatic fallback, and the
settlement reader docstring distinguishes advertised range completion from
independent proof that an indexer returned every log. Narrative text follows
UK English and sentence-case headings.

No executable code changed during this cleanup. An AST comparison excluding
docstrings confirms the dependency matches implementation commit
`6a5c00687a5c9b1baf6e25d153ae626fa67bd55d`. Reused the completed Opus 5.5 code
review and 37 passing focused checks for that implementation; no redundant
review or provider requests were started. Both repositories pass
`git diff --check`. The dependency has a new local documentation commit;
the dependency branch was published before preparing the executor PR using its pin.


## Pull request preparation

The dependency PR is published as
[web3-ethereum-defi #1734](https://github.com/tradingstrategy-ai/web3-ethereum-defi/pull/1734).
The executor branch is `feat/lix-hypersync-event-reads`. Both repositories have
feature changelog entries dated 2026-11-06. The eight focused CLI/fallback tests
passed again after rebasing the dependency onto current master. No source
formatting, production operations or additional provider requests were needed.
