# Vault test trade robust-results plan

## Objective

Make `trade-executor vault-test-trade --auto-simulated` produce results that
are reproducible, accurately classified and useful to both eth-defi maintainers
and external reporting processes. The command must distinguish an adapter or
on-chain failure from a trade-executor bookkeeping, receipt-analysis or
simulation-lifecycle failure.

This plan follows the 2026-07-22 cross-chain master-vault run after eth-defi
PR #1347. The relevant trade-executor findings were:

| Finding | Count | Why it reduces confidence |
|---|---:|---|
| State/position inference reported no trades | 57 | A real execution outcome can be hidden behind test-result bookkeeping. |
| Async simulation stopped after deposit | 28 | Request success is not distinguished from settlement, claim or redemption compatibility. |
| Receipt analysis failed after a successful receipt | 6 | A status-1 transaction is reported alongside reverts instead of as a separate analysis gap. |
| Fork provenance was only available in process logs | all | An external process cannot reliably reproduce a result from state alone. |

The currently implemented `vault_test_attempt.error` payload is the first
part of this plan. It preserves a redacted Python traceback, exception chain,
per-chain blocks at failure, and any newly created transaction/revert trace.
The remaining work makes that evidence complete and makes results semantically
reliable.

## Scope and non-goals

In scope:

- The standalone vault test command and its persisted `State` results.
- Simulated and real attempt provenance, classifications and reporting rows.
- Correct ownership of positions/trades created by a test attempt.
- Anvil-only orchestration for async vault request/settlement/claim flows.
- Focused unit, Anvil and CLI black-box regression coverage.

Out of scope:

- Implementing protocol adapters or decoding protocol-specific errors in
  eth-defi. The eth-defi follow-up plan owns those changes.
- Pretending that a vault is tradeable when admission controls, settlement
  privileges or RPC limitations prevent a real lifecycle.
- Altering production execution behaviour or normal strategy state semantics.
  Vault-test metadata remains isolated to the dedicated vault-test state.

## Result contract

Every requested vault must produce one authoritative result position whose
`other_data["vault_test_attempt"]` carries a stable, JSON-safe schema.
External reporters must be able to consume it without trade-executor source or
the original terminal log.

Add and document these fields:

```json
{
  "schema_version": 2,
  "vault_id": "42161-0x...",
  "simulated": true,
  "operation": "deposit",
  "phase": "receipt_analysis",
  "result": "receipt_analysis_failed",
  "provenance": {
    "trade_executor_commit": "...",
    "eth_defi_commit": "...",
    "execution_mode": "auto_simulated",
    "run_started_at": "...",
    "fork_blocks": {"1": 25588651, "42161": 123456789},
    "anvil_generation": 2,
    "lagoon_deployment": {
      "primary_chain_id": 1,
      "vault_address": "0x...",
      "module_address": "0x...",
      "satellite_modules": {"42161": "0x..."}
    }
  },
  "error": {
    "captured_at": "...",
    "exception": {"type": "...", "message": "..."},
    "traceback": "...",
    "chain_blocks": {"1": {"block_number": 25588657}},
    "transactions": []
  }
}
```

Rules:

1. Do not store RPC URLs, private keys, signed transaction bytes or provider
   headers. Apply the existing redaction to every persisted error string.
2. Store raw token integers as strings where added in metadata, preserving the
   state file's JavaScript integer-safety requirement.
3. Keep prior state files readable through an explicit reader normalisation
   function. Missing `schema_version`, `provenance` or `error` means
   legacy/incomplete diagnostics, not a parse failure. Treat missing
   `schema_version` as version 1. Keep an unknown legacy `result` string
   unchanged in persisted state and expose `legacy result` only as an in-memory
   display/report classification. Never write a normalised status over the raw
   value during a later state save.
4. `detail` remains a compact human table summary; it must never be the only
   persisted source of failure information.

## Work item 1: make attempt ownership and state inference reliable

### Problem

The run produced 57 `no trades` state/position inference failures. The command
can create a closed diagnostic position while the attempted position/trades
are absent, belong only to a discarded simulated copy, or cannot be connected
to the diagnostic result. This obscures whether the failure happened before a
trade, during a transaction, or after a successful receipt.

### Changes

1. In `vault_test_trade_runner.py`, create an immutable `VaultAttemptContext`
   at the start of `_run_vault()`. Record the attempt id, baseline position and
   trade ids, requested operation, target vault id and simulated Anvil
   generation.
2. Pass this context through `_prepare_attempt()`, `_execute_real()`,
   `_execute_simulated()`, terminal-result handling and exception handling.
   Do not infer ownership by scanning all state after the fact.
3. For simulated failures, extract transactions and newly created target/bridge
   positions from the fork copy before `evm_revert`. Import a closed,
   diagnostic-only copy into persistent state, preserving transaction and
   receipt fields. Link it to the authoritative result with
   `source_position_id` and a new `attempt_id`.
4. For real failures, attach the result metadata to the actual target position
   when one exists. Only create a placeholder diagnostic position when the
   failure occurred before position creation.
5. Make the post-trade state verification return typed outcomes rather than
   raising a generic assertion. `no_trades_created` is valid only when the
   context proves no trade was constructed; otherwise return
   `state_inference_failed` with the expected and observed ids.
6. Ensure closing a simulated diagnostic cannot discard a real target, bridge
   position or prior attempt. Keep the existing baseline-id guard and add
   explicit assertions for it.

### Tests

- Unit test a preflight failure with no position/trade.
- Unit test a simulated receipt revert where the fork copy has a transaction,
  asserting the persisted result links to the imported transaction evidence.
- Unit test a real receipt-analysis exception where the target position remains
  the authoritative result instead of a second placeholder.
- Regression test the historical 57-row failure shape with a minimal mocked
  state and assert `state_inference_failed`, not an unqualified `failed`.

## Work item 2: classify failures by lifecycle stage

### Problem

`failed` currently combines metadata lookup, preflight, transaction build,
gas-estimation revert, receipt revert, receipt analysis and bookkeeping errors.
This prevents reliable coverage tables and directs eth-defi work to the wrong
layer.

### Changes

1. Define string constants local to the vault-test module. Persist their values
   as plain strings; do not serialise a Python enum into state, because an old
   or future unknown status must remain readable. Initial terminal values are:

   - `metadata_failed`
   - `preflight_failed`
   - `transaction_build_failed`
   - `gas_estimation_reverted`
   - `broadcast_failed`
   - `transaction_reverted`
   - `receipt_analysis_failed`
   - `state_inference_failed`
   - `execution_failed`
   - `infrastructure_failed`
   - existing explicit outcomes such as `deposit_closed` and
     `simulation_unsupported_async`

2. Keep `failed` as a backwards-compatible presentation alias only; new
   persisted terminal results must use a specific value whenever the phase is
   known.
3. Introduce a small exception-to-stage classifier in
   `vault_test_trade_state.py` or a dedicated diagnostics module. It must
   prefer explicit transaction evidence over exception text and must not
   classify normal admission reverts as infrastructure failures.
4. Add `operation`, `phase`, `attempt_id` and `result` to every row emitted by
   `_append_result()`. Keep the existing compact terminal table, but provide a
   machine-readable JSONL or `--report-json PATH` output containing the full
   authoritative state result for each requested vault.
5. Update the TUI status display to show the specific status and a short
   lifecycle phase. Do not render tracebacks in the TUI table.

### Tests

- Parametrised classifier tests for each status, including a status-1 receipt
  whose event analysis raises `receipt_analysis_failed`.
- CLI test that the tabular result remains compact while JSON output exposes
  the full structured record.
- State round-trip tests for each result value and legacy attempts with only
  `result="failed"`.
- A checked-in, pre-schema-version state JSON fixture loaded through the normal
  state store. Cover absent fields, `result="failed"` and an unknown legacy
  result value, then assert the TUI/table/report reader keeps the state usable.
  Perform a load → save → reload cycle and assert the unknown raw result value
  is byte-for-byte preserved in the persisted attempt metadata.

## Work item 3: persist complete, reproducible provenance

### Problem

Failure-time block numbers are useful but insufficient: a reporter needs the
initial fork height, source revisions, deployment topology and execution mode
for both successes and failures.

### Changes

1. Capture the upstream block number for every configured source chain before
   Anvil launches. Store it in a run-level record under
   `state.other_data["vault_test_run"]` and copy its immutable subset into each
   attempt's `provenance`.
2. In `vault_test_trade_simulation.py`, expose the Anvil generation and source
   fork block map as ordinary serialisable runtime data. Never derive a fork
   block from the mutable local Anvil height after contracts have been deployed.
3. Record the trade-executor `HEAD` commit and the checked-out eth-defi
   submodule commit at command startup. If either cannot be resolved, store
   `null` plus a non-fatal provenance warning.
4. Record the simulated Lagoon primary vault/module/satellite module addresses
   that were used for the attempt. For real mode record the deployment-artifact
   addresses instead.
5. Add `--report-json PATH` as an export with stable JSON key ordering of
   requested result rows plus their full attempt metadata. It must not include
   the state file's unrelated portfolio history or credentials.
6. Document an independent reproduction recipe generated from this record:
   selected vault id, source commit, eth-defi commit, chain fork blocks, mode,
   amount and target transaction hashes.

### Tests

- Unit tests for provenance capture with multiple chains and a replacement
  Anvil generation.
- State/JSON export round-trip test, including a redaction assertion for a
  provider URL in an exception.
- Black-box simulated CLI test asserting stable report JSON keys for a known
  synchronous vault.

## Work item 4: capture failed-call context when there is no receipt

### Problem

Gas estimation and pre-broadcast failures have no transaction receipt or
transaction hash. A Python traceback alone is not enough for an adapter author
to reproduce the call.

### Changes

1. At transaction build/broadcast boundaries, retain a JSON-safe call summary:
   chain id, sender, target, function selector, wrapped target/selector, value,
   nonce if allocated and unsigned calldata hash. Store full calldata only when
   it is already present in state transaction details and contains no secret.
2. On a build or estimation failure, store the call summary in
   `error["call_context"]` and capture the node's revert payload/error data if
   it is available from the exception.
3. For a broadcast receipt revert on Anvil, retain the existing symbolic
   `BlockchainTransaction.stack_trace`. Where an RPC supports
   `debug_traceTransaction`, store a redacted, JSON-safe raw trace only behind
   an explicit opt-in diagnostic option; traces can be very large and not all
   production providers support the method.
4. Never re-submit a failed call only to collect diagnostics. Capture only the
   call that the tester already attempted.

### Tests

- Unit test a synthetic estimation exception with call context and encoded
  custom-error data.
- Anvil test for a reverting deposit that preserves transaction hash, receipt
  block, revert reason and symbolic stack trace.
- Test that a credential-bearing URL and signed transaction bytes are absent
  from both state and report JSON.

## Work item 5: add honest async lifecycle coverage

### Problem

The tester records 28 `simulation_unsupported_async` results after a deposit
request. This is honest but not sufficiently realistic for evaluating generic
ERC-7540/Lagoon integration.

### Changes

1. Add an opt-in `--settle-async-on-anvil` mode. Keep the existing default
   deposit-only result until the complete lifecycle is supported for the
   selected manager; no implicit success conversion is allowed.
2. Define an Anvil-only protocol hook in eth-defi's test support that accepts a
   request ticket, depositor/receiver and fork web3. It must operate the real
   deployed contracts: obtain/impersonate only the necessary settlement role,
   update valuation if required, settle, then return ordinary ticket/analysis
   objects.
3. In trade-executor, call the hook only from the vault-test simulation runner.
   Continue using `perform_test_trade()` and the normal settlement resolver to
   claim the deposit and request/claim the redemption. Do not add test-only
   branches to production routing.
4. Report each phase independently: `deposit_requested`, `deposit_claimed`,
   `redemption_requested`, `redemption_claimed`. A failed phase becomes a
   specific terminal status with its attempt evidence.
5. When a manager has no eligible Anvil settlement hook, retain
   `simulation_unsupported_async` with an explicit capability reason. This is
   coverage information, not a test failure.

### Tests

- Anvil fork integration test for one synchronous vault, confirming no
  regression in the normal round trip.
- Anvil fork integration test for one Lagoon/ERC-7540 vault covering request,
  forced settlement, claim, redemption request, settlement and claim.
- Cross-chain test with the async vault on a satellite chain, asserting the
  source CCTP bridge and destination settlement use their correct connections.
- Regression test that real mode never invokes the Anvil-only hook.

## Delivery order

1. Land work item 1 with the existing error-persistence change and the state
   inference regression tests. This unblocks trustworthy identification of
   what actually happened in the 57 rows.
2. Land work item 2 and the compact-table/report-JSON contract. This enables
   comparison of future full runs without log parsing.
3. Land work item 3 and 4 together, because provenance and failed-call context
   form one independent reproduction record.
4. Implement work item 5 only after eth-defi exposes an Anvil-only settlement
   hook with its own fork tests. Keep that cross-repository dependency in a
   separate PR pair.
5. Re-run the identical 129-vault list after each completed work item. Compare
   result status counts, not just command exit code, and attach the report JSON
   to the PR comment after checking it contains no secrets.

## Acceptance criteria

- No result reaches an unqualified `failed` state when its lifecycle phase is
  known.
- A reporter can reproduce any simulated result using only state/report JSON,
  public commits and their own archive RPCs; no trade-executor terminal log is
  required.
- Every post-broadcast failure includes transaction hash, chain and receipt
  block; every Anvil receipt revert includes its symbolic stack trace when the
  node provides one.
- A successful receipt followed by analysis failure is classified separately
  and preserves its receipt evidence.
- The historical `no trades` shape is either proved to be a true pre-trade
  failure or recorded as `state_inference_failed` with expected/observed ids.
- The default async simulation remains conservative, while an enabled,
  supported Anvil async flow executes a full request-to-redemption lifecycle.
- Existing vault-test state files and the TUI remain readable without a data
  migration.
