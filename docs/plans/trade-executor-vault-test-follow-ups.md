# Trade-executor vault test follow-up plan

## Objective

Make `trade-executor vault-test-trade` consume eth-defi vault capabilities and
flow results through their shared interfaces, and ensure the command reports
trade-executor bookkeeping or preflight outcomes accurately.

This plan covers only changes in the trade-executor repository. Protocol
adapters, protocol ABIs, custom event decoding and new eth-defi capability
fields are owned by the separate eth-defi workstream documented in
`docs/plans/eth-defi-vault-test-follow-ups.md`.

The 2026-07-23 cross-chain rerun covered 129 vault IDs. It exposed these
trade-executor issues:

| Priority | Finding | Observed impact |
|---|---|---:|
| P0 | Vault-test diagnostic positions enter normal statistics serialisation | 56 of the 64 `receipt_analysis_failed` rows |
| P0 | Synchronous routing bypasses `VaultDepositManager.analyse_deposit()` and `analyse_redemption()` | Five successful Ember deposits and one successful YieldNest deposit were reported as failures |
| P1 | A successful transaction from an earlier lifecycle step makes a later failure look like receipt analysis | D2 availability and cSigma capacity were both reported as `receipt_analysis_failed` |
| P1 | Explicitly unsupported adapter directions reach execution | Upshift multi-asset became `execution_failed` |
| P1 | Async simulation output does not distinguish an unrequested full lifecycle from a manager without an Anvil settlement driver | 28 Lagoon rows used the same broad unsupported status |

## Scope

In scope:

- `tradeexecutor/cli/vault_trade/` attempt preparation, execution and result
  classification.
- `tradeexecutor/cli/testtrade.py` support for running a test trade without
  updating strategy statistics.
- `tradeexecutor/ethereum/vault/vault_routing.py` consumption of the shared
  eth-defi receipt-analysis contract.
- Shared trade-executor helpers needed to translate eth-defi flow analysis into
  `State.mark_trade_success()`.
- Persisted vault-test result strings, report details and focused regression
  tests.
- A final rerun of the same 129-vault list.

Out of scope:

- Implementing or changing Ember, YieldNest, Upshift, cSigma, D2, Lagoon or
  other protocol adapters in eth-defi.
- Decoding protocol-specific custom errors in trade-executor.
- Treating an on-chain revert as successful because the vault is otherwise
  supported.
- Changing global `TradingPosition.is_spot()`, `TradeAnalysis` or equity-curve
  semantics to understand vault-test diagnostic positions.
- Automatically enabling privileged Anvil settlement without the existing
  `--settle-async-on-anvil` opt-in.
- Changing normal `perform-test-trade` statistics behaviour.

## Dependency contract with eth-defi

The implementation must use shared eth-defi contracts and must not branch on a
protocol name or vault address.

The current submodule already provides:

- `VaultDepositManager.analyse_deposit(tx_hash, ticket)`.
- `VaultDepositManager.analyse_redemption(tx_hash, ticket)`.
- `DepositRedeemEventAnalysis` and `DepositRedeemEventFailure`.
- `VaultDepositManagerCapability` with directional deposit and redemption
  support.
- `VaultBase.get_deposit_manager_capability()`.
- `VaultFlowUnavailable` with structured direction, requested amount and
  available amount fields.
- `UnsupportedVaultSimulation`.
- `VaultBase.fetch_deposit_closed_reason()`.

The separate eth-defi agent is expected to provide or correct:

1. Protocol-specific receipt analysis for Ember and YieldNest.
2. An explicit unsupported capability and reason for Upshift multi-asset
   deposits until an implementation exists.
3. A protocol-neutral immediate deposit/redemption capacity query.
4. An explicit capability describing safe Anvil settlement support by
   direction or ticket type.
5. Structured availability errors for adapters such as D2 where a pricing
   estimate is currently rejected with a plain `ValueError`.

eth-defi is a development submodule dependency of trade-executor. Coordinate
the two workstreams by updating the submodule pointer after the eth-defi agent
lands each shared API. Do not add runtime version checks, minimum-version
requirements, `hasattr()` compatibility paths or fallbacks for older eth-defi
revisions.

Within the current shared contract:

- `None` capability means semantically unknown, not unsupported.
- Generic exception classification remains the final failure fallback.
- No fallback may parse protocol-specific exception messages.

The receipt-analysis cutover requires the current eth-defi manager contract.
`VaultDepositManager.analyse_deposit()` and `analyse_redemption()` are abstract,
and the generic `ERC4626DepositManager` supplies the standards-compliant default.
Concrete managers therefore cannot be instantiated without implementations.
Use the methods directly from the checked-out submodule. Do not retain a
fallback from a manager analysis error to
`analyse_4626_flow_transaction()`: that would silently reinterpret a
protocol-specific parser failure as a generic ERC-4626 success and recreate the
dispatch bug this plan removes.

## Result contract

Add these persisted outcomes to `VAULT_TEST_RESULTS`:

| Result | Meaning |
|---|---|
| `adapter_unsupported` | eth-defi explicitly states that the requested direction is not implemented |
| `redemption_capacity_limited` | The requested immediate redemption exceeds current protocol capacity |
| `async_request_only` | The configured simulation confirmed an async request without requesting its full lifecycle |

Retain the existing outcomes:

- `deposit_closed` for a known closed deposit window, paused deposit or zero
  current deposit capacity.
- `redemption_unavailable` for a known closed redemption window.
- `async_request_only` when the default simulation deliberately verifies only
  an async request.
- `simulation_unsupported_async` only when a full async lifecycle was requested
  but no safe Anvil settlement driver is available.
- `receipt_analysis_failed` only when a mined status-1 transaction exists and
  its protocol adapter cannot produce a valid flow analysis for that same
  transaction.
- `execution_failed` only when no more specific typed or evidence-based outcome
  applies.

Every typed outcome must preserve a concise `detail`. Capacity results must also
retain requested and available raw amounts in JSON-safe form. Add a structured
`outcome_data` mapping to `vault_test_attempt` for such fields instead of
requiring reporters to parse `detail`.

Example:

```json
{
  "result": "redemption_capacity_limited",
  "detail": "Immediate redemption capacity is below the requested amount",
  "outcome_data": {
    "direction": "redeem",
    "requested_raw_amount": "907757",
    "available_raw_amount": "45388"
  }
}
```

Keep older state readable. Unknown future result values and missing
`outcome_data` must continue to round-trip unchanged.

## Work item 1: isolate vault-test diagnostics from statistics

### Problem

`make_test_trade()` serialises long/short statistics after several execution
steps. A dedicated vault-test state also contains closed diagnostic positions
created by `record_attempt_result()`. These positions deliberately have no
trades, so normal analytics eventually call `TradingPosition.is_spot()` and
assert:

`Cannot determine if position is long or short because there are no trades`

This is a caller-boundary problem. General analytics should continue to assume
that normal portfolio positions contain trades.

### Changes

1. Add a keyword-only `update_statistics_after_trade: bool = True` argument to
   `make_test_trade()`.
2. Extract the repeated statistics update into one small helper and guard all
   current buy, sell, short and cross-chain call sites with the new option.
3. Pass `False` from both `VaultTestBatchRunner._execute_simulated()` and
   `_execute_real()`.
4. Keep all other callers on the default `True`, including the public
   `perform-test-trade` command and trade UI.
5. Do not remove diagnostic positions from state and do not teach statistics
   modules about `vault_test_attempt` or `TradingPosition.simulated`.

### Tests

- Unit-test that `make_test_trade()` does not call statistics serialisation when
  the option is false.
- Unit-test that the default path still updates statistics.
- Add a vault-test regression with an existing no-trade diagnostic position and
  verify a later attempt reaches its real execution outcome.

## Work item 2: consume adapter receipt analysis

### Problem

The synchronous branch of `VaultRouting.settle_trade()` calls
`analyse_4626_flow_transaction()` directly. This ignores the already shared
`VaultDepositManager` analysis interface and rejects successful protocol event
variants.

The asynchronous settlement resolver already calls the manager methods, but it
has separate amount-to-trade conversion logic. The synchronous and asynchronous
paths must agree on signs, prices and token amounts.

### Changes

1. Dispatch overridden manager receipt analysers with:
   - `deposit_manager.analyse_deposit(HexBytes(tx_hash), None)` for a
     synchronous deposit; and
   - `deposit_manager.analyse_redemption(HexBytes(tx_hash), None)` for a
     synchronous redemption.
   Keep the existing generic ERC-4626 analyser only for managers that inherit
   it unchanged: the Safe wrapper path needs a ticket that synchronous trades
   do not persist yet.
2. Extract a pure trade-executor helper with a signature equivalent to
   `convert_vault_flow_analysis(direction, analysis)` that converts
   `DepositRedeemEventAnalysis` into:
   - positive reserve and share quantities for a deposit;
   - positive reserve and negative share quantity for a redemption; and
   - a positive reserve-per-share execution price.
3. Use the conversion helper in both synchronous routing and
   `settlement_retry.py`. Keep async-only bookkeeping, such as pending-capital
   allocation, refunds and clearing `vault_settlement_pending_at`, in the
   settlement resolver.
4. Keep receipt handling outside the pure conversion helper. The routing or
   settlement caller owns receipt-status validation and, for the synchronous
   path, calculates gas cost from `gasUsed * effectiveGasPrice`. Do not require
   the eth-defi analysis object to imitate `TradeSuccess`.
5. Introduce a trade-executor `VaultReceiptAnalysisError` at this boundary. It
   must identify the direction and transaction hash without embedding the full
   receipt. Raise it when adapter analysis throws or returns an invalid result
   for a status-1 receipt.
6. Treat `DepositRedeemEventFailure` as a failed trade with its supplied
   evidence when the receipt itself failed. A failure object for a status-1
   receipt is an invalid analysis result and raises `VaultReceiptAnalysisError`.
7. Validate non-zero denomination and share amounts before marking success.
   Keep cross-chain quote-token handling in decimal amounts; do not compare the
   source-chain reserve address with a satellite token address.
8. Keep the direct generic analyser import only for the documented Safe-wrapper
   compatibility path.

### Tests

- Unit-test deposit and redemption sign/price conversion.
- Extend synchronous vault routing tests with a mocked manager returning
  `DepositRedeemEventAnalysis`; assert exact executed amounts and price.
- Assert synchronous gas cost is calculated from the matching confirmed
  receipt.
- Test reverted `DepositRedeemEventFailure`, malformed status-1 analysis and an
  adapter exception separately.
- Assert that the latter two raise `VaultReceiptAnalysisError` carrying the
  matching transaction hash.
- Keep the existing async settlement lifecycle tests green and add assertions
  showing both paths use the same conversion helper.
- Once the eth-defi adapter branch is integrated, run one Ethereum-fork Ember
  deposit to prove the adapter method is dispatched.

## Work item 3: preflight explicit adapter support

### Problem

An explicitly unsupported manager shape can currently reach pricing or
execution and become `execution_failed`. Unknown support and known unsupported
support must not be conflated.

### Changes

1. After `_choose_operation()` determines `deposit` or `redeem`, inspect
   `vault.get_deposit_manager_capability()` before pricing or transaction
   construction.
2. If the capability explicitly sets the selected direction to `False`, record
   `adapter_unsupported` as a terminal result and include the eth-defi reason
   when available.
3. If the capability is `None`, continue through the existing path. Do not infer
   support by checking whether a method is overridden.
4. Do not catch arbitrary `NotImplementedError` globally: programming errors
   and incomplete unknown adapters must remain visible.

### Tests

- Explicit unsupported deposit and redemption directions produce
  `adapter_unsupported` without constructing a transaction.
- Unknown capability continues into normal execution.
- A supported direction is not blocked because the opposite direction is
  unsupported.

## Work item 4: normalise availability and capacity outcomes

### Problem

The generic evidence classifier is intentionally phase-based. It cannot by
itself distinguish a closed vault or capacity limit from an execution bug.
D2 currently reaches buy pricing before its live funding-window reason becomes
a terminal result. cSigma reaches transaction construction with an amount above
`maxRedeem(owner)`.

### Changes

1. Add a vault-test-specific typed-outcome normaliser that accepts the original
   exception and walks its cause chain before calling
   `classify_vault_test_failure()`.
2. Map `VaultReceiptAnalysisError` to `receipt_analysis_failed`. This explicit
   boundary, not the presence of any earlier status-1 receipt, establishes that
   receipt analysis failed.
3. Map structured `VaultFlowUnavailable` using its fields:
   - deposit availability to `deposit_closed`;
   - redemption availability to `redemption_unavailable`; and
   - a redemption with requested and lower available raw amounts to
     `redemption_capacity_limited`.
4. Preserve the structured raw amounts in `outcome_data` as decimal strings.
5. Before requesting a buy price, query
   `vault.fetch_deposit_closed_reason()`. A non-empty reason produces
   `deposit_closed` with the exact reason. Catch only the documented
   unavailable/read exceptions; unexpected adapter errors must still fail.
6. Once the shared eth-defi capacity query is available, run it before creating
   a synchronous redemption. Do not silently resize the redemption in this
   change: partial redemption changes the requested lifecycle and must remain an
   explicit future feature.
7. Keep status-0 transaction evidence authoritative so a real revert cannot be
   hidden by a broad preflight catch.
8. Remove the rule that any status-1 transaction implies
   `receipt_analysis_failed`. A multi-step attempt may have a successful bridge
   or deposit followed by an unrelated pricing, construction or statistics
   error.

Classification precedence must be deterministic:

1. A status-0 transaction created by the failing operation is
   `transaction_reverted`.
2. `VaultReceiptAnalysisError` is `receipt_analysis_failed`.
3. Structured `VaultFlowUnavailable` maps by direction and capacity fields.
4. Explicit capability or live closure preflights terminate before execution.
5. The phase/evidence classifier handles everything else.

When an attempt contains transactions from multiple lifecycle operations,
status-0 evidence must be associated with the current operation or transaction
boundary. An earlier reverted or successful bridge/deposit must not relabel a
later failure. Record the current operation's starting trade/transaction IDs in
`VaultAttemptContext`, or advance an equivalent evidence cursor before each
operation, and classify only evidence after that cursor.

### Tests

- D2-style dynamic closure is recorded before pricing and retains the next-open
  reason.
- Structured deposit closure, redemption closure and redemption capacity map to
  distinct outcomes.
- Capacity raw amounts survive state JSON round-trip and report export.
- A status-0 receipt still wins as `transaction_reverted`.
- A successful earlier deposit followed by a capacity or pricing failure does
  not become `receipt_analysis_failed`.
- A `VaultReceiptAnalysisError` for a matching status-1 receipt does become
  `receipt_analysis_failed`.
- An unrelated `ValueError`, `AssertionError` or `NotImplementedError` is not
  normalised by message text.

## Work item 5: clarify async simulation outcomes

### Problem

Without `--settle-async-on-anvil`, an async deposit-only simulation is a chosen
test scope, not evidence that the adapter lacks simulation support. With the
flag, a manager may still lack a safe driver.

### Changes

1. Keep `--settle-async-on-anvil` as the explicit full-lifecycle opt-in.
2. When the flag is false, record `async_request_only` with detail such as
   `Full async lifecycle was not requested`; do not claim the manager is
   unsupported.
3. When the flag is true and eth-defi explicitly reports no safe driver, record
   `simulation_unsupported_async` with direction and manager detail.
4. After the eth-defi workstream adds `supports_anvil_settlement`, update the
   submodule and use the field directly as a preflight. Keep
   `UnsupportedVaultSimulation` handling for a supported driver that rejects
   the concrete provider or ticket at runtime.
5. Do not auto-enable force settlement based solely on protocol or on the
   presence of a `force_settle` method.

Do not overload `success_simulated`, because the full deposit/redemption
lifecycle did not complete.

### Tests

- Async request-only mode records its configured scope in `detail`.
- Full-lifecycle mode with a supported driver proceeds to claim/redemption.
- Full-lifecycle mode with `UnsupportedVaultSimulation` records a terminal
  unsupported result and continues the batch.
- A settlement exception other than `UnsupportedVaultSimulation` remains a
  real execution or transaction failure.

## Work item 6: reporting and compatibility

1. Extend report and TUI presentation for `adapter_unsupported` and
   `redemption_capacity_limited`.
2. Export `outcome_data` unchanged in the JSON report.
3. Keep existing result strings readable and retain unknown future values.
4. Ensure report rows do not infer a result from display text when the raw
   attempt result exists.
5. Document the result meanings in the vault-test command/module documentation.

### Tests

- New result strings round-trip through state JSON and retain display labels.
- `outcome_data` exports unchanged through `export_vault_test_report()`.
- TUI/report rows display both new outcomes without deriving the persisted raw
  result from display text.
- An unknown future result and missing `outcome_data` remain backwards
  compatible.

## Implementation order

1. Isolate vault-test statistics updates. This removes the largest false error
   bucket without waiting for eth-defi.
2. Consume manager receipt analysis and share amount conversion between sync
   and async settlement.
3. Add result schema and typed outcome normalisation.
4. Add explicit adapter support and capacity preflights as their eth-defi
   capability changes land.
5. Clarify async simulation reporting.
6. Update report presentation and run focused tests.
7. Rerun the 129-vault list.

Keep commits reviewable along these boundaries. Isolate each eth-defi submodule
pointer update from the trade-executor consumer changes that follow it; do not
introduce a separate package-version requirement.

## Verification

Use the parent repository Poetry environment and put this worktree first on
`PYTHONPATH`:

```shell
source /home/mikko/code/trade-executor/.local-test.env
PYTHONPATH="/home/mikko/code/trade-executor-pr1574-ethdefi-vault-support:$PYTHONPATH" \
  poetry run pytest tests/cli/test_vault_test_trade.py -n auto
```

Run focused routing and settlement tests individually:

```shell
source /home/mikko/code/trade-executor/.local-test.env
PYTHONPATH="/home/mikko/code/trade-executor-pr1574-ethdefi-vault-support:$PYTHONPATH" \
  poetry run pytest tests/ethereum/test_vault_settlement_lifecycle.py -n auto
```

Run the existing simulated CLI fork test after unit coverage:

```shell
source /home/mikko/code/trade-executor/.local-test.env
PYTHONPATH="/home/mikko/code/trade-executor-pr1574-ethdefi-vault-support:$PYTHONPATH" \
  poetry run pytest tests/lagoon/test_lagoon_e2e.py \
  -k vault_test_trade_simulated_deploys_lagoon_on_fork
```

Finally rerun the same 129 explicit vault IDs twice:

1. Request-only/default mode, to verify configured async outcomes.
2. `--settle-async-on-anvil`, to verify capability-aware full lifecycle
   handling.

Compare the new report with
`/tmp/vault-test-pr1351-rerun-zbzRqL/report.json` when that temporary artefact is
still available.

## Acceptance criteria

- No vault result fails because long/short statistics inspect a diagnostic
  no-trade position.
- Synchronous routing uses overridden `VaultDepositManager` receipt analysis;
  unchanged generic ERC-4626 manager implementations retain the documented
  Safe-wrapper compatibility analyser.
- Ember status-1 deposits become successful once the eth-defi adapter change is
  present.
- YieldNest status-1 deposits become successful once its eth-defi ABI/parser
  change is present.
- D2 closed funding windows report `deposit_closed` before pricing.
- cSigma insufficient immediate redemption capacity reports
  `redemption_capacity_limited` with raw requested and available amounts once
  the shared eth-defi capacity query or structured exception is available.
- Explicitly unsupported Upshift multi-asset deposits report
  `adapter_unsupported` once eth-defi publishes the capability.
- Async request-only and unsupported full-lifecycle simulations are
  distinguishable.
- `receipt_analysis_failed` requires a successful mined receipt plus failed or
  invalid adapter analysis.
- A real status-0 receipt remains `transaction_reverted`.
- The batch continues after every non-infrastructure terminal outcome.
- Existing state files and unknown result values continue to round-trip.
