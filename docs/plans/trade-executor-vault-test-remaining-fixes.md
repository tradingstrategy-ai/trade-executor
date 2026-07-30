# Trade-executor vault test remaining fixes

## Objective

Complete the remaining trade-executor work exposed by the 129-vault cross-chain
simulation run after integrating the eth-defi vault support from PR #1357.

The implementation must:

- execute synchronous vault flows through the selected eth-defi deposit manager;
- settle asynchronous request trades without assuming standard ERC-4626 selectors;
- classify unsupported adapters and closed deposit windows before pricing;
- bridge back only the USDC actually received from a satellite vault redemption;
- preserve transaction-evidence precedence within the operation that failed; and
- rerun adapter-dependent failures after the corresponding eth-defi fixes land.

The target is trustworthy simulation and diagnostics, not an artificial 100%
success rate. Live deposit closures, allow-list restrictions and deliberate
request-only asynchronous results remain valid terminal outcomes.

## Evidence and scope

This is a historical implementation plan. Its result counts are the original
129-vault baseline that motivated the work, not the latest published matrix.
For the later rerun, see
[`cross-chain-vault-test-2026-07-25-rerun-results.md`](../reports/cross-chain-vault-test-2026-07-25-rerun-results.md).

The baseline run covered 129 vaults and produced:

| Result | Count | Interpretation |
|---|---:|---|
| `success simulated` | 49 | Complete deposit and redemption simulation |
| `async_request_only` | 29 | Expected request-only result without forced settlement |
| `receipt_analysis_failed` | 2 | eth-defi event analysis gap |
| `execution_failed` | 10 | Mixed trade-executor and adapter gaps |
| `transaction_reverted` | 5 | Three satellite bridge-back sizing failures plus two adapter-specific failures |
| `deposit_closed` | 20 | Live vault restriction |
| `whitelisting-needed` | 14 | Live account restriction |
| `simulation_unsupported_async` | 0 | Previous generic async simulation gap removed |

This plan covers the trade-executor changes needed to resolve or correctly
classify the actionable rows. Protocol-specific transaction construction and
receipt parsing remain in eth-defi.

The ten `execution_failed` rows and the two non-bridge
`transaction_reverted` rows have the following disposition:

| Vault id | Vault | Current cause | Expected result after fixes | Owner |
|---|---|---|---|---|
| `1-0xd5d097f278a735d0a3c609deee71234cac14b47e` | cSigma USD | Generic synchronous redemption bypasses the manager capacity check | `redemption_capacity_limited` | trade-executor request routing |
| `1-0x438982ea288763370946625fd76c2508ee1fb229` | cSuperior Quality Private Credit USDC | Withdrawal is modelled as an immediate synchronous redemption | An honest queued/request-only result or typed unavailability | eth-defi lifecycle |
| `1-0x2b13311fd553e74b421d4ccc96e348f71e179dcf` | Ember Apollo ACRED | Redemption is below Ember's minimum | Typed `redemption_unavailable` with minimum/requested amounts | eth-defi failure contract |
| `1-0x9be9294722f8aad37b11a9792be2c782182cafa2` | Ember Earn | `redeemShares` is not selected for settlement | `async_request_only` or settled success, depending simulation mode | trade-executor request identity; eth-defi ticket metadata |
| `1-0x0b9342c15143e8f54a83f887c280a922f4c48771` | Ember Polymarket | `redeemShares` is not selected for settlement | `async_request_only` or settled success, depending simulation mode | trade-executor request identity; eth-defi ticket metadata |
| `1-0xf3190a3ecc109f88e7947b849b281918c798a0c4` | Ember Third Eye | `redeemShares` is not selected for settlement | `async_request_only` or settled success, depending simulation mode | trade-executor request identity; eth-defi ticket metadata |
| `1-0x373152feef81cc59502da2c8de877b3d5ae2e342` | Ember UDL | `redeemShares` is not selected for settlement | `async_request_only` or settled success, depending simulation mode | trade-executor request identity; eth-defi ticket metadata |
| `8453-0xad20523a7dc37babc1cc74897e4977232b3d02e5` | Gains gTrade USDC | `makeWithdrawRequest` is not selected for settlement | `async_request_only` or settled success, depending simulation mode | trade-executor request identity; eth-defi ticket metadata |
| `42161-0xd3443ee1e91af28e5fb858fbd0d72a63ba8046e0` | Gains gTrade USDC | `makeWithdrawRequest` is not selected for settlement | `async_request_only` or settled success, depending simulation mode | trade-executor request identity; eth-defi ticket metadata |
| `42161-0x75288264fdfea8ce68e6d852696ab1ce2f3e5004` | D2 HYPE++ | Closed funding window reaches pricing as a plain exception | `deposit_closed` with next-open detail | trade-executor preflight; eth-defi structured reason |
| `143-0x7cd231120a60f500887444a9baf5e1bd753a5e59` | Accountable Hyperithm Delta Neutral Vault | Deposit reverts with undecoded custom error `0x5945ea56` | Decoded `transaction_reverted` or typed preflight unavailability | eth-defi adapter |
| `1-0x74ad2f789ed583dbd141bbdafc673fe1f033718b` | Upshift Sentora USD Earn | Generic ERC-4626 manager cannot deposit USDC into the multi-asset application flow | `adapter_unsupported` until the eth-defi USDC manager lands, then simulated success | trade-executor preflight; eth-defi P0 implementation |

The three remaining transaction reverts are the Aerodrome USDC, Autopilot USDC
Morpho and Pharaoh USDC cross-chain close-accounting regressions covered by work
item 4. The two receipt-analysis rows are covered by work item 6.

## In scope

- `tradeexecutor/ethereum/vault/vault_routing.py`
- `tradeexecutor/ethereum/swap.py`, only where a generic compatibility change is
  unavoidable
- `tradeexecutor/cli/vault_trade/runner.py`
- `tradeexecutor/cli/vault_trade/state.py`
- `tradeexecutor/cli/testtrade.py`
- CCTP and portfolio accounting helpers used by the cross-chain test-trade path
- focused unit, state lifecycle and fork tests
- rerunning the same 129 explicit vault ids and publishing a comparable result
  table

## Out of scope

- Implementing Upshift multi-asset USDC transactions in trade-executor
- Adding protocol-specific receipt decoders in trade-executor
- Treating a closed or allow-listed vault as a software failure
- Automatically reducing a requested cSigma redemption to its current capacity
- Forcing asynchronous settlement on live chains
- Changing the semantics of the 29 deliberate `async_request_only` results

## Dependency contract

Trade-executor must consume the public eth-defi manager interfaces instead of
reimplementing protocol rules:

- `get_deposit_manager_capability()` describes whether deposit and redemption
  lifecycles are implemented and whether each lifecycle is synchronous or
  asynchronous.
- `fetch_deposit_closed_reason()` returns a current, human-readable closure
  reason when the adapter can determine one.
- `create_deposit_request()` and `create_redemption_request()` own amount checks
  and return the ordered protocol calls.
- A request's `funcs` are the authoritative ordered lifecycle calls. A call
  returned in `funcs` is always tagged and parsed as a request call, even if its
  selector happens to be `approve`.
- Any token allowance transaction which is not part of `funcs` is an executor
  prerequisite with a separate `vault_approval` role. For a non-vault spender,
  eth-defi must expose explicit approval requirements on the request; the
  compatibility default for older request types remains the selected asset and
  vault address. Trade-executor must not guess a spender from a selector.
- `VaultFlowUnavailable` carries a typed direction, phase, requested amount and
  available amount.
- `DepositRequest.parse_deposit_transaction()` and
  `RedemptionRequest.parse_redeem_transaction()` interpret the complete ordered
  request transaction-hash list.
- `analyse_deposit()` and `analyse_redemption()` return a valid
  `DepositRedeemEventAnalysis` or a typed failure.

Unknown capability metadata must retain the existing best-effort execution path
for older adapters. An explicit unsupported capability must fail closed as
`adapter_unsupported`.

Upshift multi-asset vaults are a priority dependency: until eth-defi advertises
and implements the selected USDC deposit lifecycle, trade-executor must report
`adapter_unsupported` before pricing. Once eth-defi adds the USDC manager,
trade-executor should route through it without an Upshift-specific branch.

## Work item 1: use manager-owned request construction

Refactor `VaultRoutingModel.make_direct_trade()` so both synchronous and
asynchronous flows begin with the selected manager's request builder.

1. Resolve the executable vault and its deposit manager once.
2. Call `create_deposit_request(owner=..., raw_amount=...)` for buys and
   `create_redemption_request(owner=..., raw_shares=...)` for sells.
3. Sign the request's ordered `funcs` rather than recreating generic
   `deposit()` or `redeem()` calls with `approve_and_deposit_4626()` and
   `approve_and_redeem_4626()`.
4. Build any separate allowance transaction from the request's explicit
   approval requirement, falling back to the existing asset-to-vault allowance
   only for legacy request types. Preserve existing guarded Safe/Lagoon wrapper
   behaviour and do not assume every request call is sent directly to
   `vault_address`.
5. Keep synchronous settlement synchronous, but retain sufficient request
   metadata for its manager analyser.
6. Let typed `VaultFlowUnavailable` exceptions propagate to the vault runner.

This makes the cSigma capacity-aware manager authoritative. A redemption above
the available immediate capacity must become
`redemption_capacity_limited`; trade-executor must not silently resize it.

Unit tests must use a specialised synchronous manager whose request function and
target differ from the generic ERC-4626 call. They must prove that the manager
builder is invoked, calls retain their order and a typed capacity failure
survives transaction building. Add asynchronous buy and sell cases proving both
manager request builders are used and the legacy generic helpers are not.

## Work item 2: make async receipt selection manager-aware

Remove the dependency between asynchronous vault settlement and the global
fixed-selector list in `is_swap_function()`.

At construction time, tag every signed transaction with a durable
`vault_approval` or `vault_request` role. Persist the request ordinal and its
position within the trade's transaction list as identity; record a transaction
hash as execution evidence only, never as the durable identity because retry
signing can change it. Continue storing the raw amount, owner and direction
required to reconstruct the eth-defi request.

At settlement time:

1. Detect `vault_async_flow` before calling `get_swap_transactions()`.
2. Select all transactions tagged `vault_request`, ordered by their persisted
   request ordinal. A separate prerequisite is excluded because its role is
   `vault_approval`, not because its selector looks like an approval.
3. Verify that every selected request transaction has a successful receipt.
4. Pass the complete ordered hash list to
   `parse_deposit_transaction()` or `parse_redeem_transaction()`.
5. Use the final request receipt for the lifecycle timestamp unless the parsed
   ticket exposes a more precise timestamp.
6. Keep `get_swap_transactions()` for non-async swaps and synchronous
   compatibility only.

Do not add `redeemShares`, `makeWithdrawRequest` or future protocol names to the
global selector set. The design must support a manager returning one or several
arbitrarily named calls.

Tests must cover:

- approval plus one arbitrary request call;
- a request containing multiple ordered protocol calls;
- Ember-like `redeemShares` and Gains-like `makeWithdrawRequest` selectors;
- serialise-and-reload before settlement;
- a reverted or missing receipt in the selected request set; and
- unrelated transactions attached to the same trade not being parsed as part
  of the request.

Persist these additions in optional `other_data` fields so existing state files
deserialise without a schema migration. For a legacy pending async trade with no
roles, use the existing `vault_request_tx_count` once to identify the final
ordered request transactions, stamp the new roles, and persist the upgraded
state before retrying. Test loading and settling a pre-change state fixture.

## Work item 3: preflight the resolved executable vault

Make capability, admission and closure checks operate on the same resolved vault
adapter that pricing and routing will execute.

1. Resolve the routed vault once during attempt preparation and store it on the
   attempt.
2. Read capability and deposit-window methods from that resolved object instead
   of relying on optional methods on discovery metadata.
3. Run adapter capability, allow-list and deposit-closure checks before
   `pricing_model.can_deposit()` or any estimate call.
4. Include the adapter's unsupported or closure reason and serialised capability
   in `outcome_data`.
5. Preserve best-effort fallback only when the adapter genuinely has no
   capability API.

Expected regressions:

- D2's “Funding phase closed” condition is `deposit_closed`, with its opening
  detail, rather than a pricing `RuntimeError`.
- An unsupported Upshift multi-asset USDC flow is `adapter_unsupported` and does
  not reach pricing or `get_deposit_manager()`.
- A future supported Upshift USDC manager follows the normal routing path with
  no trade-executor protocol special case.

Unit tests must cover an explicitly unsupported capability, a closed deposit
window, an admitted supported adapter, and an older adapter with no capability
API. The older-adapter case must continue down the best-effort execution path
instead of failing closed.

## Work item 4: bridge exact satellite redemption proceeds

Correct the cross-chain close path in `_make_cross_chain_test_trade()` and its
close-only resume path. `TradingPosition.get_available_bridge_capital()` is an
accounting value; it must not be the sole source for a CCTP burn amount when the
wallet received less USDC than planned.

1. Read the satellite denomination-token balance immediately before the
   redemption request or synchronous close, write it to the attempt's
   `other_data`, and durably sync the state store before signing or broadcasting
   any redemption transaction.
2. Do not create a bridge-back trade while an asynchronous redemption remains
   pending.
3. After successful settlement, calculate newly received USDC from the persisted
   pre-redemption balance and current wallet balance, then compare it with the
   close trade's `executed_reserve`. Never substitute a fresh post-settlement
   balance for a missing baseline.
4. Reconcile the bridge allocation with the actual settled proceeds. Retain the
   smaller safe amount when RPC balance evidence and analysed event evidence
   differ, and record the discrepancy for diagnostics.
5. Build the CCTP close for only that reconciled amount, converted with
   `TokenDetails.convert_to_raw()`. Leave fee and rounding dust on the
   satellite chain rather than requesting an impossible transfer.
6. Persist the measured balance baseline and reconciled bridge-back amount so a
   resumed close is idempotent and does not measure the same proceeds twice.
7. Before constructing a burn, find a persisted bridge-back trade for this
   attempt. If it already has a signed or broadcast transaction, inspect and
   resume that transaction or its CCTP attestation rather than creating a
   second burn. Persist the trade id, transaction identity and lifecycle phase.
8. Represent unbridged raw-unit dust as a residual quantity on the still-open
   satellite bridge position, with the measured token balance recorded in
   `other_data`. It may fund a later attempt on that chain. Do not write it off
   or allow portfolio availability to exceed the on-chain balance; assert the
   two agree within one raw token unit after reconciliation.
9. Keep the general portfolio bridge accounting consistent: profit may make
   allocated bridge capital negative, while a loss or fee must not leave a
   phantom amount available to burn.

Store the balance baseline, reconciled amount, residual and resume identity as
optional versioned `other_data` fields. Missing fields in older bridge
positions mean “not yet measured” only when no redemption transaction has been
signed. If a legacy or inconsistent state has a signed or settled redemption but
no baseline, derive the bridge-back amount solely from the manager's analysed
redemption event evidence on the persisted transaction hashes. If neither a
persisted baseline nor valid event analysis is available, raise a typed
`bridge_proceeds_unavailable` outcome and do not construct a CCTP burn. No new
required dataclass field may break loading an existing state file.

Add unit tests for profit, loss, protocol fee, decimal dust, zero proceeds and a
resumed asynchronous redemption. The resume test must include a crash after burn
broadcast but before receipt or attestation processing, and prove that no second
burn is signed. Add a forced legacy-state crash fixture with a settled redemption
but no baseline: valid analysed event evidence must recover the amount, while
missing analysis must produce `bridge_proceeds_unavailable` without signing a
burn. The dust test must assert both the open residual state and its agreement
with the wallet balance. Add focused fork regressions for at least one Base
failure and the Avalanche Pharaoh failure. The three current regression vaults
are Aerodrome USDC and Autopilot USDC Morpho on Base, and Pharaoh USDC on
Avalanche.

## Work item 5: scope failure evidence to the current operation

Keep a status-zero transaction as the highest-priority classification, but only
when that transaction belongs to the operation that raised the failure.

1. Extend `VaultAttemptContext` with an operation evidence cursor captured
   immediately before each preflight, build, execute or settlement operation.
2. Capture only transactions and call context newer than that cursor when
   recording the operation's failure.
3. Preserve the current precedence within that evidence:
   `transaction_reverted`, `broadcast_failed`, `gas_estimation_reverted`, then
   typed or phase-derived failure.
4. Persist the operation and phase with the evidence so reports remain
   explainable after the Anvil fork is gone.

Tests must prove:

- a successful earlier CCTP bridge does not hide a later typed capacity result;
- an unrelated earlier revert does not turn a later preflight failure into
  `transaction_reverted`;
- a status-zero transaction produced by the current operation overrides a
  wrapped typed exception; and
- a settlement failure sees request or claim transactions from that settlement
  operation, but not unrelated transactions from another vault attempt.

## Work item 6: rerun adapter-dependent receipt failures

Do not add Yearn or YieldNest event parsing to trade-executor. After eth-defi
returns valid `DepositRedeemEventAnalysis` values for Yearn Arche USD redemption
and YieldNest RWA MAX deposit:

1. add focused routing tests that exercise the manager analyser result;
2. verify `VaultRoutingModel.settle_trade()` records executed shares, reserves
   and price from that result; and
3. rerun those explicit vault ids before the full matrix.

If eth-defi still returns `DepositRedeemEventFailure` or raises during analysis,
trade-executor must retain the typed `receipt_analysis_failed` outcome with the
transaction hash and direction.

## Test strategy

Implement the smallest focused layers first:

1. Unit-test manager request construction and persisted async request identity.
2. Unit-test resolved-adapter preflight and operation-scoped classification.
3. Unit-test bridge-back reconciliation and resume state.
4. Run the existing CLI vault and settlement lifecycle modules.
5. Run focused Base and Avalanche fork regressions.
6. Run the same 129 explicit vault ids, using the same funding, block and async
   settlement settings as the baseline.

The final report must include counts by terminal result and a protocol-level
table for every remaining actionable gap. Each row must state the vault id,
chain, protocol, operation, result, short cause and owning repository.

## Implementation order

1. Introduce manager-owned request construction and request transaction roles.
2. Update asynchronous parsing to consume those roles.
3. Resolve and preflight the executable adapter.
4. Add operation evidence cursors.
5. Reconcile actual satellite redemption proceeds before CCTP bridge-back.
6. Add adapter-dependent regression tests after the matching eth-defi changes.
7. Run focused tests, then the unchanged 129-vault matrix.

This order establishes the transaction and diagnostic contracts before changing
cross-chain accounting, which makes later failures attributable to one
operation.

## Acceptance criteria

- Specialised synchronous managers, including cSigma, own request validation and
  transaction construction.
- Capacity-limited cSigma redemptions are
  `redemption_capacity_limited`, with requested and available raw shares.
- Arbitrarily named asynchronous request functions settle without changes to
  `is_swap_function()`.
- D2 funding-window closures are `deposit_closed` before pricing.
- Upshift multi-asset USDC is `adapter_unsupported` until eth-defi advertises a
  supported USDC manager, then automatically uses that manager.
- Satellite bridge-back never exceeds the USDC actually made available by the
  completed redemption.
- A missing pre-redemption baseline never falls back to a post-settlement wallet
  read: valid manager event analysis recovers the amount, otherwise the attempt
  stops as `bridge_proceeds_unavailable` without a burn.
- The Aerodrome, Autopilot and Pharaoh bridge-back regressions no longer revert
  with an insufficient token balance.
- Only current-operation status-zero evidence takes precedence over typed
  outcomes.
- Yearn and YieldNest receipt results become successful when their eth-defi
  analysers return valid event analysis; otherwise they remain explicit
  `receipt_analysis_failed` rows.
- The baseline's 20 `deposit_closed`, 14 `whitelisting-needed` and 29
  `async_request_only` outcomes are not treated as acceptance failures.
- Every vault in the twelve-row disposition table reaches its stated expected
  outcome, or remains as an explicitly owned eth-defi dependency with the
  transaction evidence needed to reproduce it.
- The final matrix accounts for all 129 requested vault ids without an
  unclassified exception.

`whitelisting-needed` is an existing persisted public result literal, not an
enum member or value introduced by this work. Renaming it is outside this plan
because it would require a report-schema compatibility migration.

## Verification commands

Run tests from this worktree through the parent repository's Poetry environment,
with the worktree and its eth-defi submodule first on `PYTHONPATH`:

```shell
source /home/mikko/code/trade-executor/.local-test.env
cd /home/mikko/code/trade-executor
poetry run bash -c 'cd "$1"; export PYTHONPATH="$PWD:$PWD/deps/web3-ethereum-defi:$PYTHONPATH"; exec pytest -n auto tests/cli/test_vault_trade_*.py' bash /home/mikko/code/trade-executor-pr1574-ethdefi-vault-support
```

```shell
source /home/mikko/code/trade-executor/.local-test.env
cd /home/mikko/code/trade-executor
poetry run bash -c 'cd "$1"; export PYTHONPATH="$PWD:$PWD/deps/web3-ethereum-defi:$PYTHONPATH"; exec pytest -n auto tests/ethereum/vault/test_vault_settlement_*.py' bash /home/mikko/code/trade-executor-pr1574-ethdefi-vault-support
```

Run new focused test files individually with the same environment. Use
`--log-cli-level=info` only while diagnosing a failure, not in committed test
configuration. When invoking a multi-test command through the execution tool,
set its timeout to at least 360 seconds; use at least 180 seconds for an
individual test command.
