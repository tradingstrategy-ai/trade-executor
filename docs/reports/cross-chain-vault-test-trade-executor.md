# Trade-executor cross-chain vault test report

## Evidence and scope

The 129-vault cross-chain matrix was re-run on 2026-07-25 with the updated
eth-defi vault support:

- trade-executor `f311d0fa` (merged PR #1577 `0e84f313` plus the eth-defi
  submodule bump);
- eth-defi master `b42ef5747` (#1368 "close vault simulation adapter gaps");
- worktree-first imports verified for both `tradeexecutor` and the eth-defi
  submodule; and
- `--auto-simulated --settle-async-on-anvil`.

The machine-readable report is
`docs/reports/cross-chain-vault-test-2026-07-25.report.json`, the ordered vault
list is `docs/reports/cross-chain-vault-test-vault-ids.txt`, and the full
per-vault table is
`docs/reports/cross-chain-vault-test-2026-07-25-results.md`. This report covers
the executor-side work; protocol adapter work is in the separate eth-defi
report.

## Corrected matrix output

| Status | Vaults | Baseline (eth-defi b5803bdc5) | Interpretation |
|---|---:|---:|---|
| Success (simulated) | 43 | 43 | Deposit and redemption lifecycle completed |
| Deposit closed | 51 | 51 | Current on-chain admission (mostly Yearn `maxDeposit=0`) |
| Whitelisting needed | 14 | 14 | Simulated executor Safe is not admitted |
| Simulation unsupported async | 8 | 3 | Async redemption ticket left pending, no Anvil driver |
| Transaction reverted | 6 | 5 | Home-chain or satellite redemption/deposit revert |
| Execution failed | 3 | 2 | Typed capacity/minimum/preflight failures |
| Broadcast failed | 3 | 3 | Lagoon `settleDeposit()` allowance reverts |
| Redemption unavailable | 1 | 1 | Plutus accepted the deposit but offered no redemption |
| Redemption pending | 0 | 6 | Now resolved to typed async-unsupported/reverted |
| Adapter unsupported | 0 | 1 | Upshift now routed to a manager (see below) |

The **total gap count is unchanged at 21**. The lifecycle completion count (43
success) is identical to the baseline, so the eth-defi update did not regress
any previously working vault. What improved is *classification*: the six
baseline "redemption pending" rows (four Ember, two Gains) that were not
recognised as asynchronous are now terminal typed results, and Ember's minimum
withdrawal and async capability are surfaced with structured metadata.

## What the eth-defi update changed (observed)

- **40acres: 2 → 1 gap.** 40acres Aerodrome USDC (Base) now completes deposit
  and redemption — the PR #1577 redemption-share reconciliation fix is
  effective on that path.
- **Ember / Gains async classification.** Four Ember vaults and Gains on Base
  now report `simulation_unsupported_async` with
  `VaultDepositManagerCapability` metadata (`deposit_flow=synchronous`,
  `redemption_flow=asynchronous`) instead of the previous
  "Failed to analyse vault tx" receipt errors.
- **Ember Apollo ACRED minimum** is now a typed `execution_failed`
  ("redemption shares 904 are below minimum 9170000") rather than a generic
  `ValueError`.

## Gap rows by protocol (new run)

| Protocol | Gaps | Baseline | Owner of remaining fix |
|---|---:|---:|---|
| Lagoon Finance | 6 | 6 | eth-defi (settlement diagnostics + allowance path) |
| Ember | 5 | 5 | eth-defi (Anvil settlement driver + minimum typing) |
| cSigma Finance | 2 | 2 | eth-defi (capacity typing + async withdrawal) |
| Gains Network | 2 | 2 | eth-defi (Anvil settlement + custom error) |
| YieldNest | 1 | 1 | eth-defi (custom error decode) |
| Plutus | 1 | 1 | eth-defi (live-state vs adapter) |
| Accountable | 1 | 1 | eth-defi (Monad selector) |
| Upshift | 1 | 1 | trade-executor + eth-defi (preflight routing) |
| 40acres | 1 | 2 | trade-executor (satellite reconciliation) |
| IPOR Fusion | 1 | 0 | trade-executor (satellite close diagnostics) — **new** |

## Remaining trade-executor work

No protocol-specific settlement logic belongs in trade-executor. The executor
work below is diagnostics, cross-chain reconciliation and provenance.

### 1. Surface the real revert reason for home-chain vault redemptions

`cSuperior Quality Private Credit USDC`
(`1-0x438982ea288763370946625fd76c2508ee1fb229`) and `YieldNest RWA MAX`
(`1-0x01ba69727e2860b37bc1a2bd56999c1afb4c15d8`) both display only
`Test sell failed` in the result table, even though the executor log records the
actual reverts:

- cSuperior: `execution reverted: Withdrawal pending`;
- YieldNest: `execution reverted: custom error 0xb8b8b59c: 0000…a2b04c6a…`.

Satellite-chain closes already surface the revert reason
("Satellite close failed: execution reverted: custom error 0x…"). Home-chain
vault trades must carry the same `revert_reason` into the diagnostic result
detail so the table is actionable and eth-defi can be given the exact selector.

### 2. Apply redemption-share reconciliation on the satellite close path

`40acres Pharaoh USDC` (`43114-0x124d00b1ce4453ffc5a5f65ce83af13a7709bac7`,
Avalanche) still fails redemption with
`Satellite close failed: execution reverted: ERC20: transfer amount exceeds
balance`. This is the same class of shortfall the PR #1577 reconciliation fix
resolved for the home chain (40acres Aerodrome now succeeds). The satellite
redemption/close path must reconcile the planned share quantity against the
actual satellite module balance before building the burn, reusing the existing
`get_available_bridge_capital()`/epsilon reconciliation rather than requesting
the planned quantity.

### 3. Route the Upshift deposit preflight through the eth-defi manager

`Sentora USD Earn` (`1-0x74ad2f789ed583dbd141bbdafc673fe1f033718b`) now fails at
deposit with `The function 'maxDeposit' was not found in this contract's abi`.
eth-defi #1368 added a dedicated Upshift multi-asset `VaultDepositManager`, but
the executor's deposit-availability preflight still calls generic ERC-4626
`maxDeposit()` on this non-standard vault. The preflight (`can_deposit` /
`fetch_deposit_closed_reason`) must ask the resolved manager for its deposit
capability/preview when the manager provides one, and fall back to generic
`maxDeposit()` only for standard ERC-4626 vaults. (eth-defi should also expose a
`maxDeposit` shim or preview on the Upshift vault reader; see the eth-defi
report.)

### 4. Investigate the new IPOR Fusion satellite-close regression

`Autopilot USDC Morpho (Base)`
(`8453-0xd6701905c59ee618dc36dc747506bce0a4ac760a`) is a new gap that was not
present in the baseline: it deposits, then the Base satellite redemption reverts
with a bare `Satellite close failed: execution reverted` (no reason). IPOR
Fusion redeems synchronously and succeeds on Ethereum in the same run, so the
failure is specific to the cross-chain satellite close. Surface the decoded
revert reason first, then determine whether this is a satellite guard/slippage
issue in executor routing or a genuine protocol state to be typed.

### 5. Continue mapping typed eth-defi exceptions to stable terminal results

Ember minimum and async-unsupported now flow through cleanly as typed results.
Extend the same mapping to the cSigma capacity assertion and the cSuperior
`Withdrawal pending` case once eth-defi raises `VaultFlowUnavailable` /
`UnsupportedVaultSimulation` for them, so the executor never has to infer
protocol policy from generic RPC assertion text.

### 6. Record the eth-defi commit in report provenance

The report JSON still records `eth_defi_commit=null`. The executor imports the
submodule at a known commit (this run: `b42ef5747`) and must persist it in
`run` provenance so a matrix can be reproduced without external notes.

## Verification

| Check | Result |
|---|---:|
| Full 129-vault matrix rerun | 129/129 completed, exit 0 |
| trade-executor imports vs eth-defi master | clean (`vault_routing`, `testtrade`, `runner`, command) |
| Previously working vaults | 43/43 success retained, no regression |
| Focused smoke (Morpho/Ember/cSigma/Lagoon) | matches full-run classifications |
