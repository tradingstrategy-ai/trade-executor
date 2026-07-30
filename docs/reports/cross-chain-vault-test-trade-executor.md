# Trade-executor cross-chain vault test report

## 2026-07-30 settlement update

The 2026-07-25 matrix below is a historical baseline, not the current Ember or
Plutus support status. With the follow-up eth-defi settlement driver and
executor direct-payout recovery, a targeted rerun completed the selected five
previously unresolved async redemptions:

| Protocol | Vaults | Current result |
|---|---:|---|
| Ember | 4 | Previously `simulation_unsupported_async`; now success (simulated) after operator direct-payout recovery |
| Plutus | 1 | Previously redemption unavailable; now success (simulated) through the Safe module |

The Ember result can include synthetic Anvil denomination-token liquidity. It
proves that the request, operator settlement, receipt analysis and executor
lifecycle are wired correctly; it does **not** prove that the live vault or
operator can fund its FIFO queue. The selected rerun is not a replacement for a
new 129-vault matrix. The source files were `/tmp/async-vault-settlement-final-4/report.json`
and `/tmp/async-vault-settlement-final-4/run.log`, using trade-executor branch
`fix/plutus-satellite-settlement` with eth-defi branch
`fix/async-vault-anvil-settlement` first on `PYTHONPATH`.

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

## 2026-07-25 historical matrix output

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

## What the 2026-07-25 eth-defi update changed (observed)

- **40acres: 2 → 1 gap.** 40acres Aerodrome USDC (Base) now completes deposit
  and redemption — the PR #1577 redemption-share reconciliation fix is
  effective on that path.
- **Ember / Gains async classification.** Four Ember vaults and Gains on Base
  then reported `simulation_unsupported_async` with
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
| Ember | 5 | 5 | eth-defi (Anvil settlement driver + minimum typing); four queue rows resolved in the 2026-07-30 targeted rerun |
| cSigma Finance | 2 | 2 | eth-defi (capacity typing + async withdrawal) |
| Gains Network | 2 | 2 | eth-defi (Anvil settlement + custom error) |
| YieldNest | 1 | 1 | eth-defi (custom error decode) |
| Plutus | 1 | 1 | eth-defi (live-state vs adapter) |
| Accountable | 1 | 1 | eth-defi (Monad selector) |
| Upshift | 1 | 1 | trade-executor + eth-defi (preflight routing) |
| 40acres | 1 | 2 | eth-defi (liquidity preflight, PR #1378) — see section 2 |
| IPOR Fusion | 1 | 0 | trade-executor (gas limit) — **resolved**, see §4 |

## Trade-executor work items

Protocol-specific settlement actions belong in eth-defi. Trade-executor owns the
generic lifecycle around them, including scanning closed positions and recovering
an operator-finalised direct payout through the manager's validated transaction
lookup. The items below are diagnostics, cross-chain reconciliation and
provenance. Sections marked *done* were implemented on this branch; section 2
was investigated and found not to be an executor defect at all.

### 1. Surface the real revert reason for home-chain vault redemptions — done

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

### 2. 40acres Pharaoh — not a reconciliation bug (superseded)

An earlier revision of this report claimed the satellite close path needed
redemption-share reconciliation, by analogy with the PR #1577 home-chain fix.
**That was wrong.** The existing reconciliation already runs correctly on the
satellite path and requests exactly what is held:

```text
Vault redeem. Position quantity 0.916656, trade quantity -0.916656,
              onchain balance 0.916656, position planned quantity 0.916656
Onchain balance covers the planned shares to redeem: planned 0.916656, onchain 0.916656
```

The real cause is vault-side: `43114-0x124d00b1ce4453ffc5a5f65ce83af13a7709bac7`
holds **zero** idle underlying while reporting ~510k USDC of `totalAssets`, so
`redeem()` cannot transfer the underlying. That is an adapter preflight gap,
addressed by eth-defi #1378, which returns a typed
`redemption_capacity_limited` scoped to this exact deployment so 40acres
Aerodrome continues to succeed.

No executor change is required.

### 3. Route the Upshift deposit preflight through the eth-defi manager — done

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

### 4. IPOR Fusion satellite-close regression — resolved

`Autopilot USDC Morpho (Base)`
(`8453-0xd6701905c59ee618dc36dc747506bce0a4ac760a`) deposited, then its Base
satellite redemption reverted with OpenZeppelin `FailedInnerCall()`
(`0x1425ea42`).

This was **our own gas cap, not a vault or adapter defect**. The redemption
spends 10,489,529 gas because the PlasmaVault requests liquidity from its Morpho
market fuses, just over the previous 10,000,000
`VaultRouting.vault_interaction_gas_limit`. The out-of-gas inner call surfaced
through OpenZeppelin's wrapper error, which reads like a vault liquidity failure
from the outside.

Fixed by raising the limit to 15,000,000 — above the measured cost, below Base's
16,777,216 per-transaction cap. Verified: with that change alone, and no adapter
preflight, the vault completes a full deposit and redemption lifecycle.

The lesson generalises: an unmetered `eth_call` preflight cannot observe a
caller-side gas limit, so gas used versus the caller's cap must be ruled out
before concluding that a vault cannot service a redemption.

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
