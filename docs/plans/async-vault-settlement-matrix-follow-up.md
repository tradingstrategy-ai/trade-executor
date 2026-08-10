# Async vault settlement matrix follow-up plan

## Objective

Support a full simulated redemption trial for every selected vault. Each
successful trial is a valid matrix success, but the persisted result must say
which simulation concession, if any, made it possible. The runner must try the
following modes in order for every vault that reaches its redemption leg:

1. **Simulated ok** — settle and claim/payout against the unmodified fork.
2. **Simulated, liquidity added** — repeat only after the strict mode cannot
   settle because the fork lacks redemption liquidity; inject the required
   fork-only liquidity and settle/claim/payout.
3. **Simulated, bypass closed status** — repeat only after the prior modes are
   blocked by a closed/paused/admission status; use an Anvil-only state or
   policy override, then settle/claim/payout.

All three modes are valid success results for trade-executor. They must remain
distinct so the report never implies that a live vault was liquid or open when
the fork trial supplied that condition.

## Verdict to correct

The 2026-07-29 follow-up comment correctly treats the affected outcomes as
simulation results, not proof that a live user transaction reverted. Its
Lagoon proposal is incomplete rather than unsafe: the existing
`LagoonDepositManager.force_settle(..., ignore_liquidity=True)` deliberately
mints fork-only Safe liquidity and returns both
`synthetic_assets_injected_raw` and `liquidity_constraints_ignored`. The
matrix should use that path as its second, explicitly labelled success mode,
not as an unqualified replacement for strict settlement.

The count of 15 must also be reproduced from one identified `--rerun` report
before it is used as an acceptance metric. The committed original report has
eight unsupported rows (four Ember, three Lagoon and one Base Gains), while
the committed targeted-rerun report carries five preserved terminal rows.
Neither artefact alone substantiates the comment's four Ember + nine Lagoon +
one Gains + one Plutus breakdown. The implementation must derive counts and
root-cause groups from the persisted latest result, rather than copy a
transient aggregate into a review comment.

## Scope and design

1. Re-run the named 15 vaults with the exact eth-defi and trade-executor
   revisions, fork blocks and `--rerun` inputs recorded in the artefact. Add a
   small report reducer that selects the latest terminal row per vault and
   groups it by `result` and `outcome_data.unsupported_reason`. Preserve the
   raw result and publish the derived table, so a matrix count is auditable.

2. Implement one ordered fallback state machine in `vault-test-trade`, used
   for every protocol and every vault rather than a Lagoon-only branch:

   - Run strict settlement first, with no balance or policy mutation. On a
     terminal claim/payout record `success_simulated` with
     `simulation_mode="simulated_ok"`.
   - If the adapter reports a typed liquidity limitation, snapshot the original
     balance, provision only the deficit on Anvil, and retry the same request.
     On a terminal claim/payout record `success_simulated` with
     `simulation_mode="simulated_liquidity_added"`, the injected raw amount,
     original balance and shortfall.
   - If a typed closed, paused, capped, or admission/whitelist status prevents
     the request, apply the protocol adapter's Anvil-only status bypass and
     retry from a clean fork snapshot. On a terminal claim/payout record
     `success_simulated` with
     `simulation_mode="simulated_bypass_closed_status"`, the exact bypassed
     status and original preflight evidence.

   A fallback applies only after the preceding mode reports its matching typed
   condition. Do not combine concessions in one attempt: the matrix must make
   it clear whether liquidity, closed status, or neither was required.

3. Thread a structured `SimulationTrialMode` rather than independent booleans
   from `vault-test-trade` through `VaultTestRunner`, `perform_test_trade()`,
   `_force_vault_settlement_and_resolve()` and eth-defi manager calls. Expose
   automatic ordered trials in matrix mode and explicit single-mode selection
   for focused debugging. Assert Anvil at the runner and adapter boundary, and
   reject all liquidity/status overrides in real test trades.

4. Replace the boolean-only settlement promise with a terminal-postcondition
   contract for supported async adapters. A manager may advertise Anvil
   settlement only if it can return enough evidence for the executor to move
   the trade out of `vault_settlement_pending`: either a request-specific
   claimable state followed by a successful claim, or a protocol-specific
   direct-payout result containing the matching request id/event and the
   receiver's positive asset-balance delta. On a failed postcondition raise
   `UnsupportedVaultSimulation` with a stable reason before the matrix records
   success.

5. Fix the Gains capability discrepancy first. The Arbitrum driver currently
   advertises `supports_anvil_settlement=True`, but the matrix can leave the
   satellite trade pending after the epoch hook. Trace the exact destination
   fork transaction sequence, including connection selection, epoch advance,
   claim and final trade status. Either make it complete and prove the terminal
   postcondition, or withdraw/defer the advertised capability for deployments
   that cannot be preflighted. Retain a typed reason for an unavailable
   operator/oracle path; never let a pending trade fall through as an execution
   failure.

6. Give every async adapter the same three trial-mode hooks, with safe default
   no-ops for concessions it does not need. Lagoon implements the liquidity
   hook through its existing `ignore_liquidity=True` provisioning path. Add
   Anvil-only closed-status bypass hooks for capped/paused/whitelisted vaults,
   recording the exact storage/policy mutation made. Treat Ember and Plutus as
   protocol-specific terminal-payout implementations: for Ember, identify the
   configured operator and request-sequence processing call, then verify the
   matching `RequestProcessed` event and receiver balance delta; for Plutus,
   discover the fulfilment role/holder from verified state and indexed
   `RoleGranted` events, impersonate only on Anvil, call
   `fulfillRedeem(requestId)`, and claim. These adapters must complete the
   same ordered trial contract rather than remain permanent exclusions from the
   all-vault matrix.

## Verification

- Add focused eth-defi fork coverage for each enabled adapter: request,
  settlement action, protocol-specific terminal postcondition and final
  executor claim/payout. Include the Gains destination-chain selection.
- Add Lagoon tests for no Safe liquidity, a one-raw-unit shortfall and a
  material shortfall. Strict mode must preserve the unmodified fork; the
  second trial must settle only on Anvil and persist the injected amount,
  original balance and `simulated_liquidity_added` mode.
- Add adapter tests for all three trial hooks. Cover closed, paused, capacity
  and whitelist preflight outcomes, verify a bypass starts from a clean fork
  snapshot, and prove no balance/policy override reaches a non-Anvil provider.
- Add positive Ember and Plutus fork coverage that completes the direct payout
  or fulfil-and-claim lifecycle. Keep a typed unsupported result only for a
  genuinely unimplemented adapter path, never for a vault merely requiring one
  of the ordered simulation trials.
- Add trade-executor tests for order, stop conditions, option plumbing and
  result normalisation, then run the targeted matrix with `--rerun`. The final
  report must list exact revisions, fork blocks, per-mode success counts and
  the remaining unsupported-reason grouping.
