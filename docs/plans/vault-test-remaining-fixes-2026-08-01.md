# Vault test remaining fixes

## Objective

Resolve the remaining issues from the 2026-08-01 ordered 129-vault production
matrix and add a successful alternative Anvil simulation for every implemented
redemption lifecycle that is blocked by current vault state.

The alternative path must prove the real adapter request, settlement and claim
mechanics. It may use disclosed synthetic capital, time advancement or an
authorised protocol action on an Anvil fork. It must not conceal the production
constraint that made the natural path unavailable.

## Baseline

The verified rerun used:

- trade-executor `a073b35769ad2dd27de57831c25a949849f8d1c6`;
- eth-defi `dce56272cbd189d5812c0880cf7620a455192eca`;
- 1,001 USDC per vault;
- `--auto-simulated` and `--settle-async-on-anvil`; and
- the 129 ordered ids in
  `docs/reports/cross-chain-vault-test-vault-ids.txt`.

The machine report is
`/tmp/vault-test-pr1597-rerun-2026-08-01-worktree-1001/report.json`.

| Result | Count | Disposition |
|---|---:|---|
| `success (simulated)` | 87 | No action |
| `success-deposit-closed` | 18 | Retain Guard validation |
| `success_simulated_with_intervention` | 2 | Retain disclosure and economic exclusion |
| `whitelisting-needed` | 14 | Expected production gate; no action |
| `below_minimum` | 1 | Add a minimum-aware alternative amount |
| `incompatible_deposit_asset` | 1 | Add an accepted-asset alternative run |
| `redemption_capacity_limited` | 1 | Add disclosed liquidity intervention |
| `redemption_not_yet_matured` | 1 | Add time-advanced alternative run |
| `redemption_window_closed` | 1 | Advance to the next request window |
| `redemption_zero_payout` | 1 | Diagnose and add source-proven payout preparation |
| `transaction_reverted` | 1 | Fix satellite redemption or classify it before broadcast |
| `infrastructure_failed` | 1 | Re-run alone and fix only if reproducible |

The headline target is not to relabel every production constraint as ordinary
success. It is to retain an honest observed result and, where the adapter is
implemented, also prove a successful alternative lifecycle.

## Result model

Replace the current single-outcome compromise with two explicit layers in the
machine report:

```json
{
  "observed_result": "redemption_window_closed",
  "observed_detail": "Gains withdrawal request window is closed",
  "alternative_simulation": {
    "succeeded": true,
    "interventions": [
      {
        "kind": "time_advanced",
        "timestamp_before": "2026-08-01T00:00:00",
        "timestamp_after": "2026-08-02T00:00:00"
      }
    ],
    "transactions": ["0x..."],
    "assets_returned_raw": "1000000000"
  }
}
```

For compatibility, `row.result` remains the terminal matrix result. When the
natural path is constrained and the alternative succeeds, set it to
`success_simulated_with_intervention`; preserve the original classification in
`observed_result`. Existing consumers that only read `row.result` continue to
work, while research and production-readiness reporting retain the warning.

An intervention record must include:

- `kind`;
- chain, protocol and vault address;
- target contract or account;
- token and raw amount for synthetic capital;
- actor, role, function and transaction hash for privileged actions;
- time before and after any fork time advancement;
- the original typed failure; and
- the real request, settlement and claim transaction hashes.

Rows with interventions must remain excluded from return, slippage, solvency
and other economic aggregates.

## Safety invariants

1. Interventions are available only in automatic simulation on Anvil.
2. The unmodified natural lifecycle is always attempted first.
3. Live and manual execution never inject assets, alter time, impersonate an
   operator or relax a protocol constraint.
4. A successful alternative must broadcast and analyse the adapter's real
   calls. Do not fabricate receipts or mark a trade successful directly.
5. Generic storage writes are not an acceptable protocol-state override.
   Protocol-specific eth-defi drivers must identify the token, role, window or
   state transition they change.
6. Every failed alternative retains the original typed result plus structured
   `alternative_failure` evidence.

## Work item 1: general alternative simulation orchestration

### eth-defi contract

Add a manager-owned method such as
`prepare_redemption_simulation(owner, raw_shares, failure)` that performs its
protocol-specific Anvil intervention and returns a structured disclosure
record. The intervention may be:

- `liquidity_injected`;
- `minimum_aware_amount`;
- `time_advanced`;
- `authorised_phase_action`.

The base manager must remain unsupported. Each protocol adapter opts in with a
tested implementation.

For asynchronous redemptions, extend `force_settle()` so its returned
`VaultForcedSettlementResult` contains all synthetic assets, ignored liquidity
constraints and time/actor interventions. `is_terminal_success()` remains the
gate: a claimable ticket or proven positive direct payout is required.

### trade-executor orchestration

Refactor the current redemption retry in
`tradeexecutor/ethereum/vault/vault_routing.py` into one helper that:

1. catches a typed `VaultFlowUnavailable` from the natural redemption;
2. verifies the route is Anvil and automatic simulation enabled intervention;
3. asks the manager to apply its protocol-specific intervention;
4. retries the unchanged request and completes normal settlement and receipt
   analysis; and
5. attaches the original failure and interventions to the trade.

The runner must serialise `observed_result` before the retry and promote the
terminal row to `success_simulated_with_intervention` only after real receipt
analysis succeeds.

## Work item 2: capital, capacity and zero-payout paths

### Pharaoh USDC

Vault `43114-0x124d00b1ce4453ffc5a5f65ce83af13a7709bac7`
currently reports `redemption_capacity_limited`.

1. In the 40acres manager, identify the concrete denomination-token source
   used by redemption.
2. Calculate the shortfall from the requested shares and previewed assets.
3. Inject only that shortfall on Anvil.
4. Retry the same share quantity and require a positive analysed payout.
5. Record the target, token, raw shortfall and original capacity result.

### Arche USD

Vault `1-0x33ffc177a7278ff84aab314a036bc7b799b7cc15`
mines successfully but returns zero USDC.

1. Reproduce the receipt on a pinned Ethereum fork.
2. Determine whether the zero payout is caused by a queue, strategy debt,
   locked profit, withdrawal limit or insufficient token balance.
3. Add an intervention only for the source-proven cause. A generic vault token
   top-up is not sufficient evidence.
4. Retry through the Yearn manager and require a positive payout event and
   balance delta.

### Existing Morpho interventions

Keep Apyx USDC and Saturn USDC as regression fixtures. General orchestration
must preserve their `liquidity_injected` evidence and unchanged redemption
transactions.

## Work item 3: time and phase paths

### Gains withdrawal window

Vault `8453-0xad20523a7dc37babc1cc74897e4977232b3d02e5`
reports `redemption_window_closed` with `EndOfEpoch`.

1. Add a Gains helper that reads the next request-window opening from chain
   state.
2. Advance the Anvil timestamp to the first valid request block and mine it.
3. Create the real withdrawal request, advance through its required settlement
   interval, settle and claim it.
4. Record both time jumps and all lifecycle transactions.
5. Retain `redemption_window_closed` as the observed production result.

### YieldNest RWA MAX maturity

Vault `1-0x01ba69727e2860b37bc1a2bd56999c1afb4c15d8`
matures on 2026-10-15.

1. Keep the natural pre-maturity result as
   `redemption_not_yet_matured`.
2. In the alternative run, advance Anvil to the first valid post-maturity
   timestamp.
3. Execute the real request/readiness/claim lifecycle and require a positive
   analysed payout.

## Work item 4: amount and asset alternatives

### Ember Apollo ACRED

Vault `1-0x2b13311fd553e74b421d4ccc96e348f71e179dcf`
reports `below_minimum` because the 1,001 USDC round trip mints fewer shares than
the redemption minimum.

1. Read the minimum shares before the alternative deposit.
2. Use `previewDeposit()` plus bounded rounding headroom to calculate the
   smallest deposit that will mint enough shares.
3. Materialise that alternative USDC amount on Anvil, then complete the normal
   deposit and redemption lifecycle.
4. Record the original matrix amount, computed alternative amount, minimum
   shares and actual minted shares.

### Sentora USD Earn

Vault `1-0x74ad2f789ed583dbd141bbdafc673fe1f033718b`
does not accept native USDC.

1. Preserve `incompatible_deposit_asset` for the standard USDC matrix.
2. Add a second automatic simulation using an accepted asset, preferring RLUSD
   from the production vault metadata.
3. Materialise the accepted asset on Anvil, use the real multi-asset manager
   calls and complete redemption back to the same asset.
4. Report the accepted-asset lifecycle as an alternative, not as proof that the
   vault accepts USDC.

This is a deposit-asset run override, not a redemption intervention kind.

## Work item 5: raw failures

### Aerodrome USDC satellite close

Vault `8453-0xb99b6df96d4d5448cc0a5b3e0ef7896df9507cf5`
reverts with `ERC20: transfer amount exceeds balance`.

1. Pin the Base fork block from the verified rerun.
2. Compare planned shares, Safe share balance, escrowed shares and the exact
   amount passed to redemption.
3. If the share amount exceeds the available balance only within the existing
   epsilon, reconcile it before constructing the satellite request.
4. If the balance is sufficient and diagnosis instead proves a payout-capital
   shortage, add a protocol-specific 40acres intervention for this vault.
5. Require the subsequent CCTP bridge-back amount to equal the analysed USDC
   payout, not the planned redemption value.

### DeTrade Core USDC infrastructure failure

Vault `8453-0x8092ca384d44260ea4feaf7457b629b8dc6f88f0`
exhausted its replacement attempt after an Anvil read timeout.

1. Re-run the exact vault alone to distinguish deterministic adapter latency
   from RPC instability.
2. If it reproduces, identify the slow call and make the smallest timeout,
   provider or adapter correction required by that evidence.
3. Add a focused regression only for the reproduced cause.

## Testing

### eth-defi

- Unit-test the intervention disclosure schema and base unsupported behaviour.
- Add one pinned fork test per protocol intervention: 40acres, Yearn, Gains,
  YieldNest, Ember and the existing Morpho path, plus the Upshift accepted-asset
  run.
- Each test first proves the natural typed constraint, then applies the
  intervention, then proves a real positive payout.
- Assert every helper rejects a non-Anvil provider.
- Assert time advancement, impersonation and synthetic token amounts are
  disclosed exactly.

### trade-executor

- Unit-test natural failure capture, successful alternative promotion and
  failed-alternative retention.
- Test JSON serialisation and table rendering of both outcome layers.
- Test that manual and live modes never invoke intervention methods.
- Test that intervention rows are excluded from economic aggregates.
- Add focused satellite reconciliation coverage and a runtime regression only
  if the DeTrade failure reproduces.
- Follow the repository pytest documentation convention: one happy-path and
  one bad-path test where practical, with ordered docstring steps repeated as
  comments.

### Matrix verification

After both repositories pass focused tests:

1. update the trade-executor eth-defi submodule to the merged dependency;
2. rerun all 129 ordered vault ids with 1,001 USDC;
3. rerun any infrastructure failure alone before accepting it as terminal;
4. publish ordinary successes, intervention successes, expected gates and raw
   failures as separate totals; and
5. attach the machine report and exact fork blocks to the pull request.

## Acceptance criteria

- Every implemented redemption lifecycle that is closed by capacity,
  liquidity, time, maturity or an actionable protocol phase has a successful,
  disclosed alternative Anvil simulation.
- The original production constraint remains machine-readable and visible in
  the human report.
- Aerodrome USDC no longer ends in an untyped transaction revert.
- DeTrade Core USDC either succeeds after deterministic retry or has a
  reproducible, typed adapter limitation rather than a generic infrastructure
  failure.
- Ember Apollo ACRED completes with a computed minimum-aware amount.
- Sentora USD Earn completes with an accepted asset while the USDC attempt
  remains correctly incompatible.
- No intervention is callable in live or manual execution.

## Delivery order

1. Add the eth-defi intervention contract and protocol drivers on a dedicated
   eth-defi feature branch and merge that pull request first.
2. Add trade-executor orchestration, reporting and regression tests on a new
   feature branch based on current master, verified as `a073b357` when this plan
   was written.
3. Update the eth-defi submodule pointer and add a dated `CHANGELOG.md` feature
   entry.
4. Run focused tests, then the full 129-vault matrix.
5. Open the trade-executor feature pull request with Why, Lessons learnt and
   Summary sections, and post the final matrix as a pull request comment.
