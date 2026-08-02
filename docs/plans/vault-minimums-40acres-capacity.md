# Vault minimums and 40acres capacity simulation

## Objective

Coordinate eth-defi and trade-executor changes so vault minimum amounts are a
shared adapter capability and 40acres redemption-capacity shortfalls can be
completed honestly on Anvil.

The work is complete when:

- `VaultBase` exposes deposit and redemption minimums with unambiguous units;
- Accountable and Ember implement the shared accessors instead of hiding their
  minimum reads inside request construction;
- `vault-test-trade --auto-simulated` raises its per-vault deposit amount when
  the requested amount cannot mint the protocol's minimum redeemable shares;
- a 40acres redemption-capacity shortfall is funded on the fork, the unchanged
  redemption transaction succeeds, and the terminal outcome is
  `simulated_success_redemption_capacity_limited`;
- the report records the original requested amount, available capacity,
  denomination-token shortfall and exact synthetic amount added; and
- real execution never changes balances, raises limits or silently increases
  the operator's requested deposit.

This is a two-repository change. Protocol reads, unit conversion, preflight
semantics and Anvil-only capacity intervention belong in eth-defi. Per-vault
test sizing, result naming, persistence and reports belong in trade-executor.

## Working branches and worktrees

| Repository | Branch | Worktree | Starting commit |
|---|---|---|---|
| trade-executor | `plan/vault-minimums-40acres-capacity` | `/home/mikko/code/trade-executor-vault-minimum-capacity-plan` | `a073b35769ad2dd27de57831c25a949849f8d1c6` |
| web3-ethereum-defi | `plan/vault-minimums-40acres-capacity` | `/home/mikko/code/web3-ethereum-defi-vault-minimum-capacity-plan` | `9e9a83dcb41058908028d2d246e5d3ba2f8d9b1a` |

Both branches start from their fetched `origin/master` revisions on
2026-08-02. The trade-executor worktree has its required `.local-test.env`
copy.

## Current evidence

| Vault | Current value | Our value | Consequence |
|---|---:|---:|---|
| Accountable Hyperithm | Effective deposit minimum is 1,000 USDC from `loan().minDeposit` | Default test deposit is 1,001 USDC | Currently passes because of a hard-coded command default, not a shared adapter contract |
| Ember Apollo ACRED | Redemption minimum is 9.170000 eACRED shares | A 1,001 USDC deposit minted 0.905390 shares | `below_minimum`; the test needs about 10.1 times more shares/deposit value |
| 40acres Pharaoh USDC | Immediate redemption capacity was 917.740583 shares | Requested 917.926957 newly minted shares | Shortfall 0.186374 shares, about 0.0203%; currently `redemption_capacity_limited` |
| 40acres Aerodrome USDC | No structured capacity value was captured | Full position minted from 1,001 USDC | Raw `ERC20: transfer amount exceeds balance` revert; requested and available amounts are missing from the report |

The Pharaoh delta is small enough to look like an asset/share conversion or
rounding boundary, but that is not yet proven. The implementation must record
`previewRedeem`, the vault's idle denomination balance and both raw share
amounts before mutating the fork. Do not call the discrepancy a rounding error
solely because it is small.

## API contract for vault minimums

Add the following methods to `eth_defi.vault.base.VaultBase`:

```python
def fetch_minimum_deposit(
    self,
    block_identifier: BlockIdentifier = "latest",
) -> Decimal | None:
    ...

def fetch_minimum_redemption(
    self,
    block_identifier: BlockIdentifier = "latest",
) -> Decimal | None:
    ...
```

The two concepts deliberately expose exact and display forms:

- deposit values use denomination-token units and redemption values use
  vault-share-token units;
- both values are returned as decimals; and
- managers convert a known value with `denomination_token` or `share_token`
  for exact comparisons and transaction construction; and
- the default returns `None`, meaning that the adapter does not expose a known
  minimum. Callers must not interpret `None` as proof that the protocol has no
  minimum. `vault-test-trade` may continue with the requested amount when a
  minimum is unknown, but must persist the corresponding raw provenance field
  as JSON `null` and describe it as unknown, never as zero or absent.

Keep the methods overridable and block-aware. Do not cache a latest-block value
because governance can change minimums. Do not infer a minimum by sending a
reverting transaction when a source-proven getter is unavailable.

## Work item 1: promote Accountable minimums to the shared API

Change eth-defi first.

1. Verify every use of `MIN_AMOUNT_WEI()` against the verified vault source
   before promoting it. The getter is a raw scalar, not intrinsically an asset
   or share amount: retain it in the deposit accessor only if the deposit path
   compares it directly with raw assets, and retain it in the redemption
   accessor only if `requestRedeem` compares it directly with raw shares. Cite
   both source call sites in the adapter documentation. If either use is not
   proven, return `None` for that shared accessor side instead of converting or
   guessing, but retain the manager's existing direct preflight guard as legacy
   behaviour until the source question is resolved. Do not remove an existing
   guard merely because it cannot yet be promoted to the shared API.
2. Keep `AccountableVault.fetch_minimum_deposit()` as an override of the new
   base method. The binding deposit minimum remains the maximum of vault
   `MIN_AMOUNT_WEI()` and strategy `loan().minDeposit` only after both are
   proven to be direct raw-asset thresholds.
3. Add `fetch_minimum_redemption()`. Use the vault-level `MIN_AMOUNT_WEI()` only
   after the source check above proves that the redemption path compares this
   scalar directly with raw shares.
4. Investigate the strategy tuple's `loan().minRedeem` against verified source
   and the exact Hyperithm deployment. Include it in the effective redemption
   minimum only if its unit is proven to be raw shares. If it is assets, convert
   through the protocol-defined preview at the same block. If its semantics
   remain unproven, retain only `MIN_AMOUNT_WEI()` and document why.
5. After the relevant unit proofs pass, make
   `AccountableDepositManager.create_deposit_request()`,
   `create_redemption_request()` and the `can_create_*` predicates consume the
   vault accessors. Remove duplicate direct reads from the manager.
6. Preserve `VaultFlowUnavailable(preflight_result="below_minimum")` and its
   exact `requested_raw_amount` and `minimum_raw_amount` fields.

Tests in `tests/erc_4626/vault_protocol/test_accountable.py` must cover:

- deposit minimum below, exactly at and above the effective value;
- decimal and raw accessor agreement;
- redemption minimum below and exactly at the confirmed share threshold;
- a strategy without `loan()` falling back to the vault getter;
- source-backed assertions that each `MIN_AMOUNT_WEI()` call site compares the
  value in the claimed asset or share context; and
- a pinned or source-backed assertion for `minRedeem` units before that field
  participates in the result.

## Work item 2: expose Ember redemption minimums

1. Implement `EmberVault.fetch_minimum_redemption()` using
   `minWithdrawableShares()` and decimalise it with the share token.
2. Leave Ember's deposit minimum as `None` unless a separate contract-enforced
   deposit threshold is found. `maxDeposit()` is a capacity getter, not a
   minimum.
3. Replace the direct `minWithdrawableShares()` reads in
   `EmberDepositManager.create_redemption_request()` and
   `can_create_redemption_request()` with the shared vault accessor.
4. Preserve the existing typed `below_minimum` failure for callers that do not
   auto-size.

Extend `tests/erc_4626/vault_protocol/test_ember_deposit_redeem.py` to assert
that the accessor returns 100,000 raw shares in the existing fixture, that its
decimal value uses the share-token decimals, and that request preflight uses
the same value.

## Work item 3: resolve a minimum-aware per-vault test amount

Implement this in trade-executor without changing strategy or live-trading
position sizing.

1. Add a small helper in `tradeexecutor/cli/vault_trade/` that receives the
   user-requested denomination amount, executable vault, deposit manager and
   executor owner.
2. Read the deposit minimum and start from the greater of the CLI amount and
   that minimum.
3. Read the redemption minimum. Use the manager's deposit estimator to check
   how many shares the candidate deposit will mint. If the estimate is below
   the minimum, increase the denomination amount proportionally, round upward
   in denomination-token base units and estimate again. Use a bounded loop and
   fail with both values if the adapter cannot produce a positive estimate.
4. Require the estimated minted shares to be at least the raw redemption
   minimum. The exact share balance received from the deposit remains the
   redemption input; do not manufacture shares to pass the minimum.
5. Run the deposit manager's ordinary owner-balance and deposit-capacity
   preflights against the effective amount before execution. Do not clamp the
   amount and proceed when the available balance or accepted capacity is below
   the amount required to mint the redemption minimum. Preserve a typed failure
   and report both the required and available raw amounts. The current simulated
   runtime seeds the Safe with 100 times the requested amount, but the resolver
   must not assume that this always covers a future protocol minimum.
6. Apply automatic increases only to `--auto-simulated`. For `--auto-real` and
   interactive/real actions, retain the operator's amount and return the typed
   minimum failure. Never silently commit more real capital than requested.
7. Store these values in attempt provenance and the JSON report:
   `requested_deposit_raw`, `effective_deposit_raw`,
   `minimum_deposit_raw`, `minimum_redemption_raw`, and
   `estimated_shares_raw`. Preserve unknown minimums as JSON `null`.
8. Keep the CLI's 1,001 USDC default as a convenient baseline, but remove the
   assumption that this one amount clears every protocol.

Add focused tests in `tests/cli/test_vault_test_trade.py` for:

- no known minimum, leaving the amount unchanged while persisting both minimum
  provenance values as JSON `null`;
- Accountable's 1,000 USDC minimum with a 1,001 USDC request;
- an amount below Accountable's minimum being raised only in simulation;
- Ember Apollo increasing the effective amount until at least 9.17 shares are
  estimated;
- exact base-unit rounding at the threshold;
- the required effective amount exceeding the owner balance or manager-reported
  deposit capacity, producing a typed failure with required and available
  values; and
- real mode refusing rather than increasing the amount.

## Work item 4: implement the 40acres Anvil capacity intervention

The simulation must increase the vault's denomination-token payout capacity,
not edit user share balances or bypass the real redemption call.

1. Extend the 40acres manager with a source-proven
   `force_redemption_liquidity()` implementation. It must reject non-Anvil
   providers and any failure whose `preflight_result` is not
   `redemption_capacity_limited`.
2. Before mutation, record at one block:
   - owner share balance;
   - requested raw shares;
   - `maxRedeem(owner)`;
   - `previewRedeem(requested_raw_shares)` in raw denomination assets;
   - vault denomination-token balance;
   - capacity converted to raw shares; and
   - the raw share and asset shortfalls.
3. Treat
   `max(previewRedeem(requested_shares) - vault_asset_balance, 0)` only as the
   initial lower bound. A direct asset transfer increases 40acres
   `totalAssets()`, changes the share price and therefore increases the assets
   returned by `previewRedeem`; the initial shortfall is not generally the
   final injection amount. Derive the fixed point from the exact verified
   deployment's conversion formula, including any virtual asset/share terms,
   or find the smallest sufficient raw-asset injection with a bounded Anvil
   snapshot search against the actual post-injection capacity getter. Disable
   the intervention if neither route is source-proven. After funding, re-read
   `totalAssets`, `totalSupply`, `previewRedeem`, denomination balance and raw
   share capacity. For every candidate, snapshot-test the unchanged real
   `redeem(shares, receiver, owner)` call as well as requiring the capacity
   getter to cover the requested shares. If that call still hits the exact
   liquidity revert, continue the bounded search; abort on an unrelated revert.
   Record the total injection and whether any final base-unit rounding bump was
   needed.
4. Build the same `redeem(shares, receiver, owner)` call after the intervention.
   The call must be broadcast and analysed normally. A balance mutation alone
   is never success.
5. Return structured intervention evidence with at least:

   ```text
   kind=redemption_capacity_increased
   token
   target
   requested_raw_shares
   available_raw_shares_before
   available_raw_shares_after
   requested_raw_assets
   requested_raw_assets_after
   available_raw_assets_before
   available_raw_assets_after
   total_assets_before
   total_assets_after
   total_supply_before
   total_supply_after
   injected_raw_assets
   shortfall_ratio
   original_preflight_result
   original_reason
   ```

6. Keep this fork-only and disclosed. Do not change a live vault's cap, call a
   privileged production method, ignore a revert or mark a failed receipt as
   successful.

### Deployment scope

Pharaoh already uses `FortyAcresDepositManager` and supplies typed capacity
evidence. Implement and prove this path first.

Aerodrome currently uses the generic ERC-4626 manager because an earlier fork
proved that it could redeem without meaningful idle balance. Do not apply
Pharaoh's direct-balance rule to Aerodrome by protocol name alone. First replay
the current Base failure and inspect verified source/traces to establish
whether the vault can pull loan liquidity or now pays directly from idle USDC.

- If Aerodrome's source-proven path has a measurable immediate capacity, bind a
  deployment-appropriate 40acres manager so the same intervention can run.
- If the contract can source liquidity but fails for another reason, keep a
  separate typed cause and do not hide it with a balance top-up.
- In either case, replace the current raw assertion with structured requested,
  available and failed-transaction evidence.

Tests in `tests/erc_4626/vault_protocol/test_forty_acres.py` must prove:

- a naturally sufficient redemption uses no intervention;
- the current Pharaoh-sized shortfall records both sides of the comparison;
- direct injection's share-price movement is included in the fixed-point
  calculation, and the smallest sufficient injected amount makes capacity
  cover the original shares;
- the unchanged redemption succeeds and returns assets;
- the intervention is rejected off Anvil and for minimum, maturity, whitelist
  or unrelated failures; and
- a large shortfall remains visibly quantified rather than being described as
  rounding.

Add or extend the exact Aerodrome Base-fork regression after its deployed path
is understood.

## Work item 5: add the dedicated trade-executor outcome

1. Add `simulated_success_redemption_capacity_limited` to
   `VAULT_TEST_RESULTS` and all terminal/renderable result collections.
2. In the simulated-success finalisation path, inspect recorded intervention
   evidence. Use the new result only when:
   - the intervention kind is `redemption_capacity_increased`;
   - the original result is `redemption_capacity_limited`; and
   - the subsequent real redemption transaction and receipt analysis succeed.
3. Keep `success_simulated_with_intervention` for other intervention kinds,
   such as the existing Morpho liquidity injection. Do not rename historical
   results.
4. Keep plain `redemption_capacity_limited` when no intervention was requested,
   the manager does not support one, the run is real, or the post-intervention
   redemption still fails.
5. Persist the complete intervention dictionary in `outcome_data` and show a
   concise table detail such as “simulated success after adding the recorded
   denomination-token redemption shortfall”; the machine report retains the
   exact human-readable and raw values.
6. Ensure cross-chain failure handling no longer loses the redemption call's
   requested raw shares and available capacity before raising from the
   satellite close.

Extend `tests/cli/test_vault_test_trade.py` with one happy-path and one bad-path
test:

- supported 40acres capacity intervention followed by a successful redemption
  produces `simulated_success_redemption_capacity_limited`; and
- an intervention followed by a reverted or unanalysable redemption remains a
  failure and can never be upgraded to simulated success.

Also assert that cSigma and other protocols still return
`redemption_capacity_limited` unless they implement and complete this exact
intervention contract.

## Verification sequence

Run focused eth-defi tests first from its worktree. During local cross-repository
development, run trade-executor with
`PYTHONPATH=/home/mikko/code/web3-ethereum-defi-vault-minimum-capacity-plan:$PYTHONPATH`
so it imports the eth-defi worktree. Do not commit a trade-executor submodule
pointer to a local-only SHA: update the pointer only after the eth-defi commit is
pushed to a remotely fetchable branch (normally after its PR is available or
merged). Then rerun the focused consumer tests against that exact pointer.

1. Accountable accessor and request tests.
2. Ember accessor, minimum and lifecycle tests.
3. 40acres Pharaoh manager and guarded lifecycle tests.
4. 40acres Aerodrome diagnostic/fork test.
5. Trade-executor minimum-aware amount unit tests.
6. Trade-executor intervention result/persistence tests.
7. Exact Pharaoh, Ember Apollo and Aerodrome `vault-test-trade` reruns.
8. The same ordered 129-vault simulation matrix using fresh production JSON.

The final report comparison must show:

- Ember Apollo no longer ending at `below_minimum` solely because 1,001 USDC
  mints too few shares;
- Pharaoh changing from `redemption_capacity_limited` to
  `simulated_success_redemption_capacity_limited`, with before/after values;
- Aerodrome either succeeding naturally or receiving the same disclosed result
  only after its capacity mechanism is proven;
- no change to closed, whitelist, maturity or incompatible-asset outcomes; and
- no new raw transaction, receipt-analysis or infrastructure failures.

## Non-goals and safety rules

- Do not auto-size ordinary strategy trades; this is a `vault-test-trade`
  simulation facility.
- Do not increase real deposits or mutate live protocol capacity.
- Do not hard-code token decimals; use `TokenDetails` conversions.
- Do not treat `maxDeposit`, `maxRedeem`, denomination balance and protocol
  minimums as interchangeable concepts.
- Do not classify a successful balance mutation as a successful lifecycle; the
  original protocol transaction and receipt analysis must complete.
- Do not generalise Pharaoh's direct-idle-liquidity model to every 40acres
  deployment without source or trace evidence.
- Do not call a discrepancy a rounding error unless the recorded conversion
  path proves it.
