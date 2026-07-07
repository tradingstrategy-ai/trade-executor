# Remove Lagoon cycle oracle from live e2e

## Goal

Fix the phase-aware Lagoon live e2e test so it no longer uses
`is_lagoon_operational_for_cycle()` or any synthetic cycle schedule as the
behavioural oracle. The test may still mutate Anvil deterministically, but
trade-executor must learn whether vault deposits and redemptions are possible
through the normal live pipeline:

`VaultPricing -> VaultDepositManager -> chain reads`

Final assertions must be made from persisted state JSON observations and direct
chain evidence.

## Constraints

- Do not add Lagoon-specific logic to production trade-executor core.
- Do not add a new test-only hook to `ExecutionLoop.tick`.
- Keep direct Web3 reads only as chain evidence, not as the strategy decision
  source.
- Keep the test deterministic with a bounded max cycle count.
- Keep queue-vault assertions in the state JSON.

## Plan

1. Confirm the executor-core availability path first.
   - Verify the test strategy has access to the live
     `StrategyInput.pricing_model`.
   - Verify that this pricing model is the `VaultPricing` path that calls
     `VaultDepositManager.can_create_deposit_request()` and
     `VaultDepositManager.can_create_redemption_request()`.
   - Do this before deleting the cycle oracle; this is the load-bearing path
     the refactor is meant to test.

2. Find all callers of `is_lagoon_operational_for_cycle()`.
   - Use `rg` across the whole repository and submodules.
   - Confirm the helper lives in trade-executor and not in the
     `web3-ethereum-defi` submodule.
   - Remove every caller. Any assertion or helper using this function must be
     rewritten to use persisted core availability observations and direct
     chain evidence instead.
   - Delete the helper from `tradeexecutor/ethereum/lagoon/testing.py`.
   - Keep `set_lagoon_vault_open_for_testing()`.

3. Replace schedule-oracle usage with a private scenario controller.
   - Add a `LagoonScenarioController` in
     `tests/erc_4626/test_phase_aware_lagoon_live_e2e.py`.
   - The controller owns Anvil mutations only.
   - It must not expose `is_open(cycle)` or any equivalent method used by
     assertions.
   - Make call order explicit:
     1. `controller.before_tick(cycle)` mutates Lagoon open/paused state.
     2. direct pre-tick chain observations are collected.
     3. normal strategy tick runs.
     4. during the strategy decision function, core availability observations
        are collected using the same `StrategyInput.pricing_model` and
        strategy timestamp that the tick uses for decisions.
     5. `controller.after_tick(cycle)` performs fresh post-tick direct chain
        reads and settles newly pending requests.
   - Persist the direct pre-tick observations, core availability observations,
     and post-tick settlement evidence in the existing per-cycle
     `state.other_data` observation log.

4. Prefer event-driven scenario progression.
   - Start all target Lagoon vaults closed.
   - Open one target vault at a time in deterministic address/list order.
   - Before implementing this controller, confirm the mock strategy target
     weights are independent of vault availability. If targets are
     availability-dependent, do not wait for an observed target to choose the
     next vault; use deterministic list-order opening so the closed-all-vaults
     initial state cannot deadlock progression.
   - Once fresh post-tick direct chain reads show
     `pendingDepositRequest > 0` for the current vault, settle its deposit in
     `after_tick(cycle)`.
   - After shares exist, open vaults in deterministic order for redemptions.
   - Keep redemption windows closed for at least one observed cycle after
     shares exist, so the state JSON captures `can_redeem == False` while the
     executor holds shares.
   - Once fresh post-tick direct chain reads show
     `pendingRedeemRequest > 0`, settle the redemption in
     `after_tick(cycle)`.
   - Stop only after all target vaults have pending and processed deposit and
     redemption evidence, or fail at the existing max cycle count.
   - Verify the max cycle count before implementation. Strict one-vault-at-a
     time progression needs roughly five or more cycles per target vault
     (open deposit, request, settle, observe closed redemption with shares,
     open redemption, request, settle), so either increase the test cycle cap
     or allow safe overlap between vaults.

5. Capture executor-core availability from the strategy path.
   - Use the live `StrategyInput.pricing_model` already available in the test
     strategy/decision function.
   - Add a small generic helper in a production testing submodule, for example
     `tradeexecutor/ethereum/vault/testing.py`:
     `collect_vault_availability(pricing_model, ts, pairs) -> dict`.
   - For each vault pair, store JSON-primitive fields:
     - `can_deposit`
     - deposit block reason if the pricing API exposes one; otherwise keep
       only the boolean
     - `can_redeem`
     - redemption reason code if present
     - redemption max amount if present
   - Do not use raw Web3 in this helper.
   - Use the strategy tick timestamp from `StrategyInput` so the availability
     snapshot matches the decisions made in the same cycle.

6. Keep direct Web3 observations as independent evidence.
   - Continue collecting direct `eth_call` values for:
     - `paused`
     - `pendingDepositRequest`
     - `pendingRedeemRequest`
     - `maxDeposit`
     - `maxRedeem`
     - share balance
   - Store these next to core availability in the state JSON observation.
   - Assert the pre-tick direct chain observation reflects the controller
     mutation before interpreting core availability.
   - Use separate fresh post-tick direct chain reads inside
     `controller.after_tick(cycle)` to decide whether to settle deposit or
     redemption requests.

7. Rewrite the final state JSON assertions.
   - Remove all assertions based on `deposit_open_cycle`,
     `deposit_settle_cycle`, `redemption_open_cycle`, or
     `redemption_settle_cycle`.
   - For each target vault, assert:
     - core observed `can_deposit == False` at least once before or while
       capital was parked,
     - core observed `can_deposit == True` at least once,
     - a pending deposit trade exists,
     - a processed deposit claim exists,
     - core observed `can_redeem == False` while shares existed,
     - core observed `can_redeem == True` at least once,
     - a pending redemption trade exists,
     - a processed redemption claim exists.
   - For Lagoon-specific consistency, assert only the safe implication:
     - when direct chain `paused` is true, core `can_deposit` is false and
       core `can_redeem` is false.
   - Do not assert that `paused == False` always means core availability,
     unless also guarded by protocol evidence such as positive `maxDeposit`
     or `maxRedeem`.

8. Keep queue-vault assertions.
   - Assert queue deposits exist.
   - Assert queue redemptions exist.
   - Assert queue vault balance or TVL was positive during the run.
   - Keep the final positive queue-vault utilisation assertion only if this is
     intended by the strategy after all promotions and redemptions have run;
     otherwise replace it with the during-run utilisation assertion.

9. Keep generic DepositManager tests.
   - Keep the existing `test_vault_pricing.py` coverage for independent
     deposit/redemption request gates and owner-unavailable behaviour.
   - Extend only if the new JSON serialisation helper needs focused coverage.

10. Verify.
   - Run the focused vault pricing tests.
   - Run
     `tests/erc_4626/test_phase_aware_lagoon_live_e2e.py::test_phase_aware_lagoon_live_e2e`.
   - Run `git diff --check`.
   - Run the equivalent diff check inside `deps/web3-ethereum-defi`.

## Expected result

The live e2e remains deterministic and still drives real Lagoon vault state on
fresh Anvil. However, no assertion depends on a hand-coded cycle oracle. If
trade-executor stops observing Lagoon availability through the generic
`DepositManager` abstraction, the persisted state JSON assertions fail even
though the test controller continues to mutate the chain correctly.
