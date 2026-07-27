# Trade-executor eth-defi master vault fixes

## Objective

Finish the trade-executor work needed to consume the vault support currently
available on eth-defi master, with particular focus on full asynchronous
simulation.

The implementation must:

- use an async vault manager's persisted request ticket when forcing settlement
  on Anvil;
- call the manager-owned `force_settle()` implementation only when the manager
  explicitly advertises that capability;
- retain the existing Ostium V1.5 test helper as a compatibility path until
  eth-defi exposes an equivalent manager implementation;
- return `simulation_unsupported_async` for other async managers instead of
  making a signer-less RPC call or falling through to a generic assertion; and
- retain the state-inference, cSigma, and revert-evidence fixes already present
  on this branch.

## Corrected baseline

The report in `docs/reports/cross-chain-vault-test-trade-executor.md` cannot be
used as direct evidence for this branch's current behaviour. Although its prose
names trade-executor commit `248e1e20`, every persisted attempt records
trade-executor commit `42781eea`, and all 1,163 traceback paths point to the
parent checkout rather than this worktree.

The editable Poetry installation placed the parent checkout ahead of the
worktree. Future verification must run from this worktree with both source
roots explicitly first:

```shell
source .local-test.env && \
PYTHONPATH="$(pwd):$(pwd)/deps/web3-ethereum-defi:$PYTHONPATH" \
poetry --project /path/to/parent/trade-executor run pytest ...
```

Before accepting a matrix rerun, verify `tradeexecutor.__file__`,
`eth_defi.__file__`, and the report's `trade_executor_commit` provenance all
refer to this worktree and its current commit.

## Existing behaviour to preserve

These requested fixes are already implemented at `248e1e20` and should receive
regression coverage or verification, not a second implementation:

1. `VaultTestBatchRunner` calls `perform_test_trade()` with
   `update_statistics_after_trade=False`, preventing diagnostic positions with
   no trades from reaching long/short statistics.
2. Real execution changes the attempt phase to `state_inference` only after
   `perform_test_trade()` returns. Simulated execution records
   `state_inference_failed` if no target position/trade was created.
3. `VaultRouting.deposit_or_redeem()` uses the selected manager's
   `create_deposit_request()` / `create_redemption_request()` calls. cSigma's
   `VaultFlowUnavailable` therefore reaches
   `normalise_vault_flow_failure()` and maps to
   `redemption_capacity_limited` with requested and available raw amounts.
4. Failure capture retains transaction status, hash, chain, target/wrapped
   target, function selectors, revert reason, stack trace, and unsigned call
   context. A status-0 Accountable transaction remains
   `transaction_reverted`; decoding selector `0x5945ea56` remains an eth-defi
   responsibility.

Do not weaken `TradingPosition.is_long()` or `is_short()`, add a second cSigma
capacity API, or decode Accountable protocol errors in trade-executor.

## Work item 1: reconstruct the manager ticket for forced settlement

Refactor `_force_vault_settlement_and_resolve()` in
`tradeexecutor/cli/testtrade.py`.

The eth-defi `VaultDepositManager` abstraction deliberately owns both deposit
and redemption requests, tickets, claims, and forced settlement. There is no
separate redemption-manager lookup: direction chooses methods on the same
manager instance.

The exact API was verified in the pinned eth-defi master worktree before
writing this plan:

- `VaultBase.get_deposit_manager_capability()`;
- `VaultDepositManagerCapability.supports_anvil_settlement`;
- `VaultDepositManager.reconstruct_deposit_ticket()`;
- `VaultDepositManager.reconstruct_redemption_ticket()`; and
- `VaultDepositManager.force_settle(ticket)`.

Implementation steps:

1. Resolve the vault and its single bidirectional deposit/redemption manager
   using the vault-chain Web3 connection already passed to the function.
2. Read `vault_direction` from `trade.other_data`.
3. Reconstruct the exact persisted request ticket with
   `reconstruct_deposit_ticket(trade.other_data)` or
   `reconstruct_redemption_ticket(trade.other_data)`.
4. Read `vault.get_deposit_manager_capability()` and inspect
   `supports_anvil_settlement`.
5. When the capability is `True`, call `deposit_manager.force_settle(ticket)`.
   Do not pass the executor owner as a settlement caller. Lagoon's manager
   implementation is responsible for impersonating its deployed valuation
   manager and Safe and for validating that the selected ticket becomes
   claimable.
6. Persist a compact JSON-safe summary of the forced settlement result in
   `trade.other_data`, including whether settlement was required, the
   before/after status values, and transaction hashes. This is diagnostic
   evidence, not durable transaction identity for the subsequent claim. Store
   enum statuses through their string `.value` and transaction hashes as
   `0x`-prefixed hex strings; do not persist enum or `HexBytes` objects.
7. Call `check_and_resolve_vault_settlements()` after the manager has made the
   ticket claimable. Continue passing `web3config` so a satellite claim uses
   the vault chain's transaction builder.

Do not append the manager's force-settlement transactions to
`trade.blockchain_transactions`: the retry module currently interprets
post-request transactions as claim/reclaim attempts. Store them in
`trade.other_data` unless that retry ownership model is deliberately refactored
with corresponding migration tests.

## Work item 2: preserve the Ostium compatibility path and fail closed elsewhere

The eth-defi master baseline only implements manager-owned `force_settle()` for
Lagoon. Ostium V1.5 is already supported by
`force_ostium_v15_settlement()` in trade-executor and must not regress.

1. If `supports_anvil_settlement is True`, always use the generic manager path.
2. Otherwise, if the selected vault is Ostium V1.5, retain the existing
   permissionless settlement loop using an Anvil-unlocked development account.
   Then run the shared claim resolver.
3. Treat an absent capability method, a `None` capability, or a capability
   whose `supports_anvil_settlement` is missing, `None`, or `False` as not
   advertising generic forced settlement. Only the explicit Ostium V1.5
   compatibility branch may proceed from that state.
4. For every other asynchronous manager, raise
   `UnsupportedVaultSimulation` before issuing a settlement transaction.
   Include the manager class, vault address, direction, and advertised
   capability in the message.
5. Let the vault-test runner's existing `UnsupportedVaultSimulation` branch in
   `normalise_vault_flow_failure()` convert the exception to
   `simulation_unsupported_async`. Retain a focused test for this existing
   mapping so the forced-settlement change cannot bypass it.

This removes the direct Lagoon `force_lagoon_settle(vault, owner)` branch that
caused `No Signer available`, while keeping a bounded compatibility exception
for Ostium.

## Work item 3: strengthen failure and regression tests

Add focused tests without network access.

1. In the home-chain async settlement test module, model a pending trade with a
   serialised ticket and a vault whose capability advertises
   `supports_anvil_settlement=True`.
2. Verify `_force_vault_settlement_and_resolve()` reconstructs the correct
   direction-specific ticket, calls `manager.force_settle(ticket)`, persists a
   JSON-safe result summary, and invokes the shared claim resolver.
3. Add the redemption-direction sibling within the same test where practical,
   using parametrisation only if it keeps the ordered test steps readable.
4. Model an async manager without forced settlement support and verify
   `UnsupportedVaultSimulation` is raised before any settlement or claim call.
5. Model an Ostium V1.5 vault whose capability does not advertise manager-owned
   Anvil settlement. Verify the compatibility loop calls
   `force_ostium_v15_settlement()` with an Anvil-unlocked development account,
   calls the shared claim resolver afterwards, and does not call
   `manager.force_settle()`.
6. Retain or adapt the existing control-flow tests proving home-chain and
   satellite resolvers use the correct Web3 connection.
7. Add a runner normalisation assertion showing the unsupported exception maps
   to `simulation_unsupported_async`.
8. Keep the existing cSigma capacity and transaction-evidence classifier tests
   green.

If the mocking needed for `_force_vault_settlement_and_resolve()` becomes
excessive, extract one small helper to reconstruct the ticket and perform
manager settlement. Do not introduce protocol abstractions owned by eth-defi.

## Work item 4: retain the failing inner vault operation

The corrected 129-vault run exposed a reporting ownership issue: an automatic
`deposit` attempt normally performs both a deposit and a redemption, but a
failure in the latter half still inherits the outer `operation=deposit` label.

1. When recording an exception, inspect only trades created after the attempt's
   original trade-id snapshot.
2. Select the newest vault trade, excluding CCTP bridge and unrelated trades.
3. Derive `deposit` from a buy trade and `redeem` from a sell trade.
4. Use the derived value for the attempt failure record. If no new vault trade
   exists, retain the outer operation because the failure happened before vault
   request construction.
5. Add a focused test with an old vault trade, a new deposit, a new redemption,
   and a later non-vault bridge trade. The helper must report `redeem`.

This is reporting-only and must not affect execution, transaction
classification, or resume ownership.

## Work item 5: reconcile redemption share shortfalls correctly

The corrected matrix exposed status-0 share transfers for the Base Aerodrome
USDC and Avalanche Pharaoh USDC 40acres vaults. The executor requested its
planned share quantity even though the wallet held slightly fewer shares.

`VaultRouting.deposit_or_redeem()` intends to round a small shortfall down to
the onchain balance, but its condition is
`onchain_balance + swap_amount < 0`. Both values are positive during redemption,
so this condition can never be true and the epsilon branch is dead.

1. Extract or implement an explicit redemption reconciliation that accepts a
   positive planned raw/decimal share amount, positive onchain balance, and the
   configured relative epsilon.
2. When the balance is equal to or greater than the plan, retain the planned
   amount; do not redeem unrelated surplus shares.
3. When the balance is below the plan within epsilon, use the onchain balance.
4. When the balance shortfall exceeds epsilon, retain the current accounting
   assertion.
5. Add one focused test covering exact/surplus balance, an epsilon-sized
   shortfall, and an excessive shortfall.

This must use the existing token conversion/request path and must not hide a
material accounting mismatch.

## Work item 6: recognise manager-declared asynchronous directions

The corrected matrix leaves four Ember and two Gains redemptions as
`redemption pending`, even with `--settle-async-on-anvil`. Their static pair
kind is synchronous, while their eth-defi manager capabilities correctly
declare a synchronous deposit and asynchronous redemption. The batch runner
currently uses only `pair.is_async_vault()` to decide whether to request a full
forced lifecycle, so it never asks the shared settlement path to resolve or
reject these redemption tickets.

The concrete capability attributes were verified against pinned eth-defi
master: `VaultDepositManagerCapability.deposit_flow` and
`VaultDepositManagerCapability.redemption_flow`, each typed as
`VaultDepositFlow | None`. The matrix's human-readable `redemption pending`
status is derived from a trade whose internal status is
`TradeStatus.vault_settlement_pending`.

1. Add a small runner helper that treats a vault lifecycle as asynchronous when
   either the pair kind is async or its manager capability declares an
   asynchronous deposit or redemption flow. The pinned eth-defi API defines
   `VaultDepositFlow` as the string literal
   `Literal["synchronous", "asynchronous"]`; compare these fields directly to
   that verified API.
2. Treat an absent capability API, `None` capability, or missing flow fields as
   no additional async evidence; retain the pair-kind fallback.
3. Use this combined result for `complete_async_lifecycle` in simulated batch
   execution so a mixed synchronous-deposit/asynchronous-redemption vault is
   allowed to perform the round trip. Do not change live/manual execution
   semantics. This flag permits the lifecycle and enables forced settlement,
   but `_resolve_home_chain_async_settlement()` and
   `_resolve_satellite_async_settlement()` call the manager settlement path
   only when the individual direction's trade status is
   `vault_settlement_pending`. A successful synchronous deposit therefore
   bypasses forced settlement; only the pending asynchronous redemption reaches
   the direction-specific ticket path.
4. With `--settle-async-on-anvil`, allow the existing manager settlement path
   to resolve managers that advertise support and raise
   `UnsupportedVaultSimulation` for Ember/Gains managers that do not. This
   converts misleading durable `redemption pending` rows to the actionable
   `simulation_unsupported_async` result.
5. Add focused tests for a synchronous pair whose capability declares an
   asynchronous redemption, a synchronous-only capability, missing capability,
   and an intrinsically async pair. Retain coverage that the settlement resolver
   is a no-op for a direction whose trade is already successful, so mixed-flow
   detection cannot force-settle the synchronous half.

This is lifecycle detection only. Do not invent settlement support for a
manager that does not advertise it.

## Verification

Run the focused tests individually with the worktree-first environment:

```shell
source .local-test.env && \
PYTHONPATH="$(pwd):$(pwd)/deps/web3-ethereum-defi:$PYTHONPATH" \
poetry --project /path/to/parent/trade-executor run pytest \
  tests/units_tests/test_home_chain_async_settlement.py

source .local-test.env && \
PYTHONPATH="$(pwd):$(pwd)/deps/web3-ethereum-defi:$PYTHONPATH" \
poetry --project /path/to/parent/trade-executor run pytest \
  tests/units_tests/test_cross_chain_satellite_async_settlement.py

source .local-test.env && \
PYTHONPATH="$(pwd):$(pwd)/deps/web3-ethereum-defi:$PYTHONPATH" \
poetry --project /path/to/parent/trade-executor run pytest \
  tests/cli/test_vault_test_trade.py
```

Then run a Lagoon full-lifecycle fork test, if its required RPC environment is
available, before repeating the 129-vault matrix.

For the matrix rerun, first print the imported module paths in the same shell.
Accept the output only when:

- `trade_executor_commit` is this branch's current commit;
- traceback paths, if any, point into this worktree;
- Lagoon no longer reports `No Signer available`;
- unsupported async managers end as `simulation_unsupported_async`;
- cSigma over-capacity redemption ends as
  `redemption_capacity_limited`; and
- successful receipt analysis cannot be overwritten by long/short statistics.

## Non-goals

- Implementing new eth-defi protocol adapters or error decoders.
- Claiming all ERC-7540 managers can be force-settled.
- Changing live-chain asynchronous settlement.
- Treating protocol closures or unsupported simulation as successful trades.
- Editing global position long/short invariants.
