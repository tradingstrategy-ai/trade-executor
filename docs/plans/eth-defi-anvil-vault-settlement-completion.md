# Anvil vault settlement completion plan

## Purpose

Complete the `eth-defi` work needed to simulate the protocol side of the
remaining asynchronous vault lifecycles found by the 2026-07-29
`trade-executor vault-test-trade` matrix.

This plan is deliberately limited to `eth-defi`. It defines protocol adapter
behaviour, Anvil-only settlement helpers, typed unsupported reasons,
protocol-level terminal evidence and focused exact-address tests. It does not
define how Trade Executor orders simulation trials or interprets Trading
Strategy vault JSON.

The source matrix contained 129 vaults. Fifteen ended with
`simulation unsupported async`:

| Root cause | Count |
|---|---:|
| Ember operator-finalised redemption has no claimable-ticket proof | 4 |
| Lagoon Safe lacks redemption liquidity during strict fork settlement | 9 |
| Gains redemption remains pending after forced epoch settlement | 1 |
| Plutus redemption fulfilment is AccessControl-gated | 1 |

## Repository boundary

### In scope for eth-defi

- Preserve and exercise Lagoon's existing synthetic-liquidity settlement.
- Implement or precisely reject Ember operator processing with direct-payout
  evidence.
- Discover and verify the Plutus fulfilment role and holder, then fulfil and
  claim on Anvil when possible.
- Complete Gains settlement on the Web3 connection supplied for the vault's
  destination chain and enforce protocol-level postconditions.
- Make every unsupported simulation return a stable machine-readable reason
  with protocol, vault, direction and phase.
- Enforce Anvil-only guards at every impersonation, balance injection, time
  warp, storage mutation and privileged settlement boundary.
- Add protocol-specific closed or capped deposit overrides only where the
  exact deployment and mutation are understood and fork-tested.
- Add exact-address unit, mock and mainnet-fork tests for the affected
  deployments.

### Out of scope for eth-defi

- Loading or interpreting Trading Strategy vault JSON.
- Comparing JSON `deposit_permission` or closed-status metadata with onchain
  state.
- Defining Trade Executor result names such as `whitelisting-needed`,
  `whitelisted-incorrectly` or `success_simulated`.
- Defining `SimulationTrialMode`, fallback order or report aggregation.
- Managing Trade Executor state, positions, trades, retries, provenance or
  report files.
- Managing the Trade Executor matrix-level multichain Anvil lifecycle.
- Deciding whether an adapter result replaces or supplements a Trade Executor
  diagnostic.

No `eth-defi` API added by this work may accept Trading Strategy metadata,
Trade Executor result strings or a Trade Executor simulation-mode enum.

## Exact affected deployments

### Ember

| Chain | Vault | Vault ID |
|---|---|---|
| Ethereum | Ember Earn | `1-0x9be9294722f8aad37b11a9792be2c782182cafa2` |
| Ethereum | Ember Polymarket | `1-0x0b9342c15143e8f54a83f887c280a922f4c48771` |
| Ethereum | Ember Third Eye | `1-0xf3190a3ecc109f88e7947b849b281918c798a0c4` |
| Ethereum | Ember UDL | `1-0x373152feef81cc59502da2c8de877b3d5ae2e342` |

Current reason:
`ember_operator_settlement_has_no_claimable_ticket_status`.

### Lagoon

| Chain | Vault | Vault ID |
|---|---|---|
| Ethereum | Gami USDC | `1-0xdae854d0896ad2fee335689a3f7b4a95fd1a3e46` |
| Ethereum | Hub Capital USDC vault | `1-0xca790385506b790554571cbc9da73f0130cdcfd5` |
| Ethereum | Moon Digital AM USDC | `1-0xa00f63e85b3d242568a9edecb48f5e2cf879b07b` |
| Ethereum | Syntropia <> Resupply | `1-0xa96bc6e084aad6976d25df9431525ed2c4d3cae4` |
| Ethereum | Syntropia USDC | `1-0xd17049ed25d8f99fe3bfd10cef2263da9995cfd8` |
| Base | For Yield v2 | `8453-0x2bff679b1a9fbcc202316c1402172747ba2fbf56` |
| Base | MoneyFi FlowForge Vault | `8453-0x4efc07dca8697792119484af33549f33ab11bf3c` |
| Base | RB Capital Yield | `8453-0x63b04d3ce2c14f6d308657ab73ac92fc1a0b1075` |
| Arbitrum | Angmar Capital | `42161-0x1723cb57af58efb35a013870c90fcc3d60174a4e` |

Current reason: `lagoon_settlement_insufficient_liquidity`.

### Gains

| Chain | Vault | Vault ID |
|---|---|---|
| Arbitrum | gTrade USDC | `42161-0xd3443ee1e91af28e5fb858fbd0d72a63ba8046e0` |

Current symptom: the redemption ticket or consuming application remains
pending after the adapter's forced epoch settlement.

### Plutus

| Chain | Vault | Vault ID |
|---|---|---|
| Arbitrum | Plutus Hedge Token | `42161-0x58bfc95a864e18e8f3041d2fcd3418f48393fe6a` |

Current reason:
`plutus_redeem_fulfilment_is_access_control_role_gated`.

## Existing contracts to preserve

Build on the existing contracts in `eth_defi/vault/deposit_redeem.py`:

- `VaultDepositManagerCapability` describes whether an asynchronous manager
  can be settled on Anvil.
- `UnsupportedVaultSimulation` carries the stable adapter-level reason and
  vault context.
- `VaultForcedSettlementResult` carries the ticket, status before and after,
  settlement transaction hashes, synthetic assets injected and whether
  liquidity constraints were ignored.
- `VaultDepositManager.force_settle(ticket, ignore_liquidity=False)` is the
  narrow consumer entry point.

Do not add a generic matrix-mode enum to these APIs.

The base implementation must remain fail-closed:

- Unsupported settlement raises `UnsupportedVaultSimulation`.
- Unsupported synthetic liquidity raises
  `liquidity_bypass_simulation_not_implemented`.
- Unsupported closed/capped override raises a stable typed reason.
- A manager must not advertise `supports_anvil_settlement=True` unless its
  exact production-fork path proves a terminal protocol postcondition.

## Shared eth-defi changes

### Stable unsupported reasons

Audit the affected adapters so every expected refusal sets:

- `unsupported_reason`
- `protocol`
- `vault_address`
- `direction`
- `phase`

Do not require consumers to parse exception text. Replace reasonless
`UnsupportedVaultSimulation` raises in the affected paths with stable reasons.
At minimum add or preserve:

- `anvil_provider_required`
- `anvil_settlement_driver_not_implemented`
- `liquidity_bypass_simulation_not_implemented`
- `lagoon_settlement_insufficient_liquidity`
- `ember_operator_not_discoverable`
- `ember_operator_processing_not_reproducible`
- `ember_direct_payout_not_proven`
- `plutus_fulfilment_role_not_discoverable`
- `plutus_fulfiller_not_authorised`
- `plutus_fulfilment_not_claimable`
- `gains_epoch_did_not_advance`
- `gains_redemption_not_claimable_after_epoch_advance`
- `closed_status_override_not_implemented`

Use a more specific deployment reason where the same adapter has materially
different versions.

### Protocol terminal evidence

Keep `VaultForcedSettlementResult` independent of Trade Executor state.
Strengthen it so an asynchronous settlement can prove one of two protocol
outcomes:

1. The request moved to `AsyncVaultRequestStatus.claimable`.
2. The protocol paid the receiver directly and produced request-specific
   direct-payout evidence.

Add a frozen, slotted direct-payout evidence dataclass only if Ember's
production flow needs it. It should contain:

- request identifier
- receiver address
- denomination token address
- raw balance before
- raw balance after
- matching settlement event name
- settlement transaction hash

The positive balance delta and matching request event are both mandatory.
`status_after=none` without this evidence is not a successful terminal
settlement.

Add validation tests proving a forced-settlement result cannot claim terminal
success with:

- a still-pending ticket
- a direct payout with a zero or negative balance delta
- a request/event identifier mismatch
- no transaction evidence when settlement was required

### Anvil-only enforcement

Every public or internal helper that performs any of the following must check
`is_anvil(web3)` itself:

- account impersonation
- native-token funding for an impersonated account
- ERC-20 storage balance injection
- direct storage mutation
- time warp or manual mining
- privileged operator, manager or fulfiller transaction

The outer caller's check is not sufficient. Add negative tests using a
non-Anvil provider stub and prove the function raises
`UnsupportedVaultSimulation(unsupported_reason="anvil_provider_required")`
before any mutation or broadcast.

## Workstream 1: Lagoon synthetic liquidity

The synthetic-liquidity implementation already exists. Do not replace it with
a new mode or API.

### Required changes

1. Preserve strict behaviour:
   `force_settle(ticket, ignore_liquidity=False)` must inspect the Safe balance
   before settlement and raise
   `lagoon_settlement_insufficient_liquidity` without partially settling the
   round.
2. Preserve explicit synthetic behaviour:
   `force_settle(ticket, ignore_liquidity=True)` may inject only the observed
   deficit on Anvil.
3. Return exact evidence:
   - `synthetic_assets_injected_raw == needed_raw - original_raw`
   - `liquidity_constraints_ignored is True` when injection occurs
   - `status_before == pending`
   - `status_after == claimable`
   - settlement and approval transaction hashes are retained
4. Make all Lagoon Anvil and ticket-type refusals use stable structured
   reasons.
5. Verify each of the nine exact deployments. Do not infer coverage from one
   representative Lagoon vault because Safe balances, vault versions,
   valuation managers, allowances and settlement roles differ.

### Lagoon tests

Add focused tests for:

- sufficient existing Safe liquidity: strict settlement succeeds with zero
  injection
- one-raw-unit shortfall: strict settlement refuses; explicit synthetic
  settlement injects the exact deficit
- material shortfall: strict settlement refuses; explicit synthetic
  settlement injects the exact deficit
- zero Safe balance
- missing Safe allowance
- non-Anvil rejection
- all nine exact vault IDs

Each successful exact-address test must progress the selected ticket to
claimable and complete the protocol claim.

## Workstream 2: Ember operator direct payout

Ember is not a generic claimable-ticket protocol. Its operator processes the
request and pays the receiver directly, so `status_after=claimable` is not the
right postcondition.

### Required research

For each exact Ember deployment:

1. Resolve the proxy implementation and verified ABI.
2. Identify the configured operator or operator role from current onchain
   state.
3. Identify the production request-processing function and its access
   restrictions.
4. Identify the request-specific processed event and payout asset.
5. Verify whether the active operator can be discovered and impersonated
   deterministically on Anvil.

If historical role discovery is needed, use Hypersync for `RoleGranted` or
protocol-specific operator events. Do not use JSON-RPC `eth_getLogs`.

### Required implementation

When the operator path is reproducible:

1. Require Anvil before impersonation.
2. Verify the discovered account currently holds the required role or matches
   the configured operator.
3. Fund and impersonate only that verified account.
4. Snapshot the receiver's denomination-token balance.
5. Execute the production processing call for the exact request sequence.
6. Decode the matching processed event.
7. Re-read request state and receiver balance.
8. Return direct-payout evidence only when the event request identifier
   matches and the receiver balance increased.

When the path is not reproducible, keep
`supports_anvil_settlement=False` and publish a deployment-specific stable
reason. The existing generic reason is acceptable only after the exact
operator path has been investigated and the missing prerequisite is recorded
in the test.

### Ember tests

- Keep local mock coverage for deterministic operator processing.
- Add exact-address fork characterisation for all four Ember vaults.
- Add a positive production-fork lifecycle test for every deployment whose
  operator is reproducible.
- Add an explicit negative capability test for every deployment whose
  operator cannot be discovered or safely impersonated.
- Assert request identifier, event, payout token, receiver and exact raw
  balance delta.

## Workstream 3: Plutus role fulfilment and claim

Plutus `fulfillRedeem(requestId)` is AccessControl-gated. Support requires
verified role discovery, not arbitrary account impersonation.

### Required research

1. Resolve the exact vault proxy implementation and verified ABI.
2. Determine the role constant protecting `fulfillRedeem`.
3. Determine whether the contract implements enumerable role membership.
4. If it does not, discover candidate holders from indexed `RoleGranted`
   events through Hypersync.
5. Verify every candidate against current onchain `hasRole` state.

### Required implementation

When an active holder is verified:

1. Require Anvil.
2. Fund and impersonate the verified holder.
3. Read the request status before fulfilment.
4. Call `fulfillRedeem(requestId)`.
5. Require the exact request to become claimable.
6. Return the fulfilment transaction hash and pending-to-claimable evidence.
7. Build and execute the ordinary user claim in the focused lifecycle test.
8. Verify the receiver's denomination-token balance increased.

When no holder can be verified, retain
`supports_anvil_settlement=False` and raise
`plutus_fulfilment_role_not_discoverable` before broadcast.

### Plutus tests

- Positive mock fulfil-and-claim lifecycle.
- Exact-address role discovery.
- Positive exact-address fulfil-and-claim when a current holder exists.
- Stale `RoleGranted` candidate rejected when `hasRole` is false.
- Missing holder produces the stable typed unsupported reason.
- Non-Anvil provider refuses before impersonation or broadcast.

## Workstream 4: Gains destination-chain settlement

`GainsDepositManager.force_settle()` operates on the Web3 instance bound to
the vault. It must prove the exact destination-chain protocol lifecycle and
must not know about Trade Executor bridge or satellite-trade state.

### Required changes

1. Verify the manager's Web3 chain ID matches the vault chain ID before
   settlement. Raise a stable typed reason on mismatch.
2. Reproduce the exact Arbitrum gTrade request at a fixed block.
3. Record the ticket's current and unlock epochs.
4. Warp time and call the permissionless epoch transition through the vault's
   configured open-PnL contract.
5. Require every iteration to increase the epoch.
6. Stop at the safety cap with a stable typed reason.
7. Require `can_finish_redeem(ticket)` and
   `status_after == claimable` before returning.
8. Execute `finish_redemption(ticket)` in the focused test and verify the
   receiver's asset balance increased.
9. Add stable structured reasons to every current reasonless failure in this
   path.

If the exact deployment depends on an oracle or keeper action that cannot be
reproduced, change its advertised capability to false and publish the concrete
reason. Do not return a successful settlement result for a ticket that is
still pending.

### Gains tests

- Exact Arbitrum vault request, forced epoch transition and claim.
- Wrong-chain Web3 rejection.
- Epoch does not advance.
- Safety cap reached before unlock.
- Ticket becomes claimable but claim fails.
- Non-Anvil rejection before time warp or broadcast.

## Workstream 5: Protocol-specific closed or capped overrides

This is an optional adapter capability, not a universal fallback and not a
whitelist bypass.

### Design rules

- Do not implement any override for account whitelisting or admission.
- Do not consume JSON closed-status metadata.
- Base managers must raise
  `closed_status_override_not_implemented`.
- An adapter may advertise an override only for an onchain blocker it can
  identify and alter deterministically on Anvil.
- Prefer a verified protocol admin/configuration call from an impersonated,
  currently authorised account.
- Use direct storage mutation only when the exact proxy implementation,
  storage layout and slot are verified for the tested deployment.
- Return structured evidence describing the original onchain blocker, the
  exact mutation or transaction, and the post-override onchain state.
- Never weaken the normal `create_deposit_request()` preflight. The override
  must be an explicit Anvil-only method.

### Suggested narrow API

If at least one production adapter can support this safely, add an optional
manager method such as:

```python
apply_anvil_deposit_availability_override(
    owner: HexAddress,
    raw_amount: int,
) -> VaultAnvilDepositOverrideResult
```

The base method raises typed unsupported. The result should contain:

- blocker kind (`closed`, `paused` or `capped`)
- vault and implementation addresses
- state before and after
- transaction hashes, if any
- storage mutation description, if any

Do not add this API speculatively if no exact production deployment can
satisfy the contract.

### Override tests

- Exact supported deployment and blocker.
- Normal preflight refuses before override.
- Override refuses outside Anvil.
- Override changes only the documented policy/cap state.
- Normal request construction succeeds after override.
- Snapshot revert restores the original blocker.
- Unknown implementation/version remains unsupported.

## Test infrastructure

Follow the repository's shared Anvil fork rules from
`eth_defi/testing/anvil_fork_pool.py`.

- Use `AnvilForkPool`, fixed block constants and matching `xdist_group`
  markers.
- Use `evm_snapshot_revert` around every mutating test.
- Share expensive deployments once per session where appropriate.
- Do not fork `latest`.
- Do not launch one private Anvil process per test module.
- If an observed deployment state is absent from the canonical midnight block,
  use one explicitly named exceptional fixed block and xdist group, document
  why it is required, and warm that block's RPC cache.
- If a pooled fork is recycled, recreate any post-launch baseline deployment
  before continuing.

Use normal repository fixtures and `create_multi_provider_web3()` conventions.
Never construct a raw `Web3(HTTPProvider(...))` for these tests.

## File-level work

Expected shared files:

- `eth_defi/vault/deposit_redeem.py`
- shared settlement-result and capability tests under `tests/vault/`

Expected protocol files:

- `eth_defi/erc_4626/vault_protocol/lagoon/deposit_redeem.py`
- `eth_defi/erc_4626/vault_protocol/ember/deposit_redeem.py`
- `eth_defi/erc_4626/vault_protocol/ember/vault.py`
- `eth_defi/erc_4626/vault_protocol/plutus/deposit_redeem.py`
- `eth_defi/erc_4626/vault_protocol/plutus/vault.py`
- `eth_defi/erc_4626/vault_protocol/gains/deposit_redeem.py`
- `eth_defi/erc_4626/vault_protocol/gains/vault.py`

Expected focused tests:

- `tests/erc_4626/vault_protocol/test_ember_deposit_redeem.py`
- `tests/erc_4626/vault_protocol/test_plutus.py`
- Gains protocol tests under `tests/erc_4626/vault_protocol/`
- Lagoon lifecycle tests under `tests/lagoon/`

If implementation needs a new external contract ABI, read
`eth_defi/abi/README.md` first, store the verified ABI under the protocol's ABI
directory and record its canonical source. Do not use a broad inline ABI.

## Implementation order

1. Add missing stable reasons and protocol-level result validation.
2. Verify Lagoon's nine exact deployments using the existing strict and
   synthetic-liquidity paths.
3. Reproduce and fix the exact Gains Arbitrum lifecycle.
4. Research and implement Ember direct-payout evidence.
5. Research and implement Plutus role fulfilment and claim.
6. Add only the closed/capped overrides proven feasible during exact-address
   research.
7. Run focused tests and update the vault protocol support documentation with
   the proven capabilities and remaining typed limitations.

This order delivers the existing Lagoon path and the likely Gains correction
before the operator-discovery work required by Ember and Plutus.

## Acceptance criteria

- Every affected adapter mutation has an internal Anvil guard.
- Every expected unsupported path has a stable machine-readable reason and
  complete vault context.
- Lagoon strict settlement refuses all observed shortfalls before partial
  settlement.
- Lagoon explicit synthetic settlement injects only the exact shortfall,
  reports it and makes the exact ticket claimable.
- All nine Lagoon deployments have exact-address coverage.
- Gains either completes request, epoch settlement and claim for the exact
  Arbitrum deployment, or advertises false capability with a
  deployment-specific reason.
- Each Ember deployment either returns verified direct-payout evidence for its
  exact request or advertises false capability with a deployment-specific
  operator limitation.
- Plutus either completes fulfilment and claim using a currently authorised
  holder or advertises false capability with a verified role-discovery reason.
- No successful forced settlement returns a pending ticket.
- No direct-payout success lacks a matching request event and positive receiver
  asset-balance delta.
- No whitelist or Trading Strategy JSON behaviour is added to `eth-defi`.
- No Trade Executor result strings, modes, persistence or retry logic are added
  to `eth-defi`.
- Focused tests use fixed pooled forks and snapshot isolation.

## Verification commands

Run focused tests only, using the repository environment:

```shell
source .local-test.env
poetry run pytest tests/vault/test_deposit_redeem.py
poetry run pytest tests/erc_4626/vault_protocol/test_ember_deposit_redeem.py
poetry run pytest tests/erc_4626/vault_protocol/test_plutus.py
poetry run pytest tests/gains/test_gtrade_usdc.py
poetry run pytest tests/lagoon/test_erc_7540_deposit_redeem.py
```

Run each mainnet-fork module with a three-minute command timeout.

After implementation, run Ruff formatting on changed Python files and inspect
the final diff for:

- accidental Trade Executor or JSON coupling
- missing stable unsupported reasons
- missing Anvil guards
- capability claims without exact-address proof
- terminal results without protocol-level evidence
