# Recover Hyper-AI's interrupted 28 September deposit

## Outcome and scope

Make the existing `repair` and `correct-accounts` commands take the stopped
Hyper-AI state from an interrupted, partially executed decision to a state
that can safely restart on the next two-day slot. Keep the commands' existing
division of labour: `repair` fixes transactions that never started, while
`correct-accounts` verifies and recovers money that left the Safe. Do not add a
recovery CLI, a general HyperCore phase-resumption state machine, or automatic
replay of the interrupted decision.

This plan is for the 2026-09-28 Fadorador incident. Its proof must be a test
against a copy of the **actual pre-repair state**, not only a fabricated trade.
No implementation or test may broadcast to production or modify that original
snapshot.

## Evidence to reproduce

- The untouched local copy is
  `/tmp/hyper-ai-correct-accounts.nV23ya/state/hyper-ai.pre-repair.json`,
  SHA-256 `84b6a8e22b1ff36731a93b9b5a175de007bb4d60492ee3554c09eb4ac1eb0a30`.
  It is an incident input, not a file to commit. The partially changed
  `hyper-ai.json` beside it is **not** the starting fixture.
- The state has `cycle=123`, a pending logical slot of 2026-09-28 00:00 UTC,
  three successful same-slot trades (#1726, #1727, #1730), four planned buys
  without transactions (#1729, #1731, #1732, #1733), and Fadorador buy #1728
  in `started` status with only its phase-1 transactions persisted. Position
  #532 has zero quantity. The at-risk marker keeps 3,023.713174 USDC allocated.
  The deployed strategies repository's `strategy/hyper-ai-v9.py` uses
  `CycleDuration.cycle_2d`; the separate `strategies/hyper-ai.py` example in
  trade-executor uses a daily cycle and is **not** this deployment.
- The executor log records phase 2 spot-to-perp verified, followed by phase 3
  reverting with `Hypercore vault not allowed`. The four real HyperEVM receipts
  are phase 1 approval at block 47,080,085 (success), phase 1 deposit at
  47,080,088 (success), phase 2 at 47,080,099 (success), and phase 3 at
  **47,080,105 (revert)**. The failed phase-3 hash is
  `0x0d5a389f0dd981c38c4e848e808479ce7ce2e1953246e6bd320c636f2c32fb2d`.
  The phase-2 hash is
  `0xc2d0aeb0cba1b7bfd6535097b15205acf232fc47570f444510bc806881905e53`.
- At the failure, the log showed Safe EVM USDC 10,775.588216, HyperCore perp
  withdrawable 3,024.126339, spot free 0.008197, no active perp position,
  and zero Fadorador vault equity. The canonical HyperEVM USDC `balanceOf`
  at block 47,080,105 is 10,775.588216. These are **incident-time** values,
  not a promise about production balances when an operator later runs a command.
- On a copy, current `repair --auto-approve` saved the four no-transaction
  repairs, then raised because `repair_zero_quantity()` called
  `close_position_with_empty_trade()` on #532 although its opening trade was
  not successful. The follow-up `correct-accounts --dry-run` planned
  3,023.626339 USDC perp-to-spot and 3,023.624536 spot-to-EVM, but printed
  zero accounting corrections **before those transfers**. Startup still
  refuses the pending slot because it contains trades.

## Command changes

### 1. `repair`: fix no-fill positions without touching stranded capital

In `tradeexecutor/state/repair.py`, make zero-quantity cleanup distinguish a
never-filled position from a position whose capital location is unresolved:

- A position with any started/broadcasted transaction or unresolved HyperCore
  accounting marker is deferred. Do not call the successful-opening-trade
  close helper and do not refund its planned reserve.
- After a planned/started trade **without transactions** has been repaired,
  close its now-empty position directly with an explanatory note. Do not invent
  a successful trade merely to satisfy the existing helper's assertion.
- Guard `repair_trade()` itself against at-risk HyperCore trades, so another
  caller cannot accidentally use its unconditional planned-reserve refund.

In `tradeexecutor/cli/commands/repair.py`, take a state backup before the first
mutation, preserve and report safe partial progress, and finish with an
explicit incomplete/non-zero result while any at-risk trade remains. Do not
extend `rebroadcast_all()` to rebroadcast #1728: its persisted phase-1 hashes
are already mined, and the later HyperCore outcome is not represented by the
two persisted transaction objects. This command must not make a Safe transfer.
Assert that the backup was actually created; the current
`backup_state(unit_testing=True)` path does not exercise backup creation.

### 2. `correct-accounts`: reconcile this at-risk opening trade and real cash

Keep the existing Safe-level transit planner and its dust/fee rules. Add a
narrow path for an at-risk HyperCore **opening buy** whose stored state is
`started` with transactions. Before any transfer, read the existing EVM
receipts and fresh Safe, HyperCore spot, perp, and target-vault balances.
Require one relevant at-risk trade, no active perp positions, and no target
vault equity. Classify it before optional small-position cleanup or the
existing transit-recovery helper can broadcast anything. Distinguish two
observed states using the saved reserve, current Safe balance, and HyperCore
balances:

- **Still stranded:** perp USDC is consistent with the at-risk amount after
  pre-existing dust; plan the existing perp-to-spot-to-EVM recovery.
- **Already returned after a crash:** perp is down to allowed dust and the
  Safe has the corresponding unaccounted USDC surplus; skip both transfer
  legs and continue the accounting/slot reconciliation.

If neither pattern fits, including a possible unrelated investor cash flow,
stop for operator review. Do not add a new persisted phase marker to solve
this one incident. The phase-2 and phase-3 hashes are present in the log but
**not** in the saved trade, so the command cannot pretend to have derived the
phase-3 revert from state. The operator must verify the known reverted phase-3
receipt and current custody before accepting the real recovery. Do not infer
HyperCore settlement from an EVM receipt alone or retry the rejected deposit.

`--dry-run` must report the trade and position, observed locations, proposed
transfer amounts, expected remaining dust, the pending slot, and **not safe to
restart yet**. It must not call a pre-transfer `0 corrections` result a clean
outcome. No transactions or state writes occur in dry-run.

On the real run, use the existing recovery helper to move only verified
transferable USDC perp-to-spot-to-Safe; wait for the Safe's actual ERC-20
increase. Then terminalise #1728 as a failed, zero-fill deposit **without**
calling generic `repair_trade()` or releasing the full planned reserve. Let
the existing reserve balance correction credit the actual Safe surplus once.
Close the empty Fadorador position after verifying its vault equity is zero.
Retain an audit note with observed balances, receipt hashes, recovered amount,
remaining dust and failed-deposit reason; clear its *unresolved* marker only
after the final Safe/vault account check passes. Skip optional HyperCore
small-position cleanup while this at-risk incident is being reconciled: it
currently runs before transit recovery and could create unrelated live trades.
Keep the marker on disk during external recovery. After applying the actual
Safe surplus in memory, verify Safe and vault balances, clear the marker in
memory, run the normal final account check, then save the resolved trade and
accounting. If anything fails before that save, rerun from the observed
**already returned** state without another transfer. That rerun must apply
the still-unrecorded Safe reserve credit, terminalise the trade, clear the
marker, and consume the slot once before saving. Only a subsequent rerun of
the saved, resolved state must make no further credit or slot change.

The current `preflight_state_for_account_correction()` permits #1728 only by
accident: it rejects started trades without transactions but misses started
trades **with** transactions. Make the exception for this controlled
reconciliation explicit. Other started/broadcasted at-risk trades must still
fail closed. Add the unresolved-marker refusal at **live startup**, not to
the general `State.check_if_clean()` predicate that `check_accounts()` also
uses during reconciliation. Scan all recorded trades there, so closing a
position prematurely cannot hide an unresolved marker. Historical repaired
trades in the same state retain `hypercore_stranded_usdc` as audit metadata;
that key alone is not a new unresolved incident after repair. An explicit
capital-at-risk or reconciliation-required marker remains blocking on a
terminal trade. The correct-accounts
path must be able to inspect and clear this specific marker, whereas a normal
trading restart must not.

### 3. Consume, never replay, the partial decision

The 28 September slot already has successful trades, so it is not safe to
retry the whole decision. Add one explicit confirmation/option to the existing
`correct-accounts` command to consume this pending HyperCore slot **after**
all its trades are terminal, no material transit capital remains unresolved,
and final account checks pass. Save the slot resolution atomically with
`pending_data_availability_slot = None`, `last_cycle_at = 2026-09-28 00:00`
and the next cycle number, plus a short audit note that the decision was only
partially executed. The next scheduled slot should be 2026-09-30 00:00 UTC.
No generic automatic slot clearing and no new command. If final verification
fails, leave the pending slot in place. Save the marker clearance, accounting,
and slot consumption together after verification; do not let an earlier
`store.sync()` expose a clean-looking but still unresolved state.

## Incident integration test

Add one CLI-level incident test that invokes `repair`, a dry run of
`correct-accounts`, and the real `correct-accounts` path against a **temporary copy**
of the untouched pre-repair state. Use a HyperEVM Anvil fork pinned to block
**47,080,105**, not a moving head. Assert its USDC Safe balance is
10,775.588216 and the four incident receipts have statuses `1, 1, 1, 0`.
Use the existing deployment record and a test-only signer; never load the
production private key.

Anvil reproduces HyperEVM receipts and ERC-20 accounting, but it does not
advance HyperCore's historical Info API or settle CoreWriter cross-domain
actions. Mock only those boundaries: return the log-captured initial
perp/spot/Fadorador balances (and stable values for the other existing vault
positions); after the planned recovery action, update the mock HyperCore
balances and credit exactly the resulting Safe USDC amount on Anvil using the
existing `fund_erc20_on_anvil()` test helper. That helper **overwrites** the
absolute ERC-20 balance: read the 10,775.588216 USDC starting balance and set
it to **starting balance plus returned amount**, then assert that total. Do
**not** mock `repair`, state serialisation, reserve correction, slot scheduling,
or the CLI entry points.
Commit a fixed-block Foundry RPC cache seed at
`tests/rpc_cache_seed/hyperliquid/47080105/storage.json`, following
`tests/mainnet_fork/README.md`. Run the focused incident test once against a
responsive archive RPC with an empty external `FOUNDRY_RPC_CACHE_DIR`, close
Anvil cleanly so it writes the captured reads, then add the resulting
`storage.json` to the branch. The test uses the repository's existing
`_seed_foundry_rpc_cache` fixture and a `warm_rpc_test_group` marker with its
own `xdist_group`; CI starts from this stored RPC state instead of cold-fetching
the incident's historical reads. Assert the pinned block, Safe balance and
receipts on the seeded fork. A cache miss may still need a live provider, so
run the full test once with a fresh Foundry cache before treating the seed as
complete. Generate and test the seed with trade-executor CI's pinned Foundry
`v1.2.3`, not eth-defi's differently pinned fork suite. Do not commit an
`anvil_dumpState` file or rely on a moving fork head.
Reuse a small in-repo incident-shaped fixture for CI; run the same scenario
locally against the 44 MB snapshot through an opt-in state-file path. Do not
commit the full production snapshot or add a fixture-generation framework.
The test docstring must say that the full-snapshot run is a required gate
before any production repair. Both fixture variants assert the incident's
cycle, slot, trade IDs/statuses, and 3,023.713174 USDC at-risk amount before
running either command.

Assertions, in order:

1. `repair` exits incomplete rather than crashing; #1729/#1731/#1732/#1733
   are repaired, their empty positions close, #1728 remains at risk, and
   Safe cash and state reserve are not increased by #1728's planned amount.
2. `correct-accounts --dry-run` explains the planned 3,023.626339 and
   3,023.624536 USDC legs and the pending slot; it leaves both the copied
   state bytes and fork balances unchanged.
3. The real command credits only the verified Safe increase, marks #1728
   failed with zero fill, closes #532, clears the unresolved marker after
   final verification, and consumes the partial slot by explicit option.
   `check_accounts()` and `State.check_if_clean()` pass; slot calculation
   returns 30 September, not a replay of 28 September.
4. Exercise the crash boundary in a fresh copy: inject one exception at the
   **first state-file save after the recovery helper has changed mock HyperCore
   balances and Anvil Safe USDC**. Rerun from the unchanged on-disk state and
   assert it recognises the already-returned funds, skips transfer, books the
   Safe surplus once, and consumes the slot. A further rerun creates no
   transfer, reserve credit, or slot increment. A focused negative case
   with non-zero Fadorador equity or inconsistent perp/Safe amounts performs
   **no** transfer or state repair.

Run this test with the local original copy before considering a production
repair. Keep production stopped during validation. Once code and test pass,
the operator separately takes a fresh production backup, runs the commands'
preview path, reviews **current** balances, performs the real correction,
and restarts only after a final account and slot check. The test is not
authorisation to transfer live funds.

## Documentation and non-goals

Update `.claude/docs/hypercore-vault.md`, `docs/hypercore-data-availability.md`,
and the two commands' help text: distinguish a dry-run plan from a clean
post-transfer state, describe the explicit partial-slot consumption, and
remove the current implication that ordinary `repair` plus `correct-accounts`
already settles this crash window. No strategy ranking changes, backtest
changes, new storage schema, generic recovery command, or automatic vault
deposit retry are part of this fix.

## Review

Kimi K3 reviewed this plan on 2026-09-28. The completed, bounded plan-only
review found two blocking gaps: a post-transfer rerun must recognise USDC
already returned to the Safe, and `fund_erc20_on_anvil()` requires an absolute
post-recovery balance. Both are addressed above. Its other suggestions led to
the startup-scoped marker guard, an exact post-transfer/pre-save crash point,
and fixture-shape assertions. A final K3 blocking-only pass caught an ambiguous
sentence about crediting after a crash; it now distinguishes the first rerun,
which must book the recovered Safe cash, from later idempotent reruns. A
broader repository-grounded Kimi run reached its time limit without a final
result; do not treat that incomplete run as an additional endorsement.
