# Lagoon external settlement accounting

## Objective

Account for a confirmed Lagoon investor settlement even when Safe owners submit
the settlement transaction. A manual GuardV0 recovery must be recorded before
the changed Safe USDC balance is used for valuation or trading.

## Implementation

At a reorganisation-buffered safe block, `LagoonVaultSyncModel` scans Lagoon
`SettleDeposit` and `SettleRedeem` events from the deployment block, or from
the previous settlement cursor plus one. The dependency scanner reads its range
in bounded JSON-RPC chunks. It excludes transaction hashes already stored on
`deposit_and_redemption` balance updates, analyses each new receipt, then
creates one normal reserve balance update per transaction.

The same in-memory update records the Safe reserve balance at that block, the
queue snapshots and the new scan cursor. A scan or receipt-analysis failure
leaves persistent accounting unchanged. The executor broadcast path reuses the
same receipt-to-event helper.

Before reconciliation, newer blocks are checked for unrecorded settlement logs.
If one exists, `LagoonUnconfirmedSettlement` ends the strategy cycle before
trading. The cursor remains below that confirmation buffer.

## State

```python
pending_deposits: USDollarAmount | None = None
last_lagoon_settlement_block_scanned: BlockNumber | None = None
```

`pending_redemptions` already exists. Queue values are export-only snapshots and
do not form part of cash, NAV or the investor-flow series.

## Test

The Base Anvil test queues 9 USDC, posts NAV, settles through the Safe, then
verifies recovery, state JSON and a cursor-rewind rescan without duplication.
