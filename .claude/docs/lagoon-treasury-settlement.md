# Lagoon treasury settlement

This document describes the strategy's own Lagoon vault: external investors
request deposits and redemptions into the strategy treasury. It is separate
from the external ERC-4626/ERC-7540 vault positions described in
[`vault-deposit-redeem.md`](vault-deposit-redeem.md).

## Settlement flow

`LagoonVaultSyncModel.sync_treasury(post_valuation=True)` first reconciles the
Safe reserve balance and calculates a fresh portfolio NAV. It then always posts
that NAV with `updateNewTotalAssets()`.

Settlement is a separate, optional `settleDeposit(uint256)` transaction through
the TradingStrategyModuleV0. Stock Lagoon v0.5 can settle both queued deposits
and redemptions through this call. A successful NAV post therefore does not
mean that an investor queue was settled.

## Persisted queue state

The executor exposes the latest observed aggregate investor queue in the state
API, not in `/metadata`:

| State field | Meaning |
|---|---|
| `sync.treasury.pending_deposits` | Underlying-token assets still held in the pending Silo. |
| `sync.treasury.pending_redemptions` | Underlying-token estimate needed for redemption shares still held in the Silo. |

Both values are human-readable underlying-token amounts. For Lagoon's required
stablecoin reserve, the executor also treats them as US dollar amounts.
`None` means that the queue has not been observed by this executor version in a
completed post-valuation treasury sync; an observed empty queue is stored as
`0.0`. After upgrading an existing executor state, `pending_deposits` remains
`None` until its first completed post-valuation treasury sync even if the older
pending-redemption snapshot is populated.

After the NAV transaction is confirmed, a deferred, manually-required or empty
queue is read at one queue snapshot block. The executor stores that exact block
in `sync.treasury.last_block_scanned`, even though no settlement transaction or
balance update occurred. A successful settlement refreshes the remaining
deposit and redemption queues at the settlement analysis block.

Queue observations do not create `BalanceUpdate` events and do not enter
portfolio cash or NAV: pending deposits remain outside the Safe, while pending
redemptions are a liquidity requirement until settlement. GuardV0 cap and
cooldown policy remains separate live `/metadata` display data.

## GuardV0 policy

TradingStrategyModuleV0 v0.5 may enable GuardV0 settlement safety. The Guard
measures the gross underlying movement, in raw token units:

```text
gross flow = assets leaving the pending Silo + assets entering the Lagoon vault
```

Deposits and redemptions are added, never netted. A settlement at exactly the
configured cap is allowed. A successful non-zero asset-manager settlement also
starts the configured cooldown, normally 24 hours. Empty settlements neither
start nor wait for cooldown.

The executor reads `getLagoonSettlementSafetyConfig()` only for the explicitly
supported module version. When a new smart-contract version is deployed, bump
the advertised module version and add explicit executor support; do not infer
support from a version string or failed feature probe.

## Executor decisions

| Condition | Action |
|---|---|
| No queue | Post NAV only |
| Guard policy disabled | Post NAV and automatically settle queued flow |
| Queue within cap and cooldown expired | Post NAV and automatically settle |
| Cooldown active | Post NAV, leave queue pending and retry automatically later |
| Gross flow over cap | Post NAV, leave queue pending and emit an error |

Before a capped non-empty settlement, the executor simulates the wrapped module
call. This preserves the Guard's exact raw-unit calculation and identifies the
amount-limit or cooldown custom error without spending gas on an expected
revert.

## Manual settlement alert

An `ERROR` that says `direct Safe-governance settlement required` means that
the queued gross flow exceeded the Guard cap. It includes the vault, Safe and
module addresses, the pending queue sizes, exact gross flow and configured cap.
The NAV update has succeeded, but the queues and Guard cooldown timestamp are
unchanged.

`lagoon-settle` is not the recovery mechanism for this case because it uses the
same asset-manager module and will be rejected by GuardV0. The Safe owners must
submit the deliberate direct Safe settlement transaction. Direct governance
execution intentionally bypasses the asset-manager policy.

## Frontend metadata

For a supported TradingStrategyModuleV0 v0.5 vault, the `/metadata` response
publishes the live GuardV0 policy under
`on_chain_data.smart_contracts.lagoon_guard_v0`. It is a display aid, not an
authorisation mechanism: the executor still simulates the actual wrapped call,
and GuardV0 remains the on-chain authority.

| Field | Meaning |
|---|---|
| `daily_automatic_settlement_limit` | Human-readable maximum gross underlying-token flow for one automatic settlement. It is `null` when GuardV0 is not applying a daily limit. |
| `daily_automatic_settlement_limit_raw` | The same exact cap in underlying-token raw units. Use this for precise comparisons. |
| `settlement_cooldown_seconds` | GuardV0's wait after a successful non-empty automatic settlement; normally 86,400 seconds. The cap plus this cooldown are why the frontend calls the field a daily limit. |
| `next_automatic_settlement_timestamp` | Unix timestamp when the next non-empty automated settlement can run. A value of zero means no cooldown has started. |
| `daily_automatic_settlement_limit_enabled` | Whether GuardV0 is actively applying the displayed daily limit. A disabled limit means the normal asset-manager settlement path is uncapped, not that automatic settlement is disabled. |
| `guard_version` | The policy name, currently `GuardV0`. |

The limit is a gross-flow cap. A 9 USDC deposit and a 2 USDC redemption have
an 11 USDC GuardV0 flow, not a 7 USDC net flow. If this exceeds the displayed
limit, only direct Safe-governance settlement may process the queue.

## Troubleshooting

When a queue does not settle, inspect:

1. the TradingStrategyModuleV0 version and whether it is explicitly supported;
2. the Guard-configured vault asset and pending Silo;
3. the raw maximum settlement amount and next eligible timestamp;
4. pending underlying deposits in the Silo; and
5. pending redemption shares in the Silo.

Do not post a NAV if frozen positions or stale valuation data make it
untrustworthy. The frozen-position safety check deliberately precedes the NAV
transaction.
