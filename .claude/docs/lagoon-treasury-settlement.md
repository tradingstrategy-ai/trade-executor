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
