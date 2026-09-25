# Lagoon treasury settlement

This document describes the strategy's own Lagoon vault: external investors
request deposits and redemptions into the strategy treasury. It is separate
from the external ERC-4626/ERC-7540 vault positions described in
[`vault-deposit-redeem.md`](vault-deposit-redeem.md).

## Settlement flow

`LagoonVaultSyncModel.sync_treasury(post_valuation=True)` first reconciles the
Safe reserve balance and calculates a fresh portfolio NAV. With no investor
queue, it posts that NAV with `updateNewTotalAssets()` when it differs from the
settled onchain `totalAssets()` by at least the configured tolerance (0.5% by
default). A nonempty investor queue bypasses this tolerance. Frozen positions,
insufficient redemption liquidity and disabled broadcasting can still stop a
transaction before it is sent.

Settlement is a separate `settleDeposit(uint256)` transaction through
the TradingStrategyModuleV0. Stock Lagoon v0.5 can settle both queued deposits
and redemptions through this call. When an empty-queue NAV update is due, the
executor also settles it to update `totalAssets()`. A successful NAV post alone
does not update `totalAssets()`.

## GuardV0 policy

TradingStrategyModuleV0 v0.6 may enable GuardV0 settlement safety. The Guard
measures gross underlying movement, in raw token units:

```text
gross flow = assets leaving the pending Silo + assets entering the Lagoon vault
```

Deposits and redemptions are added, never netted. A successful non-zero
asset-manager settlement consumes its gross flow from the configured fixed
window budget, normally 5,000 USDC over 24 hours. Empty settlements do not
open, extend or consume the window. A settlement at exactly the remaining
budget is allowed.

Modules deployed before the v0.6 upgrade use the legacy v0.5 per-settlement cap
and cooldown policy. The executor retains explicit v0.5 handling so those
vaults can continue to settle, but only v0.6 exposes the current window-budget
metadata.

The executor reads `getLagoonSettlementSafetyConfig()` only for the explicitly
supported module version. When a new smart-contract version is deployed, bump
the advertised module version and add explicit executor support; do not infer
support from a version string or failed feature probe.

## Executor decisions

| Condition | Action |
|---|---|
| No queue, positive settled NAV and change below the configured tolerance | Update treasury metadata without sending transactions |
| No queue, NAV change at least the configured tolerance | Post NAV and settle the valuation after a successful simulation |
| No queue, settled NAV is zero | Post NAV and settle; there is no positive baseline for a percentage comparison |
| Guard policy disabled, queue present | Post NAV and automatically settle queued flow |
| Queue within remaining window budget | Post NAV and automatically settle |
| Queue over remaining window budget | Post NAV, leave queue pending; wait for the window reset or use Safe governance |
| Legacy v0.5 cooldown active | Post NAV, leave queue pending and retry automatically later |
| Legacy v0.5 gross flow over cap | Post NAV, leave queue pending and emit an error |

Before a guarded settlement or an empty-queue settlement on an older module,
the executor simulates the wrapped call. This preserves the Guard's exact
raw-unit calculation and identifies expected deferrals without spending gas.
If an empty-queue simulation fails on an unlimited module, the executor logs
that the posted NAV has not reached `totalAssets()` and does not broadcast
settlement.
An empty settlement has no investor cash flow, but may mint fee shares; the
executor refreshes the vault share count from the settled block. The executor
analyses its receipt immediately; the later investor-flow scanner sees only
`SettleDeposit` and `SettleRedeem` events, which an empty settlement lacks.

## Manual settlement alert

An `ERROR` that says a queue exceeds the remaining GuardV0 settlement-window
budget includes the vault, Safe and module addresses, the pending queue sizes,
exact gross flow, amount already used and configured cap. The NAV update has
succeeded, but both queues remain pending. Wait for the fixed window to reset
or submit a deliberate direct Safe settlement.

The alert also gives a Gnosis Safe Transaction Builder-ready `settleDeposit`
ABI, specifies the Safe, vault target, zero value and call operation, and logs
the raw `_newTotalAssets` value. Safe owners must use that exact raw value for
the just-posted NAV when calling the vault directly.

`lagoon-settle` is not the recovery mechanism for this case because it uses the
same asset-manager module and will be rejected by GuardV0. The Safe owners must
submit the deliberate direct Safe settlement transaction. Direct governance
execution intentionally bypasses the asset-manager policy.

### Manual Safe preflight

Use `lagoon-manual-settle` before proposing the direct Safe transaction:

```shell
trade-executor lagoon-manual-settle
```

The default is a read-only preflight that prints Safe Transaction Builder
inputs. If the configured `PRIVATE_KEY` is an owner of the vault Safe, it can
instead create a signed Safe Transaction Service proposal:

```shell
trade-executor lagoon-manual-settle --propose-safe-transaction
```

The switch refuses an empty queue or a failed direct-call simulation, verifies
that the proposer key is a Safe owner, and posts no on-chain transaction. The
Safe owners must review the proposal, collect any additional required signatures
and execute it in Safe. This is intentionally opt-in: a normal preflight must
remain safe to run against production configuration.

The command uses the same `STRATEGY_FILE`, `STATE_FILE`, `VAULT_ADDRESS` and
`JSON_RPC_*` environment configuration as the executor. The proposal switch
also needs the normal `PRIVATE_KEY` configuration. It reads the submitted,
not-yet-settled raw NAV from
`newTotalAssets()` where available. For Lagoon versions without that getter it
recovers the value from the contract's `NewTotalAssetsUpdated` and
`TotalAssetsUpdated` events. The event scan starts at the vault deployment
block and uses the shared chunked JSON-RPC or Hypersync reader.

`SAFE_TRANSACTION_SERVICE_API_KEY` is optional. On 2026-09-19, the deployed
Lighter tutorial Safe accepted and indexed a signed, zero-value `totalAssets()`
proposal while the variable was unset. Configure it in production for Safe's
higher API rate limits and service reliability; the eth-defi Safe client reads
the variable automatically.

Safe Transaction Service proposals are available only on chains supported by
Safe's hosted service. On an unsupported chain, the command still prints the
manual Transaction Builder inputs, then reports that it cannot create a hosted
proposal. Safe takes the proposal nonce from the chain, so check for an
existing pending Safe transaction at that nonce before creating another
proposal.

Do not substitute the current `totalAssets()` value. The command reads the
current deposit and redemption queues and the Safe balance on-chain, then
selects `settleDeposit` for a deposit queue (which may also settle redemptions)
or `settleRedeem` for a redemption-only queue. It prints step-by-step Gnosis
Safe setup instructions, including the Safe, target, operation, ABI, input and
calldata. It uses `eth_estimateGas` with the Safe as the sender to check the
direct vault target call. A successful estimate only shows that call did not
revert at the reported block; it does not validate the NAV or Safe signatures,
owner policy or guards. Without `--propose-safe-transaction`, it does not sign,
post NAV, create a Safe proposal or broadcast any transaction. With the switch,
the report is printed before the Transaction Service proposal is created.

## Frontend metadata

For a supported TradingStrategyModuleV0 v0.6 vault, the `/metadata` response
publishes the live GuardV0 policy under
`on_chain_data.smart_contracts.lagoon_guard_v0`. It is a display aid, not an
authorisation mechanism: the executor still simulates the actual wrapped call,
and GuardV0 remains the on-chain authority.

| Field | Meaning |
|---|---|
| `automatic_settlement_window_limit` | Human-readable cumulative gross underlying-token budget for one fixed settlement window. It is `null` when GuardV0 is not applying a limit. |
| `automatic_settlement_window_limit_raw` | The same exact cap in underlying-token raw units. Use this for precise comparisons. |
| `settlement_window_seconds` | Fixed window duration, normally 86,400 seconds. |
| `settled_amount_in_window` | Gross underlying-token amount already consumed in the active window. |
| `remaining_automatic_settlement_budget` | Gross underlying-token amount that can still settle automatically before the window ends. |
| `settlement_window_end_timestamp` | Unix timestamp for the active window end. A value of zero means no non-empty settlement has opened a window. |
| `automatic_settlement_window_limit_enabled` | Whether GuardV0 is actively applying the displayed budget. A disabled limit means the normal asset-manager settlement path is uncapped, not that automatic settlement is disabled. |
| `guard_version` | The policy name, currently `GuardV0`. |

The limit is a cumulative gross-flow budget. A 9 USDC deposit and a 2 USDC
redemption consume 11 USDC, not 7 USDC, from the same active window. If the
remaining budget is insufficient, wait for the window reset or use direct
Safe-governance settlement.

## Troubleshooting

When a queue does not settle, inspect:

1. the TradingStrategyModuleV0 version and whether it is explicitly supported;
2. the Guard-configured vault asset and pending Silo;
3. the raw maximum settlement amount, amount consumed and window-end timestamp;
4. pending underlying deposits in the Silo; and
5. pending redemption shares in the Silo.

Do not post a NAV if frozen positions or stale valuation data make it
untrustworthy. The frozen-position safety check deliberately precedes the NAV
transaction.

## External settlement accounting

When Safe owners execute a successful, already-valued `settleDeposit()` call
with investor flow outside the executor, the next treasury sync discovers and
records that flow before it reconciles the Safe USDC balance. This is the manual
recovery procedure after GuardV0 blocks automatic settlement.

The executor creates one reserve `BalanceUpdate` per discovered flow settlement:

- `cause=deposit_and_redemption`;
- `quantity = deposited - redeemed` in the reserve stablecoin; and
- the settlement transaction hash is the idempotency key.

The Safe balance alone is not evidence of an investor flow: trading, bridges
and external account integrations can change it too. Do not run
`correct-accounts` to absorb a manual settlement, because that loses the
investor-flow history used for profitability and equity calculations.

### Confirmation safety

The model scans only through its reorganisation-buffered safe block. Before it
reconciles Safe USDC, it checks newer blocks for unrecorded Lagoon settlement
logs. Such a log raises `LagoonUnconfirmedSettlement`; the strategy runner then
ends the cycle before account checks or `decide_trades()`.

Only mined receipt events are considered, so a dropped or replaced transaction
does not defer a cycle. The settlement cursor never advances beyond the safe
block.

### State export

`state.sync.treasury` exposes:

| Field | Meaning |
|---|---|
| `pending_deposits` | Underlying waiting in the Lagoon Silo; not portfolio cash. |
| `pending_redemptions` | Current underlying estimate required for queued redemption shares. |
| `last_lagoon_settlement_block_scanned` | Highest block with fully processed Lagoon settlement logs. |

`None` means the queue has not been observed. `0.0` means it was observed
empty. These are snapshots only; receipt events remain the investor-flow source.
