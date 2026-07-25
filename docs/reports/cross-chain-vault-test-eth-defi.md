# Eth-defi cross-chain vault test report

## Evidence and scope

The 129-vault matrix was re-run on 2026-07-25 against eth-defi master
`b42ef5747` (#1368 "close vault simulation adapter gaps"), from a trade-executor
worktree at `f311d0fa` (merged PR #1577 plus the submodule bump). Both import
paths were verified worktree-first before the run. The run used
`--auto-simulated --settle-async-on-anvil`.

Machine-readable evidence:
`docs/reports/cross-chain-vault-test-2026-07-25.report.json`; full table:
`docs/reports/cross-chain-vault-test-2026-07-25-results.md`.

The run completed 43 full deposit/redemption lifecycles. 51 vaults were
deposit-closed and 14 needed whitelisting (both current on-chain admission, not
adapter defects). **21 vaults have adapter/lifecycle gaps — the same count as
the previous eth-defi report.** #1368 improved result *typing* (Ember minimum,
Ember/Gains async capability metadata) and one 40acres case now completes, but
it did not close the concrete simulation gaps below.

## Confirmed improvements from #1368

- **Ember async classification.** Ember Earn/Polymarket/Third Eye/UDL now expose
  `VaultDepositManagerCapability(deposit_flow=synchronous,
  redemption_flow=asynchronous)` and reach the direction-specific resolver,
  instead of the earlier "Failed to analyse vault tx" receipt failures.
- **Ember Apollo ACRED minimum** is now a typed
  "redemption shares 904 are below minimum 9170000" instead of a generic
  `ValueError`.
- **Gains on Base** is recognised as async (`deposit_flow=synchronous`,
  `redemption_flow=asynchronous`).

## Eth-defi work still required

### Lagoon Finance — 6 vaults (unchanged)

Pending-ticket deployments:

- Moon Digital AM USDC, `1-0xa00f63e85b3d242568a9edecb48f5e2cf879b07b`;
- Syntropia USDC, `1-0xd17049ed25d8f99fe3bfd10cef2263da9995cfd8`;
- Angmar Capital, `42161-0x1723cb57af58efb35a013870c90fcc3d60174a4e`.

`force_settle()` runs but the ERC-7540 redemption ticket stays
`pending -> pending`. Add version-aware settlement diagnostics: record the
valuation-manager and Safe calls, inspect settlement receipt/events, and decide
whether these deployed versions need another settlement method, epoch advance or
role. If a fork cannot make the ticket claimable, raise
`UnsupportedVaultSimulation` with the concrete version/role reason.

Allowance-revert deployments (Base):

- For Yield v2, `8453-0x2bff679b1a9fbcc202316c1402172747ba2fbf56`;
- RB Capital Yield, `8453-0x63b04d3ce2c14f6d308657ab73ac92fc1a0b1075`;
- TruMarket, `8453-0xbe7db44f4ce20dac83b578b94fd35087f66e9754`.

These revert inside `settleDeposit(_newTotalAssets=...)` with
`ERC20: transfer amount exceeds allowance`: the vault pulls USDC via
`transferFrom()` from the impersonated asset holder without sufficient
allowance. The Anvil driver must reproduce the deployed protocol's real approval
path before settlement, or mark the deployment unsupported. Translate the
settlement transaction failure at the manager boundary and only return
settlement hashes after checking every receipt.

### Ember — 4 async settlement vaults (unchanged)

- Ember Earn, `1-0x9be9294722f8aad37b11a9792be2c782182cafa2`;
- Ember Polymarket, `1-0x0b9342c15143e8f54a83f887c280a922f4c48771`;
- Ember Third Eye, `1-0xf3190a3ecc109f88e7947b849b281918c798a0c4`;
- Ember UDL, `1-0x373152feef81cc59502da2c8de877b3d5ae2e342`.

Async redemption tickets are created correctly, but
`EmberDepositManager` advertises `supports_anvil_settlement=False`. If the
operator processing call can be reproduced on a fork, implement
`EmberDepositManager.force_settle(ticket)` (impersonate the documented operator,
advance any required time, verify the `RequestProcessed` event, publish
`supports_anvil_settlement=True`). Otherwise keep the capability false with a
precise unsupported reason.

### cSigma Finance — 2 vaults (unchanged)

- cSigma USD, `1-0xd5d097f278a735d0a3c609deee71234cac14b47e`;
- cSuperior Quality Private Credit USDC,
  `1-0x438982ea288763370946625fd76c2508ee1fb229`.

Despite the #1368 cSigma work, cSigma USD still asserts on redemption:
`Max redeem 45368 (raw) is less than what we try to redeem 907359 (raw)` — the
capacity limit is exceeded when redeeming the full deposited position. Return
`VaultFlowUnavailable` with requested and available (`maxRedeem`) capacity
before constructing the request, rather than a raw assertion.

cSuperior deposits successfully, then `redeem()` reverts with
`execution reverted: Withdrawal pending`. This is a FIFO/queued withdrawal, not
an immediate synchronous ERC-4626 redemption. Model it as an asynchronous
redemption request/ticket (request-event parsing, status probing, claim
construction, optional Anvil settlement driver); do not treat
`Withdrawal pending` as a synchronous revert.

### Gains Network — 2 vaults (unchanged count)

- Base gTrade USDC, `8453-0xad20523a7dc37babc1cc74897e4977232b3d02e5` —
  async, but `GainsDepositManager` advertises no Anvil settlement driver.
  Implement a ticket-specific driver that advances the epoch and invokes the
  keeper/manager processing path, verifying the exact ticket becomes claimable
  before advertising `supports_anvil_settlement=True`.
- Arbitrum gTrade USDC, `42161-0xd3443ee1e91af28e5fb858fbd0d72a63ba8046e0` —
  the satellite `redeem()` reverts with `custom error 0xa73449b9`. Add the
  deployed ABI error, decode the selector and determine whether it is an epoch
  window/keeper requirement; expose it as a typed async flow or
  `VaultFlowUnavailable`.

### YieldNest — 1 vault (unchanged)

YieldNest RWA MAX, `1-0x01ba69727e2860b37bc1a2bd56999c1afb4c15d8`, deposits, then
`redeem()` reverts with `custom error 0xb8b8b59c` carrying encoded
owner/amount data. Add the deployed ABI error, decode the selector and determine
whether it represents capacity, cooldown, queueing or access policy. Expose the
corresponding preflight or asynchronous manager flow and raise
`VaultFlowUnavailable` when it is a current-state admission condition.

### Accountable — 1 vault (unchanged)

Hyperithm Delta Neutral Vault, `143-0x7cd231120a60f500887444a9baf5e1bd753a5e59`
(Monad), still reverts `deposit()` with `custom error 0x5945ea56`. #1368 added
typed Accountable minimum-deposit/preflight handling for a different deployment
(the originally reported address no longer exposes usable state), but this live
Monad vault is unhandled. Map the selector for this deployment: identify whether
the executor module needs a role/allow-list entry or whether deposits are
paused/capped, and surface it through `is_account_whitelisted()`, deposit
preflight or a typed `VaultFlowUnavailable`, preserving the decoded arguments.

### Upshift — 1 vault (regressed error shape)

Sentora USD Earn, `1-0x74ad2f789ed583dbd141bbdafc673fe1f033718b`. #1368 added an
Upshift multi-asset `VaultDepositManager`, but the simulation now fails at
deposit with `The function 'maxDeposit' was not found in this contract's abi`
instead of the previous clean "adapter unsupported". The deposit-availability
preflight reaches generic ERC-4626 `maxDeposit()`, which this multi-asset vault
does not implement. eth-defi should expose a `maxDeposit`/`previewDeposit`
capability (or deposit-limit reader) on the Upshift vault/manager so the
preflight can size and gate the deposit without the standard ERC-4626 method.
The complementary executor-side preflight routing fix is in the trade-executor
report.

### Plutus — 1 vault (unchanged)

Plutus Hedge Token, `42161-0x58bfc95a864e18e8f3041d2fcd3418f48393fe6a`, reports
`redemption_unavailable` after a successful deposit. Confirm whether this is
live protocol state (cooldown/window/capacity) or missing adapter support. If
live state, populate a typed reason and next-eligible time; if a public
request/claim flow exists, implement it and advertise the correct capability.

## Not eth-defi defects

The 51 deposit-closed and 14 whitelist-gated rows reflect current on-chain
admission and remain structured non-success results. The 40acres Pharaoh
(Avalanche) `transfer amount exceeds balance` and the IPOR Autopilot (Base)
satellite-close revert are cross-chain executor reconciliation/diagnostics items
tracked in the trade-executor report, not adapter defects.

## Acceptance criteria

For each protocol change:

1. add a focused manager-level test using the affected deployment/version;
2. publish capability metadata only for complete deposit and redemption paths;
3. use typed `VaultFlowUnavailable` / `UnsupportedVaultSimulation` instead of
   generic `ValueError`, `AssertionError` or raw settlement transaction errors;
4. persist enough ticket data to reconstruct a claim after restart; and
5. verify a forced settlement changes the selected ticket from pending to
   claimable before returning.
