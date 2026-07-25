# Eth-defi cross-chain vault test report

## Evidence and scope

The corrected 129-vault matrix ran on 2026-07-24 against eth-defi master
`b5803bdc52606190969ca44af878b25cde8e3dec`. Both import paths were verified
before the run. The invalid earlier matrix, whose tracebacks imported a
different trade-executor checkout, is not evidence for eth-defi behaviour.

The corrected run completed 43 full deposit/redemption simulations. Another 65
vaults were currently closed or required whitelisting. The actionable
protocol-level gaps are detailed below. The separate trade-executor report
covers executor fixes already implemented for settlement ownership,
failure-operation attribution, share rounding and async lifecycle detection.

## Eth-defi work required

### Lagoon Finance

Affected vaults:

- Moon Digital AM USDC,
  `1-0xa00f63e85b3d242568a9edecb48f5e2cf879b07b`;
- Syntropia USDC,
  `1-0xd17049ed25d8f99fe3bfd10cef2263da9995cfd8`;
- Angmar Capital,
  `42161-0x1723cb57af58efb35a013870c90fcc3d60174a4e`;
- For Yield v2,
  `8453-0x2bff679b1a9fbcc202316c1402172747ba2fbf56`;
- RB Capital Yield,
  `8453-0x63b04d3ce2c14f6d308657ab73ac92fc1a0b1075`;
  and
- TruMarket,
  `8453-0xbe7db44f4ce20dac83b578b94fd35087f66e9754`.

The first three reach `LagoonDepositManager.force_settle()`, but their
redemption ticket stays `pending -> pending`. Add version-aware settlement
diagnostics: record the valuation-manager and Safe calls made, inspect the
settlement receipt/events, and determine whether these deployed versions need
another settlement method, epoch advance or role. If the fork cannot make the
ticket claimable, raise `UnsupportedVaultSimulation` with that concrete
version/role reason.

The three Base vaults revert inside `settleDeposit(_newTotalAssets=...)` because
the vault calls USDC `transferFrom()` from the impersonated asset holder without
sufficient allowance. The Anvil driver must reproduce the deployed protocol's
real approval path before settlement, or explicitly mark that deployment
unsupported. Catch and translate the settlement transaction failure at the
manager boundary; do not leak a raw `TransactionAssertionError`, because the
caller then cannot distinguish a failed settlement driver from a failed user
request. Return settlement hashes only after checking every receipt.

Add focused fork tests for one successful Lagoon deposit/redemption lifecycle,
one `pending -> pending` deployment and one allowance-requiring deployment.

### Ember

Affected vaults:

- Ember Earn, `1-0x9be9294722f8aad37b11a9792be2c782182cafa2`;
- Ember Polymarket, `1-0x0b9342c15143e8f54a83f887c280a922f4c48771`;
- Ember Third Eye, `1-0xf3190a3ecc109f88e7947b849b281918c798a0c4`;
- Ember UDL, `1-0x373152feef81cc59502da2c8de877b3d5ae2e342`;
  and
- Ember Apollo ACRED,
  `1-0x2b13311fd553e74b421d4ccc96e348f71e179dcf`.

The first four correctly create asynchronous redemption tickets, but eth-defi
does not advertise or implement an Anvil settlement driver. If the operator
processing call can be reproduced on a fork, implement
`EmberDepositManager.force_settle(ticket)`, impersonate the documented operator,
advance any required time, verify the matching `RequestProcessed` event and
publish `supports_anvil_settlement=True`. Otherwise leave the capability false
and provide a precise unsupported reason.

Apollo ACRED rejects the small test redemption because 904 raw shares are below
`minWithdrawableShares() == 9,170,000`. Replace the generic `ValueError` with a
typed `VaultFlowUnavailable` carrying protocol, vault, direction, requested raw
shares and minimum raw shares. This lets consumers classify it as a current
redemption constraint rather than an adapter execution failure.

### Gains Network

Affected vaults:

- Base gTrade USDC,
  `8453-0xad20523a7dc37babc1cc74897e4977232b3d02e5`;
  and
- Arbitrum gTrade USDC,
  `42161-0xd3443ee1e91af28e5fb858fbd0d72a63ba8046e0`.

Deposits succeed and epoch redemptions become pending. Implement a
ticket-specific Anvil settlement driver that advances the epoch and invokes the
keeper/manager processing path required by the deployed version. It must verify
the exact redemption ticket becomes claimable before advertising
`supports_anvil_settlement=True`. If protocol state cannot be safely advanced,
return a typed unsupported-simulation result with the epoch and timing
requirement.

### cSigma Finance

Affected vaults:

- cSigma USD, `1-0xd5d097f278a735d0a3c609deee71234cac14b47e`;
  and
- cSuperior Quality Private Credit USDC,
  `1-0x438982ea288763370946625fd76c2508ee1fb229`.

The capacity-aware `CsigmaDepositManager` is selected only for the one hardcoded
V2 address. cSigma USD therefore falls back to the generic ERC-4626 manager and
asserts because `maxRedeem(owner)` is only 45,379 raw shares while the request
is 907,575. Detect supported cSigma deployments by contract version or
capability rather than one address, and return `VaultFlowUnavailable` with
requested and available capacity before constructing the request.

cSuperior deposits successfully, but `redeem()` reverts with
`Withdrawal pending`. This is evidence that the deployment is not an immediate
synchronous ERC-4626 redemption. Model its FIFO/queued withdrawal as an
asynchronous redemption request and ticket, including request-event parsing,
status probing, claim construction and an Anvil settlement driver if the queue
can be advanced. Do not treat `Withdrawal pending` as a generic synchronous
revert.

### YieldNest

YieldNest RWA MAX,
`1-0x01ba69727e2860b37bc1a2bd56999c1afb4c15d8`, accepts the deposit but its
`redeem()` reverts with custom error `0xb8b8b59c` and encoded owner/requested
amount data. Add the deployed ABI error, decode the selector and determine
whether it represents capacity, cooldown, queueing or access policy. Expose the
corresponding preflight or asynchronous manager flow and raise a typed
`VaultFlowUnavailable` when the condition is current-state admission rather
than an implementation error.

### Accountable

Hyperithm Delta Neutral Vault,
`143-0x7cd231120a60f500887444a9baf5e1bd753a5e59`, reverts its Monad `deposit()`
with custom error `0x5945ea56`. Add the deployed contract error ABI or selector
mapping, identify whether the executor module needs a role/allow-list entry or
whether deposits are paused/capped, and surface that condition through
`is_account_whitelisted()`, deposit-capability preflight or a typed
`VaultFlowUnavailable`. Preserve the selector and decoded arguments in the
failure.

### Upshift

Sentora USD Earn,
`1-0x74ad2f789ed583dbd141bbdafc673fe1f033718b`, is correctly reported as
adapter-unsupported because it accepts multiple assets and the generic manager
cannot choose or convert the input asset. Implement a dedicated manager that:

1. discovers accepted assets and their per-asset limits;
2. selects the requested denomination asset or performs the required conversion;
3. builds approval and deposit/request calls for that asset;
4. parses shares and denomination amounts from the deployed events; and
5. advertises support only for directions with a complete request/claim
   lifecycle.

### Plutus

Plutus Hedge Token,
`42161-0x58bfc95a864e18e8f3041d2fcd3418f48393fe6a`, reports
`redemption_unavailable` after a successful deposit. Confirm whether this is a
live protocol state (cooldown, window or capacity) or missing adapter support.
If it is live state, populate a typed reason and next eligible time where
available. If a public request/claim flow exists, implement it and advertise
the correct synchronous/asynchronous capability.

## Not eth-defi defects

The two 40acres balance reverts were caused by trade-executor requesting its
planned share quantity instead of a slightly smaller actual module balance; the
executor reconciliation is fixed. The 51 deposit-closed rows and 14
whitelist-gated rows reflect current on-chain admission and should remain
structured non-success results.

## Acceptance criteria

For each protocol change:

1. add a focused manager-level test using the affected deployment/version;
2. publish capability metadata only for complete deposit and redemption paths;
3. use typed flow or unsupported-simulation exceptions instead of generic
   `ValueError`, `AssertionError` or raw settlement transaction errors;
4. persist enough ticket data to reconstruct a claim after restart; and
5. verify a successful forced settlement changes the selected ticket from
   pending to claimable before returning.
