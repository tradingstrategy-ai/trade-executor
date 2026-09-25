# Settle Lagoon NAV when the queue is empty

Status: implemented on `fix/lagoon-empty-queue-nav`; deployment pending.

## Issue

Lighter AI calculates NAV correctly but leaves the vault's ERC-4626 `totalAssets()` stale. At 06:08 UTC on 2026-09-25, the executor valued Safe USDC at **19** and Lighter account 748071 at **14,657.858542 USDC**, then successfully [posted](https://eth.blockscout.com/tx/0x981543b6dc538c9d5823add343c370071c55501166b37665bda82ae46b239325) a NAV proposal of **14,676.858541 USDC**. The vault's settled `totalAssets()` was still **15,010.885772 USDC**, a **334.03 USDC** overstatement against that valuation.

The copied production log at `/tmp/lighter-ai-nav-investigation-20260925/lighter-ai.log` shows the cause at line 68969: **“Lagoon NAV posted without settlement because the queue is empty.”** The same outcome followed 103 NAV posts after the September 19 settlement. [Lagoon requires two steps](https://docs.lagoon.finance/vault/how-to/update-the-vault-valuation-and-settle-requests): posting proposes a valuation; `settleDeposit()` accepts it and updates `totalAssets()`. The unconditional empty-queue return in `tradeexecutor/ethereum/lagoon/vault.py::_preflight_lagoon_settlement()` skips that second step. A read-only simulation of the Lighter vault's empty-queue settlement through its module succeeded.

## Change

1. Remove the generic empty-queue skip. After a confirmed NAV post, simulate the wrapped `settleDeposit()` call and broadcast it when the preflight succeeds, including when both queues are empty. Simulate the empty call for older unlimited modules too, rather than broadcasting blindly. Keep existing Guard limits, cooldown deferrals and fail-closed handling; if a supported older vault cannot settle an empty queue, log clearly that `totalAssets()` remains stale.
2. Reuse the existing receipt analysis and treasury update path. Determine actual investor flow from the mined receipt, since a request can arrive after preflight. A genuinely empty settlement updates `totalAssets()`, has zero cash movement, and may mint fee shares; refresh share count as the current path already does. Keep the settlement scanner focused on deposit and redemption events: a zero-flow settlement emits only `TotalAssetsUpdated` and needs no cash-flow recovery.
3. Update the Lagoon treasury guide and the affected code comments to explain that an empty queue now settles NAV. Keep the separate one-micro-USDC float rounding issue out of this fix.
4. Use the existing `min_nav_change_update` setting (default 0.5%) against settled onchain `totalAssets()`: skip an empty-queue NAV cycle below the threshold, while always posting for queued investor flows.

## Verification

- Update the existing Lagoon deposit and Lighter NAV tests: an empty-queue cycle should post and settle, `totalAssets()` should equal the accepted NAV, and Safe cash and investor netflow should stay unchanged.
- Check that an empty queue below the 0.5% NAV threshold sends no transactions and a change above it settles.
- Extend the Guard test to show a zero-flow settlement leaves its used budget and window unchanged. Check share count on a fee-bearing vault, and preserve tests for queued flows and settlement deferrals.
- After deployment, check the first NAV and settlement receipts, `totalAssets()`, Safe and Lighter balances, share count, and Guard budget. Qualifying empty-queue NAV updates need one extra settlement transaction and sufficient executor ETH.

## Review

Kimi K3 reviewed the draft with high thinking effort on 2026-09-25. Its key findings are reflected above: zero-flow settlements are invisible to the investor-flow scanner, and fee shares can change on an empty settlement.
