# NB05 exclusion audit

> The findings below are addressed in the
> [eligibility-corrected rerun](nb05-eligibility-corrected-backtest-2026-11-02.md).
> This audit retains the evidence and status at the time the mistakes were found.

Audited on 2 November 2026 after the D2 exclusion was challenged. This reviews
the saved fee-aware run's 347 identities and the canonical NB05 selection code.
It does not change the saved run, restore entries or recalculate its return.

## Findings

1. **D2 was incorrectly excluded as unsupported.** The existing backtester has
   deposit/redemption window modelling (`c9e32a9d3`), phase-aware parking and
   settlement handling. HYPE++'s public economic eligibility uses its USDC
   denomination, established by `f072f5a06`. The original NB05 explicitly models
   a two-day open window every fourteen days. This is a declared simulation
   assumption, not missing execution support. Removing its schedule and then
   rejecting D2 as unmodelled is not evidence that the route cannot be backtested.

2. **The blanket unknown-permission exclusion is too broad for historical
   simulation.** Twenty otherwise eligible identities are rejected solely
   because the current metadata permission is unknown: eight AUTO Finance,
   five Fluid, six Morpho V2 and one Hypercore. All twenty have complete fee
   inputs under the existing completeness rule. The new selection rejects them
   before the shared historical admission logic can apply. In particular,
   Morpho V2 has documented unknown/allowed historical admission handling, and
   Hypercore has an explicit pre-collection assumption. Unknown current metadata
   is not evidence that every historical allocation was prohibited. These
   identities need the existing historical admission rules and their stated
   assumptions; this finding does not establish that all twenty may trade live.

3. **The deposit end-date guard uses either-direction delay.** NEMO is eligible
   but its synchronous deposit is suppressed by its thirty-day redemption
   assumption. The guard calls `is_delayed_vault()`, which includes delayed
   redemptions. The defect can affect other synchronous-deposit/delayed-exit
   families too; the NEMO allocation is the demonstrated case. The backtest's
   per-address delay override also forces both directions through a queue.

4. **Fluid admission is inconsistent.** The Arbitrum parking identity appears
   among the twenty unknown-permission exclusions but still receives trades
   through YieldManager. Directional admission and parking need the same
   applicable historical policy rather than an implicit parking exemption.

## All primary outcomes in the saved run

| Outcome | Identities | Audit conclusion |
| --- | ---: | --- |
| Eligible | 220 | Eligibility does not guarantee allocation; NEMO demonstrates a later incorrect gate. |
| Producer blacklist | 79 | Existing explicit policy; this audit does not establish that the labels are incorrect. |
| Unknown permission | 20 | Blanket historical exclusion is unjustified; use the shared admission logic. |
| Missing fee inputs | 10 | Nine Upshift and one Fluid/Kiln; fee inputs are genuinely incomplete in the capture. |
| Whitelist | 7 | Explicit restriction in the captured metadata; no contrary exact-deployment evidence established here. |
| Pair ID collision | 2 | Supported deployments disabled by a real upstream identity collision, not absent adapters. |
| Lighter execution missing | 4 | Separate unfinished live pool route; another two Lighter identities are counted under whitelist. |
| Different payout | 2 | Upshift payout-token conversion remains outside the supported USDC route. |
| D2 | 1 | Incorrect blanket exclusion; existing simulation support should be reused. |
| GMX liquidity | 1 | Unfinished lifecycle under the approved supported-route scope. |
| Symbiotic | 1 | Deployment-specific withdrawal lifecycle remains unestablished. |
| **Total** | **347** | |

The six originally traded identities excluded for unknown permission are
autoUSD, siloUSD, Base fUSDC, baseUSD, fLiteUSD and Ethereum fUSDC. Their original
profit totals $837.64. HYPE++ contributed $2,203.41 and NEMO $156.97. These are
historical contributions, not additive forecasts of restored return.

The supported-route run's 15.61% result remains a result for its recorded
restricted universe. It must not be described as having exhausted the original
strategy's supported backtest coverage.

## Evidence

- [Historical admission contract](../vault-closure.md), including Morpho V2 and Hypercore assumptions.
- [Window and phase-aware modelling](../../.claude/docs/phase-aware-alpha-model.md#backtest-window-modelling).
- [July corrective evidence](cross-chain-vault-test-2026-07-31-results.md), including the HYPE++ amendment.
- `strategy/master-vault-nb05.py` in the canonical strategies repository: `get_nb05_exclusion_reason()`, `get_window_gated_pairs()` and `apply_portfolio_stability_guards()`.
- Saved assessment and captured metadata in `vault-notebooks/data/xchain3/nb05-externalised-fees-20261102/attempt-03/`.
