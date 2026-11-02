# NB05 rerun with externalised vault fees

> Superseded by the [eligibility-corrected rerun](nb05-eligibility-corrected-backtest-2026-11-02.md):
> 241 eligible identities, 17.72% return and 11.06% CAGR. The 15.61% result below
> is retained with its original universe and source snapshot.

Completed on 2 November 2026 using a fresh process, empty indicator cache and
simulated portfolio. Evaluation: **1 April 2025–21 October 2026, exclusive end**,
568 daily observations; warm-up from 1 June 2024; starting capital **$100,000**.
The original 347 identities and the earlier authenticated metadata/price
snapshots are retained. Ranking, membership, concentration and TVL parameters
were not tuned. This is a historical simulation with current snapshot fee rates,
not evidence that those rates applied throughout the historical period.

## Results

| Metric | Previous supported run | New fee-aware run |
|---|---:|---:|
| Final equity | $115,117.55 | $115,613.55 |
| Total return | 15.12% | 15.61% |
| CAGR | 9.47% | 9.77% |
| Maximum drawdown | -0.26% | -0.13% |
| Sharpe | 12.05 | 13.74 |

The new result is **$495.99 higher**. CAGR uses
365 days over the 568-day evaluation; Sharpe uses daily whole-wallet returns,
sample standard deviation and sqrt(365), with a zero risk-free rate.

Eligibility expands from **188 to 220** identities because known externalised
fees are now modelled rather than excluded. This is not a fee-only comparison:
allocation membership changes, and the synchronous cash cap described below
also changes which purchases can execute. Earlier 17.86% original, 18.04%
current-engine full-universe and 7.23% restricted results remain historical
comparators with their separately documented universe and modelling assumptions.

## External fees paid per vault

**External management: $0.00; external performance:
$313.65; fixed withdrawal charges:
$21.35. Total tracked deductions paid:
$335.01.**
These are deductions on successful withdrawals, not estimated liabilities.
Fees internalised in share prices are not extracted into this table and are not
charged a second time. No external management fee was realised in the positions
this strategy reduced; the separate constructed-strategy validation covers
non-zero management accrual. AngmarUSD has only negligible Decimal rounding
residue (2e-25 USD), reported as zero at currency precision.

| Vault | Chain | Management | Performance | Fixed withdrawal | Total paid |
|---|---|---:|---:|---:|---:|
| DTUSDC | 8453 | $0.00 | $152.34 | $0.00 | $152.34 |
| 9SUSDC | 1 | $0.00 | $84.21 | $0.00 | $84.21 |
| gamiusdc | 1 | $0.00 | $44.27 | $0.00 | $44.27 |
| HubCap-USDC | 1 | $0.00 | $31.45 | $0.00 | $31.45 |
| AVS | 42161 | $0.00 | $0.00 | $21.35 | $21.35 |
| USDACM | 42161 | $0.00 | $1.38 | $0.00 | $1.38 |

The full CSV includes **all 53 traded identities**, including zero-fee
rows, exact chain/address identities, gross/net withdrawals and separate unpaid
estimates. A second CSV lists each paid withdrawal by trade ID and execution date.
At period end, another **$18.56** in estimated
management/performance fees is reflected in open-position net NAV and has **not
been paid**: USDACM $9.90 and Stratwise Hypercore $8.67. Stratwise has no completed
withdrawal in this run; its generic fee treatment appears in net valuation.
Lighter remains excluded from NB05 because its live public-pool execution path
is unfinished; the separate real-data constructed strategy tests its fee path.

## Allocation differences

The largest positive changes in net position PnL versus the previous supported
run are below. Net PnL includes completed withdrawals and end-period net NAV;
it is not gross interest and does not isolate fee effects.

| Vault | Chain | Old net PnL | New net PnL | Change |
|---|---|---:|---:|---:|
| DTUSDC | 8453 | $0.00 | $609.37 | $+609.37 |
| 9SUSDC | 1 | $0.00 | $336.85 | $+336.85 |
| HubCap-USDC | 1 | $0.00 | $283.08 | $+283.08 |
| USUALUSDC+ | 1 | $80.69 | $268.94 | $+188.26 |
| gamiusdc | 1 | $0.00 | $177.07 | $+177.07 |
| USDACM | 42161 | $0.00 | $101.49 | $+101.49 |
| Stratwise  | 9999 | $0.00 | $77.99 | $+77.99 |
| 40avax-USDC-VAULT | 43114 | $1,301.53 | $1,365.46 | $+63.93 |

The largest reductions offset part of these gains:

| Vault | Chain | Change in net PnL |
|---|---|---:|
| plvHedge | 42161 | $-180.22 |
| hyperUSDCd | 1 | $-170.22 |
| ymvOG-USDC | 1 | $-83.85 |
| CSUSDCCORE | 1 | $-80.90 |
| TAUBOND | 1 | $-78.11 |

Mean daily Fluid exposure is **4.29%**, versus 3.53%
previously; mean cash is **5.08%**, versus 4.93%.
The first directional investment remains 1 April 2025. Daily qualification,
allocation and historical-window evidence are retained as CSVs.

## Funding defect found and corrected

Two preserved failed attempts stopped on **10 March 2026**. NB05 returned only
an Ethereum svUSDC purchase for **$4,237.17**, while the primary reserve held
**$3,000.00** and the batch supplied no sell trades or usable bridge-back cash.
The existing synchronous cash cap was disabled in NB05. The corrected strategy
now passes `cap_buys_to_sync_cash=True` and uses the existing $3,000 cash buffer
as its headroom. This reuses the existing AlphaModel/PhaseAware funding guard;
it changes execution feasibility, not ranking or financial parameters.

The retry uses a new process, universe rebuild, indicator directory and portfolio,
with the same input hashes and the **original 900-second/four-CPU budget**.
All three attempts, failed-state diagnostics and logs are retained. The final
supervisor receipt is COMPLETED, return code 0. Eight focused cash-cap and NB05
strategy checks passed. No CCTP planner or live native-funding semantics changed.

## Verification and remaining limitations

- 719 successful trades and 3 legitimate
  pending withdrawals, 184 successful CCTP trades, no remaining
  bridge transit and no pending deposits. State integrity and clean-state checks
  passed, with pending withdrawals explicitly allowed.
- Daily cash/position/claim accounting reconciles to whole-wallet equity within
  $0.000001713. An independent replay of each paid
  external fee's proportional basis, principal-time exposure and rate matched
  saved amounts within 4 × 10⁻¹³ USD.
- Ending cash is $5,780.68; net position NAV is
  $109,832.87. Unsettled proceeds were not credited early.
  Pending exits are synUSD and aHYPERLOOP (requested 20 October, due 22 October),
  and Atoma AVS (requested 19 October, due 26 October). No settlement was forced
  past the exclusive end date.
- Equity/drawdown, allocation, eligibility and per-vault fee charts were opened
  and visually inspected. The charts use the saved financial state/equity.
  Tiny rounding-only fees remain in CSV evidence but are immaterial economically.
- Known earlier limitations remain: YieldManager uses the Fluid parking venue
  despite its unknown-permission exclusion; NEMO's synchronous entry can still
  be suppressed by its 30-day exit override/end-date guard. This rerun does not
  claim those admission/timing issues are resolved or establish live readiness.
- Fee rates and eligibility come from the captured current metadata; historical
  observations are sparse and forward-filled. Management/performance collection
  is modelled on withdrawal, with estimated fees in NAV. Earlier operator fee
  collection and weekly/monthly management debits are not simulated.

## Evidence

The saved financial run predates the subsequent
[PR cleanup](backtest-externalised-vault-fees.md#pr-cleanup-review), which corrected
legacy pricing factories and command lifecycle checks. Its original source
snapshot and financial artefacts remain unchanged; cleanup did not rerun the
full NB05 simulation.

Local bundle: `/home/mikko/code/vault-notebooks/data/xchain3/nb05-externalised-fees-20261102/attempt-03`.

- [Per-vault fee CSV](/home/mikko/code/vault-notebooks/data/xchain3/nb05-externalised-fees-20261102/attempt-03/external-fees-by-vault.csv)
- [Per-withdrawal fee CSV](/home/mikko/code/vault-notebooks/data/xchain3/nb05-externalised-fees-20261102/attempt-03/external-fee-withdrawals.csv)
- [Position PnL comparison](/home/mikko/code/vault-notebooks/data/xchain3/nb05-externalised-fees-20261102/attempt-03/profit-comparison.csv)
- [Equity comparison](/home/mikko/code/vault-notebooks/data/xchain3/nb05-externalised-fees-20261102/attempt-03/charts/comparison-equity-drawdown.png)
- [Fees per vault](/home/mikko/code/vault-notebooks/data/xchain3/nb05-externalised-fees-20261102/attempt-03/charts/external-fees-by-vault.png)
- [Asset allocation](/home/mikko/code/vault-notebooks/data/xchain3/nb05-externalised-fees-20261102/attempt-03/charts/asset-allocation.png)
- [Universe eligibility](/home/mikko/code/vault-notebooks/data/xchain3/nb05-externalised-fees-20261102/attempt-03/charts/universe-eligibility.png)
- [Supervisor receipt](/home/mikko/code/vault-notebooks/data/xchain3/nb05-externalised-fees-20261102/run-budget.json)

The bundle includes the exact strategy, changed engine sources (including the
new fee module), patches, dependency/package revisions, parameters, input hashes,
state, result object, independent metrics and analysis script. Vault metadata
and prices match the previous supported-run hashes; this run does not consume
old universe pickles or indicator caches.

vault_metadata SHA256: `5f3d87fd6f3c3aaff146c332a7f1e6fb03929e0f812afc757575726bdeb63c40`.

vault_prices SHA256: `22957a698202551cd205c2d36952e38bb2a2c6fa031a162c9942df1a262a1596`.
