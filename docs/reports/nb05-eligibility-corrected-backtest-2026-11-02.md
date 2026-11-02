# NB05 backtest after eligibility corrections

Completed on 2 November 2026. All 347 original identities were assessed against
the existing mechanisms and the [universe preservation rules](../backtest-universe-preservation.md).
The demonstrated D2, unknown-permission, directional timing and parking-admission
mistakes are corrected. This is the current financial report; intermediate
reports are retained in Git history rather than duplicated here.

## Results

Evaluation is `[2025-04-01, 2026-10-21)` UTC, with warm-up from 1 June 2024
and $100,000 initial capital. Ranking, weekly membership buffering, concentration,
TVL limits, reserve configuration and the existing synchronous cash cap are retained.

| Metric | Original NB05 | Previous fee-aware run | Corrected run |
| --- | ---: | ---: | ---: |
| Final equity | $117,861.79 | $115,613.55 | **$117,724.22** |
| Return | 17.86% | 15.61% | **17.72%** |
| CAGR | 11.14% | 9.77% | **11.06%** |
| Maximum drawdown | -0.49% | -0.13% | **-0.13%** |

The corrected result gains $2,110.67 over the previous fee-aware run and is
$137.57 below the original. CAGR uses 365 days over the 568-day evaluation.
Daily Sharpe, using sample standard deviation, sqrt(365) and zero risk-free
rate, is 10.28. These comparisons combine eligibility, settlement-timing and
allocation changes; they are not single-factor estimates of fee effects.

## Corrections and eligibility audit

- **D2:** restore HYPE++ and its inherited two-day open window every fourteen
  days, anchored on 30 March 2025. The schedule is a declared simulation
  assumption. Every request in the result occurs inside that window; delayed
  settlement remains in the existing engine.
- **Unknown permission:** historical modes delegate unknown current permission
  to the shared timestamped admission rules. Live modes retain their stricter
  permission requirement. Explicit whitelists and other applicable restrictions
  remain enforced.
- **NEMO:** use the existing pair settlement estimate for its thirty-day exit
  assumption. Do not pass a two-direction engine override. The end-date guard
  now distinguishes queued deposits from delayed redemptions.
- **Parking:** YieldManager checks pair exclusions and shared deposit admission
  before positive vault adjustments. Refused deposits leave cash in reserve;
  reductions and releases of existing holdings remain available. There is no
  strategy-wide assertion that would also block exits.

Exactly **21 primary outcomes changed**: HYPE++ plus the twenty unknown-permission
identities. The other **326 outcomes remain unchanged**. There are **241 eligible
identities** and 106 exclusions. The 345 runtime identities omit two documented
upstream pair-ID collisions; the assessment retains all 347.

The checked-in [eligibility audit](nb05-eligibility-audit-2026-11-02.parquet) stores
the final run's `eligibility-audit.csv` as Zstandard-compressed Parquet, preserving
all 347 rows and their values. Read it with `pandas.read_parquet()`. Its `old_reason`
describes the previous 15.61% run and `new_reason` the corrected run;
`live_route_exclusion` records additional live route exclusions, not a
production owner/Guard certification. It preserves every identity and its
before/after reason without repeating earlier inventories.

Remaining primary exclusions are 79 producer blacklists, ten incomplete fee
records, seven whitelist restrictions, four missing Lighter pool routes, two
incompatible Upshift payouts, two pair-ID collisions, one GMX liquidity route
and one Symbiotic withdrawal lifecycle. Two additional Lighter identities are
counted under whitelist restrictions. These are explicit scope/data/implementation
limitations, not evidence that historical unknown permission should be refused.

The engine now honours positive Upshift settlement estimates generally. The
deployment audit finds **only NEMO** has a positive estimate among the eligible
Upshift identities in this capture. Its deposits remain immediate; other selected
Upshift vaults retain the default when no positive estimate exists.

## Restored allocation evidence

| Identity | Original profit | Corrected profit | Evidence |
| --- | ---: | ---: | --- |
| HYPE++, Arbitrum | $2,203.41 | **$2,262.57** | Nine trades; first deposit requested 13 April 2025; 449 buffered-member days |
| NEMO, Ethereum | $156.97 | **$168.35** | Deposit requested and executed 28 September 2026; 23 buffered-member days |
| autoUSD, Ethereum | $264.37 | $220.10 | Restored and traded |
| fUSDC, Base | $136.56 | $147.00 | Restored and traded |
| baseUSD, Base | $98.76 | $184.32 | Restored and traded |
| fLiteUSD, Ethereum | $93.08 | $89.77 | Restored and traded |
| fUSDC, Ethereum | $16.28 | $14.85 | Restored and traded |

siloUSD is eligible but receives no buffered membership in this run. The same
is true for original contributors Stratwise, CSUSDCCORE, Base ymvOG-USDC and
AngmarUSD. Their saved inclusion/membership counts distinguish non-selection
from a blanket universe exclusion.

Original-to-current profit reconciles at currency precision:
$17,861.79 minus $705.93 from seven originally traded identities now untraded,
plus $412.34 net improvement in the 47 continuing identities, plus $156.02 from
four newly traded identities, equals $17,724.22. These are ledger contributions,
not predicted gains from re-enabling each vault. The untraded group includes the
deliberately excluded Lighter pool ($216.26 originally) and blacklisted gtusdcrwa
($98.63 originally), as well as the eligible non-selected identities above.

## Fees paid

| Vault | Chain | External performance | Fixed withdrawal |
| --- | ---: | ---: | ---: |
| DTUSDC | 8453 | $142.66 | $0.00 |
| 9SUSDC | 1 | $79.42 | $0.00 |
| AVS | 42161 | $0.00 | $51.92 |
| gamiusdc | 1 | $35.12 | $0.00 |
| HubCap-USDC | 1 | $29.22 | $0.00 |
| USDACM | 42161 | $1.77 | $0.00 |

External management paid is zero; external performance paid totals **$288.19**.
Fixed withdrawal costs total **$51.92**, giving **$340.11** total tracked paid
deductions. Another **$9.37** of estimated external fees is reflected in open
net NAV and has not been paid. The per-vault CSV includes all 51 traded identities,
including zero-fee rows. Internalised fees remain in observed share prices.

## Validation and reproducibility

- **90 focused tests pass**, covering canonical NB05 policy/timing, inherited
  windows, async lifecycle, phase-aware backtests, CCTP cash awareness, external
  fees, YieldManager admission/releases and directional classification. The
  command includes `tests/backtest/test_phase_aware_backtest.py` and
  `tests/units_tests/test_yield.py`. Five canonical NB05 and YieldManager tests
  also pass after final cleanup, covering deposit timing and exits from an
  ignored parking venue.
- Both financial attempts are preserved. The second adds the shared parking
  gate and produces identical daily equity to the first across all 568 dates.
  They complete within the **same original 900-second budget**, using four CPUs;
  the final receipt records COMPLETED, exit 0, after 836.70 seconds overall.
- Each attempt uses a fresh process, universe rebuild, empty indicator cache and
  portfolio. Normal loaders consume the same authenticated Nov2 metadata/prices
  as the 15.61% run, hashed before execution. No old universe pickle is input.
- **716 trades succeed**, including 177 CCTP trades. No pending withdrawals,
  deposits or bridge transit remain. State integrity and clean-state checks pass.
  Final cash is $5,884.07 and net position NAV $111,840.15.
- Daily whole-wallet cash/assets/claims reconcile within $0.000001634. The fee
  oracle independently matches realised fee amounts within 1.02 × 10⁻¹² USD.
  Queued trades execute after their request cycle; pending proceeds are not
  credited early. The first directional investment remains 1 April 2025.
- Equity/drawdown, allocation, universe eligibility and fee charts were opened
  and inspected. Mean daily Fluid exposure is 4.77%; mean cash exposure is 5.06%.

The shared parking gate applies to other YieldManager strategies too: vault
deposits now consult the existing admission provider, while withdrawals retain
their normal path. In live NB05, Fluid's current unknown permission keeps idle
capital in reserve until admission evidence is resolved; it does not halt exits.
No production hosting, funded transactions or deployment validation was performed.
Current metadata policies and fee rates are scenario inputs, not a historical
permission or fee archive. Sparse NAV and forward-fill limitations remain.

## Production-readiness boundary

Earlier guarded fork checks passed for Atoma (five cases), Upshift (five cases)
and the generic public CLI matrix including Fluid (one case). They exercised
existing focused strategies and impersonated operators, not funded NB05
production operation.
The public test-trade command now selects lifecycle handling from manager
capabilities; `check-universe` uses the strategy's primary chain.
Reserve parking refuses venues with delayed exits as well as queued deposits.

The actual NB05 notebook was converted to `strategy/master-vault-nb05.py` in
the canonical `tradingstrategy-ai/strategies` repository. That module and its
tests are separate from this executor PR. Delivered classification fixes keep
Gains/Ember exits delayed, Accountable/Plutus deposits synchronous, Plutus Hedge
exits delayed and Plutus Dolomite synchronous in both directions.

On 2 November at 09:45 UTC, live freshness checks refused Hyperithm and Yearn
CompoundV3 histories whose latest real prices were 27 October. That dated
observation must be refreshed before launch. Validate current data, selected
custody owners, funding, proxy versions and Guards. The production migration
journal, outstanding Upshift migrations 018/019, metadata publication and
funded rehearsal were not checked or performed. Follow the
[readiness plan](../plans/nb05-live-readiness.md) for remaining work.

## Evidence

Bundle: `/home/mikko/code/vault-notebooks/data/xchain3/nb05-eligibility-corrected-20261102/attempt-02/`.

The checked-in audit is the portable identity/reason summary. The full local
bundle contains the authenticated inputs, state, source snapshots and charts;
these large artefacts are not included in this PR. Executor baseline is
`ee7b1fd325f28359a149ac5a1a54d371c6849ad2` plus the saved engine patch,
eth-defi is `27c16e26dd9325427bb35af8e1c5e2efcb63db60` and trading-strategy is
`f80f684758413d27b5350a57718d3ca594a91574`.

- `eligibility-audit.csv`, `eligibility-corrections.json`: all 347 outcomes and live/historical distinction.
- `original-versus-corrected.csv`, `restored-vault-trades.json`, `eligible-untraded-originals.json`: profit and selection evidence.
- `external-fees-by-vault.csv`, `external-fee-withdrawals.csv`: exact per-vault and per-trade fee amounts.
- `upshift-timing-audit.json`: every Upshift estimate and exclusion in the rebuilt universe.
- `input-receipt.json`, `strategy.py`, `engine-source/`, `engine.patch`: source, revisions, parameters and consumed input hashes.
- `backtest-state.json`, `equity.csv`, `result.pickle`, `verification.json`: financial state and independent checks.
- `daily-qualification.csv`, `daily-window-evidence.csv`, `daily-asset-values.csv`, `charts/`, `review-receipt.json`: membership, historical rules, accounting and visual review.
- `retry-comparison.json` and parent `run-budget.json`: repeat comparison and original deadline receipt.

The [earlier assessment and intermediate reports](https://github.com/tradingstrategy-ai/trade-executor/tree/6110b48061b95e688c47719ce89a5beacb2fab53/docs/reports)
remain available at their original revision, including all 347 historical
capability observations and the nine-vault/188-route inventories from
[PR #1700](https://github.com/tradingstrategy-ai/trade-executor/pull/1700). Their dated
findings are not current admission evidence. Prior restricted and frozen-input
controls returned 7.23% and 18.04%; neither is a matched fee-only comparison.
