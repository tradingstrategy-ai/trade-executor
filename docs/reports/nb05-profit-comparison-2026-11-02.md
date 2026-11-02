# NB05 profit comparison

> Historical comparison before automatic externalised fee deduction. See the
> [fee-aware rerun](nb05-externalised-fees-backtest-2026-11-02.md) for the subsequent
> results, restored vault allocations and fees paid.

Compare the original full NB05 saved replay with the first broad supported-route
simulation over the same 568-day calendar and $100,000 starting capital.
Amounts below include realised and unrealised position profit, aggregated by
chain/address across successive positions. They describe the two ledgers;
they are not estimates of the marginal return from re-enabling a vault.

| Profit reconciliation | USD |
| --- | ---: |
| Original full NB05 profit | 17,861.79 |
| Original profit from 17 identities no longer traded | -5,256.92 |
| Net increase from the 37 identities traded in both runs | +1,681.67 |
| Profit from 17 newly traded identities | +831.01 |
| **Supported-route profit** | **15,117.55** |
| **Net shortfall** | **2,744.24** |

The original return was 17.86%, compared with 15.12% in this run. The full-universe
current-engine diagnostic returned 18.04%; relative to that control, the
shortfall is $2,918.74. Full per-identity columns for both baselines are saved
with the supported-route run.

## Vaults no longer traded

| Vault | Chain/venue | Original profit | Current reason |
| --- | --- | ---: | --- |
| HYPE++ | Arbitrum | $2,203.41 | D2 eligibility asset and real epoch behaviour unverified |
| DTUSDC | Base | $702.23 | External fees not charged to simulation equity |
| 9SUSDC | Ethereum | $383.14 | External fees not charged to simulation equity |
| HubCap-USDC | Ethereum | $276.87 | External fees not charged to simulation equity |
| autoUSD | Ethereum | $264.37 | Deposit permission metadata unknown |
| siloUSD | Ethereum | $228.59 | Deposit permission metadata unknown |
| Steady Wealth Builder | Lighter | $216.26 | Pool execution integration missing |
| gamiusdc | Ethereum | $183.50 | External fees not charged to simulation equity |
| NEMO nUSDC | Ethereum | $156.97 | Eligible, but end-boundary timing guard suppresses the buy |
| fUSDC | Base | $136.56 | Deposit permission metadata unknown |
| USDACM | Arbitrum | $111.00 | External fees not charged to simulation equity |
| baseUSD | Base | $98.76 | Deposit permission metadata unknown |
| gtusdcrwa | Ethereum | $98.63 | Producer blacklist |
| fLiteUSD | Ethereum | $93.08 | Deposit permission metadata unknown |
| Stratwise Multi-Asset Public | Hypercore | $87.28 | External fees not charged to simulation equity |
| fUSDC | Ethereum | $16.28 | Deposit permission metadata unknown |
| AngmarUSD | Arbitrum | $0.00 | External fees not charged to simulation equity |

Unknown permission is a metadata/validation gap, not proof of private access or
permanent closure. External fee exclusions are simulation-model restrictions;
they do not establish missing live adapter support. HYPE++ is the largest
missing contributor, but its former synthetic schedule is not live evidence.

## Continuing contributors

The major surviving contributors generally earned more, offsetting part of the
removed profit:

| Vault | Chain | Original profit | Supported-route profit | Change |
| --- | --- | ---: | ---: | ---: |
| Plutus Hedge | Arbitrum | $1,604.20 | $1,992.88 | +$388.68 |
| 40acres VAULT | Base | $1,664.97 | $1,688.88 | +$23.91 |
| Gains gUSDC | Base | $812.52 | $902.73 | +$90.21 |
| Gains gUSDC | Arbitrum | $783.15 | $877.80 | +$94.65 |
| Syntropia synUSDx | Ethereum | $910.88 | $918.13 | +$7.25 |
| Syntropia synUSD | Ethereum | $539.65 | $570.15 | +$30.50 |
| wildUSD | Ethereum | $478.79 | $609.75 | +$130.96 |

Among continuing identities, seven earned less: Arbitrum Fluid -$160.35,
Ethereum Yearn `ysUSDC` (`0xf6e2…ad5ddc`) -$99.81, Atoma Index -$64.93,
Accountable aHYPERLOOP -$59.63, USUALUSDC+ -$49.27, Ember eY10K -$22.15
and Avalanche 40acres -$10.86. These losses total $467.00; increases in the
other shared identities total $2,148.67.

Fluid's average equity allocation fell from 5.52% to 3.53%, so its lower
contribution reflects less reserve parking. Plutus Hedge's average allocation
rose from 5.99% to 8.20%. These allocation changes must not be interpreted as
an isolated measurement of changes in each vault's underlying return.

## Correctness issues found by this comparison

**Existing fee simulation was missed in the initial investigation.** Hyper-AI
v6–v13 already contains strategy-level redemption accounting. In v13,
`get_remaining_cost_basis()` handles partial reductions pro rata,
`refresh_vault_redemption_accounting()` values holdings after profit-based
performance fees and a separate capital-cost allowance, and each sell snapshots
`backtest_vault_redemption_fee`. The shared backtest resolver already consumes
that snapshot at async settlement. This is existing machinery to reuse, not
an absent platform capability. NB05 does not call these helpers: zero of its
304 original reductions and 241 clean-run reductions carry that fee snapshot.
NB05 does charge Atoma's separate 0.5% exit tax. The broad claim that the
backtester does not simulate vault reduction fees was incorrect; the issue is
NB05 wiring and per-vault fee treatment, with annual management fees requiring
their own verification. Exclusion should not substitute for that integration.

**NEMO timing.** NEMO first enters current buffered membership at cycle 546,
28 September 2026, matching its original first-entry date. The final AlphaModel
signal records `deposit_suppressed_after_backtest_end=9215.69768560501` and zero
adjustment. The new 30-day override makes the entry fail the end-boundary guard.
That guard uses delayed-vault classification, including delayed redemptions,
while the existing per-vault override forces the simulated asynchronous path
for both directions. NEMO deposits are synchronous. Its exit-delay assumption
must not create an artificial deposit queue or suppress an otherwise synchronous
entry on that basis. This requires a directional simulation correction.

**Fluid parking admission.** The actual Arbitrum parking venue
`0x1a996cb54bb95462040408c06122d45d6cdb6096` is marked
`deposit_permission_unknown` in the captured universe, yet the YieldManager
path traded it 188 times. Its profit was $258.38. The directional selection
ignores that pair, but the explicit cash venue still receives parking trades.
Consequently the reported 188 eligible identities do not include this venue,
although the portfolio uses it. Parking admission must be made consistent with
the chosen policy, resolving evidence through existing mechanisms rather than
silently exempting the venue.

The financial ledger, metrics, cold-cache repeat and daily accounting checks
remain reproducible. These additional findings mean the run has **not** fully
passed the intended admission and directional-timing acceptance criteria.
The 53 existing focused tests did not cover these two combinations. No strategy
or executor behaviour was changed during this profit comparison.

## Evidence

The original state is in canonical vault-notebooks at
`data/xchain3/master-vault-multichain-v2/attempt-02/`; the full current-engine
control is under `data/xchain3/nb05-readiness-attribution-20261102/`.
Supported-route evidence is in
`data/xchain3/nb05-production-supported-clean-20261102/attempt-02/`.
`profit-comparison.json` and `profit-comparison.csv` retain complete addresses,
both comparison baselines, profit, trade counts, deposited capital, first-entry
dates and average equity weights. Vault-profit totals agree with wallet gains
within $0.00004 in each state. No financial backtest was rerun for this analysis.
