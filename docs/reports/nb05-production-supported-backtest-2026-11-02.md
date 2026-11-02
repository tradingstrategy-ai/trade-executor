# NB05 clean backtest with production-supported routes

> Historical result before automatic externalised fee deduction. The subsequent
> [fee-aware rerun](nb05-externalised-fees-backtest-2026-11-02.md) admits 32 more
> identities and supersedes this financial result. The observations below describe
> this earlier run and its implementation.
>
> A subsequent [per-vault profit audit](nb05-profit-comparison-2026-11-02.md)
> found Fluid parking despite its metadata exclusion and a directional timing
> problem suppressing NEMO entries. The financial/accounting result below is
> reproducible, but admission and timing acceptance remain incomplete.
>
> The [eligibility-corrected run](nb05-eligibility-corrected-backtest-2026-11-02.md)
> supersedes both runs. It corrects the unsupported D2 and unknown-permission
> exclusions below and restores NEMO's synchronous entry.

Completed on 2 November 2026. This supersedes the nine-vault financial result
in the [initial readiness report](nb05-live-readiness-2026-11-02.md).
“Supported” means an existing execution route with compatible funding and
modelled fee/timing treatment; production owner, Guard and funded deployment
validation remain separate requirements.

## Result

| Measure | Clean supported-route run |
| --- | ---: |
| Evaluation | 1 April 2025–21 October 2026, exclusive end |
| Initial equity | $100,000 |
| Final equity | **$115,117.55** |
| Return | **15.12%** |
| Maximum drawdown | **-0.26%** |
| Daily equity observations | 568 |
| Trades | 643: 642 successful, one pending redemption |
| Successful CCTP trades | 171 |
| Vault identities traded | 54, including the Fluid cash venue |
| First directional investment | **1 April 2025** |
| Mean daily Fluid allocation | **3.53%** |
| Mean daily reserve cash | **4.93%** |

The corrected universe invests from the beginning of the evaluation. The old
nine-vault subset first invested outside Fluid on 10 December 2025 and parked
70.94% of equity in Fluid on average.

| Historical comparison | Return | Meaning |
| --- | ---: | --- |
| Original saved NB05 | 17.86% | Original frozen inputs and earlier engine |
| Full original universe on the current engine before this follow-up | 18.04% | Historical diagnostic control retaining research assumptions |
| Superseded nine-vault subset | 7.23% | Narrow lifecycle validation sample |
| Clean supported-route universe | **15.12%** | Current captured inputs and explicit execution/fee/admission exclusions |

The clean run improves on the narrow subset by 7.89 percentage points. Its
remaining difference from the historical controls combines changed eligibility,
current inputs and the documented simulation assumptions. These are not matched
single-factor estimates of individual protocol or timing effects. Matching the
original count of 54 traded identities does not imply matching their membership.

## Universe and implementation

The strategy retains all **347 recorded source identities** and its original
ranking, weekly buffering, concentration, TVL, reserve and funding parameters.
It uses current authenticated loaders without rerunning the inherited
lifetime-performance screen. There are **188 eligible routes** and **159 explicit
exclusions**. The follow-up audit found that the separately configured Fluid
parking venue is among the exclusions but was still used by YieldManager.
Exact guarded lifecycle evidence per address
was not used as a blanket admission prerequisite.

| Primary outcome | Identities |
| --- | ---: |
| Eligible existing route | 188 |
| Producer blacklist | 79 |
| External fees not charged to simulated equity | 32 |
| Unknown deposit permission | 20 |
| Incomplete fee metadata | 10 |
| Requires whitelist | 7 |
| Lighter pool execution missing | 4 |
| Different-payout Upshift route requires conversion | 2 |
| Upstream cross-chain pair ID collision | 2 |
| D2 eligibility asset and epoch model unverified | 1 |
| GMX liquidity execution missing | 1 |
| Symbiotic withdrawal lifecycle unverified | 1 |
| **Total** | **347** |

Primary outcomes preserve existing blacklist/whitelist reasons. Additional
route exclusions are retained in the assessment, so these counts do not count
every overlapping restriction. The [inventory](nb05-live-adapter-coverage.json)
retains the original assessment fields unchanged. This run's review is now
`superseded_188_route_review`; the current `readiness_review` follows the
eligibility-corrected run. The nine-vault review also remains historical
documentation. No execution code consumes this report.

Eligible families include Morpho, Yearn, IPOR, Upshift, Lagoon, Euler, Ember,
Accountable, 40acres, Gains, Nova, Atoma and Plutus. Hypercore was assessed using
its existing native route: the captured snapshot excludes eleven sources for
uncharged external performance fees, one for whitelist requirements and one for
unknown admission. It was not excluded merely because it is native.

The existing `returns_net` indicator adjusts ranking for known external fees,
but does not charge those fees to portfolio equity. Such routes are therefore
excluded. Atoma retains its modelled 0.5% exit fee. NEMO is eligible with an
explicit 30-day settlement override; other supported queues retain existing
engine estimates and documented fallbacks. No synthetic windows were supplied.
D2's adapter requires real phase/eligibility evidence, which the original
synthetic schedule and queued-deposit approximation do not establish.

Current metadata classifies Nova separately from Gains. Its pinned adapter
inherits the Gains epoch manager, so the shared delayed-proceeds classification
now includes `nova_markets_like`. Regression coverage exercises immediate
deposits, delayed exits, mixed rebalances and satellite funding.

The pinned client derives vault pair IDs from address alone. The source address
`0x46ce9aa8167fa3e154395c25d19db25ddfcfeab4` occurs on Ethereum and Arbitrum;
one identity was silently lost when loaded together. Both now have the specific
collision exclusion and are omitted before pair construction. A held colliding
identity causes an explicit refusal. No universal ID/state migration was added.

In the original saved ledger, identities now excluded earned $5,518.69:
$2,203.41 from D2, $1,744.01 from routes with uncharged external fees,
$1,256.37 from unknown-permission routes, $216.26 from Lighter and $98.63
from a currently blacklisted route. These are original contributions, not
counterfactual marginal effects: the clean strategy reallocates to other vaults.

## Clean execution and provenance

- Executor baseline `ee7b1fd325f28359a149ac5a1a54d371c6849ad2` plus the saved
  local patch, including the directional fixes and Nova classification.
- eth-defi `27c16e26dd9325427bb35af8e1c5e2efcb63db60`;
  trading-strategy `f80f684758413d27b5350a57718d3ca594a91574`.
- Original source revision `243217e7fa935baee18894048f84daa3d000626d`;
  warm-up begins 1 June 2024. The API harness preserves the 1 April first cycle.
- Authenticated metadata and prices were acquired once at about 12:17 UTC,
  copied and hashed before execution. Metadata generation is
  `2026-11-02T11:19:29Z`. The final repeat used those same captured bytes.
- Full SHA-256 values, packages, parameters and supporting-market inputs are in
  the pre-execution receipt. Vault metadata hash begins `5f3d87fd6f3c` and price
  hash begins `22957a698202`. The executed strategy hash is
  `9e2bdad35d029535394cd08f890ad40a6a6de9d83b87198bc763d322bc718e8f`.
- The existing supervisor enforced one **900-second budget and four CPUs**.
  It completed with exit code 0. The sequence was the initial financial run,
  a saved-result reporting review, and the final fresh financial repeat after
  excluding the aliased identities. All attempts remain under the same budget.
- Both financial runs began with a new process, new portfolio and empty
  indicator cache. The final repeat rebuilt through normal loaders and used
  no prior universe pickle, state or successful-result shortcut.

Final evidence is in canonical vault-notebooks at
`data/xchain3/nb05-production-supported-clean-20261102/attempt-02/`.
Its parent preserves acquisition, the initial result, original charts and review.
The final directory contains source and engine patch, input receipt/snapshots,
universe assessment, state, equity, result, allocations, daily qualification,
historical window evidence, pending claims, summary, verification and charts.
The module subsequently received a dated result docstring; the executed source
copy and hash remain preserved.

## Verification and charts

State integrity and cleanliness checks passed, allowing the one explicitly
reported queued redemption. Return and drawdown were independently recalculated
from the saved equity CSV. Cash, asset values and queued deposits reconciled
to whole-wallet equity on every date within **$0.00000145**. Input hashes remained
unchanged. All 171 bridge trades succeeded; no bridge capital remains in transit.

The final repeat has **identical trade economics and byte-identical daily equity**
to the initial clean financial run. The removed colliding identities were not
traded. All final chart bytes match the visually inspected corrected exports.
The reporting review deduplicated repeated Fluid position labels and separated
small closure counts from larger unknown-evidence counts; state/equity were
unchanged by that review.

Final reserve cash is $5,755.88 and position NAV is $109,361.67, totalling
$115,117.55. The pending operation is Upshift wildUSD redemption **trade 639**,
requested on 19 October, with estimated settlement on **21 October**, outside
the evaluation. Its requested reserve estimate is $2,026.60. It has no executed
reserve output and remains valued within the position, rather than credited
to cash or added to NAV a second time.

Final chain NAV is Arbitrum $37,593.27, Ethereum $36,478.73, Base $16,688.15,
Monad $15,895.74 and Avalanche $8,461.66. The chain totals reconcile to the wallet.

Inspected exports are `charts/equity-drawdown.png`,
`charts/asset-allocation.png` and `charts/universe-eligibility.png`.
Allocation displays the fifteen largest mean weights and an explicitly grouped
remainder; the CSV retains every asset. Qualification uses the prior daily candle
and separates first checks, tail gates, entry threshold and buffered membership.
Historical coverage, unknown permission evidence, observed closures and policy
blocks are separate series. No synthetic schedule is included.

Focused validation: **53 tests passed**—50 timing/backtest/CCTP cases and three
canonical strategy cases. These cover broad membership, concrete exclusions,
held-vault retention and the Nova timing extension. This work did not rerun all
previously delivered guarded fork suites or certify the 188 deployments.

These remain historical simulations using current retrospective membership
metadata, sparse/forward-filled NAVs and documented settlement assumptions.
Unknown historical evidence can be admitted by the engine's existing explicit
protocol policies; it is not proof of owner access. Current data freshness,
owner capacity, actual Guard configuration and funded operation require their
own live checks. No hosting changes, funded transactions or external publication
were performed.
