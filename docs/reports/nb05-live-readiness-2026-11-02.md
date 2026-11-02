# NB05 readiness implementation evidence

> Latest financial and universe results are in the
> [eligibility-corrected rerun](nb05-eligibility-corrected-backtest-2026-11-02.md):
> 241 eligible identities, 17.72% return and -0.13% maximum drawdown.
> The nine-vault conversion and diagnostic controls below are retained as history.
> The [exclusion audit](nb05-exclusion-audit-2026-11-02.md) records the selection
> errors corrected in that rerun; apply the [universe preservation guide](../backtest-universe-preservation.md)
> before interpreting these results as complete supported-route coverage.

Checked on 2 November 2026 against executor `ee7b1fd325f28359a149ac5a1a54d371c6849ad2`
plus the accompanying changes, eth-defi `27c16e26dd9325427bb35af8e1c5e2efcb63db60`
and trading-strategy `f80f684758413d27b5350a57718d3ca594a91574`.
This records implementation of the [approved plan](../plans/nb05-live-readiness.md).
Production hosting and rollout were explicitly excluded from this work by the user.

**Initial result (superseded):** the restricted NB05 strategy is converted, directional timing is
corrected, and local regression and guarded fork checks pass. Current market-data
freshness prevents live admission. These results do not establish funded production
readiness or certify the historical 347 deployments.

## Implementation

- Gains and Ember exits no longer supply same-cycle buying or bridging cash.
  Accountable and Plutus deposits remain synchronous even with ERC-7540 flags.
  Plutus Hedge exits are delayed; Dolomite remains synchronous. Historical
  Plutus deposit gates and their chart grace period are removed. Explicit
  simulation overrides, including zero-delay next-cycle settlement, are retained.
- `check-universe` now reuses the existing default-chain configuration helper,
  selecting a cross-chain strategy's primary chain with multiple configured RPCs.
- The actual NB05 notebook was converted to
  `strategy/master-vault-nb05.py` in the canonical `strategies` repository.
  The original `master-vault-multichain-v2.py` research export remains unchanged.
  Normal authenticated loaders replace frozen research inputs. Live synthetic
  windows are removed. Ranking, buffered weekly membership, floors, concentration
  limits, TVL sizing, AlphaModel, YieldManager and CCTP paths are retained.
  AST comparison confirms 31 notebook functions, including `decide_trades`,
  membership and weighting, are unchanged; four input/timing helpers are adapted.
- The candidate subset is applied before allocation. Existing holdings outside
  the subset remain in the universe for reconciliation and exits, marked with
  existing ignore reasons. Entry-fee screening does not reject held-only vaults.
- Fluid was added to the existing guarded manual CLI matrix to exercise parking,
  cash release and investor redemption. No new persistence, admission service,
  conversion route or settlement controller was introduced.

## Assessment and candidates

The [historical inventory](nb05-live-adapter-coverage.json) retains the original
347 records, capability observations, dates, gap labels and summary counts
unchanged. The initial review, now retained as `superseded_nine_vault_review`,
contains one row per original identity, with nine candidates and 338 exclusions.
The current `readiness_review` follows the eligibility-corrected run linked above.
The initial review distinguishes implementation and
validation gaps, deliberate exclusions and observed temporary unavailability.
It is documentation and is never loaded by execution. The historical assessment's
reproduction scripts remain in PR #1693.

| Chain | Candidate address | Selected route |
| --- | --- | --- |
| Arbitrum | `0x4b6f1c9e5d470b97181786b26da0d0945a7cf027` | Hyperithm USDC, Morpho V1 |
| Arbitrum | `0x55a2b207b0074e13adcb858950a81b7a04775e0f` | Gauntlet USDC Balanced, Morpho V2 |
| Arbitrum | `0x2b0b6376083c6e1f376c7439f328436a673f333c` | yPT-aUSDC, Yearn V3 |
| Arbitrum | `0xcacc53bacce744ac7b5c1ec7eb7e3ab01330733b` | USDC CompoundV3 Lender, Yearn compounder |
| Arbitrum | `0xcc56410e1a136af0eceb7241c6ae394f4d8b581c` | Atoma Index |
| Arbitrum | `0x1c788e14d8e5b446e3f71b5142e2edabcab36da1` | Atoma RWA |
| Arbitrum | `0x1a996cb54bb95462040408c06122d45d6cdb6096` | Fluid fUSDC cash venue |
| Ethereum | `0x74ad2f789ed583dbd141bbdafc673fe1f033718b` | Upshift Sentora USD Earn |
| Ethereum | `0xe9b725010a9e419412ed67d0fa5f3a5f40159d32` | Upshift Core USDC |

Read-only current RPC observations confirm the selected adapters, native-USDC
denominations and synchronous deposits for all nine. Atoma and Sentora report
asynchronous redemption; Core currently reports synchronous redemption but keeps
the executor's conservative delayed-proceeds classification. Observed blocks were
Ethereum 26326251 and Arbitrum 520530318–520530328. Block numbers were read after
calls, so these are provider observations, not atomic pinned snapshots or owner
capacity checks. Guarded fork tests below establish separate execution evidence.

USPC Custody remains excluded because of its sender allow-list. Axis USDx, Tori
trUSD and earnAUSD remain excluded because payout differs from USDC; accepted USDC
deposits are insufficient. Tori's upstream adapter and payout are now identified,
but its owner admission is not certified. Sentora is USDC-denominated and is not
one of these three different-payout deployments. NEMO was excluded in this run;
the [eligibility-corrected run](nb05-eligibility-corrected-backtest-2026-11-02.md)
restores its synchronous entry and configures its 30-day redemption assumption.
Lighter pools, GMX liquidity,
Symbiotic and other unvalidated deployments remain expansion work.

## Validation

All counts below are local results, not CI results. Fork transactions used Anvil;
no production transactions were sent.

| Check | Result | Evidence boundary |
| --- | --- | --- |
| Async backtests, directional regression, Upshift units, CCTP cash awareness | 59 passed | Includes mixed rebalance and satellite funding with pending exits, overlapping flags and explicit delay overrides |
| Atoma public CLI fork suite | 5 passed | Both deployed ABIs, operator settlement, restart, haircut/zero payout and refusal/reconciliation paths |
| Upshift public CLI fork suite | 5 passed | Sentora/Core execution, pending restart and standalone simulation; NEMO covered by existing test but not selected |
| Generic manual public CLI fork matrix | 1 passed | Six targets including Fluid; deployment, deposit/close, account checks and investor redemption |
| Canonical NB05 module tests | 2 passed | Normal module loading, registries and retaining an excluded holding for exits |
| Existing `check-universe` CLI regression | 1 passed | Legacy single-chain command still works |
| Actual NB05 `check-universe` | Expected freshness refusal | Primary-chain selection and live universe creation succeed; current data is too old |

Total: **73 passing tests**. Fork operator impersonation and test settlement
drivers prove mechanics, not availability of a real operator. NB05-produced
trades were exercised in the financial backtest; guarded live tests use the
existing focused strategies and public commands, not a funded NB05 deployment.

Relevant test modules are `tests/backtest/test_backtest_async_vault.py`,
`tests/units_tests/test_vault_directional_timing.py`,
`tests/units_tests/test_upshift_execution.py`,
`tests/ethereum/test_cctp_bridge_cash_aware.py`,
`tests/erc_4626/test_atoma_async_cli.py`, `tests/erc_4626/test_upshift_cli.py`,
`tests/lagoon/test_erc4626_manual_cli_e2e.py`,
`tests/cli/test_cli_check_universe.py`, and canonical strategies'
`tests/test_master_vault_nb05.py`. Tests used the sourced local test environment,
Poetry and worktree-first `PYTHONPATH`; the 59-case group used `-n auto`.

## Financial backtest

Run date: 2 November 2026. Source notebook revision:
`243217e7fa935baee18894048f84daa3d000626d`,
`notebooks/xchain3/05-backtest-high-frequency-high-win-rate-native-vaults.ipynb`.
The run completed under the notebook research deadline supervisor, exit code 0.

| Metric | Restricted readiness run | Original research reference |
| --- | ---: | ---: |
| Period | 1 April 2025–21 October 2026, exclusive end | Same calendar |
| Initial equity | $100,000 | $100,000 |
| Final equity | $107,231.82 | $117,861.79 |
| Return | 7.23% | 17.86% |
| Maximum drawdown | -0.62% | -0.49% |
| Trades | 107 successful, zero pending | 797 |

This is a restricted-universe run with refreshed inputs, not a parity replay or
an isolated measurement of the timing fix. Ending positions were Fluid $46,679.50,
Hyperithm $19,781.60, Gauntlet $19,770.19 and Sentora $15,640.26, with about $5,360.28
remaining cash. Atoma Index also traded and was closed; the other candidates did
not receive allocations. There were two bridge trades. No native pool or
different-payout position was opened.

Local artefacts are under
`/home/mikko/code/vault-notebooks/data/xchain3/nb05-readiness-20261102/`:
runner and supervisor receipt, input parameters/module hash, universe pickle,
state, equity CSV, summary and allocation summary. `input-hashes.json` records
download/source hashes after the run, not immutable original download snapshots.
The module subsequently received documentation and a held-only fee-screen fix;
the latter does not affect this initially empty, selected-only backtest.

### Explanation of the lower return

A comparison of the saved original and restricted states shows a substantial
change in portfolio composition:

| Measure | Original | Restricted |
| --- | ---: | ---: |
| Distinct vault identities traded | 54 | 5 |
| Mean daily equity parked in Arbitrum Fluid | 5.52% | 70.94% |
| Mean daily free cash / equity | 5.09% | 5.01% |

Identities outside the new subset generated $17,196.70 of the original
$17,861.79 position profit (96.3%). Examples include HYPE++ ($2,203.41),
Plutus Hedge ($1,604.20) and the two Gains vaults ($1,595.67 combined).
These figures describe the original portfolio; they do not measure the
counterfactual marginal return from restoring each vault.

The restricted portfolio earned $4,979.57 from Fluid and $2,252.25 from the
other four traded vaults. Its small eligible membership and retained 20%
concentration ceiling leave much of the capital in the cash venue. The
readiness subset therefore materially changes the strategy's economics.
The comparison supports universe restriction and increased parking as the
principal explanation. The controlled comparisons below subsequently confirmed
that subset selection causes the large difference, while refreshed inputs
produce identical trades and equity for this subset. This drop must not be
presented as the measured cost of the directional timing fixes.

The calculation is saved locally as `return-comparison.json` alongside the run
artefacts. Daily exposure means use the saved portfolio and position statistics.

## Review against earlier fixes

The nine-candidate choice was too narrow to represent NB05's allocation strategy.
It was drawn from lifecycle validation targets rather than prioritising the
recorded strategy's investments. The approved subset approach permits a staged
launch, but does not make an arbitrary test matrix economically representative.

- [Vault-notebooks #3](https://github.com/tradingstrategy-ai/vault-notebooks/pull/3)
  deliberately preserved inherited membership and verified the ledger when adding
  tagged sources. [Strategies #65](https://github.com/tradingstrategy-ai/strategies/pull/65)
  verified all 797 trades and 568 equity observations. Preserving decision
  function bodies alone does not preserve allocations after replacing membership.
- The broad `outside_initial_subset_pending_validation` exclusions removed
  identities responsible for $16,736.19 of the original profit. Only $216.26 came
  from Lighter and $87.28 from Hypercore; GMX and Symbiotic were untraded. Missing
  native integrations therefore explain little of the removed historical profit.
  Most omitted return came from existing adapter families awaiting additional
  deployment validation, including 40acres, Lagoon, D2, Plutus and Gains.
- [Executor #1694](https://github.com/tradingstrategy-ai/trade-executor/pull/1694)
  proves small-amount lifecycles for specific deployments. That evidence is useful
  for implementation validation, but its matrix is not a strategy allocation
  specification or proof of capacity at NB05's portfolio size.
- [Executor #1686](https://github.com/tradingstrategy-ai/trade-executor/pull/1686)
  distinguishes unknown admission, account restrictions, policy exclusions and
  closure. Missing validation must remain an explicit validation gap. It must not
  become a claim of missing implementation or permanent unavailability.
- [Executor #1566](https://github.com/tradingstrategy-ai/trade-executor/pull/1566)
  warns against a cash cap that cannot see managed-yield releases. NB05 retains
  PhaseAwareAlphaModel's queue-cash hook and YieldManager's release path. Average
  idle cash stayed near 5%; the increase was parked Fluid capital. There is no
  demonstrated reason here to remove funding safeguards or increase concentration.
- [Executor #1688](https://github.com/tradingstrategy-ai/trade-executor/pull/1688)
  already modelled Atoma's delayed exits and fee in the original replay. The
  10.63 percentage-point return difference is not a newly measured Atoma cost.

### Controlled historical comparison

Two additional financial cases ran with the original module callbacks, its
hash-verified frozen vault metadata/prices and the current executor/dependencies
including the local timing fixes. The restricted case changes only the ignore
flags for vaults outside the nine-candidate subset; all original pairs remain
available for diagnostics. Evaluation dates, capital and parameters are identical.

| Case | Final equity | Return | Trades |
| --- | ---: | ---: | ---: |
| Saved original replay, earlier engine | $117,861.79 | 17.86% | 797 |
| Original vault inputs and full universe, current engine | $118,036.29 | 18.04% | 806 |
| Original vault inputs, restricted subset, current engine | $107,231.82 | 7.23% | 107 |
| Refreshed inputs, restricted readiness module | $107,231.82 | 7.23% | 107 |

All trades in the two additional cases succeeded, with no pending trades.
Their last equity observation is 20 October 2026. The two restricted runs have
**identical trade economics and byte-identical equity CSVs**. Refreshed inputs
therefore contribute no observed difference for this subset. Restriction costs
10.80 percentage points against the full current-engine case. The combined
engine/dependency changes improve the frozen full-universe reference by about
0.17 percentage points; this does not separately identify each timing fix's cost.

The restricted strategy first invests outside Fluid on 10 December 2025.
Sentora first enters on 6 April 2026, Gauntlet on 11 May and Atoma Index on
10 August. Selecting current lifecycle test deployments left the early part
of the historical evaluation without eligible directional investments.

Evidence is under local `data/xchain3/nb05-readiness-attribution-20261102/` in
vault-notebooks: runner, input/module hashes, current engine patch, per-case
state/equity/summary, exact restricted ledger comparison and supervisor receipt.
The supervisor completed with exit code 0 across three attempts. The first
attempt could not use the old universe pickle's legacy metadata; the second
rebuilt through the module's hash-verified loader and completed the full case,
then failed while reconstructing another universe in the same process with
float token decimals. The third retained the completed full case and ran the
restricted case in a fresh process. These harness failures are retained and
are not successful financial cases. No shared engine checkout was reset.

The full-universe cases retain the source research's synthetic windows,
idealised native funding and uncharged native performance fees. They are
diagnostic controls and do not establish live execution readiness.

The appropriate correction is to choose and validate a subset against actual
NB05 allocation coverage, retaining specific exclusions and the existing risk
limits. Unsupported routes must stay excluded. A representative subset requires
deployment validation of the routes the strategy would actually select, rather
than using the existing lifecycle test matrix as its investment specification.

## Remaining readiness blockers

1. **Fresh market data.** At 09:45 UTC, the strategy's existing 36-hour freshness
   gate rejected Hyperithm and Yearn CompoundV3: both had last real daily prices
   on 27 October and TVL above the $5,000 checking threshold. Six candidates were
   older than the loader's 24-hour diagnostic threshold. Normal CLI preflight
   also refused its default 24-hour data-age check. Thresholds were not weakened.
   Restore current histories and repeat preflight before live allocation.
2. **Current admission.** Candidate membership is not an allocation guarantee:
   fee availability, TVL, whitelist, owner capacity and execution limits still
   apply. Upshift Core USDC has missing fee metadata and is retained as ignored
   data (`lacks_fee_data`). Validate actual custody owner permissions, proxy/Guard compatibility and
   funding configuration when deployment is selected.
3. **Operational evidence outside this task.** Migration journal inspection,
   outstanding metadata migrations 018/019 and publication, production Guard
   inspection and funded rehearsal remain unperformed. Follow the existing
   deployment procedures; no production host work was requested in this scope.
4. **Expansion.** The remaining historical identities and separate native/queue
   integrations remain explicitly incomplete. No blanket certification or funded
   test requirement for all 347 identities has been introduced.
