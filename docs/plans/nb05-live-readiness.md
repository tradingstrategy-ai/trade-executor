# NB05 live-readiness plan

## Goal and baseline

Launch NB05 with a validated subset of vaults, then expand coverage through
separate protocol milestones. Preserve the original 347 identities and exclusion
reasons as assessment evidence; supporting every candidate is not a launch prerequisite.

Baseline: executor master `ee7b1fd32`, with eth-defi `27c16e26d`, checked on
2 November 2026. This replaces the outstanding-work requirements in the
[seven-family plan](nb05-erc4626-live-support.md) and the original
[12-step PR comment](https://github.com/tradingstrategy-ai/trade-executor/pull/1693#issuecomment-6499577978).
The older plan remains historical technical research. External PR publication is separate.

Implementation and the corrected financial result are recorded in the
[2 November backtest report](../reports/nb05-eligibility-corrected-backtest-2026-11-02.md).
The timing fixes and local strategy conversion are implemented; deployment
validation, current admission and funded rehearsal remain outstanding. The
table and delivery steps below record the approved baseline scope. The linked
report records current results and remaining operational checks.

The [multichain account-check implementation record](multichain-account-checks.md)
adds read-only hub/satellite custody checks and bounded settled corrections to
the mixed Lagoon deployment. Missing observations and unresolved operations now
refuse a clean result; generic correction is not a bridge, vault or pool recovery
tool. The record contains the fixed-fork acceptance coverage and validation gaps.
This closes a command-coverage gap without replacing the remaining production
configuration, admission and funded-rehearsal requirements below.

The launch-selection criteria below must not silently become historical backtest
exclusions. For strategy conversion and financial comparisons, apply
[backtest universe preservation](../backtest-universe-preservation.md) and its
required review checks.

## Corrected status and scope

| Area | Status at baseline `ee7b1fd32` | Requirement at baseline |
| --- | --- | --- |
| Atoma execution | Delivered by #1696/#1697 | Validate production accounts/deployments and complete funded rehearsal. |
| Upshift execution | Delivered within denomination-matching scope by #1698 | Publish corrected metadata, configure strategy timing and complete funded rehearsal. |
| Persistence and recovery | Delivered for supported flows | Reuse state, tickets, callbacks and resolver; extend only for a demonstrated missing lifecycle. |
| Admission and selection | Existing mechanisms sufficient | Apply NB05 exclusions through universe selection/ignore reasons; retain live owner/capacity checks. |
| Directional timing | Concrete defects remain at baseline | Correct Gains/Ember delayed exits and Accountable/Plutus deposit and variant classification. |
| Guard deployment | Atoma/Upshift selection delivered | Check deployment compatibility; Lighter pool mapping remains separate unfinished work. |
| Conversion assets | Intentional exclusion | Keep three different-payout Upshift vaults excluded from USDC strategies. |
| Deployment coverage | Evidence incomplete | Validate the selected launch subset and distinct implementations. |
| Lighter pools, GMX liquidity, Symbiotic | Separate unfinished integrations | Retain expansion milestones; exclude from initial allocation. |
| NB05 strategy and launch | Outstanding | Convert actual NB05, validate decisions/funding and rehearse production operation. |

Remove these requirements from the active backlog:

- A new certification registry or admission framework.
- Aggregate-ticket allocation/reconciliation: the supported model deliberately halts on aggregates.
- A new settlement controller, execution journal or recovery database.
- Upshift accepted-token overrides and automatic conversion support.
- Automatic modelling of every vault's live queue schedule.
- Universal state migrations or Guard replacement machinery without an identified compatibility problem.
- Funded round trips for all 347 historical candidates before any launch.

## Ordered delivery

### 1. Reconcile the assessment and select the launch universe

- Refresh the existing #1693 assessment against current dependency and merged evidence.
  Preserve dated historical observations rather than rewriting them as current results.
- Distinguish implementation gaps, validation gaps, temporary unavailability and deliberate exclusions.
- Select using supported execution, compatible reserve/payout assets, custody-chain
  infrastructure and current admission evidence.
- Reuse selection and ignore reasons. Retain excluded candidates in diagnostics and
  supported exit handling for existing positions.
- Exclude USPC, different-payout Upshift vaults and unfinished native/queue integrations.
- Resolve Tori's actual adapter, denomination and admission before inclusion;
  missing evidence means exclusion.

Completion: each selected identity has an execution route and evidence reference;
each excluded identity has a reason. The assessment is documentation, not execution state.

### 2. Fix the demonstrated timing errors

Extend existing pair classification and backtest methods:

- Gains/Ember: synchronous deposits and delayed redemption proceeds.
- Accountable: synchronous deposits despite ERC-7540 metadata.
- Plutus Hedge: synchronous deposits and delayed exits.
- Plutus Dolomite: synchronous deposits/exits, distinguished using verified deployment/feature evidence.
- Remove the blanket Plutus historical deposit-settlement requirement.
- Preserve Atoma and Upshift's conservative delayed-proceeds rule, including zero-lag live exits.
- Keep manager capabilities authoritative for live calls, and existing delay settings
  for simulation; preserve explicit simulation overrides.

No new public CLI, persistent lifecycle schema or capability service is required.
Completion: allocation, cash management, satellite bridging and backtests agree on
spendable proceeds; synchronous deposits do not acquire an artificial queue.

### 3. Validate the selected executor paths

- Reuse delivered generic ERC-4626, Atoma and Upshift lifecycle evidence.
- Exercise existing managers before adding implementation: Gains request/claim,
  Ember operator payout, Accountable settlement and Plutus variants.
- Use guarded fork coverage per distinct implementation and deployment-specific
  address, asset, proxy, owner, limit and Guard checks.
- Verify reserve parking/release for the selected Fluid/Kiln deployment.
- Cover public deployment/test-trade commands and strategy-produced trades;
  describe the branches actually exercised.
- Exclude deployments whose funding, infrastructure or settlement cannot be established.

Completion: selected paths deposit, exit, recover after restart and account for
actual payouts through existing machinery.

### 4. Connect and validate the NB05 strategy

- Convert the actual NB05 source, not the separate EXP03 conversion in #1679.
- Preserve allocation rules, replacing frozen inputs with normal loaders and
  synthetic live windows with existing admission/window checks.
- Apply the subset before allocation, retain holdings for reconciliation/exits,
  and reuse AlphaModel, YieldManager and CCTP funding.
- Set material simulation assumptions, including NEMO's 30 days if selected;
  zero-delay overrides still resolve on a later cycle.
- Rerun the financial backtest after timing/universe changes and report changed allocations/metrics.

Completion: the strategy runs through the normal CLI and respects funding,
pending claims and deployment restrictions.

### 5. Complete rollout and funded rehearsal

- Inspect the production migration journal first; apply outstanding Upshift
  migrations 018/019 through the existing runbook, then verify publication and USPC exclusion.
- Inspect selected production Guards/pending state. Use existing governance only
  where incompatibilities or broad legacy permissions require changes.
- Prepare a concrete canary run sheet with addresses, custody accounts, amounts,
  fees and completion checks before funded execution.
- Rehearse representative lifecycles, including selected Atoma/Upshift deployments,
  real operator settlement, restart, reserve reconciliation and investor redemption.
- Monitor pending requests and unresolved transactions using existing diagnostics.

Completion: the production configuration has attributable transaction evidence
and reconciled NAV/cash; unresolved operations prevent launch completion.

## Expansion milestones

These remain full-readiness work but do not block the validated subset.

| Milestone | Required outcome |
| --- | --- |
| Lighter public pools | Implemented in [the Arbitrum multi-vault plan](arbitrum-multi-vault-live-plan.md); a funded pmalt round trip completed on 3 November. [Idle-collateral NAV and strategy-cycle testing](multi-vault-nav-and-start-testing.md) cover the remaining bootstrap dilution and `start` validation work. Inclusion in NB05 still requires deliberate strategy selection and deployment-specific evidence. |
| GMX liquidity | Implement selected GM liquidity request/execution and Guard permissions through upstream adapters/existing paths; perpetual-account support is insufficient evidence. |
| Symbiotic | Identify deployed withdrawal mechanism first; add manager, queue ownership and partial payouts only where required; prove restart, remaining entitlement and final payout accounting. |
| Additional deployments | Extend assessment/evidence incrementally and enable only when concrete gaps are resolved. |

Each new integration receives a focused implementation subplan after deployment research.

## Validation and acceptance

- Compact classification regressions cover affected families and overlapping flags.
- A mixed rebalance proves pending exits cannot fund purchases or satellite transfers.
- Changed settlement behaviour receives lifecycle and interruption/refusal coverage,
  including repeated recovery without duplicate credit.
- Reuse Atoma loss/aggregate-refusal and Upshift recovery/snapshot tests; rerun
  affected tests when shared code changes.
- Distinguish unit, fork, provider and funded evidence.
- Launch acceptance applies to the subset; broader coverage remains incomplete
  until expansion milestones are delivered.

Retain one state writer and one pending redemption per custody account/vault,
existing persistence/settlement, USDC-compatible funding and deliberate exclusions.
