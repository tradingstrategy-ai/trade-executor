# Automatic externalised vault fees in backtests

Status: implemented and locally validated, 2 November 2026. See
[implementation evidence](../reports/backtest-externalised-vault-fees.md) for
observed results and modelling limitations. The sections below retain the
approved implementation scope.

## Outcome

Automatically deduct externalised management and performance fees whenever a
backtest reduces a vault position, including a partial withdrawal. Strategies
must not install pricing wrappers, write a performance-fee `sell_tax`, or call
a fee helper from `decide_trades()` to receive correct accounting. The same
calculation must serve ERC-4626 vaults, Hyperliquid/Hypercore vaults and Lighter
public vault pools, using their normal dataset identities and pair translation.

Assume externalised management and performance fees are collected on withdrawal.
Estimate the unpaid fees when valuing an open position, but debit them only from
the withdrawal proceeds. Add this comment beside the calculation and in the
backtest documentation:

> Vault operators may deduct externalised fees before withdrawal. This simulator
> currently collects them only on withdrawal. Future work may model management
> fee deductions weekly or monthly; periodic collection is not simulated here.

This change concerns fees paid by a strategy investing in another vault, not
fees charged to investors in the strategy's own Lagoon treasury. Existing entry
and withdrawal charges remain separate and must be applied once. It does not
implement Lighter's unfinished live public-pool execution or enable unsupported
NB05 production routes.

## Existing code to reuse

- [Backtest pricing](../../tradeexecutor/backtest/backtest_pricing.py) already
  quotes sell-side fees, and [valuation](../../tradeexecutor/backtest/backtest_valuation.py)
  uses sell quotes. Pricing currently lacks position-specific cost and age inputs.
- [Backtest execution](../../tradeexecutor/backtest/backtest_execution.py)
  credits net synchronous proceeds and honours the caller-supplied
  `backtest_vault_redemption_fee` rate at asynchronous settlement. That rate alone
  cannot calculate settlement-time profit or accrued management fees.
- `strategy/hyper-ai-v13.py` in the canonical strategies repository contains
  `get_remaining_cost_basis()`, `refresh_vault_redemption_accounting()` and
  `install_vault_redemption_pricing()`. Reuse the successful-trade replay and
  proportional cost-basis reduction, replacing its strategy-local monkeypatch
  and fixed fee assumptions with shared metadata-driven accounting.
- [Fee completeness checks](../../tradeexecutor/strategy/vault_fee_data.py)
  already distinguish internalised, known-zero, externalised and unknown fees.
  Pair translation carries `VaultMetadata`; verify this for native pool entries
  as well as ERC-4626 entries.
- The pinned eth-defi `calculate_net_profit()` uses principal-based, simple
  annual management accrual, then performance fees on positive profit after
  management fees. Its year is 365.25 days. Use these conventions consistently;
  it is a research-return helper, not a replacement for the position ledger.
- Follow [vault lifecycle guidance](../../.claude/docs/vault-deposit-redeem.md),
  [state persistence](../state-persistence.md) and
  [strategy module conventions](../strategy-parameters.md).

## Fee and accounting rules

1. Resolve fee mode and rates from the loaded vault data, not a protocol-name
   allow-list. Internalised management/performance fees are already in share
   prices and receive no additional deduction. Known zero is valid; unknown,
   non-finite, negative or inconsistent inputs must not silently become zero.
   Keep existing missing-data admission reasons and fail clearly before a
   financial mutation if an entered/held externalised vault cannot be priced.
2. Use snapshotted constant rates for this implementation and label that
   assumption in the run results. `VaultMetadata` explicitly warns that its
   latest rates are not historical rates: the constructed test is a fee
   accounting scenario, not proof of historical fee schedules. Preserve any
   dated observations in the sample provenance; variable-rate integration is
   separate work when a consumer requires it. Do not introduce a fee-history
   service or silently represent current rates as historical facts.
3. Reconstruct remaining share quantity, acquisition principal and unpaid
   principal-time exposure from successful trades. Buys add actual acquisition
   principal when shares settle; successful reductions remove each component
   proportionally. Later buys start their own age, and an earlier partial sell
   does not reset the age or erase unpaid fees on surviving shares. Failed,
   cancelled and merely requested trades do not consume cost basis.
4. For a reduction of fraction `f` of the held shares, compute in the vault's
   denomination currency, converting to portfolio USD only for reporting:
   `C = f × remaining acquisition principal`, `G = sold shares × gross price`,
   `M = f × sum(remaining principal × annual management rate × elapsed years)`,
   and `P = performance rate × max(G − C − M, 0)`. Apply the existing withdrawal charge once to
   `max(G − M − P, 0)`. Cap collected fees at available proceeds, report nominal
   versus collected amounts when capped, and never create a negative payout.
   Acquisition principal is the amount actually invested after any entry fee,
   excluding gas/bridge costs. Report the full investor cash outlay separately
   for PnL; an entry charge must not increase the performance-fee basis.
5. Use proportional weighted-average cost basis, not FIFO or a new tax-lot
   database. This is a withdrawal-time performance-fee approximation; do not
   claim protocol-specific hurdle/high-water-mark crystallisation is simulated.
   Profitable partial withdrawals incur fees only on the redeemed portion;
   repeated reductions must not charge previously realised profit again.
6. Open-position NAV and sell quotes use the same hypothetical net withdrawal
   calculation. This is valuation, not a cash deduction or a second liability
   subtracted from equity. At zero quantity or a total loss, handle zero proceeds
   without division by zero. Cash plus net positions and existing pending claims
   must reconcile to equity.
7. For asynchronous exits, snapshot the requested shares and the relevant basis
   and fee assumptions in the existing trade record. Calculate final amounts at
   settlement using the settlement price and elapsed time through settlement;
   request-time estimates are not final fees. Pending redemption NAV follows the
   same convention and remains unspendable. A repeated resolver call or a state
   reload cannot deduct fees or credit proceeds twice. Existing one-pending-exit
   restrictions remain in force.

## Implementation sequence

### 1. Add the shared calculation and metadata handling

First select and inspect the real samples described in step 4. Verify that fee
mode/rates survive the loader and that the price series matches that mode,
including native identities. The pinned fee matrix calls Lagoon externalised;
this classification alone does not prove that every deployed Lagoon price
series is gross. Resolve contradictory evidence upstream or record a blocked
sample; never force an externalised flag merely to obtain a non-zero test fee.
If native fee fields are missing, inspect the existing provider/adapter output
and fix the smallest necessary metadata propagation gap. No guessed rates or
hand-authored replacement metadata qualify as real-data detection evidence.

Put the calculation in one small module under `tradeexecutor/backtest/`, with
an explicit result containing gross proceeds, attributable principal, management
fee, performance fee, existing exit charge and net proceeds. Keep units and
timestamps explicit. Derive accounting from existing positions/trades and use
their normal serialisation for any settlement inputs and realised breakdowns.
Use `TradeExecution.other_data["backtest_vault_externalised_fees"]` for a
versioned serialisable breakdown: fee mode/rates and provenance, requested
quantity, attributable principal and principal-time exposure at request,
calculation timestamp, gross proceeds, each nominal/collected fee and net
proceeds. Request fields are fixed; realised fields are written only on
successful settlement. Absent keys remain valid for pre-change state. Reuse
the existing serializer's Decimal/datetime conventions. No separate registry,
ledger, persistence service or public CLI is needed.

Start with ledger replay and measure it on the bounded strategy run. Add only
a derived in-memory cache if profiling demonstrates a material cost; do not
introduce another persisted accumulator merely to avoid a hypothetical slowdown.

Inspect `dex_data_translation.py` and native universe construction. Preserve
fee mode, rates, chain and vault identity through translation and state reload.
Lighter's `lighter-pool-*` identities and Hypercore's synthetic chain must reach
the generic vault fee calculation; do not gate it on an EVM address, ERC-4626
feature flag, or a live routing implementation. Exchange-account transfers,
perpetual trades and bridge transfers are not vault withdrawals.

### 2. Wire quotes, valuation and both execution paths

Use normal backtest setup to provide pricing with the position information it
needs, rather than attaching methods or mutable fee rates to shared pair objects.
Call the shared calculation from net sell pricing/valuation and from synchronous
and asynchronous execution. Cover normal batch and CCTP sequential execution.
Recompute for the actual executed quantity after sizing/rounding adjustments.

Allocation and same-cycle funding must use net synchronous proceeds. Pending
exits cannot fund purchases or satellite transfers. Record realised fee
components on the trade without treating them as AMM liquidity-provider fees
or deducting them again through reserve updates. Refresh remaining-position NAV
from its own surviving basis after a partial settlement.

Preserve existing fixed entry/exit costs, especially Atoma's 0.5% exit charge.
Trace whether each charge came from pair pricing or metadata; use one source
for each component. A legacy `backtest_vault_redemption_fee` snapshot must not
stack with automatic management/performance fees. Retain its existing behaviour
for old pending trades, while new trades use the shared calculation. Reject
ambiguous mixed legacy/new inputs before execution rather than guessing.
Release the engine and affected strategy changes together. Refuse the known
Hyper-AI pricing wrapper/fee-map marker with an actionable migration message.
Check at setup and after strategy callbacks before execution, because Hyper-AI
installs its wrapper inside `decide_trades()`. Removing it is required before
the active strategy uses the new engine.
Audit its unmarked pre-wrapper `sell_tax` setup too. Keep ordinary fixed exit
taxes valid, but do not infer that an arbitrary manual tax is a performance fee.
No new global engine-version switch is required. Test the known legacy caller
refusal explicitly before automatic fees can reach a wallet mutation.

### 3. Remove duplicate fee handling from affected callers

Update Hyper-AI v13 and NB05 in the canonical strategies repo
to use shared accounting. Remove the dynamic method wrapper, duplicate basis
replay and per-cycle mutation of `sell_tax` once their regression coverage
passes. Keep any deliberate capital-friction assumption separately named and
applied once; do not turn it into a protocol management/performance fee.

The implemented migration covers v13. The active `hyper-ai.py` alias still
selects v12, whose legacy wrapper must be migrated before backtesting it on the
new engine. Live execution does not use this backtest fee path.

Remove NB05's blanket exclusion for otherwise supported externalised fees when
the engine can model them. Preserve independent admission, funding and timing
exclusions. Ranking on net historical returns remains a signal calculation;
it must not alter execution candles or apply another portfolio deduction.
Update stale claims that the backtester cannot charge externalised fees.

Limit cleanup to touched fee code and documentation: remove redundant helpers,
temporary compatibility hacks for unpublished code, repeated explanations and
misleading comments. Preserve necessary historical strategy versions and avoid
unrelated formatting or broad rewrites. Record the intended behaviour change
for old backtests with previously uncharged fees.

### 4. Capture real samples and run a constructed strategy

Acquire a small real dataset through the authenticated vault loaders. Save
unaltered metadata rows and the required price/fee observations for Lagoon,
IPOR, a Lighter public vault pool and a Hyperliquid managed vault. Include an
internalised or zero-fee control. Prefer actual non-zero management and
performance fees across the selected samples; do not invent either fee when a
protocol reports zero. If the available samples cannot exercise a component,
add a separately labelled synthetic case and report the real-data coverage gap.

Record exact chain/vault identities, denomination, metadata generation and
capture timestamps, observed fee mode/rates, price convention, sample period,
source and hashes. Confirm whether the consumed share-price series is gross or
already fee-net; metadata/price conflicts require resolution before accepting
the sample. Store only the small redistributable fixture and provenance under
`tests/backtest/fixtures/vault_externalised_fees/`, with no credentials or
complete private dataset. Save/hash inputs before execution; offline tests
verify the hashes. Store the constructed schedule in test support code, not
protocol fee overrides in strategy `Parameters`.

Add a constructed module under `strategies/test_only/` and exercise it through
the normal strategy loader and backtest runner. Load the frozen real rows via
the ordinary translation path, preserving their identities and fees. The
module deposits, tops up on a later date, partially reduces, and finally exits
each sample. It must contain no fee mutation or fee-calculation calls.

Use a bounded sample interval and explicit settlement assumptions. Where the
live native funding route is unfinished, label the Lighter run as pool fee
simulation with simulated funding; do not spoof an ERC-4626 identity or claim
a live sequencer/Guard round trip. Required management/performance deductions
must be detected from real metadata rather than hardcoded strategy rates.

### 5. Verify and report

Add compact tests covering these distinct behaviours:

- Independent numeric oracle: dated top-up, profitable partial withdrawal,
  remaining unpaid fees, final exit, and a loss/zero-payout boundary. For example,
  a 1,000-unit deposit with 2% annual management and 20% performance fees, gross
  value 1,200 after 365.25 days, and no entry/exit charge yields management 20,
  performance 36 and net 1,144. A half withdrawal at that timestamp collects
  management 10, performance 18 and net 572, leaving principal 500 on the
  remaining shares. A same-time 500-unit top-up makes remaining principal 1,000,
  with unpaid management 10 on the older half and zero on the new half. After
  another 365.25 days, unpaid management is 30. Choose the final price explicitly
  in the test to calculate its performance fee independently.
- Constructed strategy: real snapshot rates survive loading and automatic fee
  detection for Lagoon, IPOR, Lighter and Hyperliquid; internalised/zero-fee
  controls receive no duplicate deduction. Assert both realised amounts and
  pre-exit net NAV, not just the presence of metadata.
- Delayed redemption: change the price while pending, reload state, settle,
  then repeat resolution. Verify settlement-time fees, no duplicate credit,
  and no purchase/bridge funding from pending proceeds. Include a same-cycle
  synchronous sell/buy case whose affordability depends on net proceeds.
- Missing/invalid fee inputs fail before mutation; existing fixed exit fees
  and legacy pending snapshots are neither lost nor doubled. A non-vault trade
  and exchange-account transfer retain their existing behaviour. Load a
  pre-change state without the new key and test the legacy-caller refusal.
  A control run without externalised vaults retains its previous financial
  results. Missing metadata on unused universe members does not abort a run.

Use focused existing vault pricing, async settlement, metadata translation and
cash-aware CCTP tests alongside the new cases. Follow repository Poetry,
environment, worktree `PYTHONPATH` and pytest conventions. No fork suite or
funded transaction is required to prove this simulation change.

Save a concise result table by vault and withdrawal: detected rates, gross
proceeds, basis, accrued management fee, performance fee, other exit cost, net
cash and remaining NAV. Independently reconcile the final wallet/equity and
sum of realised fees against the saved state. Record command, revisions, input
hashes and test results. Keep real-data observations, constant-rate modelling
assumptions and synthetic boundary cases clearly distinguished.
Write the report to `docs/reports/backtest-externalised-vault-fees.md`. Use
Decimal for monetary bookkeeping, token-precision checks for wallet amounts,
and `pytest.approx()` with explicit absolute tolerances for floating reporting
values (at most 0.000001 denomination units for these USDC-sized samples).

## Acceptance

An ordinary constructed strategy automatically applies the detected fee mode
on partial and full vault reductions across all four named families, including
zero extra management/performance deduction for internalised samples, with no
strategy-specific fee code. Real externalised samples must exercise non-zero
management and performance deductions across the fixture set; a missing real
sample is an explicit acceptance gap. Net valuation, cash available for allocation and
settled wallet amounts agree. Repeated recovery is idempotent, internalised fees
are not charged twice, and the real-data fixture and numeric results reproduce
offline. The weekly/monthly management-fee limitation is documented beside the
implementation. Production deployment and a new full NB05 financial run are
separate from this focused fee implementation and validation. The active legacy
strategy cannot silently combine its wrapper with automatic deductions.

## Plan review

Claude CLI reviewed the initial plan with `claude-fable-5-1` on 2 November 2026.
The review was text-only, after local code inspection by the primary agent.
Its completed result prompted earlier sample verification, an explicit
cross-repository migration guard, concrete trade evidence, a defined principal
basis and stronger numerical/reload controls. Variable-rate integration was
deferred. Protocol-wide claims about internalisation or accrual ending at request
were not accepted without deployment evidence: this plan explicitly assumes
accrual through settlement. Persisted incremental accounting and a new engine
version gate were not added without a demonstrated need.

A second text-only review of the revised plan with `claude-fable-5-1` completed
with **no blocking findings**. Both CLI runs returned successful terminal
results (`end_turn`, `completed`). No implementation or backtest was run as
part of writing and reviewing this plan.
