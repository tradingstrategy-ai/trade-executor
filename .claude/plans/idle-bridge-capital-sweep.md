# Idle CCTP bridge capital sweep-back

- Issue: https://github.com/tradingstrategy-ai/trade-executor/issues/1562
- Status: implemented
- Date: 2026-07-17 (implemented 2026-07-18)

## Problem

A phase-aware cross-chain strategy can finish with settled native USDC sitting
idle on a satellite-chain CCTP bridge position instead of being productive in
the yield-managed queue vault. Observed in NB15
([getting-started PR #40](https://github.com/tradingstrategy-ai/getting-started/pull/40)).

Three verified root causes, all in the current hub-and-spoke planner flow:

1. **The planner only bridges idle capital back on demand.**
   `inject_cctp_bridge_trades()` (`tradeexecutor/ethereum/cctp/planner.py:207`)
   injects a bridge-back only when a satellite chain has a *net sell* this
   cycle (step 2, line 358) or when the *primary chain has a cash shortfall*
   (line 398). Capital that becomes idle on a satellite — an async vault
   redemption settling cycles after its sell trade, a withheld net-sell excess
   (the comment at line 374 promises it "bridges back once the deposits have
   settled", but no mechanism does this), or `bridge_capital_allocated` going
   negative on a profitable round-trip — stays parked on the bridge position
   forever if no later demand happens to pull it.

2. **Quiet cycles never plan bridges at all.** The runner gate
   (`tradeexecutor/strategy/runner.py:977`) is
   `if primary_chain is not None and len(rebalance_trades) > 0:` — when
   `decide_trades` returns no trades, the planner is not even called, so idle
   satellite capital cannot be recovered on a quiet cycle.

3. **YieldManager cannot see bridge positions.**
   `gather_current_yield_positions()`
   (`tradeexecutor/strategy/pandas_trader/yield_manager.py:298`) collects only
   the default (hub) reserve position and the configured venue positions.
   Satellite bridge capital is neither, so `available_for_yield` never counts
   it and the sweep never touches it. This is the chain-aware YieldManager gap
   already documented in `phase_aware.py:300` and
   `.claude/docs/phase-aware-alpha-model.md` ("Chain-aware YieldManager"
   follow-up).

## Design decision: hub-first sweep

Issue #1562 proposes depositing settled bridge cash into *that satellite
chain's* queue vault (`sweep_bridge_cash_to_queue`). We deliberately implement
the simpler **hub-first** variant instead, per maintainer direction: **all
idle satellite bridge capital is bridged back to the primary chain, where the
existing YieldManager sweep parks it in the (hub) queue vault.**

Rationale:

- The queue venue is required to be a synchronous hub-chain vault
  (`.claude/docs/phase-aware-alpha-model.md`, "YieldManager interplay" —
  invariant-4 same-cycle funding depends on it). Satellite-side queue vaults
  need a chain-aware `YieldRuleset`/`YieldManager`, which is a documented
  follow-up with its own correctness questions (a hub venue swept with cash a
  same-cycle satellite bridge still needs, etc.). We do not take that on here.
- Capital consolidated at the hub is deployable to *any* chain next cycle with
  one bridge hop; capital queued on a satellite needs a satellite redeem plus
  a bridge-back before it can fund anything elsewhere.
- No new component owns cash: the CCTP planner already owns bridge liquidity
  and YieldManager already owns the cash home (the one-owner-per-concern
  architecture rule). The fix composes the two existing owners; no phase-aware
  model change is needed.

The satellite-side queue-vault deposit remains a follow-up under the
chain-aware YieldManager work; this plan keeps it out of scope but leaves the
diagnostics able to express it (the "why not queued" reason codes below).

Consequence for the issue's acceptance criteria: "deposit into that chain's
queue vault" becomes "bridge back to the hub, then the hub queue vault", with
one extra cycle of latency between settlement and the queue deposit (see
"Latency and churn" below). The issue's other criteria (settlement respected,
buffers respected, works with no later allocation, configurable + on by
default, visible in diagnostics) are all met directly.

## Changes

### 1. Planner: unconditional idle-capital sweep phase

`tradeexecutor/ethereum/cctp/planner.py`

- New keyword arguments on `inject_cctp_bridge_trades()`:
  - `sweep_idle_bridge_capital: bool = True` — feature flag, on by default per
    the issue.
  - `bridge_sweep_min_usd: USDollarAmount = 1.0` — minimum idle amount worth a
    bridge trade; below this the cash is deliberately left (dust buffer). The
    existing per-pair dust epsilon and raw-unit floor
    (`_is_meaningful_bridge_trade_amount`) still apply on top.
- New step between the primary-shortfall reservations (line 419–454) and the
  bridge-back trade creation loop (line 456): for every chain in
  `liquidity_by_chain` (which already includes every open bridge chain via the
  `_open_bridge_chain_ids` seeding at line 350), reserve **all remaining**
  `free_idle_bridge_capital` as a bridge-back when it clears the thresholds:

  ```python
  if sweep_idle_bridge_capital:
      for chain_id in sorted(liquidity_by_chain):
          liquidity = liquidity_by_chain[chain_id]
          bridge_pair = bridge_pairs.get(chain_id)
          if bridge_pair is None:
              continue
          amount = _floor_to_raw_units(liquidity.free_idle_bridge_capital, bridge_pair.base)
          if amount < Decimal(str(bridge_sweep_min_usd)):
              continue  # dust buffer — recorded by diagnostics
          if not _is_meaningful_bridge_trade_amount(amount, bridge_pair):
              continue
          liquidity.reserve_bridge_back(amount)
          total_bridge_back += amount
          sweep_amounts[chain_id] = amount
  ```

  The existing creation loop then emits one combined bridge-back per chain
  (`liquidity.bridge_back_amount`), so the sweep merges with any net-sell or
  shortfall bridge-back into a single trade, and the existing
  `closing` logic (`_is_token_dust(available - amount)`) closes the bridge
  position when the sweep empties it.

- Correctness properties this placement preserves for free:
  - **No double-reserve.** `ChainLiquidity.reserve_bridge_back()`
    (`planner.py:81`) decrements `free_idle_bridge_capital` as it accumulates
    `bridge_back_amount`, so by the time the sweep runs, the net-sell (step 2)
    and primary-shortfall (step 3) reservations have already been subtracted —
    "all remaining `free_idle_bridge_capital`" is genuine remainder
    semantics, and the existing creation-loop assert
    (`amount <= available_before_buy`, `planner.py:472`) backstops it.
  - Capital reserved for same-cycle satellite buys is excluded —
    `free_idle_bridge_capital` is initialised as
    `max(available_before_buy - satellite_buys, 0)` in
    `ChainLiquidity.prepare()` before any reservation.
  - Capital committed to unsettled async satellite deposits is excluded —
    `_available_bridge_capital()` uses
    `position.get_available_bridge_capital()` which subtracts
    `bridge_capital_allocated`.
  - Sweeps sort as ordinary bridge-backs (−30M bump), before bridge-outs and
    buys, so a same-cycle cross-satellite rebalance can consume the proceeds
    in the backtest (synchronous settlement).
  - Adding the sweep to `total_bridge_back` after the `NotEnoughMoney` check
    only widens `fundable_primary` for step-4 bridge-outs — same
    already-documented backtest-vs-live optimism caveat as existing
    bridge-backs (line 505), no new failure mode.

- Tag every injected bridge trade with its planning-reason **breakdown** so
  diagnostics can distinguish sweeps. A single bridge-back can legitimately
  combine net-sell, primary-shortfall and sweep portions (the creation loop
  emits one trade per chain from `liquidity.bridge_back_amount`), so a scalar
  reason is not enough:
  `trade.other_data["cctp_planning_amounts"] = {"net_sell": "12.5", "primary_shortfall": "0", "idle_sweep": "88.1"}`
  (Decimal-as-string values, JSON-safe), plus `"bridge_out"` for step-4
  trades. Track the per-chain reservation amounts per reason as the phases
  run.

### 2. Runner: plan bridges on quiet cycles too

`tradeexecutor/strategy/runner.py:972`

- Read the flag from strategy parameters:
  `sweep = self.parameters.get("sweep_idle_bridge_capital", True) if self.parameters else True`
  and `min_usd = self.parameters.get("bridge_sweep_min_usd", 1.0) ...`.
- Change the gate so the planner runs even with an empty trade list when the
  sweep is enabled:

  ```python
  if primary_chain is not None and (len(rebalance_trades) > 0 or sweep):
      rebalance_trades = inject_cctp_bridge_trades(..., sweep_idle_bridge_capital=sweep, bridge_sweep_min_usd=min_usd)
  ```

  With no trades and nothing to sweep the planner returns an empty list and
  the existing "no action taken" shortcut still fires. The planner with an
  empty input list is cheap: the liquidity ledger is seeded only from open
  bridge positions.

### 3. Settlement safety: no double-sweep while in transit

In live trading a sweep bridge-back goes `TradeStatus.cctp_in_transit` and
settles on a later cycle. The next cycle's planner must not see the swept
amount as still-idle and sweep it again. **Verified — already safe, no code
change needed:** `State.mark_bridge_in_transit()`
(`tradeexecutor/state/state.py:1108`) locks the burned amount for bridge-back
sells via `bridge_capital_allocated += abs(planned_quantity)` explicitly "so
`get_available_bridge_capital()` returns zero for it", and
`check_and_retry_cctp_in_transit()` (`tradeexecutor/ethereum/cctp/retry.py:235`)
reverses the lock at settlement immediately before `mark_trade_success`
reduces the position quantity. A unit test pins this (test 1c below).

Conversely, inbound in-transit transfers are not yet part of
`get_quantity()`, so "wait for settlement" (issue criterion) holds by
construction: only settled, minted USDC is ever swept. Note also that live
execution halts the trade batch when a bridge trade goes in transit
(`tradeexecutor/ethereum/execution.py:754`) and resolution happens via
restart/retry — sweeps make this operational path routine (see "Latency and
churn").

### 3b. Live startup ordering fix (blocking, found in review)

`tradeexecutor/cli/loop.py:1347–1405`: the startup sequence currently runs
treasury sync (`sync_treasury_on_startup`, line 1347) and the startup
accounting check (`check_accounts`, line 1383) **before** the CCTP in-transit
retry block (line 1394+), even though that block's own comment says unresolved
transfers must be resolved first "otherwise capital accounting is incorrect".
Consequences today: a satellite→hub bridge-back that settled on-chain while
the executor was down raises the real hub USDC balance, but state credits the
reserve only when the retry calls `mark_trade_success()` — so a startup
treasury sync first adjusts the reserve to the on-chain balance and the retry
then credits it *again* (double count), while with sync disabled the account
check sees a spurious mismatch. This is a pre-existing bug, but automatic
sweeps make in-transit bridge-backs routine, so it must be fixed in this PR:
**move the CCTP retry (and its unresolved-transfer halt) above both the
treasury sync and the accounting check** in the startup sequence. Regression
coverage: extend the existing in-transit lifecycle tests
(`tests/ethereum/test_cctp_in_transit_lifecycle.py`,
`tests/test_cctp_in_transit_status.py`) with a settled-while-down bridge-back
resolved before reserve reconciliation; a loop-level ordering test with a
mocked sync model if practical, otherwise assert ordering via the lifecycle
test and document the constraint in `loop.py`.

### 4. Diagnostics

`tradeexecutor/analysis/cctp.py`

- `build_bridge_trade_dataframe()`: add a `reasons` column summarising the
  non-zero entries of `trade.other_data.get("cctp_planning_amounts")` (blank
  for old state files).
- New `analyse_idle_bridge_capital(state, bridge_sweep_min_usd=1.0,
  sweep_enabled=True) -> pd.DataFrame`: one row per open bridge position —
  chain, quantity, `bridge_capital_allocated`, available idle capital,
  in-transit value, and a `why_not_swept` classification:
  `swept` (below threshold remains) / `below_min_sweep` /
  `reserved_for_async_deposit` / `in_transit` / `sweep_disabled`. Strategy
  parameters are **not** persisted in state, so `sweep_enabled` and the
  threshold are explicit caller arguments (the notebook/test passes its
  `Parameters` values); the function never guesses them from state. This is
  the end-of-backtest acceptance check the issue asks for ("identify any
  remaining bridge cash and explain why it was not queued") and what NB15
  will display.

No chart changes required: `equity_curve_by_chain` already renders bridge
capital inside its chain band and the queue venue as its own band, so the
before/after effect of the sweep is visible with existing charts.

### 5. Documentation

- `.claude/docs/phase-aware-alpha-model.md`: update the "Chain-aware
  YieldManager" limitation to note idle satellite capital is now swept back to
  the hub by the CCTP planner (the remaining follow-up is satellite-side
  queue venues and same-cycle chain-aware funding).
- `inject_cctp_bridge_trades()` docstring: document the new sweep step and
  flags.
- `tradeexecutor/state/README-bridged-position.md`: one paragraph on the idle
  sweep.

## Latency and churn

- **Latency.** In the backtest, swept cash lands in the hub reserve the same
  cycle (`simulate_bridge` settles synchronously). In live trading a bridge
  trade halts the execution batch when it goes in transit
  (`tradeexecutor/ethereum/execution.py:754`) and resolves through the
  restart/retry path, so live latency is CCTP attestation time plus the
  retry/restart cadence — not a fixed cycle count. In both modes YieldManager
  runs *inside* `decide_trades`, before the planner, so it first sees the
  recovered cash on a following cycle and only then deposits it into the
  queue vault. Net: idle capital reaches the queue vault with bounded delay —
  versus never, today. No YieldManager change is needed. Sweeps make the live
  in-transit/retry operational path routine; the startup ordering fix (3b) is
  a precondition for that to be safe.
- **Live operational model (explicit decision).** When any bridge trade goes
  in transit, live execution raises `ExecutionHaltableIssue`
  (`tradeexecutor/ethereum/execution.py:754`) and the executor saves state and
  waits for an external restart (`cli/commands/start.py` halt handling). This
  is the *existing* operational contract for every live cross-chain strategy —
  demand-driven bridge trades already halt the same way — so the sweep stays
  **on by default in all execution modes** and does not add a new requirement,
  only frequency: live cross-chain deployments must (and already do) run
  under restart supervision. Operators who cannot tolerate the extra restarts
  set `sweep_idle_bridge_capital = False` or raise `bridge_sweep_min_usd` in
  `Parameters`. Document this in the planner docstring and the phase-aware
  doc update.
- **NAV during transit.** While a sweep bridge-back is in transit, the bridge
  *position's* equity excludes the locked amount
  (`position.get_equity()` subtracts `bridge_capital_allocated`,
  `tradeexecutor/state/position.py:918`), but portfolio
  `calculate_total_equity` adds it back via `get_in_transit_value()`
  (`tradeexecutor/state/portfolio.py:758`), which sums `planned_reserve` over
  `cctp_in_transit` trades — so total NAV is conserved provided bridge-back
  sells carry `planned_reserve`. Tests 1c and 2a assert total-equity
  conservation across the burn→settle window to pin this; per-chain charts
  legitimately show the amount leaving the chain band while in transit, and
  `analyse_idle_bridge_capital`'s in-transit column is the operator's
  explanation for it.
- **Churn.** If the alpha model re-targets the satellite the very next cycle,
  the sweep causes a bridge round-trip (back then out). Same-cycle demand is
  protected (`free_idle_bridge_capital` already nets out same-cycle satellite
  buys), so this only occurs across cycles, costs gas but no CCTP fee
  (burn/mint is 1:1), and is accepted: the maintainer's direction is that idle
  capital must never persist on bridges. A minimum-idle-age damping knob is a
  possible follow-up if live gas cost proves material.

## Tests

Per repo test rules: few tests, docstrings stating what/why with numbered
steps mirrored as body comments, type-hinted fixtures local to the module (no
imports from `tests/`), `pytest.approx()` for money, no stdout/log output,
5-minute pytest timeouts when running.

1. **Planner unit tests** — new `tests/ethereum/test_cctp_idle_bridge_sweep.py`,
   fixture style copied from `tests/ethereum/test_cctp_bridge_cash_aware.py`
   (SimulatedWallet + BacktestExecution + direct `inject_cctp_bridge_trades`
   calls; Arbitrum hub, Base satellite):
   - a. **Happy path:** settled idle capital on Base, *empty* input trade
     list → exactly one bridge-back injected, `closing=True`, with
     `cctp_planning_amounts` recording the full amount under `idle_sweep`;
     execute through the simulated wallet; the bridge position's available
     capital ends below `bridge_sweep_min_usd` (raw-unit/dust exception) and
     the hub reserve gains the amount.
   - b. **Guard paths (one test, several asserts):** (i) idle capital partly
     reserved by a same-cycle satellite buy → only the free remainder swept;
     (ii) capital committed to an unsettled async vault deposit
     (`bridge_capital_allocated`) → not swept; (iii) idle amount below
     `bridge_sweep_min_usd` → no trade; (iv) `sweep_idle_bridge_capital=False`
     → planner output identical to today; (v) combined case: same-cycle
     net sell **plus** extra idle capital → one merged bridge-back whose
     `cctp_planning_amounts` splits `net_sell` and `idle_sweep` correctly.
   - c. **In-transit no-double-sweep:** extend the existing in-transit
     lifecycle coverage (`tests/ethereum/test_cctp_in_transit_lifecycle.py` /
     `tests/test_cctp_in_transit_status.py`) rather than a standalone test:
     with an open `cctp_in_transit` bridge-back, the sweep sees zero
     available capital (the `mark_bridge_in_transit` allocation lock) and
     plans nothing; after retry resolution the lock is reversed and
     accounting balances; portfolio `calculate_total_equity` is conserved
     across the burn→settle window (the `get_in_transit_value` term). Also
     covers the 3b startup-ordering regression.

2. **Integration backtest test** — new
   `tests/backtest/test_cctp_idle_sweep_backtest.py`. Note: no existing
   backtest test builds a cross-chain universe through `run_backtest_inline`
   (`tests/backtest/test_cctp_backtest_sequential.py` drives
   `BacktestExecution` by hand and never touches the runner), so the fixture
   is new work: a synthetic `TradingStrategyUniverse` with `primary_chain`
   set (Arbitrum hub + Base satellite), hand-built
   `TradingPairKind.cctp_bridge` pairs (fixture shapes from
   `tests/ethereum/test_cctp_bridge_cash_aware.py`), synthetic candles/TVL and
   the YieldManager/queue-venue wiring of
   `tests/backtest/test_phase_aware_backtest.py` (hub queue venue,
   `calculate_yield_management_safe` in `decide_trades`). Running through
   `run_backtest_inline` exercises the real `StrategyRunner.tick()` planner
   gate. If universe plumbing turns out to block `run_backtest_inline`
   (e.g. candle/routing assumptions for bridge pairs), fall back to driving
   cycles manually as `test_cctp_backtest_sequential.py` does **plus** a
   focused unit test on the runner gate logic, and record the reason in the
   test docstring:
   - a. **Happy path:** cycle 1 allocates to a Base position (bridge-out +
     satellite buy); cycle 2 the signal exits the satellite position and the
     sell proceeds settle to the bridge position; a later **quiet cycle**
     (decide_trades emits no directional trades) still emits the sweep
     bridge-back (this exercises the runner gate change); final state: no
     bridge position has available capital at or above `bridge_sweep_min_usd`
     (raw-unit/dust exception), the hub queue vault holds the recovered
     capital, `analyse_idle_bridge_capital` reports nothing unswept, and
     total equity is conserved across every cycle (no sweep-created NAV
     jump).
   - b. **Bad path:** same run with `sweep_idle_bridge_capital=False` in
     `Parameters` → idle capital remains on the bridge position (reproduces
     issue #1562) and `analyse_idle_bridge_capital` classifies it
     `sweep_disabled`; also assert the no-queue-vault variant (no yield
     rules): swept cash safely accumulates as hub reserve, nothing crashes —
     this covers the issue's "absent queue vault" criterion under the
     hub-first design.

3. **Existing-test audit.** The default flip changes planner output wherever a
   fixture leaves free idle capital. Derive the audit list, do not enumerate
   it: (a) grep for direct `inject_cctp_bridge_trades` callers (currently
   `tests/ethereum/test_cctp_bridge_cash_aware.py` and
   `tests/ethereum/test_cctp_bridge_planner.py`); (b) grep tests **and any
   strategy modules/fixtures they load** for universes that set
   `primary_chain` — the runner-gate change affects those even when the test
   file itself never mentions the planner; (c) run all ~12
   `tests/**/test_cctp_*` modules plus the phase-aware suites. Where a test
   asserts leftover idle capital as *intended ledger arithmetic*, pass
   `sweep_idle_bridge_capital=False` explicitly; where the leftover was
   incidental, update expectations. No test may silently rely on the old
   leak.

## NB15 verification

After the fix, in the getting-started repo (PR #40 branch):

```shell
TQDM_LOGGABLE_FORCE=stdout poetry run jupyter-execute-agent \
  scratchpad/xchain2/15-backtest-phase-aware-hype-gains-lagoon-ipor-ember-yearn-40acres-csigma-yieldnest-plutus-no-subvaults-25pct.ipynb \
  --timeout=900
```

Acceptance: the notebook completes; the CCTP trade diagnostics show
bridge-backs with `idle_sweep` planning amounts; `analyse_idle_bridge_capital`
(added to the notebook's diagnostics section) shows no bridge position with
available capital at or above `bridge_sweep_min_usd` at the end of the run;
the queue-venue band in `equity_curve_by_chain` absorbs what used to sit
idle.

## Implementation notes (lessons from the NB15 run)

Lessons from the NB15 acceptance run and the implementation reviews:

1. **`get_available_bridge_capital()` IS the physical satellite balance.**
   Satellite buys and sells only ever mutate `bridge_capital_allocated`; the
   bridge position quantity stays at the gross bridged amount. So
   `available = quantity − allocated` tracks the physical USDC, *including*
   realised profits (allocated gone negative). A first-cut fix clamped the
   sweep to `min(available, get_quantity())` on the mistaken theory that
   quantity bounds the physical balance — that under-sweeps profits and, once
   the gross quantity is burned to zero, strands them unsweepable forever
   (caught in review). The sweep uses `available` net of same-cycle satellite
   buys and already-reserved bridge-backs, and never counts same-cycle sell
   proceeds (the demand-driven net-sell bridge-back's domain; an async sell's
   proceeds do not settle the same cycle) — any idle surplus a sync sell
   displaces is swept on the next quiet cycle.

2. **Full-balance sweeps hit backtest state/wallet rounding drift.** Bridging a
   chain's *entire* settled balance lands exactly on the boundary where the
   state's tracked balance and the simulated wallet balance differ by a
   fraction of a raw unit (accumulated `Decimal` rounding); NB15 crashed with
   `OutOfSimulatedBalance` at −2.2e-7 USDC. `simulate_bridge` now clamps the
   burned side to the wallet balance within a two-raw-unit dust bound (via
   `fix_sell_token_amount`, the spot-sell precedent) — a clamp rather than an
   `update_balance` epsilon, so tiny legitimate one-raw-unit burns still
   execute and a genuine over-burn beyond the bound still raises. A
   backtest-simulation fix, not a live-execution change; demand-driven
   bridge-backs never hit this because they are bounded by trade size.

3. **A positive `bridge_capital_allocated` is not an async-deposit marker.**
   Every satellite buy — synchronous included — allocates bridge capital for
   the position's lifetime, so diagnostics must not read `allocated > 0` as
   "reserved for an unsettled deposit". `analyse_idle_bridge_capital` gates
   the `reserved_for_async_deposit` reason on an actual
   `vault_settlement_pending` buy on that chain.

## Out of scope / follow-ups

- Satellite-side queue vaults and a chain-aware `YieldRuleset`/`YieldManager`
  (the issue's literal `sweep_bridge_cash_to_queue`) — future work; the
  reason-code diagnostics are designed to extend to it.
- Minimum-idle-age damping to reduce live bridge churn.
- Tracking sweep totals separately from `total_bridge_back` so live planning
  can exclude them from same-cycle `fundable_primary` (the existing
  bridge-back live-settlement optimism, `planner.py:505`, is unchanged by
  this plan; a conservative live refinement flagged in review).
- The live `reclaim_satellites` CLI command remains as manual recovery for
  balances outside portfolio state; unchanged.

## Execution order

1. Planner sweep phase + planning-amount tagging (+ unit tests 1a–1b).
2. Live startup ordering fix in `loop.py` (3b) + in-transit lifecycle test
   extension (1c).
3. Runner gate + parameter threading.
4. Integration backtest tests 2a–2b.
5. Existing-test audit sweep (test group 3).
6. Diagnostics (`reasons` column, `analyse_idle_bridge_capital`).
7. Docs updates.
8. NB15 run and acceptance check.
9. PR prep: `feat:` title, one-line `CHANGELOG.md` entry (dated), PR
   description with Why / Lessons learnt / Summary sections.
