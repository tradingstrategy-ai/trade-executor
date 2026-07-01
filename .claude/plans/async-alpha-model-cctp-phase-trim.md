# Fix CCTP bridging-in-wrong-phase: execute bridges in correct order

Branch: `async-alpha-model`

## Why

Cross-chain rebalances fail (`NotEnoughMoney` from the CCTP planner) because the
CCTP bridge trades execute in the **wrong phase** of the cycle. Trades run in the
sorted order of `Trade.get_execution_sort_position()` (`trade.py:1358`). Today:

```
credit withdraw     -trade_id - 200M
closes              -trade_id - 100M
vault withdrawals   -trade_id -  50M
BRIDGE-BACKS        -trade_id -  30M   <- phase 4  (WRONG: before sells)
spot sells          -trade_id           <- phase 5
buys                +trade_id           <- phase 6
BRIDGE-OUTS         +trade_id +  30M   <- phase 7  (WRONG: after buys)
vault deposits      +trade_id +  50M
credit supply       +trade_id + 200M
```

Two ordering faults:

- **Bridge-backs run before spot sells** (`trade.py:1392`). A satellite chain's
  spot-sell proceeds do not exist yet when the bridge-back fires, so they cannot
  be carried to the primary chain this cycle. The planner compensates by
  capping bridge-backs to `free_idle_bridge_capital` (idle only) and reports a
  primary-chain shortfall.
- **Bridge-outs run after buys** (`trade.py:1395`). Primary-chain buys drain the
  reserve before the bridge-out fires, so `fundable_primary` is computed as
  `reserve + primary_sells + total_bridge_back − primary_buys` (`planner.py:480`)
  and can go short.

The shortfall is therefore an **artifact of the execution order**, not a real
cash deficiency: total deployable cash is correct (the alpha model already caps
buys to it). The planner currently raises `NotEnoughMoney`
(`planner.py:412`, `planner.py:533`).

**The fix is to bridge in the correct order — there is nothing to trim.** CCTP is
same-cycle (bridges sort between sells and buys by design, `trade.py:1389`), so
if every sell releases cash *before* the bridges reposition it, and the bridges
complete *before* the buys spend it, every buy is funded and no shortfall arises.

## Fix

### Correct phase order (target)

```
credit withdraw
closes
vault withdrawals
spot/primary SELLS          <- release all cash first
BRIDGE-BACKS  (sat->primary) <- carry idle + same-cycle sell proceeds to hub
BRIDGE-OUTS   (primary->sat)  <- fund satellite buys, before any buy spends
spot/primary BUYS            <- all cash now positioned on the right chain
vault deposits
credit supply
```

i.e. `sells → bridge-backs → bridge-outs → buys`.

### Change 1 — `tradeexecutor/state/trade.py` `get_execution_sort_position()`

Open a clean band between sells and buys for the two bridge sub-phases. Proposed
constants (trade_id stays small vs the bumps, as today):

- The `is_reduce` branch (`trade.py:1408`) — covers **all** non-credit,
  non-vault, non-closing reduces: move from `-trade_id` into their own band
  **before** bridge-backs, e.g. `-trade_id - SPOT_SELL_ORDER_BUMP` with
  `SPOT_SELL_ORDER_BUMP = 40_000_000` (sits between vault withdrawals −50M and
  bridge-backs −30M). Per Codex, this branch is what carries both ordinary spot
  sells **and non-closing leveraged long reduces** — both feed the CCTP ledger as
  satellite sells, so both must land in the −40M band. Reduce types that do *not*
  go through this branch and their handling (so the invariant is airtight):
  short reduces are `planned_quantity > 0` → the planner treats them as *buys*,
  not sells (`planner.py:306-310`); vault withdrawals (−50M) and closes (−100M)
  already sort earlier; async vault sells and HyperCore vaults are excluded before
  the ledger (`planner.py:287-299`); a credit sell lacking the close/open flag
  would fall through to `+trade_id` (`trade.py:1417-1422`) — it must never reach
  the CCTP ledger, and the follow-up invariant assert (see Simplifications)
  enforces that.
- Bridge-backs (`trade.py:1392`): keep `-trade_id - 30M`; now correctly **after**
  the −40M sells.
- Bridge-outs (`trade.py:1395`): move from `+trade_id + 30M` to
  `-trade_id - CCTP_BRIDGE_OUT_ORDER_BUMP` with
  `CCTP_BRIDGE_OUT_ORDER_BUMP = 20_000_000` — **after** bridge-backs (−30M),
  **before** buys (~0).

Resulting order: `−50M vault-withdraw < −40M sells < −30M bridge-back <
−20M bridge-out < ~0 buys < +50M vault-deposit`. Update the inline comments.

Band assumption (unchanged from today): the 10M gaps assume `trade_id` stays well
below 10M within a cycle. This is inherent to the existing bump design (credit
±200M, close ±100M, vault ±50M all rely on it); the reorder does not weaken it.
Codex-approved as non-blocking.

Note: moving all counted satellite reduces below −30M makes
`_trade_releases_before_bridge_back()` (`planner.py:175`, tests
`pos < -CCTP_BRIDGE_ORDER_BUMP`) return True for every satellite spot sell and
long reduce — the corrected semantics (their proceeds are available to
bridge-backs). This wires the Change 2 accounting for free; the follow-up
refactor then collapses the now-always-true branch behind an assert.

### Change 2 — `tradeexecutor/ethereum/cctp/planner.py` accounting follows the order

With sells now releasing before bridge-backs and bridge-outs before buys:

- `early_satellite_sells` now includes same-cycle satellite spot sells (via the
  helper above), so `free_idle_bridge_capital` (`planner.py:80`) lets a
  bridge-back carry sell proceeds, not just idle USDC. The async-deposit
  starvation cap on bridge-back amount (`planner.py:342-352`) stays — that is a
  separate, still-valid constraint. Update the now-wrong old-order comment at
  `planner.py:348-350`.
- **`fundable_primary` keeps the `− primary_buys` term** (`planner.py:480`).
  Codex review correction: this is a *conservation* constraint, not an ordering
  one. Cash bridged out leaves the primary chain for the rest of the cycle and
  cannot fund primary buys regardless of execution order, so the cap must still
  reserve room for primary buys. The win from moving bridge-outs **before** buys
  is that freshly-bridged cash is now on the satellite in time to fund
  *same-cycle satellite buys* (previously the satellite buy at ~0 ran before the
  +30M bridge-out and could only use pre-existing idle capital). The cap *amount*
  is unchanged.
- The two `NotEnoughMoney` raises (`planner.py:412`, `planner.py:533`) **stay as
  genuine guards** — the reorder removes only the *ordering artifact*, not real
  shortages. They still (correctly) fire for: total primary cash genuinely
  insufficient, a per-chain bridge-out exceeding `fundable_primary`, async vault
  sells excluded from same-cycle funding (`planner.py:180-193, 292-299`), and
  in-flight satellite deposits reserving bridge capital (`planner.py:77-83`).
  Action: confirm the notebook failure was the *ordering-artifact* case (fixed by
  the reorder) and not one of these real-shortage cases.

No trimming, no buy resizing, no new diagnostic flags.

### Verify execution honours the new order (both engines)

- **Backtest:** `BacktestExecution` currently has a CCTP-specific re-sort that
  promotes bridge-outs to key `0` (`backtest_execution.py:741-762`). This
  workaround is **removed** (see B) so the backtest executes trades in the exact
  global `get_execution_sort_position()` order — identical to live. After the
  reorder, `withdrawals → sells → bridge-back → bridge-out → buys` holds without
  any special-casing, so bridge positions are funded before the satellite buys
  that allocate from them.
- **Live:** live CCTP is *not* unconditionally same-cycle. The Ethereum executor
  halts remaining trades when a bridge is `cctp_in_transit`
  (`execution.py:744-764`), and the planner already flags `+ total_bridge_back`
  as optimistic live (`planner.py:472-479`). The reorder makes the *backtest /
  notebook* correct and is still the right shape for live, but live funding
  ultimately falls back to halt-and-resume across cycles — call this out, don't
  claim the reorder makes live single-cycle.

### Blast radius — full codebase impact map

Verified by four parallel codebase sweeps. Categorised by action.

**A. Current vs target order (the actual bug).** With `trade_id=5`: bridge-back
`-30,000,005` < spot sell `-5` < spot buy `+5` < bridge-out `+30,000,005`, i.e.
execution runs **bridge-back → sell → buy → bridge-out** today. Target:
**sell → bridge-back → bridge-out → buy**. Note: the *backtest* already
half-corrects this (see C) — the dominant uncorrected fault in the
backtest/notebook is **bridge-backs running before sells**, so satellite sell
proceeds can't be bridged back. Live is uncorrected on both sides.

**B. MUST change (code):**
- `tradeexecutor/state/trade.py:30,1386-1422` — `get_execution_sort_position()`
  bands + bump constants; inline comments (`:1391`, `:1394`).
- `tradeexecutor/backtest/backtest_execution.py:741-762` — **DELETE the
  `_sequential_sort_key()` workaround entirely.** It only exists to compensate
  for the old global sort (bridge-outs at `+30M`, after buys). With the corrected
  global sort the plain `get_execution_sort_position()` already yields the right
  sequential order (`withdrawals −50M → sells −40M → bridge-back −30M →
  bridge-out −20M → buys ~0 → deposits +50M`). Replace the whole block with:
  ```python
  trades = sorted(trades, key=lambda t: t.get_execution_sort_position())
  ```
  Principle: backtest and live must use the **same** global ordering — no
  backtest-specific bridge special-casing. Also drop the old-order comment block
  (`:741-748`). Verify no other backtest path re-sorts or special-cases bridges.

**C. MUST review (order-coupled logic — Codex + sweeps flagged these):**
- `tradeexecutor/ethereum/cctp/planner.py:37-89` — `ChainLiquidity` properties
  (`available_before_bridge_back`, `satellite_buy_bridge_back_time_need`,
  `free_idle_bridge_capital`) encode the *old* sell↔bridge-back timing. Re-verify
  each once spot sells move below `-30M`.
- `tradeexecutor/ethereum/cctp/planner.py:175-177` — `_trade_releases_before_bridge_back()`
  now returns True for all spot sells (intended); update its comment + the
  stale comments at `:348-350`, `:472-479`.
- `tradeexecutor/ethereum/cctp/planner.py:480` — keep the `− primary_buys`
  conservation term (do **not** drop). Keep the `NotEnoughMoney` guards
  (`:412`, `:533`) — real-shortage backstops only.
- `tradeexecutor/state/state.py:784-806` — `start_execution()` capital routing:
  satellite spot/vault buys draw from the bridge position **if it exists**, else
  reserves. This assumes bridge-outs have funded the bridge position before the
  satellite buy — re-validate under the new order.
- `tradeexecutor/state/state.py:1003-1013` (satellite-sell → bridge-position
  return) and `:1108-1133` (`mark_bridge_in_transit` locking bridge-back
  capital) — verify bridge positions still exist at the right moment.

**D. Auto-inherits, no change (just confirm):**
- Sort call sites that read `get_execution_sort_position()` and apply it:
  `strategy/runner.py:1401-1404` (`prepare_sorted_trades`, also sets
  `trade.sort_index`), `alpha_model.py:2321`,
  `pandas_trader/rebalance.py:140`, `qstrader/portfolio_construction_model.py:212`.
- Injection call site `strategy/runner.py:978` — bridges injected *before* the
  sort, so the reorder applies to them automatically.
- Live sequential caller `ethereum/execution.py:791` → `:707` relies on
  **pre-sorted** input (it does not re-sort); with normal strategy flow sorting
  at `runner.py:1401`, bridge-outs at `-20M` arrive before satellite buys and
  `start_execution()` funds them correctly (`state.py:784`, `:797`), with the
  in-transit halt (`execution.py:744`) as the safety valve.
- `exchange_account/allocation.py:187` (`get_in_transit_value` exclusion) —
  cross-cycle only, unaffected by same-cycle order.

**E. Tests to update / run:**
- Hardcoded sort ints — `tests/test_cctp_in_transit_status.py`: `:182` bridge-out
  `+30M` **will fail → update to the new band**; `:202`, `:208` bridge-back
  `-30M` **still pass** (unchanged).
- `tests/units_tests/test_trade_execution_sort.py:77-99` — add CCTP bridge cases
  to the tier-ordering test.
- Behaviour (no hardcoded ints, must be re-run, may need assertion tweaks):
  `tests/ethereum/test_cctp_bridge_cash_aware.py` (largest risk),
  `tests/ethereum/test_cctp_bridge_planner.py`,
  `tests/backtest/test_cctp_backtest_sequential.py` (comments assert
  bridge-out-before-spot).

**F. Docs to refresh:**
- `.claude/cctp-roadmap.md:27` (names the `-30M/+30M` bands),
  `.claude/docs/cctp-async-vault-accounting-plan.md` (review),
  `.claude/docs/vault-deposit-redeem.md` (review if it states bridge order).

**G. Extra acceptance candidate already in-repo:**
- `strategies/test_only/159-backtest-xchain-master-vault-cctp.ipynb` — existing
  CCTP backtest notebook; run alongside the capped-waterfall one as a second
  regression.

**H. Residual risk — ordering depends on the caller.** Live `execute_trades()`
does not sort internally; it trusts pre-sorted input. Normal flow sorts at
`runner.py:1401`, but any direct CLI/manual/test caller of `execute_trades()`
that bypasses `prepare_sorted_trades()` would rely on its own trade order —
confirm none feed CCTP batches unsorted. (Codex review, 2026-07-01.)

## Simplifications enabled by the correct ordering

The `ChainLiquidity` ledger (`planner.py:37-88`) exists almost entirely to model
the *old* hazard that bridge-backs could run **before** some satellite sells, so
sells had to be split into "early" (release before bridge-back) vs "late". Once
the sort guarantees every non-async satellite reduce sorts below `-30M`
(spot sell `-40M`, vault withdraw `-50M`, close `-100M`) and async sells are
already excluded (`planner.py:293-299`), line 304 is **always** true, so:

```
early_satellite_sells == satellite_sells_before_buy   (always)
```

That makes several members degenerate — this is dead machinery to remove, not
just comments to fix:

- `satellite_sells_after_bridge_back_before_buy` (`:54-57`) → `max(x − x, 0)` =
  **always 0**. Remove the property and its only consumer.
- `available_before_bridge_back` (`:49-52`) becomes identical to
  `available_before_buy` (`:59-62`). **Merge into one** `satellite_side_available`
  property.
- `satellite_buy_bridge_back_time_need` (`:64-70`) reduces to `satellite_buys`.
  Inline it into `prepare()` → `free_idle_bridge_capital = max(available − buys, 0)`.
- `early_satellite_sells` field (`:44`) → removable (equals
  `satellite_sells_before_buy`); the `if _trade_releases_before_bridge_back(...)`
  branch at `:304-305` collapses to a single unconditional add.
- `_trade_releases_before_bridge_back()` helper (`:175-177`) → no longer needed
  for branching, but **repurpose it into a warn-and-skip guard** in the
  sell-classification loop: a satellite sell that does not sort below
  `-CCTP_BRIDGE_ORDER_BUMP` is skipped (and logged) rather than counted. This is
  the correctness guard for the `early == all` collapse — everything counted is
  early. (Implemented as warn+skip, not a hard `assert`, after review: an
  uncommon-but-legitimate shape that is `is_sell()` yet does not release spot cash
  before bridge-backs — a short-position increase, or a zero-quantity repair sell
  — would fall through to `+trade_id` (`trade.py:1417-1422`). `master` tolerated
  these; a hard assert would turn them into a crash, so we skip+warn instead,
  which is strictly safer than both the assert and `master`.)

Codex sanity check (2026-07-01) confirmed the removals are self-contained:
`satellite_sells_after_bridge_back_before_buy`, `available_before_bridge_back`, and
`early_satellite_sells` are each read only by the properties / `prepare()` /
bridge-back assert inside `planner.py`; nothing outside the module imports
`ChainLiquidity`. Caveat it raised: `early == all` is **false until the reorder
lands** (ordinary reduces still sort at `-trade_id` today), so this refactor must
strictly follow the reorder — never precede it.

Net: `ChainLiquidity` drops from 7 fields + 5 derived properties to ~4 fields +
2 properties (`satellite_side_available`, `satellite_bridge_shortfall`, with
`free_idle_bridge_capital` as their complement). `free_idle_bridge_capital` and
`satellite_bridge_shortfall` become the two sides of `available − satellite_buys`.

Plus the already-listed deletions:
- `backtest_execution.py:741-762` `_sequential_sort_key()` — deleted (Fix §B).
- Stale old-order comments in `planner.py` (`:224`, `:348-350`, `:472-479`).

**Not** simplified (leave as-is): the `primary_shortfall` idle-bridge-back top-up
(`planner.py:365-421`) still handles the distinct case where deployable capital
exists only as idle satellite USDC with no satellite net sell to trigger a
bridge-back; and the `− primary_buys` conservation term + `NotEnoughMoney` guards
stay. Do these simplifications in a **separate follow-up commit** after the
behavioural reorder + tests are green, so the diff that changes behaviour stays
small and the refactor is independently reviewable.

## Test 1 — manual notebook run (acceptance)

Prove the real capped-waterfall notebook from getting-started PR #34 runs
end-to-end on the fixed code.

1. Download the notebook from the merged branch into the repo's notebook test
   convention (`strategies/test_only/`):
   ```shell
   gh api \
     repos/tradingstrategy-ai/getting-started/contents/scratchpad/xchain2/08-backtest-capped-waterfall.ipynb?ref=vault-capped-waterfall-universe \
     --jq '.content' | base64 -d \
     > strategies/test_only/08-backtest-capped-waterfall.ipynb
   ```
2. Adapt only what is required to run inside trade-executor's test env (data /
   universe cache paths, any getting-started-relative imports). Keep strategy
   logic untouched.
3. Run it:
   ```shell
   source .local-test.env && poetry run jupyter execute \
     strategies/test_only/08-backtest-capped-waterfall.ipynb --inplace --timeout=900
   ```
4. Add a smoke test modelled on `tests/test_hyperliquid_waterfall_notebook.py` →
   `tests/test_capped_waterfall_notebook.py`: copy notebook to `tmp_path`,
   `jupyter execute --inplace`, assert return code 0, extract failing-cell
   diagnostics on error, assert outputs exist.
5. Expected: fails today with `NotEnoughMoney` from the CCTP planner; passes
   after the reorder.

## Test 2 — minimal integration test (regression)

A fast test that reproduces the exact phase-ordering failure without the full
notebook. Fixtures modelled on `tests/backtest/test_cctp_backtest_sequential.py`
(Arbitrum primary + Base satellite; synthetic USDC/WETH/CCTP pairs).

New file: `tests/backtest/test_cctp_bridge_phase_order.py`.

Construct a one-cycle trade set that only funds if bridges run in the correct
order:

1. A satellite (Base) spot **sell** whose proceeds are needed on the **primary**
   chain to fund a primary-chain buy this cycle.
2. A primary-chain buy sized so it is only fundable once the satellite sell
   proceeds have been bridged back — i.e. `current primary reserve` alone is
   short, but `reserve + bridged-back satellite sell` covers it.
3. (Optional second leg) a satellite buy that needs a bridge-out, to exercise the
   bridge-out-before-buy path.

Assertions:
- **Ordering:** `bridge_back.get_execution_sort_position()` is greater than the
  satellite spot sell's and less than the bridge-out's; the bridge-out's is less
  than every buy's. (Locks the corrected phase order.)
- **Bug guard:** with the *old* sort constants the same scenario raised
  `NotEnoughMoney` (assert against the pre-fix behaviour in a focused check, or
  document it in the docstring) — proving the test reproduces the original bug.
- **After fix:** `inject_cctp_bridge_trades` returns injected bridge trades with
  no exception; the bridge-back amount equals the satellite sell proceeds (within
  dust); `fundable_primary >= 0` and the primary buy is fully funded — no buy was
  resized.
- **End-to-end (optional):** run the trades through `BacktestExecution` and
  assert every trade `is_success()` and reserves reconcile.

Single happy-path + single bad-path, multiple asserts, full docstring with
numbered steps mirrored as inline comments, type-hinted fixtures, no stdout — per
repo pytest rules.

Run:
```shell
source .local-test.env && PYTHONPATH="$(pwd):$PYTHONPATH" poetry run pytest \
  tests/backtest/test_cctp_bridge_phase_order.py --log-cli-level=info
```

## Lessons learnt (to record in PR)

- CCTP is same-cycle by design: bridge trades sort *between* sells and buys
  (`trade.py:1389-1395`). The bug was purely that bridge-backs sorted *before*
  sells and bridge-outs *after* buys, so cash was moved at the wrong time.
- The primary-chain "shortfall" the planner raised on was an artifact of that
  ordering, not a real cash deficit — so the correct fix is to reorder the
  bridges, not to trim or resize any buys.
- The planner's per-chain accounting was modelled around the old order; once the
  sort is fixed, the existing `_trade_releases_before_bridge_back` helper folds
  satellite sell proceeds into bridge-back funding automatically.
- `fundable_primary` must still subtract primary buys — that term is conservation
  (bridged-out cash leaves the hub), not ordering. The reorder fixes *which* cash
  is available *when*; it does not change the total-cash bookkeeping, so the
  `NotEnoughMoney` guards stay valid for genuinely insufficient cash, async
  exclusions, and in-flight-deposit reservations.
- Live CCTP can settle across cycles (executor halts on `cctp_in_transit`); the
  reorder is a backtest/notebook correctness fix and the right live shape, but
  live still relies on halt-and-resume.
- The backtest's `_sequential_sort_key()` override (bridge-outs → key `0`) was a
  band-aid over the wrong global sort, and it made backtest and live disagree on
  the bridge-out phase. Fixing the global order lets us delete the workaround —
  execution order should have a single source of truth
  (`get_execution_sort_position()`) shared by both engines.

## Summary (what will change)

- Reorder CCTP bridge trades in `get_execution_sort_position()` so sells →
  bridge-backs → bridge-outs → buys.
- Refresh CCTP planner comments to the corrected order; keep `fundable_primary`
  and the `NotEnoughMoney` guards (they fire only on real shortages). The
  reorder removes the ordering artifact that caused the notebook crash.
- Delete the backtest-only CCTP sort workaround so backtest and live share one
  ordering, and update all hardcoded sort-order test expectations.
- New notebook acceptance test + copied capped-waterfall notebook.
- New minimal CCTP integration test that reproduces the phase-ordering failure.

## Implementation outcome — the reorder was necessary but NOT sufficient

Running the real capped-waterfall notebook after the reorder still raised
`NotEnoughMoney`, short by **$3.83**. Instrumented diagnosis (failing cycle
2026-01-05, primary chain Ethereum, satellites Base + Arbitrum):

- The reorder DID its job: it carried Arbitrum's synchronous $1,353 sell back to
  the hub, shrinking the shortfall from **$1,357 → $3.83**. So the ordering
  component is fixed.
- The residual $3.83 is a genuine **~0.0225% over-allocation**, not an ordering
  or accounting bug. Its precise source: `AlphaModel._cap_buys_by_async_sell_proceeds`
  scales buys to `cash + sync_sell_usd`, but `sync_sell_usd` is **mark-to-market**
  (`position_adjust_usd`) while execution realises slightly less (fees, price,
  raw-unit rounding, CCTP bridge floors). Scaling to 100% of the mark budget
  over-commits by the mark-vs-realisable gap.
- The strategy's `cross_chain_cash_buffer_usd` and `allocation_pct` are applied
  only to the **target value** (`deployable_target_value`) in `decide_trades`,
  not to this same-cycle cash cap — so they leave zero same-cycle headroom when
  the cap binds during an async-redemption rebalance.

### Fix added (Codex-reviewed and agreed)

Give `_cap_buys_by_async_sell_proceeds` a same-cycle cash buffer:
`budget = max(cash + sync_sell_usd - same_cycle_cash_buffer_usd, 0.0)`.

- Threaded as a kwarg on `generate_rebalance_trades_and_triggers(...,
  same_cycle_cash_buffer_usd=0.0)` → `_cap_buys_by_async_sell_proceeds`.
- Only bites when async proceeds constrain the cycle (the method already returns
  early when `async_sell_usd <= 0`), so no permanent under-deployment; `0.0`
  preserves prior behaviour.
- The capped-waterfall notebook passes it from `cross_chain_cash_buffer_usd`.
- The CCTP planner `NotEnoughMoney` raise stays as the final invariant backstop —
  no planner trim.
- New regression: `tests/backtest/test_backtest_async_vault.py::test_backtest_async_vault_same_cycle_cash_buffer`.

**Result:** the capped-waterfall notebook now runs end-to-end (23/23 cells, no
`NotEnoughMoney`). Full affected suite green (49 tests).
