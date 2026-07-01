# PhaseAwareAlphaModel — decouple allocation from deposit via a yield-bearing queue venue

Branch: `phase-aware-alpha-model` (worktree `/home/mikko/code/trade-executor-phase-aware-alpha`, off `master` @ `67b65935`).

Status: **design plan**, nothing implemented. This is a clean rewrite; it folds in three review rounds
(Codex ×2 + Opus) and three follow-up refinements (drop the resolver, be allocation-method-agnostic,
refactor shared functions for reuse). Provenance is in **Review history** at the end. File anchors are
from `master` — verify at implementation time (they drift).

---

## Why

The capped-waterfall cross-chain vault backtest (`strategies/test_only/08-backtest-capped-waterfall.ipynb`,
getting-started PR #34/#35) runs end-to-end after the CCTP fix (#1549) but **leaks undeployed cash**:
idle USDC climbs to **24.0% of equity** by the last cycle vs the **~10%** implied by
`allocation_pct = 0.90` ([analysis](https://github.com/tradingstrategy-ai/getting-started/pull/35#issuecomment-4854703611)).

Idle cash is four flavours of *capital decided-for but not placeable this cycle*, all falling back to
**0%-yield USDC**:

1. **Async redemption-delay pinning.** `carry_forward_non_redeemable_positions()`
   (`alpha_model.py:815-887`) pins any position reporting `can_redeem=False` or with an in-flight
   settlement and subtracts it from the deployable target, so the strategy **can't rotate out of a
   laggard** while its redemption is delayed. Evidence: 157/344 cycles carried a pin across 10 vaults;
   `settlement_pending` ×273, `cannot_redeem` ×29. *Plutus Hedge Token* alone: **$249,497** of redemptions
   attempted over 28 cycles, blocked every time.
2. **Same-cycle sync-only financing.** `_cap_buys_by_async_sell_proceeds()` (`alpha_model.py:922-998`)
   scales buys down to `cash + sync_sell_usd`; an async vault's proceeds arrive days later, so the
   withheld capital sits in reserve until settlement.
3. **Capacity overflow.** The size/pool caps (`max_concentration_pct=0.12`, `per_position_cap_of_pool_pct=0.20`)
   leave capital that can't be placed into shallow top-CAGR vaults undeployed (~$18k at the final cycle).
4. **Structural slack.** The `(1−allocation_pct)` slice (~10%) + `cross_chain_cash_buffer_usd` are never
   targeted, by design.

Final cycle: equity **$106,903**, idle **$25,664 (24%)**, invested **$84,698** — of which only **$602** is
physically pending; the rest is cumulative churn withheld across settlement windows. 15/20 slots used.

Separately, the product motivation: **D2 (HYPE++)**, **Gains**, **Ostium** open for deposits/redemptions
only a few days per month, misaligned with the 1-day cycle. Today the model **skips** a closed-window
deposit (`_should_skip_signal_rebalance`, `:1904-1915` → `cannot_deposit` + `missed_deposit_usd`), so the
intended allocation reverts to idle cash.

**Both share one root:** allocation intent is computed every cycle, but *execution* is gated (window,
settlement, capacity), and the in-between capital is unproductive. `PhaseAwareAlphaModel` decouples the
two and routes otherwise-idle capital into a yield-bearing **queue venue** that earns while it waits.

---

## Goal & scope

**Fixes.** Idle 0%-yield cash → a yield-bearing venue. With `position_allocation = allocation_pct`,
YieldManager keeps the required `equity·(1−allocation_pct)` (~10%) as reserve cash and **sweeps the
*excess* idle above it** (the capacity-overflow + settled-but-not-redeployed proceeds that pushed idle to
24%) into the venue, returning idle to the ~10% floor. (Earning yield on the ~10% reserve too needs a
higher `position_allocation` with the zero-release handling — a tuning follow-up.)

**Enables.** Allocating to window-gated vaults (D2/Gains/Ostium) by *deferring then depositing* when they
open — the original ask.

**Does NOT change.** The pinning of genuinely non-redeemable positions (we still can't force an early
Plutus redeem); async settlement delays; the capacity caps. It gives *waiting/freed* capital a productive
home and lets allocation intent persist across windows — it does not shorten any queue.

**Locked decisions.**
- **Both directions:** deposit is *active* (defer → deposit-on-open); redemption is *passive* — reuse the existing pin, only **claimed**
  proceeds swept into the venue.
- **Poll `can_deposit()` / `check_redemption()` each cycle** to detect an open window; `deposit_next_open`
  is ETA/diagnostics only. Correctness scope: backtest (`vault_state`) + Hypercore; live generic ERC-4626
  windowing is a follow-up (see Openness detection).
- **Venue = a synchronous ERC-4626 / Aave money market**, configured as a `YieldManager` `YieldRuleset`
  (no bespoke resolver — see Architecture).
- **Allocation-method-agnostic:** works with any `normalise_weights` variant, not just waterfall.

---

## Architecture — one owner per concern

The load-bearing decision (an earlier draft had two components owning the same cash):

- **`YieldManager` owns the cash home.** *Every* queue-venue trade — sweeping idle cash **in**, releasing
  cash **out** to fund directional buys — is generated by the existing
  `tradeexecutor/strategy/pandas_trader/yield_manager.py`, called after the alpha model in the same
  `decide_trades` (the established two-step pattern: `trades += YieldManager.calculate_yield_management(...).trades`).
  **`PhaseAwareAlphaModel` never emits a queue-venue trade.**
- **`PhaseAwareAlphaModel` owns the intent.** It (a) **defers** a directional buy into a window-closed
  vault instead of skipping it, (b) appends a **park/promote event** to a durable log, (c) **re-emits**
  the buy when the window opens, and (d) **tags** venue positions so charts/accounting separate them.

Net: idle reserve becomes a yield-bearing venue position; the alpha model decides *what* to deploy and
*when*, YieldManager moves the cash. They interact numerically in exactly one place — the same-cycle cash
cap (below).

**Venue configuration = `YieldRuleset`, not a resolver.** The venue set is declared by the strategy's
`create_yield_rules(parameters, strategy_universe) → YieldRuleset` (YieldManager's existing convention):
a waterfall of `YieldWeightingRule`s, the last (`max_concentration=1.0`) being the instant-liquidity
fallback. This *is* the "configurable function, evolvable to per-chain" the earlier decision wanted. (The
current `YieldRuleset` is a *global ordered* waterfall, not target-chain-aware — `yield_manager.py:426,448`
— so per-chain venues need chain-matching added in **YieldManager**, a follow-up; still no alpha-model
resolver.) The alpha model needs neither to choose nor to trade the venue; it only needs to **identify** venue positions
(the `is_queue_vault` tag / `YieldRuleset` membership) and an **opt-in** (it is the phase-aware subclass
with a YieldRuleset wired; otherwise it degrades to the base `AlphaModel` skip behaviour).

**Allocation-method-agnostic.** `PhaseAwareAlphaModel(AlphaModel)` inherits every normalisation variant —
`_normalise_weights_simple` (`:1046`), `_normalise_weights_size_risk` (`:1074`),
`_normalise_weights_size_risk_positions` (`:1195`), `_normalise_weights_waterfall` (`:1335`) — and the
phase-aware pass runs *after* whichever the strategy picked, on the computed `self.signals`. It does not
care how targets were produced. The only requirement is that normalisation is **window-agnostic** — it
does not filter a vault out *because its window is closed*. (It may still zero a low-ranked vault via the
max-position / size-risk caps — that vault then gets no target and nothing to park, which is correct.)
Every current method is window-agnostic; see the slot-holding invariant below.

---

## The mechanism

### Phase 1 — park (defer + log)

For each target vault `V` whose deposit **cannot execute this cycle because its window is closed**
(`can_deposit(V)` false) — *not* because it is capacity-capped (see note), and **distinct from an
in-flight settlement** (already handled by the settlement pin, `alpha_model.py:843`, `:2034` — do not
re-park):

- **Defer** (don't skip, today's `:1904-1915`): zero `V`'s same-cycle `position_adjust_usd`.
- **Log a park event** to the durable event log (`state.other_data`: target vault, USD, cycle) and flag
  the `V` signal `parked_in_queue_vault` (a per-cycle diagnostic). The event log is the **single durable
  structure** — it drives the waiting-deposit chart and lets Phase 2 detect a promotion. It is **not** a
  cash-withholding earmark.
- The unspent cash stays reserve and is **swept into the venue by YieldManager**, earning yield while `V`
  is closed.

**Slot-holding invariant (why no earmark is needed, method-agnostic).** Normalisation runs *before* the
phase-aware pass and is window-agnostic: whenever `V` earns a **positive post-normalisation target** (it
ranks high enough to survive the max-position / size-risk caps, regardless of window), the other vaults get
their own targets *in the same pass*. Deferring `V`'s buy therefore leaves *V's* target-cash unspent
(→ venue) without reallocating it to anyone else. When `V` opens *and still holds a positive target that
cycle*, the buy is funded from the venue (a stale park event whose vault no longer earns a target is
closed, not promoted — see Phase 2). Holds for simple / size-risk / waterfall alike. **The one thing that
would break it:** a method that re-normalised over *open vaults only* — none do; if one is added, it must
keep closed vaults in the target set.

**Note — capacity and structural slack are not parked.** Backtest `can_deposit()` only gates
closed/zero-cap vaults; `get_max_deposit()` is *reported, not enforced* (`backtest_pricing.py:606,630`);
a size/pool cap just yields a smaller *accepted target*. Capacity overflow is generic residual YieldManager
sweeps into the venue; the `(1−allocation_pct)` slack is the **retained reserve cash** (not swept unless
`position_allocation` is raised). Neither is parked to a specific vault.

### Phase 2 — deposit-on-open (promote)

At cycle start, for each vault with an **open park event**, poll `can_deposit(V)`:

- **Open + still targeted:** if `can_deposit(V)` **and** `V` earns a **positive target this cycle**,
  **re-emit `V`'s directional buy** (size to `min(target, get_max_deposit(V))` if a hard cap applies); log
  a **promote event** (closing the park event) and flag `promoted_from_queue_vault`. YieldManager releases
  venue cash to fund it (same-cycle, venue is sync). Async `V` simply *requests* and settles later — the
  request debits the freed cash now. Also require `V` is not settlement-pending (else the second guard at
  `alpha_model.py:2034-2044` zeros the re-emit).
- **Open but no longer targeted (stale event):** `V` fell out of the ranked set this cycle (CAGR dropped,
  or a cap zeroed it) — **close the park event without promoting**. Its cash stays in the venue and simply
  re-competes; nothing special to do.
- **Closed:** leave the park event open; retry next cycle.

### Redemption side — mark → claim, proceeds stay productive

**This side is passive** — `apply_phase_aware_intent` adds no redemption logic. A window-gated/async redeem
is handled by the existing settlement pin (`carry_forward_non_redeemable_positions`, which sets
`carry_forward_position=True` and excludes the position from `_get_allocatable_signals` `:811-813`, so it
never earns a fresh target and cannot collide with a park). The only net-new behaviour is sweeping
**claimed** proceeds.

- **Mark:** request the redemption (existing path). Async `V` lands in `vault_settlement_pending`
  (`state/trade.py:126`, `is_vault_settlement_in_flight()` `:1064-1088`); a window-gated sync vault retries
  until `check_redemption()` opens. `carry_forward_non_redeemable_positions` pins it meanwhile.
- **Claim:** settled proceeds land as reserve and are **swept into the venue by YieldManager**, not left
  idle — the change that attacks the "cumulative churn withheld across settlement windows."

### The sweep (YieldManager)

`calculate_yield_management(input)` computes
`available_for_yield = all_cash_like − cash_to_cover_directional − always_in_cash − pending_redemptions`
(`always_in_cash = equity·(1−position_allocation)`) and distributes it across the `YieldRuleset`. The last
slot (`max_concentration=1.0`, `YieldRuleset.validate()`-enforced) absorbs the remainder — the
instant-liquidity fallback. Promotion funding is `calculate_cash_needed_to_cover_directional_trades()`,
which releases venue cash to cover the re-emitted buy. Pool-size capping is the `size_risk_model` (note:
`YieldWeightingRule.max_pool_participation` is defined but **not** wired into `calculate_yield_positions()`,
`yield_manager.py:448` — inert). Aave works only via the `credit_supply` path (it can't be an AlphaModel
signal; `set_signal()` takes spot/vault only).

**Venue must be synchronous** (`is_async_vault()` false — none of `erc_7540_like`/`lagoon_like`/`ostium_like`,
`identifier.py:59-63`, `:915-930`): backtest routes it through `simulate_spot()` (`backtest_execution.py:371-381`),
live through the default `ERC4626DepositManager`, and `get_redeemable_capital()` (`allocation.py:85-121`)
returns full marked value with no lockup withholding — that value is the `queue_venue_redeemable` the cash
cap reads. Candidates in-universe: Gauntlet/Steakhouse/Spark/Fluid USDC. **Avoid** Lagoon/Ostium/Gains/
Centrifuge/Plutus/Hyperliquid.

---

## Correctness invariants (do NOT undo these)

The non-obvious constraints an implementer must honour. Each is grounded and has a regression test.

1. **YieldManager owns all venue trades.** The alpha model never emits one; it only defers/logs/re-emits.
2. **Exclude the venue from `update_old_weights`** (`:1699-1754`), *before* `set_old_weight()` (`:747`).
   With the reference strategy's `ignore_credit=False` it otherwise picks up the venue as a zero-signal
   position and either **sells it to zero** (a sync-vault venue) or **asserts** (a credit-supply venue —
   `set_old_weight()` accepts spot/vault only, `identifier.py:1320`) — both fight YieldManager. The
   `_count_position_in_old_weights` hook must exclude queue venues (vault *and* credit) up front.
3. **The venue stays inside `total_equity` / `deployable_target_value`.** `calculate_total_equity()`
   (`portfolio.py:758`) includes it and `calculate_portfolio_target_value()` (`allocation.py:147-189`)
   derives the deployable target from it — *intended*, so the swept cash re-competes each cycle and funds
   a window-opening `V`. **Do not subtract the venue from deployable** (that strands the cash). Combined
   with (2): in equity, not in old-weights.
3b. **The sweep is value-neutral** (cash→venue conserves equity), so (2)+(3) do not double-count; a test
   must assert a swept venue does not inflate directional `position_adjust_usd` beyond `allocation_pct·equity`.
4. **The same-cycle cap must count venue-redeemable cash.** `_cap_buys_by_async_sell_proceeds` runs
   *before* YieldManager and sees only `self.signals`; without widening its `cash` term to
   `get_current_cash() + queue_venue_redeemable`, it scales a promotion buy down to raw cash (low because
   the cash is in the venue).
5. **`position_allocation = allocation_pct`, and handle `available_for_yield ≤ 0`.** The two knobs are not
   independent: `available_for_yield > 0` is an unguarded assert (`yield_manager.py:679`) and a full venue
   release makes it exactly zero, then `weight = amount / cash_available_for_yield` divides by zero
   (`:502`). Tie them equal (the `base-ath-ipor.py:1115` convention) **and** make zero/negative an explicit
   path (zero targets + release existing yield + no div). Clamping alone is insufficient; keep the assert
   strict (`> 0`) and handle the zero case in the caller.
6. **Backtest window override > real `vault_state`.** A stale/always-open `vault_state` must not block a
   synthetic D2/Gains schedule.
7. **Normalisation stays window-agnostic** (includes closed vaults) — the slot-holding invariant.
8. **Event log is the single durable structure** (no separate dollar earmark), read as a **full-history
   fold** over `state.other_data.data`, **never `OtherData.load_latest`** — which returns only the latest
   cycle that stored anything and would silently drop open events on a quiet cycle (`other_data.py:98-107`,
   `state.py:261`). Events are JSON-primitive.

---

## How it plugs in (seams + refactor for reuse)

### Wiring (two-step `decide_trades`, unchanged pattern)

```python
alpha = PhaseAwareAlphaModel(timestamp, ...)
...  # set_signal → carry_forward → select_top → assign → normalise (any method) → update_old (excl. venue) → calculate_target
alpha.apply_phase_aware_intent(position_manager)          # promote (re-emit V buys) + park (defer + log). NO venue trades.
trades = alpha.generate_rebalance_trades_and_triggers(..., same_cycle_cash_buffer_usd=...)   # cap counts venue balance (inv. 4)
yield_result = YieldManager(position_manager, rules=create_yield_rules(...)).calculate_yield_management(yield_input)
trades += yield_result.trades                              # sweeps idle → venue, releases venue cash to fund the buys above
```

`apply_phase_aware_intent` operates on `self.signals`, does **promote + park only**, reads/writes the
event log, emits/zeros directional adjustments — never venue trades, never sweeps.

### Refactor for reuse — PR-0 (behaviour-preserving, lands first)

`AlphaModel` is a core class used by many strategies. Rather than have `PhaseAwareAlphaModel` copy-paste
large methods to change a few lines, first **extract the variation points into small protected hooks on
the base**, as a pure refactor with **no behaviour change** (guarded byte-for-byte by the existing
AlphaModel/async-vault suite):

- `_available_same_cycle_cash(position_manager)` — extracted from `_cap_buys_by_async_sell_proceeds`
  (`:975`). Base: `get_current_cash()`. Subclass: `+ queue_venue_redeemable` (invariant 4).
- `_count_position_in_old_weights(position)` — extracted from `update_old_weights` (`:1723-1733`). Base:
  the **existing composite predicate** (skip `is_cctp_bridge()`; and when `ignore_credit`, skip
  `is_credit_supply`/`is_vault`) — *not* a bare `True`. Subclass *additionally* excludes `is_queue_vault`
  (vault *and* credit) before `set_old_weight()` (invariant 2). Cleaner than threading `portfolio_pairs`.
- `_on_deposit_window_closed(signal, position_manager)` — extracted from the `can_deposit`-false branch of
  `_should_skip_signal_rebalance` (`:1904-1915`). Base: flag `cannot_deposit` + `missed_deposit_usd`.
  Subclass: defer + log park event.

Genuinely new, shared logic goes in a **new module `tradeexecutor/strategy/phase_aware.py`** (pure
functions reused by the model, the charts, and tests — one source of truth):

- the park/promote **event-log** type + read/write over `state.other_data` — the reader is a
  **full-history fold** (park minus close/promote events), never `load_latest`; events are JSON-primitive;
- `is_queue_vault(position)` (from the tag / `YieldRuleset` membership);
- `queue_venue_redeemable(portfolio)` (the value invariant 4 reads);
- the promote-detection helper the `weight.py` / `vault.py` charts need.

### Reused machinery (don't reinvent)

- **Sweep engine:** `YieldManager` — owns all venue trades.
- **Redemption-mark template:** `carry_forward_non_redeemable_positions()` (`:815-887`, branches
  `:843-860` / `:862-885`). The venue is *not* `carry_forward`-pinned — it's fully liquid, YieldManager-managed.
- **Flags:** add `parked_in_queue_vault`, `promoted_from_queue_vault` to `TradingPairSignalFlags` (`:64-109`).

### Persistence & subclass

The durable **event log** lives in `state.other_data` as park/promote events keyed by cycle.
**Critical — read it as a full-history fold, not `load_latest`:** `OtherData.load_latest(name)`
(`other_data.py:98-107`) returns only the *most recent cycle that stored anything*, and the framework
writes `decision_cycle_ended_at` **every** cycle (`state.py:261`); so on a quiet cycle the next
`load_latest("park_events")` returns `None` and silently drops every open event — breaking dedup,
promotion, and reload at once. The `phase_aware.py` reader **must fold the whole history**
(`for c in sorted(data): apply park minus close/promote events`), or re-save the full open-event set into
the current cycle slot unconditionally each cycle. Events must be **JSON-primitive** (str vault id, float
USD, int cycle — no `TradingPairIdentifier` objects, `other_data.py:24`). Nail this in PR-A.
`TradingPairSignal`/`AlphaModel` are per-cycle (`:112`, `:473`) so `signal.other_data` holds only a
per-cycle diagnostic flag; there is no separate dollar-earmark structure. Recommend the
`PhaseAwareAlphaModel(AlphaModel)` subclass (slotted `@dataclass` → subclass also `@dataclass`, minimal
new fields). Opt-in = use the subclass with a YieldRuleset; `AlphaModel` alone reproduces today's
behaviour exactly.

---

## Openness detection & backtest windows

**Polling.** Backtest: `BacktestPricing.can_deposit` (`backtest_pricing.py:630-643`) / `check_redemption`
(`:645-681`) over the historical `vault_state` sentinel arrays (`:217-240`, no look-ahead) — correct.
Live: `GenericPricing → VaultPricing.can_deposit` (`vault_live_pricing.py:228-236`) /
`HypercoreVaultPricing` (`hypercore_valuation.py:521-539`). **Correct for backtest + Hypercore only** —
`VaultPricing.can_deposit` returns `True` on unknown `maxDeposit` and has **no `check_redemption`
override**, so live generic ERC-4626 windowing (both deposit *and* redemption) needs new protocol-specific
adapters — a v1 follow-up. Scope v1 live correctness accordingly.

**Next-open getters (diagnostics only).** `deposit_next_open` / `redemption_next_open` flow to
`pair.other_data` (`dex_data_translation.py:227-228`) with no getter or consumer. Add
`get_deposit_next_open()` / `get_redemption_next_open()` (`identifier.py:960-965` neighbourhood) for the
ETA column. They are a current snapshot — never a historical backtest input (look-ahead).

**Backtest window modelling — `get_assumed_open_close_time(vault, vault_universe)`.** The historical
`vault_state` has open/closed truth + caps but no next-open column, and may mark D2/Gains always-open. A
layered resolver supplies a schedule; **precedence: explicit override > real `vault_state` >
assumed-from-metadata** (the override must beat stale/always-open data or the assumed schedule never
fires):
1. **Explicit test/backtest override** — per-vault synthetic schedule, mirroring
   `vault_settlement_delay_overrides` (`backtest_execution.py:106-128`); can override real `vault_state`.
2. **Real `vault_state`** transitions when present and not overridden.
3. **Protocol-default cadence** — `{feature/protocol: cadence}` (e.g. `7d`/`30d`/`60d`), keyed off vault
   **feature flags / metadata**, not protocol-name strings. Confirm real epochs (D2 = `epoch_end−epoch_start`,
   Gains ≈ 3d).
4. **Share-price-spike inference** (optional) — from NAV discontinuities, cached per vault; precomputed,
   never read forward.

It feeds a synthetic availability layer consulted by `BacktestPricing.can_deposit`/`check_redemption`;
**backtest-only**, look-ahead-free. (The async-redemption idle-cash case already manifests via
`DEFAULT_VAULT_SETTLEMENT_DELAY` — that half needs no synthetic data.)

---

## Diagnostics & charts

The allocation charts must stop lumping everything non-directional into one grey "USDC" band (the PR #35
chart). The undeployed slice now has structure: **idle cash**, **queue-venue (yield-bearing) allocation**,
**deposits waiting to be triggered** (deferred, window closed), **redemptions waiting to be triggered**.
Proving idle→productive is the whole point.

- **`weight.py`** — split the reserve band into **idle USDC** vs **queue-venue allocation**
  (`volatile_and_non_volatile_percent` :39, `equity_curve_by_asset` :53, `equity_curve_by_chain` reserve
  rows :94-114; weights `_calculate_and_cache_weights` :15). Venue positions are rendered as a distinct
  reserve-like band via the `is_queue_vault` tag.
- **`vault.py`** — `pending_vault_settlements()` (:192, window from `_get_trade_pending_window` :156)
  already stacks **in-flight** async buffers. Add a sibling `pending_trigger_queue` (or distinct series)
  for the **not-yet-in-flight** buffers (deferred deposits, marked redemptions), visually distinct from
  in-flight (operator/epoch-gated) vs waiting (next-flip-gated).
- **Durable event log drives the time series.** Reconstruct chart rows from the `state.other_data` event
  log (not queue-position `other_data`, which is lost on resize/close) — the widening-idle-band curve the
  PR needs. (Mirrors how `pending_vault_settlements` uses durable `vault_settlement_requested_at`.)
- **Tables.** `format_signals()` (`:2406`) gains **Parked USD**, **Waiting deposit USD**, **Waiting
  redemption USD**.

The chart + table code consumes the shared `phase_aware.py` helpers (event-log reader, `is_queue_vault`).

---

## Example notebook & acceptance

Primary acceptance = a **derived notebook** swapping the old alpha model for the phase-aware one on the
exact reference strategy, measured on real data.

- Base: `getting-started .../08-backtest-capped-waterfall.ipynb` (in-repo copy
  `strategies/test_only/08-backtest-capped-waterfall.ipynb`). Verified: **daily rebalance**
  (`TimeBucket.d1` / `cycle_1d`), `always_include` for D2 and Ostium, `LENDING_RESERVES = None`.
- Derived: `strategies/test_only/09-backtest-capped-waterfall-phase-aware.ipynb` — identical except:

1. Swap `AlphaModel` → `PhaseAwareAlphaModel` + the `apply_phase_aware_intent` pass. (It keeps the base's
   `waterfall=True` — that's the reference; the class itself is method-agnostic.)
2. Add a **sync USDC vault** as the `YieldRuleset` last slot (no extra plumbing; Aave `get_credit_supply_pair()`
   would need `LENDING_RESERVES` wiring the base lacks); wire `YieldManager` two-step with
   `position_allocation = allocation_pct` (0.90) + the zero-release handling.
3. Ensure Ostium + D2 **HYPE++** are present, tradeable (real TVL, and/or relax the pool cap for them),
   **and selected** — boost their signal / set a floor weight so both get a **positive target while
   closed** (so criterion 3 is deterministic, not vacuous).
4. Configure `get_assumed_open_close_time()` overrides for D2 and Ostium so their windows close/reopen
   across the run (override > real `vault_state`).

**Acceptance criteria (measurable):**
1. **Universe includes Ostium + D2 HYPE++** — a tradeable pair with `get_vault_protocol() == "ostium"` and
   a D2 **HYPE++** vault, each with non-zero TVL.
2. **Daily rebalance** — `cycle_duration == cycle_1d` and a daily cadence over the span.
3. **Correctly allocates to Ostium + D2 (deterministic)** — per vault: a non-zero **parked** amount
   (`parked_in_queue_vault`) on a closed cycle, **matched to a later `promoted_from_queue_vault` deposit
   for the same vault** when it opens, and a realised holding (`position.get_value() > 0`). Baseline: the
   old model skips → `missed_deposit_usd`, ~zero allocation.
4. **Idle-cash win + continuity** — end-of-run idle USDC falls from ~24% toward ~10%, the difference now in
   the venue; return ≥ base +6.9%; equity continuity (no cycle move > 1% from park/promote/sweep).

Run: `source .local-test.env && poetry run jupyter execute .../09-...ipynb --inplace --timeout=900`. Wrap
as a `tests/test_capped_waterfall_notebook.py`-style smoke test asserting criteria 1–4 from the final `state`.

---

## Tests

Repo rules (docstring numbered steps mirrored inline, typed fixtures, happy + bad path, `pytest.approx`,
no stdout):

- **Refactor (PR-0):** the existing AlphaModel + async-vault suites stay green **byte-for-byte** after the
  hook extractions — the behaviour-preserving guarantee.
- **Park:** closed `V` + positive target ⇒ no `V` buy, a park event in the `state.other_data` log, flag
  `parked_in_queue_vault`; unspent cash available to the sweep. Assert **other same-cycle trades (incl.
  sells) survive** when a *non-dominant* signal is parked (the pre-cap global `min_trade_threshold` gate at
  `:2272-2284` can otherwise cancel the whole cycle). Bad path: base `AlphaModel` ⇒ today's skip.
- **Deposit-on-open:** parked `V` flips open *and still targeted* ⇒ buy re-emitted, promote event logged,
  `promoted_from_queue_vault`; assert the same-cycle cap does **not** scale it (venue balance counted,
  invariant 4) — engineer a **coincident same-cycle async sell** so `_cap_buys` doesn't early-return
  (`:972`, `async_sell_usd<=0`) and the widening is actually exercised, not vacuous; assert `V` is **not**
  settlement-pending (else the `:2034-2044` guard no-ops it).
- **Stale park event:** a parked `V` whose window opens but which no longer earns a positive target
  (dropped from the ranked set) ⇒ the park event is **closed without a promote/buy**; venue balance
  unchanged.
- **Method-agnostic:** run the park→promote cycle under `normalise_weights` *simple* and *size-risk* (not
  just waterfall) — the slot-holding invariant holds; `V`'s cash is not reallocated while closed.
- **YieldManager owns venue trades:** all venue trades come from `calculate_yield_management`, none from
  the model; venue excluded from `update_old_weights` (no sell-to-zero); no directional over-allocation
  beyond `allocation_pct·equity` (invariant 3b).
- **Zero-release:** a full venue release (`available_for_yield == 0`) does not assert/divide-by-zero
  (invariant 5).
- **Redemption swept:** settled async redeem ⇒ freed cash swept into the venue, not idle.
- **Window cycle (integration):** closed N cycles then open (override) through `BacktestExecution` — defer
  across the window, single deposit on open, idle stays near the buffer.
- **Cross-chain (integration):** venue on the **primary/hub chain** (a satellite venue would break
  same-cycle funding — per-chain is a follow-up), `V` on a satellite ⇒ venue-redeem → CCTP bridge →
  satellite-`V`-deposit funds in one cycle, no `NotEnoughMoney`. (Composable: the venue-redeem is a sync
  sell counted into `primary_sells`, funding the bridge-out under the post-#1549 order.)
- **Charts:** `weight.py` splits idle vs venue; `vault.py` shows waiting buffers reconstructed from the
  durable event log.

---

## Implementation plan (PR sequencing)

- **PR-0 — extract seams (no behaviour change).** The three base hooks + the `phase_aware.py` module
  skeleton; regression suites green byte-for-byte.
- **PR-A — scaffolding.** `is_queue_vault` tag + predicate; next-open getters; `TradingPairSignalFlags`;
  the event-log type + read/write; `queue_venue_redeemable`. No decision logic yet.
- **PR-B — core behaviour.** `PhaseAwareAlphaModel` + `apply_phase_aware_intent` (park/deposit-on-open),
  the invariant-4 cap widening, invariant-2 old-weights exclusion, redemption-swept. Unit + method-agnostic
  tests. Same-chain only.
- **PR-C — backtest windows + charts.** `get_assumed_open_close_time()` + wiring; the `weight.py`/`vault.py`
  splits + waiting-trigger chart. Chart/diagnostic tests.
- **PR-D — cross-chain + acceptance.** Cross-chain deposit handling + the `09-...` notebook + the idle-cash
  regression.
- **Follow-ups.** Live ERC-4626 openness adapters; per-chain venues (more `YieldRuleset` rules +
  chain-matching in YieldManager); earning yield on the structural reserve (raise `position_allocation`).

---

## Risks / open questions

1. **The venue is a third position category** (always-keep, fully-liquid, YieldManager-managed) — excluded
   from the candidate set/waterfall *and* old-weights, yet in equity + `get_redeemable_capital`. Getting
   this classification wrong is the likeliest accounting bug (invariants 2+3).
2. **Churn / gas (live).** Defer→deposit + sweep add trades; respect `individual_rebalance_min_threshold_usd`
   / `absolute_min_vault_deposit_usd` + the YieldManager dust tolerance; measure trade count before/after.
3. **Single hub venue re-engages CCTP** for cross-chain deposits (latency, mark-vs-realisable gap).
   Mitigation: per-chain venues (follow-up) + `same_cycle_cash_buffer_usd` + the integration test.
4. **Backtest data coverage** for D2/Gains windows is unverified — hence `get_assumed_open_close_time`
   (override beats stale data). Verify real `vault_state` coverage before claiming *real* D2/Gains parking.
5. **Venue capacity.** A large sweep may hit the venue's `maxDeposit` / move its price; the venue must be
   deep; `YieldRuleset` can spread across venues later.
6. **`pending_redemptions` is owner-side** — YieldManager's term reads the strategy's *own* Lagoon
   treasury queue (~0 here), not our async-`V` redemptions (that's the settlement-pin). Don't conflate.
7. **Does not shorten queues.** The venue makes *waiting* capital productive; it doesn't redeem
   Plutus/Ostium faster. Keep honest in the PR.

## Open decisions

- Concrete v1 venue (deep sync USDC vault — Gauntlet/Steakhouse — recommended over Aave, no `LENDING_RESERVES`
  plumbing).
- `position_allocation` value (= `allocation_pct`) and `buffer_pct` sizing vs `cross_chain_cash_buffer_usd`.
- Protocol-default cadence values (D2/Gains/Ostium) — confirm real epochs before hard-coding.

---

## Review history & provenance

Every decision above was vetted; this records where each came from.

- **Codex `xhigh` (round 1, 122 inspections)** — 7 blocking + 3 non-blocking, all folded in:
  **[B1]** one owner (YieldManager owns venue trades; model keeps intent). **[B2]** cap must count venue
  redeemable. **[B3]** durable state in `state.other_data`, not per-cycle signal. **[B4]** exclude venue
  from `update_old_weights`. **[B5]** live openness scoped to backtest+Hypercore. **[B6]** capacity-overflow
  is swept residual, not parked. **[B7]** window override > real `vault_state`. Non-blocking: Aave only via
  credit path; `max_pool_participation` inert; charts need a durable event log.
- **Opus `ultrathink` (round 2, fresh context, 55 inspections)** — confirmed the reconciliation clean;
  two new blocking: **[B8]** `position_allocation`/`allocation_pct` not independent + the unguarded
  `available_for_yield` assert; **[B9]** venue is dual-natured (in equity/deployable AND excluded from
  old-weights). Non-blocking: live gap covers redemption too; CCTP bands are vault ±50M; CCTP-injection
  ordering already structural; promotion test must assert not-settlement-pending; `pending_redemptions`
  owner-side.
- **Codex `xhigh` (round 3)** — confirmed [B8]/[B9]/[B2] grounded and the base notebook (D2/Ostium, daily,
  `LENDING_RESERVES=None`); folded three refinements: earmark resolved to the event log (no load-bearing
  structure); zero-`available_for_yield` handling is a MUST; acceptance criterion 3 forced deterministic.
- **Post-review refinements (folded into this clean version):** (i) **dropped `queue_vault_resolver`** —
  redundant with `YieldRuleset` + `is_queue_vault`, a vestige of the pre-[B1] design; (ii)
  **allocation-method-agnostic** — the slot-holding invariant generalises to any window-agnostic
  normalisation, not just waterfall; (iii) **refactor for reuse** — extract base hooks + a shared
  `phase_aware.py`, behaviour-preserving PR-0.

- **Codex `xhigh` (round 4, on the clean rewrite)** — **zero blocking**; confirmed the 688→470 condensation
  lost no correctness (invariants 2–5 grounded), the resolver removal is sound for v1, and the three PR-0
  seams are cleanly separable. Four non-blocking clarifications folded in: the slot-holding invariant needs
  a *positive post-normalisation target* (caps can zero a vault); per-chain venues need YieldManager
  chain-matching (not just more `YieldRuleset` rules); the old-weights hook must exclude venues *before*
  `set_old_weight()` (a credit venue asserts, not sells-to-zero); and a **stale park event** (vault no
  longer targeted) is closed, not promoted.

- **Opus `ultrathink` (round 5, clean context)** — verified every load-bearing anchor; found **one blocking
  gap the prior four rounds missed** ([O1]): the event-log durability assumed an append-durable multi-cycle
  set, but `OtherData.load_latest` returns only the latest cycle that stored anything and would silently
  drop open events on a quiet cycle — the reader must be a full-history fold (now invariant 8 + Persistence).
  Non-blocking, folded in: `_count_position_in_old_weights` base is the existing composite predicate (not
  `True`); parking a *dominant* signal can trip the pre-cap `min_trade_threshold` gate (test guards it); the
  redemption side is *passive* (existing pin + claimed-proceeds sweep, no new deferral); cross-chain funding
  composes **iff the venue is on the hub chain**; the deposit-on-open test must engineer a coincident async
  sell or the invariant-4 widening is vacuous. Verified sound: invariants 2–5, the credit-venue
  `set_old_weight` assert, cross-chain ordering, opt-in subclass.

**Verdict:** implementable — the one Opus blocking ([O1], event-log reader) is folded in as invariant 8 +
a PR-A must; all five review rounds' items resolved. Start at PR-0.
