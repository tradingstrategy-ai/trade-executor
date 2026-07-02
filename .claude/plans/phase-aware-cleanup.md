# Phase-aware cleanup — close the gaps between PR #1550 and the design plan

Branch: `phase-aware-alpha-model` (worktree `/home/mikko/code/trade-executor-phase-aware-alpha`, off `master` @ `67b65935`).
PR: https://github.com/tradingstrategy-ai/trade-executor/pull/1550 (OPEN, +5,499/−16, 18 files).
Parent plan: `.claude/plans/phase-aware-alpha-model.md` (committed `24b4602b`).

Status: **cleanup plan, nothing implemented.** Produced from an adversarial plan-conformance
review (clean Fable subagent, 2026-07-02) of the delivered branch against the parent plan.
File anchors are from the branch HEAD (`4f9e03de`) — verify at implementation time, they drift.

---

## Why

The conformance review's verdict: *the core mechanism is faithfully — in places better than —
implemented* (park/promote/stale-close, the full-history-fold event log, reconcile-after-emit,
zero-release, window-override precedence), **but the branch under-delivers the plan's v1 scope,
and several drops are silent**:

1. The parent plan mandated *"each [invariant] … has a regression test"*. Invariants 5, 6, 8 have
   real tests; invariant 4's test is **exactly the vacuous version the plan forbade**; invariant
   3b's test is absent despite *"a test must assert"*; invariants 1, 3, 7 have none.
2. Acceptance criteria 3/4 were quantitatively weakened without acknowledgement (no +6.9%-over-base,
   no equity-continuity, `< 20%` instead of ~10%, no baseline run, no same-vault park→promote
   assert, no realised-holding assert).
3. The parent plan's *Diagnostics & charts* section (half of planned PR-C) is entirely absent.
   It **is** disclosed in the PR description's follow-ups list, but not in any commit/CHANGELOG.
4. Two docstrings in `phase_aware.py` are now factually wrong.
5. The promised behavioural tests (redemption-swept, gate-survival, method-agnostic under real
   normalisers, window-cycle/cross-chain integration) don't exist; end-to-end coverage lives only
   in a `slow_test_group` notebook wrapper.

**Corrections to the review, for the record** (the reviewer saw only repo artifacts):
the charts deferral and the focused acceptance window *are* disclosed in the PR body; the
overstating claim "self-verifies all four acceptance criteria" lives in the PR-D **commit
message**, not the PR description. The *criteria weakening itself* is disclosed nowhere — that
part stands.

---

## Findings register

IDs referenced by the work packages below. Severity: **B** = should land before merge,
**F** = fast-follow, **N** = note/no action.

| ID | Finding | Sev |
|----|---------|-----|
| G1 | `phase_aware.py:204` class docstring: "Same-chain only in this PR; cross-chain (CCTP) promotion is a follow-up" — contradicts PR-D (cross-chain notebook works; the actual follow-up is chain-aware YieldManager for full-history funding) | B |
| G2 | `phase_aware.py:246-249` `__init__` docstring: with `cycle=None` "the deposit deferral still happens" — **false**: `apply_phase_aware_intent` returns early, behaviour degrades to the base `AlphaModel` *skip* (`cannot_deposit` + `missed_deposit_usd`), not deferral | B |
| G3 | Notebook cell 20 asserts `gated_parked` and `gated_promoted` **separately** — does not prove the *same* vault both parked and promoted (criterion 3's "matched" requirement); no `position.get_value() > 0` realised-holding assert | B |
| G4 | Criteria retirement undisclosed: +6.9%-over-base, ~24%→~10%, no-cycle-move->1% were defined on the full reference run; the focused window (2026-02-01) invalidates them, and the substitution (`< 20%`, no baseline) is stated nowhere | B (disclosure) |
| G5 | Invariant-4 test vacuous: `test_phase_aware_alpha_model.py:120-131` calls `_available_same_cycle_cash` directly; **no phase-aware test exercises the widened budget through `_cap_buys_by_async_sell_proceeds`**. (Scope precisely: the *base* async cap path IS exercised end-to-end by `tests/backtest/test_backtest_async_vault.py:1289,:1397` — what is untested is the phase-aware widening flowing through the cap.) The parent plan explicitly warned: "engineer a coincident same-cycle async sell … not vacuous" | F |
| G6 | Invariant-3b test missing: "a test must assert a swept venue does not inflate directional `position_adjust_usd` beyond `allocation_pct·equity`" | F |
| G7 | Redemption-swept test missing: settled async redeem proceeds → swept into venue, not idle (the redemption side's only net-new behaviour is otherwise untested) | F |
| G8 | Park gate-survival test missing: a parked non-dominant signal must not let the pre-cap whole-portfolio `min_trade_threshold` gate cancel the cycle's other trades (design plausibly moots it — parking zeroes the adjust pre-generation — but the property is unproven) | F |
| G9 | Method-agnostic tests rationalised into stubs (`test_phase_aware_alpha_model.py:10-12` sets `self.signals` directly); the slot-holding invariant is never proven under a real `normalise_weights` (simple / size-risk) | F |
| G10 | Window-cycle integration test missing: closed-N-cycles-then-open through `BacktestExecution` (defer across window, single deposit on open) exists only inside the slow notebook | F |
| G11 | Charts dropped: `weight.py` idle-vs-venue reserve split, `vault.py` `pending_trigger_queue`, `format_signals` Parked/Waiting columns, chart tests. Session task #21. Disclosed in PR body only | F |
| G12 | Invariants 1 (YieldManager owns venue trades), 3 (venue in equity/deployable) have no dedicated regression tests; inv. 7 (window-agnostic normalisation) only implicit | F |
| G13 | `calculate_yield_management_safe` duplicates steps 1-3 of `calculate_yield_management` ("mirroring" acknowledged in its docstring) — drift risk between the two computations of `available_for_yield` | F |
| G14 | `_on_deposit_window_closed` PR-0 seam is never overridden by the subclass (parking moved to the pre-generation pass); the seam's stated purpose is unfulfilled and undocumented at the hook | B (doc) |
| G15 | Latent precedence inversion: a protocol-cadence schedule wired as an *override* beats real `vault_state`, inverting the parent plan's layers 2/3. Documented as deliberate in `vault_windows.py` docstrings; `protocol_cadences` currently has no caller | N |
| G16 | PR-D commit message overstates ("self-verifies all four acceptance criteria"); commits are immutable once pushed — remediate via PR-description honesty (CU-0), do not rewrite history | N |

Out of scope for this cleanup (already-named separate follow-ups): chain-aware `YieldManager`
(full-history cross-chain funding, parent-plan risk 3), live ERC-4626 openness adapters,
share-price-spike window inference (resolver layer 4).

---

## Work packages

Sequenced; CU-0 is the only pre-merge blocker (matches the reviewer's verdict). Each package =
one commit, reviewed per the protocol below.

### CU-0 — honesty + cheap fixes (pre-merge) [G1 G2 G3 G4 G14 G16]

1. **Fix `phase_aware.py:204`** class docstring: replace the "Same-chain only" sentence with:
   cross-chain deposits compose via the existing CCTP planner provided the queue venue is on the
   primary/hub chain; the chain-aware YieldManager (full-history funding) is the follow-up.
2. **Fix `phase_aware.py:246-249`** `__init__` docstring: with `cycle=None` the phase-aware passes
   are inert and behaviour degrades to the base `AlphaModel` skip (`cannot_deposit` flag +
   `missed_deposit_usd`) — *not* deferral.
3. **Document the unused seam** (`alpha_model.py` `_on_deposit_window_closed`, ~:1974): add a
   docstring note that `PhaseAwareAlphaModel` intentionally does **not** override it — parking
   happens in `apply_phase_aware_intent` *before* generation so deferred buys are excluded from
   the min-trade gate and the same-cycle cash cap; the base hook remains the skip path.
4. **Tighten notebook cell 20** (the acceptance cell of
   `strategies/test_only/09-backtest-capped-waterfall-phase-aware.ipynb`):
   - assert the same-vault match: `assert window_gated_ids <= (parked_vault_ids & promoted_vault_ids)`
     (both gated vaults must appear in *matched*, not merely one in parked and one in promoted);
   - assert a realised holding per gated vault: open position with `get_value() > 0` at run end;
   - add a short markdown note (or comment block) explicitly retiring the full-run quantitative
     criteria (+6.9% over base, ~24%→~10%, 1% equity continuity) and stating what is asserted
     instead and why (focused window, no baseline run) — the disclosure G4 requires.
   - Edit the notebook JSON programmatically (see Working instructions); re-run it end-to-end and
     confirm the acceptance cell passes, then **clear outputs** before committing.
5. **Amend the PR description** (only when the user asks to push/update — see conventions) adding
   a "Deviations from the design plan" section: (i) charts/diagnostics deferred (named follow-up);
   (ii) acceptance criteria actually asserted vs the plan's retired baseline numbers, and why;
   (iii) outstanding plan-mandated tests (G5-G10, G12) as fast-follows; (iv) end-to-end coverage
   currently lives in a `slow_test_group` notebook test.
6. Re-run focused suites (see Working instructions) + the notebook; commit as
   `fix: phase-aware docstrings + acceptance assert tightening (cleanup CU-0)`.

**Done when:** both docstrings correct; cell 20 asserts matched-vault + holding; notebook runs
clean; disclosure text drafted in the PR body file (`/tmp/pr-body-v2.md`) ready for the user's
go-ahead; suites green.

### CU-1 — plan-mandated invariant tests [G5 G6 G12]

New tests in `tests/units_tests/test_phase_aware_alpha_model.py` (or a sibling module if it grows
past ~600 lines). Repo test rules apply (numbered-step docstrings mirrored inline, typed fixtures,
happy+bad path, `pytest.approx`, no stdout).

1. **Non-vacuous invariant 4 — DELIVERED** as `test_cap_buys_widened_by_venue_through_async_sell`
   (`tests/units_tests/test_phase_aware_alpha_model.py`). Drives `_cap_buys_by_async_sell_proceeds`
   directly with (a) a same-cycle async-vault sell — detected via the position's
   `has_async_vault_flow` fallback — so `async_sell_usd > 0` and the cap does not early-return,
   (b) a $500 buy exceeding raw cash ($100) + sync sells ($0), (c) a $1000 queue-venue position.
   Base `AlphaModel` scales the buy to $100 and flags `capped_by_pending_settlement_cash`;
   `PhaseAwareAlphaModel` (with `venue_pair_ids`) leaves it at $500 unflagged — proving the widened
   budget flows through the cap, not just the `_available_same_cycle_cash` hook. This is the exact
   non-vacuous shape the parent plan demanded (G5). Passing.

2-4. **Invariants 3b, 1, 3 reassigned to CU-3 (the integration backtest)** — they are *not* cleanly
   unit-testable without the shallow stubbing that would repeat the very vacuity the conformance
   review flagged, so faithfully they belong against real machinery:
   - **inv-3b** *is* unit-testable in principle (a Codex review corrected an earlier overstatement):
     `calculate_target_positions()` → `map_pair_for_signal()` needs only a `strategy_universe`
     attribute for positive signals, no shorting helpers. But a stub-level assertion (directional
     `sum(max(adjust,0)) <=` the exact `investable_equity` handed in — not a recomputed
     `allocation_pct*equity`, since `allocation.py:170` deducts in-flight/pending) would mostly
     re-verify *base* `calculate_target_positions` maths; the *phase-aware-specific* property — a
     venue in equity yet excluded from old-weights not inflating the buy budget — only manifests with
     a venue actually holding value across `update_old_weights` + target-calc, i.e. the real run.
     Grouped into CU-3 for that reason, not because it is infeasible in a unit test.
   - **inv-1** ("the model emits no venue trade; YieldManager owns them") is only meaningful against
     real trade generation with a real venue in the portfolio.
   - **inv-3** ("venue stays in equity / deployable, not subtracted") is a construction/notebook
     property — there is no model code to unit-assert; the run must show it.

**Done when:** the invariant-4 test passes and the existing phase-aware unit suite stays green
(11 in `test_phase_aware_alpha_model.py` + 9 in `test_phase_aware.py`). Inv-1/3/3b move to CU-3.

### CU-2 — behavioural tests [G7 G8 G9]

1. **Redemption swept — DELIVERED** as `test_yield_safe_sweeps_reserve_withholding_pending_redemptions`
   (`tests/units_tests/test_yield.py`, on the existing `synthetic_universe`/`state`/`rules`
   fixtures). The safe wrapper's *sweep* branch routes idle reserve above `always_in_cash` into the
   venue (settled redemption proceeds are indistinguishable reserve cash, so the same sweep covers
   them), while the strategy's own `pending_redemptions` are withheld from that sweep: with $10,000
   reserve and $2,000 pending, `available_for_yield == 10,000 − 500 − 2,000 = 7,500`, all swept into
   venue buys. Earns the "redemption" name by exercising the one redemption-specific term in the
   calc (a review found the earlier version left `pending_redemptions` at 0, untested, and merely
   re-ran `test_yield_distribute_all` through the wrapper). Passing (G7).

2. **Park gate-survival (bad path) — DELIVERED** as `test_park_dominant_cancels_cycle_via_min_trade_gate`
   (`tests/units_tests/test_phase_aware_alpha_model.py`, existing stubs). Two reviews showed the bad
   path is cleanly unit-testable — `generate_rebalance_trades_and_triggers` returns `[]` at
   `alpha_model.py:2344` (before the cap and per-signal loop), and `_prepare_hypercore_sell_signals`
   only needs the stub's `get_current_position_for_pair` → `None` — so deferring it was the one
   reassignment that repeated the original under-delivery. It parks a dominant closed-window deposit
   ($1000) with a below-threshold open adjust ($5), then asserts the cycle cancels (`max_diff` drops
   to $5 < $50 once the parked adjust is zeroed) and `max_position_adjust_usd == 5`. Passing (G8, bad path).

3. **Gate-survival happy path + method-agnostic → CU-3.** The *happy* path (a non-dominant park
   leaving a valid cycle intact) reaches the real per-signal trade-creation loop, which the stubs
   do not provide. Method-agnostic is unit-testable for the *simple* normaliser (no pricing) but its
   size-risk variant needs a size-risk model + pricing; grouped into CU-3 for a single
   real-normaliser assertion of the slot-holding invariant. These are genuine integration/near-
   integration cases, not shallow theatre — see CU-3.

**Done when:** both new tests pass and their suites stay green (`test_yield.py` 5 tests;
`test_phase_aware_alpha_model.py` 12 tests). Gate-survival happy path + method-agnostic move to CU-3.

### CU-3 — window-cycle integration test [G10]

**DELIVERED** as `tests/backtest/test_phase_aware_backtest.py` — a synthetic, no-network
`run_backtest_inline` over two sync vaults (window-gated target `VT` + always-open queue venue `VQ`,
plus an always-open directional `VD` for gate-survival), 8 small independent tests, 1.9s. Covers:
window-cycle (park → single promote → executed deposit holds ~95% + idle at the 5% reserve floor);
inv-1 (model never trades the venue), inv-3 (venue held in equity while the target is closed, via
per-cycle observations since the venue is released to fund the deposit on open), inv-3b (directional
buys ≤ investable equity every cycle); method-agnostic (park→promote fires under both the simple and
the real size-risk *waterfall* normaliser - a `USDTVLSizeRiskModel` over deep synthetic TVL, since
`waterfall=True` alone silently falls back to simple; witnessed via `accepted_investable_equity`);
and gate-survival happy path (VD deposits while VT is parked - the cycle is not cancelled). Threading confirmed: `user_supplied_routing_model` builds `BacktestPricing` directly
with `vault_window_overrides` (backtest_runner.py:1036), so the synthetic `vault_state=None` is
irrelevant. The original recipe follows.

A standalone (non-slow) synthetic backtest: `run_backtest_inline(client=None, universe=…)` with
**two vault pairs, not one** — (a) a window-gated *target* vault under `vault_window_overrides`
(closed for N cycles then open; anchor in the closed phase, as the notebook does) and (b) a
separate always-open **synchronous queue venue** wired through `YieldManager` as a single-slot
`YieldRuleset` in `decide_trades`. Without (b) only park/deposit-on-open is testable — the
idle→venue sweep assert has nowhere to sweep to. Assert from final `state`: park events for the
target vault while closed, exactly one promote + one executed deposit on the first open cycle,
and idle cash near the reserve floor after the sweep (the venue holding the excess). Start from the harness in
`tests/backtest/test_backtest_inline_synthetic_data.py` (no network) and the vault machinery in
`tests/backtest/test_backtest_async_vault.py` — reuse, don't reinvent. If synthetic vault-pair
construction proves disproportionate, document why and keep the notebook as the integration
gate (explicitly, in the test module docstring).

Also fold in, asserted against this real run:

*Invariants reassigned from CU-1:*
- **inv-1** — no trade in the run targets a queue-venue pair id except those YieldManager generated
  (assert the venue position changes only via the yield step, e.g. the model's directional trades
  never carry a venue pair id).
- **inv-3** — the venue value is included in the cycle's total equity / deployable target (the
  strategy deploys against equity that includes the venue, not net of it).
- **inv-3b** — directional buys in a cycle never exceed that cycle's `investable_equity` (the venue,
  in equity but excluded from old-weights, does not inflate the buy budget).

*Behavioural tests reassigned from CU-2 (bad-path gate-survival already landed in CU-2):*
- **gate-survival (happy path only)** — a cycle where a *non-dominant* closed-window deposit is
  parked still emits its other same-cycle trades (the zeroed parked adjust does not lower `max_diff`
  below the gate, and the surviving trades reach real generation). Assert from the per-cycle trade
  log. (The bad path — parking the *dominant* signal cancels the cycle — is already covered by
  `test_park_dominant_cancels_cycle_via_min_trade_gate` in CU-2.)
- **method-agnostic** — run the same window-gated scenario under a non-waterfall `normalise_weights`
  (e.g. simple) as well as the waterfall/size-risk path; assert the park→deposit-on-open still
  fires and the other vaults' allocations are unaffected (the slot-holding invariant under a real
  normaliser, not asserted-by-construction).

**Land the pieces independently, not as one all-or-nothing package.** A review flagged that CU-3 now
carries six previously-independent obligations (window-cycle + inv-1/3/3b + gate-survival happy path
+ method-agnostic); if it slips or lands partial, all of them silently go untested — the exact
plan-vs-delivery failure this cleanup exists to correct. Mitigation already applied: every
cleanly-unit-testable slice was pulled forward (inv-4 → CU-1; redemption-swept, gate-survival bad
path → CU-2). Land CU-3 as several small tests (one per obligation) sharing a synthetic-universe
fixture, each committable on its own, rather than a single mega-test — and if the synthetic
two-vault backtest proves disproportionate for any obligation, assert that obligation on the
existing `09` acceptance notebook run instead and say so in the module docstring, rather than
dropping it.

**Done when:** the tests run in the default (non-slow) suite in seconds-to-a-minute, no network,
and cover the window-cycle plus the folded-in invariants and behavioural checks above.

### CU-4 — charts + diagnostics (the dropped half of planned PR-C) [G11, session task #21]

Consume the shared `phase_aware.py` helpers (event-log fold reader, venue identity) — one source
of truth; parent-plan anchors (drifted, re-verify):

1. **`tradeexecutor/strategy/chart/standard/weight.py`** — split the reserve band into **idle
   USDC** vs **queue-venue allocation** (`volatile_and_non_volatile_percent` ~:39,
   `equity_curve_by_asset` ~:53, `equity_curve_by_chain` reserve rows ~:94-114). Venue positions
   identified via `queue_vault_pair_ids`/`is_queue_vault` render as a distinct reserve-like band.
2. **`tradeexecutor/strategy/chart/standard/vault.py`** — sibling of `pending_vault_settlements()`
   (~:192): a `pending_trigger_queue` series for **not-yet-in-flight** buffers (open park events =
   waiting deposits; marked-not-settled redemptions), reconstructed from the durable event log
   (never per-position `other_data`), visually distinct from in-flight settlements.
3. **`format_signals()`** (`alpha_model.py`, was ~:2406 on master — re-locate): add **Parked USD**,
   **Waiting deposit USD**, **Waiting redemption USD** columns.
4. Chart tests: reconstruct rows from a synthetic event log; register the new charts in the 09
   notebook's `create_charts` and re-run it once to confirm rendering.

**Done when:** charts + table render from the 09 notebook run; unit tests over the reconstruction
logic pass; task #21 closed.

### CU-5 — optional refactor [G13]

Extract the duplicated steps 1-3 into a private
`YieldManager._calculate_available_for_yield(input) -> tuple[float, float, dict]` (available,
trade_cash_diff, current_positions) used by **both** `calculate_yield_management` and
`calculate_yield_management_safe`. Pure behaviour-preserving refactor: `test_yield.py` (5 tests)
must pass byte-for-byte identically. Do this last — it touches the strict method other strategies
rely on.

---

## Working instructions (read before starting any package)

**Environment.** "Parent repo" = `/home/mikko/code/trade-executor` — the main checkout that owns
the shared Poetry virtualenv. The worktree carries its own copies of `pyproject.toml`,
`poetry.lock` and `.local-test.env`, which is exactly the trap: with the shell cwd inside the
worktree, `poetry run` resolves the *worktree's* project to a fresh, empty virtualenv and fails
with `Command not found: pytest/jupyter` (this bit us three times). So: keep the shell cwd at the
parent repo, reference worktree paths absolutely, and put the worktree first on `PYTHONPATH` so
imports resolve to worktree source (the editable-install `.pth` otherwise wins for the parent):

```shell
source .local-test.env && PYTHONPATH="/home/mikko/code/trade-executor-phase-aware-alpha:$PYTHONPATH" \
  poetry run pytest /home/mikko/code/trade-executor-phase-aware-alpha/tests/units_tests/test_phase_aware_alpha_model.py -q
```

Focused regression set to keep green throughout (55 tests, ~11 s with `-n auto`):
`tests/units_tests/test_yield.py tests/backtest/test_vault_windows.py
tests/backtest/test_backtest_vault_state_pricing.py tests/units_tests/test_phase_aware.py
tests/units_tests/test_phase_aware_alpha_model.py tests/test_satellite_preflight.py
tests/backtest/test_cctp_backtest_sequential.py`. Use bash timeouts 180000 (single) / 360000
(multiple); `-n auto` when >5 tests. Never plain `python` — the editable-install `.pth` resolves
to the parent repo for a bare interpreter (pytest with `PYTHONPATH` is fine).

**Notebook.** Edit `09-…ipynb` programmatically (JSON; `source` is a list of lines — a
`.replace()` keyed on the last line of a cell silently no-ops because that line has no trailing
newline; append or assert-after-replace). Run:

```shell
source .local-test.env && PYTHONPATH="/home/mikko/code/trade-executor-phase-aware-alpha:$PYTHONPATH" \
  poetry run jupyter execute /home/mikko/code/trade-executor-phase-aware-alpha/strategies/test_only/09-backtest-capped-waterfall-phase-aware.ipynb --inplace --timeout=3600
```

Errors land in the executed notebook's cell outputs, not the jupyter log — extract
`output_type == "error"` cells to diagnose. Clear all outputs before committing. First run
downloads the 200-vault universe (slow); later runs are cached.

**Review protocol (per package).** Before ANY `codex`/`claude` CLI call, read
`.claude/docs/agent-tricks-and-troubleshooting.md` (blocking repo requirement). Then:
Codex CLI review of the package diff — `codex exec --json --sandbox read-only "…" < /dev/null
> /tmp/….jsonl` (streaming JSONL to a file, never text mode piped to tail; stdin from
`/dev/null`); plus one clean subagent review with fresh context. Address all blocking findings
before commit. Codex's read-only sandbox cannot run pytest — run the focused tests yourself and
say so in the prompt.

**Commit/PR conventions.** One commit per package, `fix:`/`feat:`/`test:` prefix, end with the
`Co-Authored-By: Claude …` trailer. **Commit only when the user asks; never push or update the PR
automatically.** PR description updates go via
`gh api repos/:owner/:repo/pulls/1550 --method PATCH -F body=@/tmp/pr-body-v2.md`
(not `gh pr edit`, which can fail on the Projects-classic GraphQL deprecation). Keep the PR body's
Why / Lessons learnt / Summary structure; add the CU-0 "Deviations from the design plan" section.
If CU-4 lands as its own PR, add a dated CHANGELOG entry. UK English throughout; headings
first-letter-capitalised only; don't format unrelated code.

**Verification bar.** A package is done only when: its new tests pass, the 55-test focused
regression set passes, the notebook still runs clean end-to-end (packages that touch it or the
model), and both reviews (Codex + clean subagent) return no blocking findings.

---

## Review history

- **Fable plan-conformance review (2026-07-02, clean subagent):** source of the findings register.
  Verdict: mechanism faithful (inv. 8 exemplary; reconcile-after-emit an improvement over the
  parent plan), scope under-delivered; CU-0 items pre-merge, the rest fast-follow. Corrections
  applied: charts deferral and focused window are disclosed in the PR body (reviewer saw repo
  artifacts only); the overstatement is in the PR-D commit message (immutable → remediated by
  PR-body honesty, G16).

- **Codex CLI review of this plan (2026-07-02, read-only, grounded):** verified G1/G2/G14 and the
  anchors (`:941`, `:2326`, `:2345` usable) against the branch. Three blocking plan-text fixes,
  all folded in: **[C1]** G5 was overbroad — the *base* async cap path is exercised by
  `test_backtest_async_vault.py:1289,:1397`; only the phase-aware widening through the cap is
  untested (register narrowed). **[C2]** the CU-2 gate-survival recipe was inverted — the gate is
  `max(abs(adjust))`, so a counted parked signal can only raise `max_diff`; the real hazard is
  parking the dominant signal lowering it below threshold (recipe rewritten with happy + bad
  path). **[C3]** CU-3 under-specified the sweep assert — it needs a window-gated target vault
  *plus* a separate sync queue venue wired through YieldManager (added). Non-blocking, folded in:
  inv-3b must assert against the exact `investable_equity` handed to
  `calculate_target_positions()` (in-flight/pending deductions at `allocation.py:170`), and the
  "parent repo cwd" environment instruction was made precise (the worktree's own
  `pyproject.toml`/`poetry.lock` are the trap). Verdict after fixes: grounded and feasible.
