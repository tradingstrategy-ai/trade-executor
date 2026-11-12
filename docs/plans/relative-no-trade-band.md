# Plan: no-trade band relative to the position

Branch `relative-no-trade-band`, worktree
`~/code/trade-executor-worktrees/relative-no-trade-band`, from master
`96e42a131`. Reviewed by Codex
`gpt-6.1-sol` and by a fresh Claude Opus 5.5 agent on 12 November 2026; see
[Independent reviews](#independent-reviews).

**Status, 12 November 2026:** steps 1–4 are done. The engine code, tests and
documentation are implemented, and the shadow comparison and migrated run on a
real backtest passed; see [Implementation status](#implementation-status). The
engine pull request is not opened yet.

## Goal

Move the rebalance no-trade band from strategy modules into `AlphaModel`, and
let its width be relative to each position instead of the whole portfolio. A
strategy's `decide_trades()` must be able to set the band at three levels:

1. **portfolio:** one default rule for every position;
2. **vault category:** a different rule for a group of vaults, for example
   high-frequency (HFT) vaults;
3. **vault:** a different rule for one vault.

Existing strategies must not change behaviour unless they opt in.

## Background

### Where the band is today

The engine has no no-trade band. It has only dollar thresholds in
`AlphaModel.generate_rebalance_trades_and_triggers()`: `min_trade_threshold`
for the whole rebalance, and `individual_rebalance_min_threshold` /
`sell_rebalance_min_threshold` per trade
(`tradeexecutor/strategy/alpha_model.py:2672`, `:2290`).

The band lives in the strategies repository, copied between modules: two on
strategies master (`master-vault-v3.py`, `master-vault-multichain.py`) and four
more on research branches (`master-vault-multichain-daily.py` and three HFT
variants). Each has an
`apply_portfolio_stability_guards()` function containing:

```python
no_trade_band_usd = equity * parameters.target_weight_no_trade_band_pct   # 0.015
...
if (signal.old_value > 0 and signal.position_target > 0
        and abs(signal.position_adjust_usd) < no_trade_band_usd):
    signal.position_adjust_usd = 0.0
    signal.position_adjust_quantity = 0.0
    signal.position_adjust_ignored = True
```

It runs after `calculate_target_positions()` and before
`apply_phase_aware_intent()` and trade generation. The 1.5% value arrived with
`master-vault-v3.py` (strategies PR #47, 12 August 2026) and has never been
tested or tuned.

### Why it needs to change

The band is a share of whole-wallet equity, so it is the same dollar amount for
every position. It was sized for carry vaults at about 10% of the portfolio.
The budgeted HFT tier holds positions of 3–5%:

| Position | Target | Band (1.5% of $120k) | Shortfall needed before a top-up |
| --- | ---: | ---: | ---: |
| Carry vault | $12,000 | $1,800 | 15% |
| HFT vault | $6,000 | $1,800 | 30% |

In the saved vault-notebooks NB06 leader run (UTC [2025-04-01, 2026-10-21)),
HFT top-ups were suppressed on 97% of decisions, about $3,100 per decision, and
HFT holdings averaged 81% of their targets. The HFT strategy module already
carries one special case: HFT reductions skip the band.

## Scope

In scope:
- a reusable band rule and policy in the engine;
- one `AlphaModel` method that applies it and records why each adjustment was
  suppressed;
- unit tests, documentation and a changelog line.

Out of scope, each a separate change:
- migrating strategy modules to the new API (strategies repository);
- choosing band values, which is a research question for vault-notebooks;
- the other strategy guards in `apply_portfolio_stability_guards()` (TVL floor,
  deposit admission, backtest-end deposit suppression);
- redistributing capital that a suppressed top-up leaves idle;
- the 12-slot limit.

## Design

### New module `tradeexecutor/strategy/no_trade_band.py`

```python
class NoTradeBandBasis(enum.Enum):
    #: Band is a share of whole-portfolio equity. Reproduces the strategies' band.
    portfolio_equity = "portfolio_equity"
    #: Band is a share of this signal's target value.
    position_target = "position_target"


@dataclass(frozen=True, slots=True)
class NoTradeBandRule:
    #: Band width as a fraction of the basis. Zero disables the band.
    pct: Percent
    basis: NoTradeBandBasis = NoTradeBandBasis.position_target
    #: Never let the band be narrower than this many dollars.
    min_usd: USDollarAmount = 0.0
    #: Suppress small top-ups of an existing position.
    apply_to_increases: bool = True
    #: Suppress small reductions of an existing position.
    apply_to_reductions: bool = True

    @staticmethod
    def disabled() -> "NoTradeBandRule":
        """A rule that never suppresses anything (``pct=0``)."""


@dataclass(slots=True)
class NoTradeBandPolicy:
    #: Portfolio level: the rule for every signal without a more specific one.
    default: NoTradeBandRule
    #: Category level: rule by category name.
    categories: dict[str, NoTradeBandRule] = field(default_factory=dict)
    #: Vault level: rule by ``(chain_id, lower-case pool address)``.
    pairs: dict[tuple[int, str], NoTradeBandRule] = field(default_factory=dict)
    #: Maps a pair to its category name, or ``None``. Supplied by the strategy,
    #: because the engine has no single notion of a vault category.
    category_function: Callable[[TradingPairIdentifier], str | None] | None = None

    def resolve(self, pair) -> tuple[NoTradeBandRule, str]:
        """Return the rule and the level it came from: "pair", "category" or "default"."""
```

Resolution is most specific first: vault, then category, then portfolio. A
level that has no entry for the pair is skipped. A category name returned by
`category_function` with no rule in `categories` falls through to the default.

The vault key is `(chain_id, address)`, the same key the engine already uses
for manual vault overlays (`PERMANENT_DEPOSIT_CLOSURES`). Internal pair ids
change between universes and are not used. The policy lower-cases the
addresses of its `pairs` keys when it is created and looks a pair up by
`pair.chain_id` and lower-cased `pair.pool_address`, so the case of a key
cannot cause a silent miss.

Rule values are validated when a rule is created:
- `pct` and `min_usd` must be finite and not negative;
- `pct == 0` disables the rule entirely, whatever `min_usd` says. A fixed
  dollar band with no relative part is not offered; the engine's dollar
  thresholds already cover that.

Behavioural values such as the percentages are strategy decisions and belong
in the strategy's `Parameters` (`docs/strategy-parameters.md`). The policy
object is built inside `decide_trades()`, because it holds a callable and may
depend on that cycle's state.

Band width:

```
basis_usd = portfolio_equity          (basis portfolio_equity)
          = signal.position_target    (basis position_target)
band_usd  = max(rule.pct * basis_usd, rule.min_usd)
```

`min_usd` is a per-rule dollar floor. The engine's own dollar thresholds are
one value for the whole rebalance and cannot differ by category or vault. A
dollar cap on the band was considered and dropped: nothing asks for it.

### New method `AlphaModel.apply_no_trade_band()`

```python
def apply_no_trade_band(
    self,
    policy: NoTradeBandPolicy,
    portfolio_equity: USDollarAmount | None = None,
) -> list[TradingPairSignal]:
    """Suppress rebalance adjustments that fall inside the no-trade band.

    Call after calculate_target_positions() and before
    generate_rebalance_trades_and_triggers().

    :return: The signals whose adjustment was suppressed this cycle.
    """
```

For each signal the method suppresses the adjustment when all of these hold:
- the signal is not carried forward (`carry_forward_position` is false);
- it adjusts an existing position: `old_value > 0` and `position_target > 0`,
  so new entries always pass;
- its weight is not below the close threshold: `normalised_weight` is neither
  zero nor under `close_position_weight_epsilon`. Trade generation closes any
  such position whatever the sign of its adjustment
  (`alpha_model.py:2564-2567`), so a positive target alone does not mean the
  position stays open. The band leaves these signals to the engine's close
  logic in both directions. This is wider than `_is_full_close_intent()`,
  which covers only negative adjustments;
- the adjustment is non-zero and not already ignored;
- the resolved rule applies to its direction (`apply_to_increases` for
  `position_adjust_usd > 0`, `apply_to_reductions` for `< 0`);
- `abs(position_adjust_usd) < band_usd` (strict, as today).

Suppression does exactly what the strategies do now: it sets
`position_adjust_usd` and `position_adjust_quantity` to zero and
`position_adjust_ignored` to true. `position_target` is left unchanged, so
diagnostics still show the gap between target and holdings.

Recorded on each suppressed signal:
- a new flag `TradingPairSignalFlags.inside_no_trade_band`;
- `signal.other_data["no_trade_band"]` with `band_usd`, `basis`, `pct`,
  `level` and `suppressed_adjust_usd`. All values are JSON-safe, because
  signals are serialised into state.

The method asserts that `calculate_target_positions()` has run: every signal
must have its `synthetic_pair` mapped. A populated `position_target` is not
enough, because `normalise_weights()` can already set it. It also asserts that
every signal it evaluates is unleveraged and not flipping. Leveraged and short
signals are out of scope, as they are for `cap_buys_to_sync_cash`.
`portfolio_equity` is required only when a resolved rule uses the
`portfolio_equity` basis; the method asserts it is given in that case.

The band applies to the adjustment the strategy requests. It is not a minimum
executed size: a later cash cap can still shrink a surviving buy below the
band. A partial fill can also leave the remaining gap inside the band, where
it waits until drift widens it.

### How a target-relative band behaves

- **It does not oscillate.** A trade that passes the band moves the holding to
  its target, unless a later cash cap or threshold shrinks it, after which the
  gap is zero or small.
- **It follows target noise.** The band is a share of the target, so a target
  that moves also moves the trade trigger. Two sources matter here. Targets
  capped by pool size move with daily TVL. A ceiling of the form
  `budget ÷ N` jumps by 17–20% when N changes between 5 and 6, which is
  outside a 15% band, so each change trades. The dollar band absorbed more of
  this. Expect more trades; turnover and trade count must be measured in the
  research variant.
- **One-sided rules bias the holding.** If reductions bypass the band and
  top-ups do not, yield drift above target is trimmed almost at once while
  shortfalls wait for the full band. Holdings then sit between about
  `(1 − pct)` and 1.0 of target. If shortfalls build up evenly, that averages
  roughly 0.92 at 15%, a rough estimate rather than a measurement; today's
  shortfall is 81% of target.

### Interaction with the phase-aware model

`apply_phase_aware_intent()` skips zeroed adjustments, and then closes any open
park event whose vault was neither re-parked nor made a promotion candidate
(`phase_aware.py:460`, `:483-499`). So when the band suppresses a top-up of a
vault that has capital parked for a closed deposit window, the park is
released. The cash stays in the queue venue and competes again next cycle.

This is the behaviour today, because the strategies' band runs at the same
place, and it matches how a refused entry releases its earmark. The plan keeps
it and documents it. The call order also stays: applying the band after the
phase-aware step would miss adjustments that parking has already zeroed.

The band is a separate, explicit call, not a new argument of
`generate_rebalance_trades_and_triggers()`. Strategies record their own state
between the guard and trade generation, and the phase-aware model runs
`apply_phase_aware_intent()` there. An explicit call keeps that order visible
and lets `decide_trades()` inspect the returned signals. The engine's dollar
thresholds still apply afterwards, unchanged.

### Control from `decide_trades()`

The strategy builds the policy each cycle, so every level can depend on
parameters or on that cycle's state:

```python
policy = NoTradeBandPolicy(
    # Portfolio level
    default=NoTradeBandRule(pct=parameters.no_trade_band_pct),
    # Category level: a narrower band for HFT vaults, in both directions
    categories={"hft": NoTradeBandRule(pct=parameters.hft_no_trade_band_pct)},
    # Vault level: one vault with its own rule
    pairs={(9999, "0x…"): NoTradeBandRule.disabled()},
    category_function=lambda pair: "hft" if is_hft(pair) else None,
)
suppressed = alpha_model.apply_no_trade_band(policy)
```

The HFT strategy today lets HFT reductions bypass the band, so that a holding
above its ceiling is always trimmed. That shape is
`NoTradeBandRule(pct=…, apply_to_reductions=False)`. It keeps ceilings tight
but biases holdings below target, as described above. Which shape to use is a
research decision; both are arms of the band-width variant.

The strategies' current behaviour is expressible as a policy, which is what
makes a regression check possible:

```python
NoTradeBandPolicy(
    default=NoTradeBandRule(pct=0.015, basis=NoTradeBandBasis.portfolio_equity),
    categories={"hft": NoTradeBandRule(pct=0.015, basis=NoTradeBandBasis.portfolio_equity,
                                       apply_to_reductions=False)},
    category_function=lambda pair: "hft" if is_hft(pair) else None,
)
```

"Equivalent" here means the same trades and the same economic result for the
same inputs. It does not mean identical diagnostics. Known differences:
- **Closes.** The strategy guard could suppress the adjustment of a position
  whose weight is below `close_position_weight_epsilon`, in either direction,
  which blocked its close; the engine method never does. This is a deliberate
  correction.
- **Zero adjustments.** The strategy guard marks them ignored and tags them
  with `stability_no_trade_band_usd`; the engine method leaves them alone.
- **Diagnostic keys.** `stability_no_trade_band_usd` and
  `hft_band_bypass_reduction` are replaced by the flag and the
  `no_trade_band` record.

A migrating strategy replaces only the band block and the HFT-reduction bypass.
The TVL, admission and backtest-end guards stay in the strategy and keep
running before the band, and the strategy passes the same
`portfolio.get_total_equity()` figure it uses today.

## Compatibility

- Nothing calls the new method by default, so existing strategies and saved
  results are unaffected.
- The new flag is an added enum member. States written before the change load
  as before.
- No state schema, persistence or recovery change.

## Validation

1. **Unit tests** in `tests/units_tests/test_alpha_model_no_trade_band.py`,
   with hand-built signals and no network or market data:
   - resolution order: vault over category over portfolio; unknown category
     falls through; pair keys match regardless of address case;
   - each basis, and the `min_usd` floor;
   - direction switches; `disabled()`; `pct == 0` with a positive `min_usd`;
   - invalid rules are rejected (negative or non-finite values);
   - a rule on the `portfolio_equity` basis without `portfolio_equity` fails;
   - new entries and carried-forward signals are never suppressed;
   - a signal below `close_position_weight_epsilon` is never suppressed, with
     a negative and with a positive adjustment;
   - a target that shrinks and then grows between cycles trades when the gap
     exceeds the band and not otherwise;
   - calling the method twice changes nothing the second time;
   - zero and already-ignored adjustments are left untouched;
   - leveraged or flipping signals and a model without calculated targets
     fail the assertions;
   - the strict `<` boundary;
   - the legacy policy gives the same suppressed non-zero adjustments as the
     strategies' formula on the same signals;
   - the flag and `other_data` record survive a state serialisation round
     trip, with `basis` stored as a string.
2. **Pipeline tests**, using the fixtures of the existing alpha-model tests:
   - a suppressed signal produces no trade from
     `generate_rebalance_trades_and_triggers()`;
   - a suppressed funding sell with a surviving buy: the cash caps size the buy
     from the sells that remain;
   - when every remaining adjustment is below `min_trade_threshold`, the
     whole-cycle gate still cancels the cycle;
   - a suppressed HyperCore reduction is skipped by sell preparation;
   - phase-aware: a suppressed top-up with an existing park releases the park,
     for both a closed window and an open-window promotion candidate.
3. **Shadow comparison on a real backtest** (vault-notebooks, after the
   engine change). This is the acceptance test. One run of a test copy of
   `master-vault-multichain-hft-budget.py` on this worktree. The copy keeps
   the strategy's own band in control of trading, so the run follows the
   baseline path on every cycle. In addition, each cycle it:
   - deep-copies the signals after the TVL, admission and backtest-end guards
     and before the strategy's band;
   - applies the engine method with the legacy policy to the copy;
   - records, per pair, whether each side suppressed the adjustment and the
     amount.

   Every cycle is therefore compared from the same starting state. Each
   mismatch must be a signal below the close threshold (the documented
   difference) or it is a defect. Two full runs cannot serve as the
   acceptance test: the first differing close changes every later trade and
   equity value, leaving nothing to compare.
4. **Migrated run, reported separately.** A second copy that trades through
   the engine method with the legacy policy. If the shadow comparison found no
   mismatches, its trades, positions, cash, park and promote events and all
   568 daily equity values must equal the first run's. If it found
   close-threshold mismatches, the two runs are expected to part at the first
   one, and the report says where. Both runs must read the same frozen inputs
   and caches, with no fresh downloads. The saved NB07 run (final $124,521.97)
   used eth-defi `9b7aa11ef` and this worktree pins `d8f439888`; whether the
   baseline still matches the saved run is recorded, but is not an acceptance
   criterion.
5. **Existing tests:** `test_alpha_model_priority.py`,
   `test_alpha_model_sync_cash_cap.py` and `test_phase_aware_alpha_model.py`
   pass unchanged.

## Documentation

- A "No-trade band" subsection in `.claude/docs/alpha-model.md`, between
  "Old weights and dollar targets" and "Trade generation", with the wiring
  order, the three levels, and the note that the band applies to requested
  adjustments, not executed sizes.
- A paragraph in `.claude/docs/phase-aware-alpha-model.md` on a band-suppressed
  top-up releasing its park.
- One `CHANGELOG.md` line, dated the day of the PR.

## Risks and open questions

1. **Hyperliquid lock-ups.** A relative band lets through smaller, more
   frequent top-ups. Whether a top-up restarts a Hyperliquid vault's
   withdrawal lock-up is a protocol fact this plan has not confirmed. What the
   code shows:
   - live redemption checks the user's lock-up expiry
     (`hypercore_valuation.py:444-461`);
   - historical redemption checks take a position argument but do not use it,
     and consult vault closure only (`backtest_pricing.py:797-809`). So a
     backtest does not simulate an account's lock-up.

   A backtest of a narrower band can therefore overstate how freely a
   Hyperliquid position can be trimmed. Modelling lock-ups is outside this
   change. Settling it is a precondition of the band-width research variant:
   that variant's plan must either confirm the reset rule or add lock-up
   handling before its results are used. `min_usd`
   limits how small top-ups can get, but it is a strategy tolerance and does
   not replace protocol minimums, fees or settlement gates.
2. **Sells that fund buys.** Suppressing a small sell removes cash a buy in the
   same cycle may rely on. This already happens with the strategies' band, and
   the cash caps run afterwards on the zeroed adjustments and the yield
   manager works from the trades that are emitted. The pipeline test and the
   shadow comparison cover it.
3. **More trades.** A target-relative band follows target noise (see above).
   The research variant must report turnover, trade count and fees beside
   CAGR, Sharpe and drawdown.
4. **Basis choice.** Only two bases are proposed. A basis of the current
   holding, or the larger of holding and target, is deferred until a research
   result asks for it.
5. **The relative band alone is a partial fix.** The band was expected to
   account for a minority of NB07's idle cash. The backtest measured it at
   about 2 of 22.5 points of equity; see
   [Implementation status](#implementation-status).

## Steps

1. Add `no_trade_band.py`, the flag and `AlphaModel.apply_no_trade_band()`.
2. Add the unit and pipeline tests; run them and the three existing
   alpha-model test files.
3. Update `.claude/docs/alpha-model.md`, `.claude/docs/phase-aware-alpha-model.md`
   and `CHANGELOG.md`.
4. In the strategies worktree, make the two test copies of the HFT budget
   module and run the shadow comparison, then the migrated run. In the
   migrated copy the band must be applied before the strategy writes its
   `hft_budget` record, so the recorded adjustments still match the trades.
5. Open the engine PR. The band-width experiment is a separate vault-notebooks
   variant.

## Independent reviews

### Codex `gpt-6.1-sol`

Codex `gpt-6.1-sol` reviewed this plan read-only on 12 November 2026 against
the engine and strategy code. Verdict: ready with listed changes. It judged the
call location, the target-relative basis and the three-level policy sound and
proportionate. Each finding was checked against the code and accepted; the plan
above includes the changes.

| Severity | Finding | Change to the plan |
| --- | --- | --- |
| MEDIUM | A positive target does not mean the position stays open: the engine fully closes sub-epsilon weights, and the band could have suppressed that close | Full closes are exempt, with a test; recorded as a deliberate difference from the strategy guard. The second review widened the exemption to both directions |
| MEDIUM | Suppressing a top-up before the phase-aware step closes its existing park | Behaviour specified (the park is released, as today), tested for both window states and documented |
| MEDIUM | "Exact legacy equivalence" was too strong: zero and already-ignored adjustments and diagnostic keys differ, and guard order matters | Equivalence defined as same trades and economics; differences listed; migration replaces only the band block |
| MEDIUM | Comparing daily equity with a saved run on different dependencies is inconclusive | Regression moved to one worktree with identical dependencies, comparing trades, positions, cash and park events; the saved-run comparison is reported separately. The second review replaced the two-run design with a shadow comparison |
| MEDIUM | Rule inputs were unvalidated, `pct == 0` with `min_usd > 0` was ambiguous, and a populated `position_target` does not prove targets were calculated | Validation rules added; zero disables; precondition is a mapped `synthetic_pair` |
| LOW | The lock-up evidence was imprecise: zero locked capital at daily decisions does not show the backtest ignores lock-ups | Risk reworded around what the code shows; the reset rule is marked unconfirmed |
| LOW | Tests and documentation should cover the downstream contract | Pipeline tests added for a suppressed funding sell, the whole-cycle gate and HyperCore; both guides updated |

### Claude Opus 5.5, fresh agent

A second read-only review on 12 November 2026, by an agent with no earlier
context, told to verify the Codex summary instead of trusting it. Verdict:
ready with listed changes, no redesign. It confirmed the code descriptions,
line references, vault key, call location and the downstream handling of a
suppressed signal. Findings were checked against the code.

| Severity | Finding | Change to the plan |
| --- | --- | --- |
| HIGH | Two full runs cannot prove equivalence: the full-close exemption changes one close, after which every trade and equity value differs | Acceptance test is now a per-cycle shadow comparison inside one baseline run; the migrated run is reported separately |
| MEDIUM | The example HFT rule (reductions bypass the band) keeps holdings at about 92% of target | Example is symmetric; the one-sided shape and its bias are described; both are research arms |
| MEDIUM | A target-relative band trades more when targets are noisy (pool caps, `budget ÷ N` changes); partial fills can stay inside the band | New "How a target-relative band behaves" section; turnover and trade count required in the research variant; a shrinking and growing target test |
| MEDIUM | The engine closes sub-epsilon positions whatever the sign of the adjustment, but `_is_full_close_intent()` covers only negative ones | Exemption now tests the weight against the close threshold in both directions |
| MEDIUM | `max_usd` has no requirement; `min_usd` overlaps engine thresholds | `max_usd` dropped. `min_usd` kept, because engine thresholds cannot differ by category or vault |
| LOW | `docs/strategy-parameters.md` does not require building callables in `decide_trades()` | Citation corrected |
| LOW | `portfolio_equity` is needed only for the equity basis | Made optional, asserted when needed |
| LOW | "Six modules" is true only of the research worktree | Reworded: two on strategies master, four on research branches |
| LOW | The lock-up risk named no owner or gate | Made a precondition of the research variant |
| LOW | The migrated module must apply the band before its `hft_budget` record | Added to the steps |
| LOW | Extra tests: idempotent second call, `basis` stored as a string | Added |

Not adopted:
- **Plan location.** The reviewer noted other plans live in `.claude/plans/`.
  Both directories exist, and recent plans are in `docs/plans/`; this one
  stays there.
- **Reporting unmatched `pairs` keys.** The method sees only that cycle's
  signals and cannot tell a mistyped address from a vault that is not
  targeted. The recorded `level` shows when a vault rule applied.

## Implementation status

12 November 2026.

Engine, this branch:
- `tradeexecutor/strategy/no_trade_band.py`: `NoTradeBandBasis`,
  `NoTradeBandLevel`, `NoTradeBandRule`, `NoTradeBandPolicy`.
- `AlphaModel.apply_no_trade_band()` and the `inside_no_trade_band` flag in
  `tradeexecutor/strategy/alpha_model.py`.
- Tests: `tests/units_tests/test_alpha_model_no_trade_band.py` (rules,
  suppression and exemptions, legacy-formula equivalence on 300 random
  signals, rejected input, HyperCore sell preparation, serialisation);
  `test_alpha_model_no_trade_band_sell_does_not_fund_buy` in
  `tests/test_rebalance.py` (real `PositionManager`, sync cash cap);
  `test_no_trade_band_releases_park_and_can_cancel_cycle` in
  `tests/units_tests/test_phase_aware_alpha_model.py`. These, the three
  existing alpha-model test files and `tests/test_rebalance.py`: 73 passed.
- `.claude/docs/alpha-model.md`, `.claude/docs/phase-aware-alpha-model.md` and
  `CHANGELOG.md` updated.

The repository's test guidance asks for few tests with several asserts each, so
the unit cases above are grouped into four tests rather than one per bullet.

Strategy test copies, `~/code/strategies-worktrees/vault-of-vaults-hft`:
- `master-vault-multichain-hft-budget-band-shadow.py`: the parent's own band
  trades; the engine method runs on a deep copy of each cycle's signals after
  the entry guards, and the comparison is saved per cycle as
  `no_trade_band_shadow` in State.
- `master-vault-multichain-hft-budget-band-engine.py`: the engine method with
  the legacy policy replaces the parent's band and HFT-reduction bypass.

Both split the parent's single guard loop into the entry guards, then the band;
signals refused by an entry guard are not revisited, as the parent's
`continue` statements ensured. An offline check, not a backtest, ran the three
guard functions on the same 20 × 400 random signals with a stub pricing model:
- the shadow copy left every signal identical to the parent (adjustment,
  ignored flag and `other_data` keys) in all 8,000 cases;
- the engine copy differed from the parent in 93 cases, all below the close
  threshold, and the shadow record listed exactly those 93 as mismatches.

### Backtest validation (validation steps 3 and 4)

Run on 12 November 2026 as vault-notebooks NB08
(`notebooks/vault-of-vaults-hft/08-backtest-relative-no-trade-band.ipynb`,
branch `vault-of-vaults-hft`), UTC [2025-04-01, 2026-10-21), on this branch at
`190dc4b27` put first on `PYTHONPATH`, with the shared checkout's
trading-strategy `c56f379bc` and eth-defi `9b7aa11ef`:
- the unmodified NB07 module on this engine reproduced the saved NB07 run
  exactly (maximum difference $0.00 over 568 daily equity values);
- **shadow comparison:** 568 cycles compared, 5,352 adjustments suppressed by
  the strategy's band and the same 5,352 by the engine band, no disagreements;
- **migrated run:** the engine band with the legacy policy equalled the
  unmodified module on every day.

The close-threshold difference did not occur in this backtest. The eth-defi
revision question did not arise, because the run used the shared checkout's
eth-defi, not this worktree's pinned submodule.

The same notebook ran five target-relative arms. They are research results,
recorded in vault-notebooks pull request 17, not acceptance evidence for this
change. Two findings bear on how the band is used:
- a rule that also holds back reductions lets holdings drift above a
  strategy's own ceilings by up to the band (HFT holdings reached 11.6% above
  their budget), so whether reductions bypass the band is a risk-limit choice;
- no Hyperliquid reduction fell inside a lock-up, assuming every top-up
  restarts it, but a 10% band produced 190 reductions within an hour after
  one ended, against 18 for the legacy band.
