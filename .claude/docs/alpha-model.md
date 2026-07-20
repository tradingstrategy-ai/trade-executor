# Alpha model

What the synchronous `AlphaModel` does, how a strategy wires it into
`decide_trades`, and where each stage of the pipeline lives. This is the base
model documentation; the window-gated subclass has its own deep-dive in
`phase-aware-alpha-model.md`, and the same-cycle financing caps are covered from
the strategy side in `vault-deposit-redeem.md`.

The canonical reference implementation is
[strategies/hyper-ai.py](../../strategies/hyper-ai.py) — a long-only
vault-of-vaults strategy whose `decide_trades` exercises every stage described
here. Line references below point at it.

## What it does

`AlphaModel` (in
[tradeexecutor/strategy/alpha_model.py](../../tradeexecutor/strategy/alpha_model.py))
turns per-pair conviction numbers into a concrete list of rebalance trades, one
strategy cycle at a time. The mental model is a four-stage funnel:

```
signals  ─►  weights  ─►  dollar targets  ─►  trades
set_signal    assign_weights   calculate_target   generate_rebalance
select_top    normalise_weights  _positions       _trades_and_triggers
```

Each cycle the strategy creates a **fresh** `AlphaModel(timestamp)`, feeds it
raw signals, lets it compute target position values, and asks it to diff those
targets against the current portfolio. The output is a sorted list of
`TradeExecution` objects (sells before buys, so sells fund the buys). The model
instance is also serialised into state per cycle, which is what powers the
diagnostics charts.

One `TradingPairSignal` object per pair carries the pair through the whole
funnel: raw `signal`, `raw_weight`, `normalised_weight`, `old_weight` /
`old_value` (current portfolio), `position_target`, `position_adjust_usd`, and a
set of diagnostic `TradingPairSignalFlags` explaining every cap or skip applied
along the way.

## Wiring in decide_trades

The stage order matters. From `hyper-ai.py` (see
[hyper-ai.py:351-452](../../strategies/hyper-ai.py#L351)), slightly abridged:

```python
alpha_model = AlphaModel(timestamp, close_position_weight_epsilon=parameters.min_portfolio_weight_pct)

for pair_id in included_pairs:                       # 1. signal generation
    alpha_model.set_signal(pair, weight_signal_value)

locked = alpha_model.carry_forward_non_redeemable_positions(position_manager)
portfolio_target = calculate_portfolio_target_value(position_manager, parameters.allocation_pct)
deployable = max(portfolio_target - locked, 0.0)

alpha_model.select_top_signals(count=parameters.max_assets_in_portfolio)   # 2. selection
alpha_model.assign_weights(method=weight_passthrouh)                       # 3. weighting

alpha_model.normalise_weights(                                             # 4. normalisation + risk caps
    investable_equity=deployable,
    size_risk_model=USDTVLSizeRiskModel(pricing_model, per_position_cap=...),
    max_weight=parameters.max_concentration_pct,
    max_positions=parameters.max_assets_in_portfolio,
    waterfall=True,
)
alpha_model.update_old_weights(state.portfolio, ignore_credit=False)       # 5. current portfolio
alpha_model.calculate_target_positions(position_manager)                   # 6. dollar targets

trades = alpha_model.generate_rebalance_trades_and_triggers(               # 7. trades
    position_manager,
    min_trade_threshold=...,
    individual_rebalance_min_threshold=...,
    sell_rebalance_min_threshold=...,
    cap_buys_to_sync_cash=True,          # hyper-ai opts into the sync cash cap
    sync_cash_headroom_usd=...,
)
```

`carry_forward_non_redeemable_positions()` must run **before**
`select_top_signals()`: it pins positions that cannot be sold this cycle
(lock-ups, pending settlements) at their current value so top-N selection does
not evict a position the strategy could not exit anyway, and returns the locked
value the strategy subtracts from its deployable target.

## Signal generation

`set_signal(pair, alpha, ...)` records one signal per pair. The signal is any
float the strategy believes in — a momentum score, a trailing Sharpe, or a
precomputed weight. Zero means "no position / close". Optional per-signal inputs
include `stop_loss`, `take_profit` and `leverage`.

The model does not compute signals itself; that is indicator work. In
`hyper-ai.py` the signal is the `age_ramp_weight` indicator (vaults ramp from
zero weight to full weight over 0.75 years of track record), and pairs first
pass an `inclusion_criteria` indicator (TVL, age, data-availability filters)
before `set_signal` is called at all.

## Selection

`select_top_signals(count, threshold)` keeps the `count` strongest signals by
absolute value (optionally requiring a minimum `threshold`) and discards the
rest. Carried-forward (non-redeemable) signals occupy slots but are never
evicted. This is where "we track 120 vaults but hold at most 20 positions"
happens.

## Weighting

`assign_weights(method)` converts raw signals to unnormalised weights. Methods
live in [tradeexecutor/strategy/weighting.py](../../tradeexecutor/strategy/weighting.py):

| Method | Weighting |
|---|---|
| `weight_by_1_slash_n` | 1/1, 1/2, 1/3… by signal rank (default) |
| `weight_equal` | equal weight to every included pair |
| `weight_passthrouh` | signal value used as-is — for strategies whose indicator already outputs a weight (hyper-ai) |
| `weight_by_1_slash_signal` | inverse of the signal value |
| `weight_by_softmax` | softmax with a temperature parameter |
| `weight_by_log` / `weight_by_blend` | log-dampened / blended variants |

## Normalisation and risk caps

`normalise_weights(...)` scales weights to sum to 1.0 and applies every
portfolio-level risk control in one pass:

- **`investable_equity`** — the dollar pool the weights allocate. The strategy
  decides what is investable (hyper-ai: `portfolio target − locked capital`).
- **`max_weight`** (+ optional `max_weight_function`) — per-asset concentration
  cap; excess weight is redistributed or discarded.
- **`size_risk_model`** — per-pair dollar caps from market depth. Hyper-ai uses
  `USDTVLSizeRiskModel` so a position may not exceed a fixed % of the vault's
  TVL. Discarded allocation is tracked (`size_risk_discarded_value`).
- **`max_positions`** — hard position-count cap after size risk shuffling.
- **`waterfall=True`** — survivor-first allocation: fill the strongest signal to
  its cap, then the next, instead of pro-rata scaling everyone down.
- **`max_protocol_weight`** / `cap_chain_allocation()` — optional per-protocol
  and per-chain allocation ceilings (flags `capped_by_protocol_allocation`,
  `capped_by_chain_allocation`).

Signals capped here carry the corresponding flag, so the diagnostics table
shows *why* a position is smaller than its raw signal implied.

## Old weights and dollar targets

`update_old_weights(portfolio)` records what the portfolio currently holds
(`old_weight`, `old_value` per signal), so the model knows the starting point.
Queue/bridge/credit positions can be excluded via predicate — see the
phase-aware subclass for an example override.

`calculate_target_positions(position_manager)` turns normalised weights into
`position_target` dollars and the per-pair difference into
`position_adjust_usd` (positive = buy, negative = sell). Settlement-aware
details (pending deposits valued into old value, non-redeemable positions
pinned) are documented in `vault-deposit-redeem.md`.

## Trade generation

`generate_rebalance_trades_and_triggers(position_manager, ...)` diffs targets
against the portfolio and emits trades, applying gates in this order:

1. **Whole-portfolio gate** — if the largest single adjustment is below
   `min_trade_threshold`, the entire rebalance is skipped (flag
   `max_adjust_too_small`). All-or-nothing, because dropping only some legs
   would break the sells-fund-buys pairing.
2. **Same-cycle cash caps** — the always-on **async cap** (queued ERC-7540 /
   Ostium redemptions pay out later, so buys are scaled to the cash that
   actually arrives; flag `capped_by_pending_settlement_cash`) and the
   **opt-in sync cap** (`cap_buys_to_sync_cash=True`: sells dropped by the
   min-trade gate free no cash either, so buys are scaled to the sells that
   actually execute; flag `capped_by_sync_cash`). Opt-in rules and the worked
   hyper-ai example live in `vault-deposit-redeem.md`.
3. **Per-signal gates** — problematic/frozen pairs, buys whose vault deposit
   window is closed (`cannot_deposit`), sells below
   `sell_rebalance_min_threshold` or below the pair dust epsilon
   (`individual_trade_size_too_small` / `individual_trade_quantity_too_small`),
   positions with a pending vault settlement (`settlement_pending`), and
   positions not yet redeemable (`cannot_redeem`).
4. **Emission** — position closes for signals under
   `close_position_weight_epsilon`, adjust trades for the rest, plus optional
   stop-loss/take-profit triggers. Trades are returned sorted by
   `get_execution_sort_position()`: credit withdrawals → closes → vault
   redemptions → sells → bridge legs → buys → vault deposits → credit supply,
   so cash is always released before it is spent.

Thresholds exist because tiny rebalances cost more in fees and execution risk
than they earn: hyper-ai floors both at Hyperliquid's $5 hard minimum
([hyper-ai.py:178-220](../../strategies/hyper-ai.py#L178)).

## Diagnostics and charts

Every stage writes its reasoning onto the signal, and the per-cycle model
snapshot is stored in state, so post-hoc "why did it trade like that?" is
answerable without re-running the strategy:

- **`format_signals(alpha_model)`** — one DataFrame row per signal: raw signal,
  old/new weight, target value, adjust, flags, pending settlement columns. This
  is the primary debugging table.
- **`alpha_model_diagnostics`** chart
  ([chart/standard/alpha_model.py](../../tradeexecutor/strategy/chart/standard/alpha_model.py))
  — renders that table for the last cycle in the web/notebook UI. Related:
  `skipped_signals` (weights blocked by closed deposit windows) and
  `missed_vault_deposit_redemption_events` / `_timeline`.
- **Weight charts** ([chart/standard/weight.py](../../tradeexecutor/strategy/chart/standard/weight.py))
  — `equity_curve_by_asset` (stacked equity bands per asset, curator/chain
  symbols in the legend), `equity_curve_by_chain`, and
  `weight_allocation_statistics`.
- **Strategy thinking** — hyper-ai composes a per-cycle text report (cash,
  redeemable capital, investable equity, allocated value, discarded-by-liquidity
  value, trades decided; [hyper-ai.py:465-514](../../strategies/hyper-ai.py#L465))
  and stores it via `state.visualisation.add_message`, surfaced by the
  `last_messages` chart.
- **Flag rollup** — `get_flag_diagnostics_data()` aggregates flag counts per
  cycle; `get_unallocatable_signals()` and `get_missed_vault_events()` expose
  liquidity-capped and window-blocked allocations.

Chart registration happens in the strategy's `create_charts()`
([hyper-ai.py:779-807](../../strategies/hyper-ai.py#L779)); everything above is
registered there and rendered in backtest notebooks and the live web UI alike.

## Cross-references

| Doc / code | What it covers |
|---|---|
| [strategies/hyper-ai.py](../../strategies/hyper-ai.py) | Reference implementation of the full pipeline (long-only, vault universe, waterfall + TVL size risk, sync cash cap opt-in) |
| `phase-aware-alpha-model.md` | `PhaseAwareAlphaModel` subclass: window-gated deposits parked in a queue vault, the three base hooks, correctness invariants |
| `vault-deposit-redeem.md` | Async vault lifecycle, settlement-aware sizing, both same-cycle cash caps and the sync-cap opt-in rules |
| [tradeexecutor/strategy/alpha_model.py](../../tradeexecutor/strategy/alpha_model.py) | `AlphaModel`, `TradingPairSignal`, `TradingPairSignalFlags`, `format_signals` |
| [tradeexecutor/strategy/weighting.py](../../tradeexecutor/strategy/weighting.py) | Weighting methods and weight normalisation helpers |
| [tradeexecutor/strategy/size_risk_model.py](../../tradeexecutor/strategy/size_risk_model.py) | Size-risk base class; `USDTVLSizeRiskModel` in `tvl_size_risk.py` |
| [tests/units_tests/test_alpha_model_sync_cash_cap.py](../../tests/units_tests/test_alpha_model_sync_cash_cap.py) | Sync cash cap unit coverage |
