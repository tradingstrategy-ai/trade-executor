# Plan: point-in-time HyperCore deposit availability

## Goal

Make Hyper-AI use the same source-backed open/closed state in backtesting and
live trading without excluding a vault from its whole history because deposits
are closed today.

For HyperCore vaults:

- before **2026-04-11 00:00 UTC**, assume deposits were open;
- from that cutoff onwards, use only deposit state that had been observed by
  the decision timestamp;
- after the cutoff, block a new deposit when state is missing or stale;
- never treat deposit closure as a reason to sell an existing position; and
- leave redemption behaviour unchanged because the historical dataset contains
  no `redemption_open` observations.

The implementation should reuse the existing vault-state DataFrame,
`PricingModel.can_deposit()` / `check_deposit()`, and Hyper-AI v7 candidate
backfill. It must not add a second availability subsystem.

## Evidence for the cutoff

The exact 2026-09-20 backtest cache at
`cache/hyper-ai-v8/vaults/downloads/vault-prices.parquet` contains 1,574,358
HyperCore rows across 610 vaults. The source is the same website export used by
the v8 run; the cache also contains non-HyperCore rows, which are excluded from
these counts.

For the Hyper-AI eligibility floor of TVL >= 7,500 USD, daily explicit
`deposits_open` coverage is:

| Period | Coverage |
|---|---:|
| 2026-04-06 | 98.19% |
| 2026-04-07 | 98.22% |
| 2026-04-08 | 98.93% |
| 2026-04-09 | 99.29% |
| 2026-04-10 | 99.29% |
| From 2026-04-11 | 100% |

Earlier rows contain sparse or retrospectively written observations, so they
are deliberately covered by the pre-cutoff assumed-open rule rather than
treated as a complete historical series.

The 2026-04-10 observations became complete only late in that UTC day. Using
`written_at` as the information-availability timestamp, all then-eligible
vaults had an explicit state at or before 2026-04-11 00:00 UTC. After rounding
observation time up to the daily decision grid, this makes 2026-04-11 the first
safe daily decision boundary.

An older 2026-09-14 local snapshot also contains 8,078 HyperCore rows with a
null `written_at`; all of those rows also have null `deposits_open`. They are
unknown rows, not explicit open/closed observations, and must be tolerated
without making them usable state.

The same parquet has no non-null HyperCore values for `redemption_open`,
`max_deposit`, or `max_redeem`. This plan must not manufacture historical
redemption restrictions or capacity limits.

## Current problems

### Historical state can be visible too early

`convert_vault_prices_to_vault_state()` currently timestamps state with the
price observation and floors the result to the start of the candle bucket. A
state written late in the day can therefore appear available to a decision at
the start of that day.

For observed HyperCore open/closed transitions, `written_at - timestamp` has a
median of about 2.4 hours and an observed maximum of about 19.8 hours. The
backtest must use when the scanner knew the state, not the earlier portfolio
mark associated with it.

### Unknown state has one rule for all dates

`BacktestPricing` currently treats missing, unknown, and stale state as open at
all times. That is the requested assumption before the cutoff, but it hides
data failures after reliable collection began.

### Current closure removes historical membership

The Hyperliquid vault curator defaults to excluding any vault whose current
metadata has `deposit_closed_reason`. This causes a closed-now vault to
disappear from all earlier backtest dates and can remove a live incumbent from
the ranking universe even though deposit closure only prohibits adding money.

### Reason text is not the state contract

HyperCore uses a low leader-fraction reason as a conservative deposit gate,
while the metadata exporter describes it as a warning for public-permission
classification. The implementation must preserve the existing Hyper-AI live
policy, but backtest and live code must both decide through
`can_deposit()`/`check_deposit()` rather than by interpreting a non-empty reason
string inside the strategy.

## Scope

In scope:

- HyperCore `deposits_open` history from the website vault-price parquet.
- Observation-time-safe daily resampling.
- The 2026-04-11 assumption cutoff.
- Backtest behaviour for missing state before and after the cutoff.
- Inclusion of currently deposit-closed vaults in Hyper-AI v7's source
  universe.
- Focused unit tests and a v7 comparison backtest.

Out of scope:

- Historical redemption availability; there is no source data.
- Partial deposit or redemption capacity; there is no source data.
- Reconstructing a second leader-fraction rule. The archived `deposits_open`
  field is already derived from the exporter’s existing HyperCore closure gate,
  including its low-leader-fraction condition.
- A new `vault-state.parquet`, manifest schema, availability service, or
  general policy registry.
- Changes to HyperCore routing, settlement, repair, or persisted strategy
  state.
- v6 compatibility, rollback loading, migrations, or dual old/new behaviour.
- Reworking `AlphaModel`; v7 already gates unheld candidates before selection
  and promotes the next ranked candidate.

## Behaviour contract

| Decision and state | Result |
|---|---|
| HyperCore decision before 2026-04-11 | Deposit allowed, regardless of archived state |
| At/after cutoff, fresh explicit `True` | Deposit allowed |
| At/after cutoff, fresh explicit `False` | Deposit blocked with the archived reason |
| At/after cutoff, missing, unknown, or stale | New deposit blocked as unavailable data |
| Non-HyperCore vault | Existing behaviour unchanged |
| Existing position in a deposit-closed vault | May remain selected; positive adjustment blocked |
| Unheld deposit-closed vault | Skipped before top-N is filled; next ranked eligible vault promoted |
| Position selected for exit | Deposit state is irrelevant; redemption rules decide whether it can exit |

The cutoff applies to deposit decisions only. It must not affect valuation,
TVL inclusion, ranking indicators, or redemptions.

## Implementation

### 1. Make state effective when it was observed

Change the website vault-history load in
`tradeexecutor/strategy/trading_strategy_universe.py` to read `written_at`
alongside `chain`, `address`, `timestamp`, and the existing vault-state
columns.

Treat `written_at` as an optional source column in
`read_vault_price_history_parquet()`, just like the existing optional state
columns. Older website caches and the bundled daily parquet do not contain it
and must continue to load. This exception must be limited to `written_at`; keep
the existing fail-fast behaviour for misspelled or genuinely required columns.

In
`deps/trading-strategy/tradingstrategy/alternative_data/vault.py`:

1. Define one named naive-UTC constant for the HyperCore deposit cutoff:
   `HYPERCORE_DEPOSIT_STATE_CUTOFF = datetime.datetime(2026, 4, 11)`.
2. Retain `chain` while converting and identify HyperCore with
   `ChainId.hypercore.value` (9999). The output schema does not need a new
   `chain` column.
3. For HyperCore state rows, calculate the effective timestamp as the later of
   `timestamp` and `written_at`.
4. When resampling to a daily or multi-hour decision grid, round the effective
   timestamp **up** to the next bucket boundary. A state observed during a
   bucket must not be visible at that bucket's opening timestamp.
5. Keep the whole latest row together so `deposits_open`, reason, and any
   future caps cannot come from different observations. When a backfilled batch
   gives several rows the same `written_at`, use the original source timestamp
   as the tie-breaker.
6. Keep the existing timestamp/floor behaviour for non-HyperCore data. Do not
   broaden this data-specific correction to protocols that have different
   collection semantics.
7. A HyperCore row with a null or absent `written_at` contributes no usable
   state row. Do not raise and do not silently fall back to the price timestamp:
   it is harmless before the cutoff and resolves to unavailable after it.

Do not change the upstream HyperCore scanner in this work. The downloaded
parquet already carries `written_at`; the bug is that the framework does not
load or use it for state availability.

### 2. Apply the cutoff in BacktestPricing

In `tradeexecutor/backtest/backtest_pricing.py`, add one small HyperCore helper
used by both `can_deposit()` and `check_deposit()`:

1. Preserve the existing explicit `_vault_window_overrides` precedence.
2. If `timestamp < HYPERCORE_DEPOSIT_STATE_CUTOFF`, return allowed without
   consulting archived deposit state. This deliberately suppresses sparse or
   retrospectively written pre-cutoff values.
3. At or after the cutoff, use `_lookup_vault_state()`.
4. Treat no sample, nullable `deposits_open`, or an out-of-tolerance sample as
   unavailable and block the new deposit.
5. Reuse `DepositBlockReason.unknown` for unavailable post-cutoff data and set
   one precise "missing, unknown, or stale" diagnostic message. Do not add a
   new enum or a second lookup result type solely to split these cases.
6. Continue to use `check_backtesting_deposit()` for explicit state and caps.

Identify HyperCore with the existing `pair.is_hyperliquid_vault()` helper,
which checks `ChainId.hypercore.value` (9999), not `ChainId.hyperliquid` (999).
Do not add new fields to `Dataset`, `TradingStrategyUniverse`, state JSON, or
every pricing-model constructor just to carry one source-data cutoff.

`can_deposit()` and `check_deposit()` must share the helper so their answers
cannot diverge.

Update `_lookup_vault_state()`'s docstring: `None` means unavailable state, not
globally "allowed". Audit its callers and keep the new fail-closed rule limited
to HyperCore deposits at/after the cutoff; cap accessors, redemptions, and
non-HyperCore deposits retain their current behaviour.

### 3. Keep closed-now vaults in Hyper-AI v7's universe

In `/Users/moo/code/strategies/strategy/hyper-ai-v7.py`, pass
`include_closed_vaults=True` to `build_hyperliquid_vault_universe()`.

Do not change the curator's global default in this work. Other strategies may
not yet perform pre-selection deposit gating.

No new selection code is needed. v7 already:

- includes held pairs in the incumbent set without checking deposit state;
- calls `pricing_model.can_deposit(timestamp, pair)` for unheld candidates;
- skips an unavailable new entry and continues down the ordered candidates;
- applies the same check to capacity-extension candidates; and
- calls the AlphaModel trade-generation gate again for any positive
  adjustment.

This means a closed incumbent can remain in the portfolio but cannot receive a
top-up. A closed unheld vault does not consume a portfolio slot.

### 4. Keep live checks authoritative

Do not apply the historical cutoff in live pricing. `HypercoreVaultPricing`
must continue to obtain current `is_closed`, `allow_deposits`, relationship,
and leader-fraction data from the Hyperliquid API.

The source universe may contain a currently closed vault, but v7's existing
pre-selection call to live `can_deposit()` excludes it from new entries. If
the live API cannot establish availability, the buy must remain blocked by the
existing live check. Cached metadata must not override a fresher live result.

### 5. Make diagnostics explicit

Update docstrings and the existing result message to distinguish:

- assumed open before the cutoff;
- explicitly open or closed historical state; and
- unavailable post-cutoff state.

Keep this in the existing `DepositCheckResult` and signal diagnostics. A
combined unavailable message is sufficient; do not add a new result type or
persisted diagnostics model.

## Focused tests

### trading-strategy dependency

Extend `deps/trading-strategy/tests/test_vault_state.py` with two focused tests:

1. A HyperCore closure whose price timestamp is before its `written_at` is not
   emitted on an earlier daily bucket; it becomes visible only at the first
   bucket boundary after observation.
2. A HyperCore row with null `written_at` yields no usable state, while a
   parquet without the optional column still loads.

Keep the existing whole-row, reopen, and nullable-boolean tests. Update the
whole-row test with `written_at` so it also proves that rows tied to one batch
choose the latest source observation; do not add a duplicate reopen test.

### trade-executor

Extend `tests/backtest/test_backtest_vault_state_pricing.py` with two tests,
using several assertions per repository test conventions:

1. HyperCore cutoff test:
   - explicit archived closure before the cutoff is allowed;
   - missing state before the cutoff is allowed;
   - explicit open after the cutoff is allowed; and
   - explicit closed after the cutoff is blocked with its reason; and
   - an explicit backtest window override still takes precedence.
2. HyperCore missing-state test:
   - missing, nullable, and stale state after the cutoff are blocked with
     `DepositBlockReason.unknown`; and
   - an equivalent non-HyperCore pair keeps the current unknown-is-allowed
     behaviour.

Extend `tests/strategy/test_vault_universe_creation.py` only if its existing
coverage does not already prove that `include_closed_vaults=True` retains a
closed HyperCore vault. Do not add a duplicate test.

The tests must also prove that redemption behaviour is unchanged.

## Verification

1. Run the focused trading-strategy state tests.
2. Run the focused trade-executor backtest-pricing and vault-universe tests.
3. Reconfirm the 2026-04-11 boundary against the exact parquet snapshot used
   for the release; stop if any then-eligible first explicit observation rounds
   to a later decision bucket.
4. Run Hyper-AI v7 over the same fixed dataset and dates as its previous v6/v7
   comparison.
5. Record, per decision:
   - source-universe size;
   - candidates skipped for deposit state;
   - selected addresses;
   - count of assumed-pre-cutoff decisions;
   - count of unavailable-post-cutoff decisions; and
   - turnover.
6. Check the known divergence dates around 27 and 29 August. Closed-now vaults
   that were open then must be present in the backtest universe, while a vault
   whose historical state was closed must be skipped and replaced by the next
   ranked candidate.
7. Confirm that live mode still queries the Hyperliquid API and that an
   incumbent does not receive a positive adjustment when live deposits are
   unavailable.

The expected result is not necessarily higher backtest profit. It is matching
membership and deposit eligibility at each decision timestamp without using
today's closure state as historical information.

## Acceptance criteria

- The first enforced HyperCore historical deposit-state boundary is
  2026-04-11 00:00 UTC.
- No state observation is available to a decision before `written_at` or the
  next decision-bucket boundary.
- Pre-cutoff HyperCore deposits are assumed open.
- Missing or stale post-cutoff HyperCore state blocks only new deposits.
- Hyper-AI v7 loads currently closed vaults but skips closed unheld candidates
  point-in-time and backfills the portfolio slot.
- A deposit-closed incumbent is not sold solely because deposits closed and is
  not increased while closure remains in force.
- Live deposit checks remain API-backed.
- Redemption behaviour and non-HyperCore pricing behaviour do not change.
- No compatibility layer, migration, rollback loader, new state file, or new
  generic availability framework is introduced.
