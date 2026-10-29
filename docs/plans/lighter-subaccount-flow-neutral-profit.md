# Flow-neutral profit for Lighter sub-account positions

## Goal

Report each Lighter account's profit, the master's and every sub-account's,
independently of the capital moved in and out of it. Deposits from the Safe,
withdrawals to the Safe and transfers between the vault's own accounts are
capital flows, not profit or loss. What an account earns or loses while
holding capital is profit, and so are the fees it pays to move capital.

Each position reports two figures:

| Figure | Definition |
|--------|------------|
| Return % | Internal share price over its initial value, minus one |
| Profit USD | Current value + capital withdrawn − capital invested |

The return is the existing internal share price: a flow mints or burns units
at the current price without moving it. The USD profit is the cash-flow
identity `TradingPosition.get_cash_flow_profit_usd()` already uses for
HyperCore vault positions, extended to the flows exchange accounts have that
trades do not record.

Assumption, from the request: every flow into or out of a sub-account goes
through the executor's own fund-movement logic: `lighter-move-funds`,
`perform-test-trade` and the `PositionManager` custody methods. A trader moving
USDC between accounts with its own key is out of scope; see "Non-goals".

## Current state

Each Lighter account is its own `TradingPosition` with an `exchange_account`
pair (`LIGHTER-ACCOUNT` for the master, `LIGHTER-<id>` for a sub-account),
valued from its own `total_asset_value`, and carries a
`PositionInternalSharePriceState`
(`tradeexecutor/state/position_internal_share_price.py`, from #1401 and
#1683):

| Event | Code | Effect on the share-price state |
|-------|------|---------------------------------|
| Safe → account deposit | `State.mark_trade_success()`, `state.py:1010` | Mints units; `total_invested` += trade value |
| Account → Safe withdrawal | Same | Burns units pro rata |
| Funding a destination account | Same | Mints units; `total_invested` += amount |
| Funding from a source account | `complete_exchange_account_funding()`, `exchange_account/state.py:340-377` | Fee as a balance update (a loss); principal burned with `apply_capital_outflow()` |
| Revaluation from Lighter | `ExchangeAccountValuator` (`valuation.py:218`), `ExchangeAccountSyncModel.sync_positions()` (`sync_model.py:248`) | Units unchanged, price moves. The first valuation of a position opened with a placeholder trade seeds `total_invested` |
| Residual written off on close | `close_exchange_account_position()`, `exchange_account/state.py:657-668` | A balance update: a loss |

The return % is therefore already flow-neutral. The USD profit is not.

### The defect

`TradingPosition.get_share_price_profit()` (`position.py:2427-2452`) computes:

```text
open:    total_assets − total_supply × initial_share_price
closed:  Σ sell trade values − total_invested          (when total_supply is 0)
```

The open formula counts profit only on the units still outstanding, so every
withdrawal or funding outflow drops the profit its units had earned. The
closed formula counts only sells as money out; a funding outflow is a balance
update, so a source account that funded others reports the principal as a
loss. `get_total_profit_usd()`, `get_unrealised_profit_usd()` and
`calculate_position_statistics()` (`statistics/core.py:80-104`) all report
this figure for exchange accounts.

Evidence, from the real share-price functions: a deposit of 100 gains 10,
then leaves in two halves. The true profit is 10 throughout.

| Step | Held | Share price | USD profit, funding outflows | USD profit, sell withdrawals |
|------|------|-------------|------------------------------|------------------------------|
| Deposit 100, gain 10 | 110 | 1.10 | 10.00 | 10.00 |
| 55 out | 55 | 1.10 | 5.00 | 5.00 |
| Remaining 55 out | 0 | 1.10 | −100.00 | 10.00 |

In the mainnet run of 2026-10-29 (`lighter-cli-20261029-092532`), with no
trading, the master reports **−4 USD**: the 4 USDC it funded the sub-accounts
with. Its true profit is 0. (The sub-accounts show +100% and −33% from the
out-of-band trader transfer, which is out of scope.)

## Design

### 1. Track the capital withdrawn

Add to `PositionInternalSharePriceState`:

```python
#: Capital taken out of the position: the reserve received for sell trades
#: and the principal moved to another account. ``None`` in a state saved
#: before this field existed.
total_withdrawn: float | None = None
```

Old state JSON without the key loads as `None`: `dataclass_json` reads
defaults through `dataclasses.fields()`, which works with `slots=True`.

Every function in `tradeexecutor/strategy/position_internal_share_price.py`
returns a new state object, so:

- `create_share_price_state()` and
  `create_share_price_state_for_exchange_account()` set `0.0`.
- `update_share_price_state()`, `apply_capital_outflow()` and
  `update_share_price_state_for_balance_update()` copy the previous value
  forward. `None` stays `None`.
- Two outflows add to it: the sell branch of `update_share_price_state()`
  adds `abs(trade.executed_reserve)`, the USDC that reached the Safe; and
  `apply_capital_outflow()` adds `amount`, the principal.
- `open_exchange_account_position()` (`exchange_account/state.py:621-625`)
  today replaces `share_price_state` whenever `reserve_amount > 1`, and
  `state.create_trade()` can return an existing position, which would reset
  the counter and the history. It creates a state only when there is none
  and otherwise mints into it with `update_share_price_state()`.

### 2. Flow-neutral USD profit

In `get_share_price_profit()`, for an exchange-account position whose
`total_withdrawn` is not `None`:

```text
profit_usd = total_assets + total_withdrawn − total_invested
```

for open and closed positions alike. With `None` the current formulas apply.
`profit_pct` stays the share-price return. Spot and vault positions are
unchanged.

Fees: a funding fee is a balance update, so it is a loss in both figures. A
withdrawal fee, the gap between the account debit and the Safe credit
(`lighter_move_funds.py:387-395`), is a loss in USD only, because units are
burned on the gross debit at a price of 1.0. Lighter charged no withdrawal
fee in either mainnet run.

### 3. Closed positions

`get_realised_profit_usd()` reports `None` or zero for exchange accounts,
because every trade is at 1.0, so open positions report their profit as
unrealised. For a closed exchange-account position, unrealised USD and % are
zero, realised USD is the flow-neutral profit, and realised and total % are
that profit over `total_invested`, as #1623 did for HyperCore. Otherwise
`TradeAnalysis` (`analysis/trade_analyser.py:917-975`), which sums unrealised
over open and realised over closed positions, drops the account's profit.

The closed % cannot be the share-price return. Closing writes off a residual
below Lighter's 1 USDC minimum as a balance update, which takes the value of
the remaining units, and so the share price, to zero. With the real
functions: deposit 10, gain 0.4, withdraw 10, then write off the 0.4 residual
leaves the share price at 0, a return of −100%, for an account whose USD
profit is 0.

Lighter positions rarely close: `close_undeclared_lighter_positions()` closes
only an account the strategy stops declaring.

### 4. Existing states

Positions saved before this change keep `total_withdrawn = None` and their
current figures; positions opened after it report flow-neutral profit from
the start. No backfill. A position open across the upgrade, such as a Lighter
account that is never closed, therefore keeps the old figures. If a
production vault needs them corrected, a backfill is a follow-up: derive
`total_withdrawn` from the position's sells and the funding trades that name
it as `funding_source_position_id`. It must not trust `total_invested`
blindly, since a state rebuilt by `migrate_share_price_state()` can count a
placeholder trade as invested. Recomputing invested capital from history is
not straightforward either: a first valuation that matches the placeholder
seeds capital without recording a balance update, and a valuation seed can
precede a real allocation.

### 5. Deposit into a position valued at zero

A valuation that finds the account empty while units remain, such as a
liquidated sub-account, sets the share price to zero. The buy branch of
`update_share_price_state()` mints `trade_value / current_share_price`, so
the next deposit raises `ZeroDivisionError` inside `mark_trade_success()`.
With the real functions: deposit 10, revalue to 0, deposit 5 raises it. This
is a crash in production today, independent of the USD change: re-funding a
liquidated sub-account would fail while completing its trade.

When the share price is zero, the buy branch burns the worthless units and
mints the deposit at the initial share price, starting a new series. The
return % then covers the period since the total loss; the USD profit still
covers the whole life, including the loss, because `total_invested` and
`total_withdrawn` are not reset.

## Changes

| File | Change |
|------|--------|
| `tradeexecutor/state/position_internal_share_price.py` | `total_withdrawn` field |
| `tradeexecutor/strategy/position_internal_share_price.py` | Set on creation, carried forward, added on sells and capital outflows |
| `tradeexecutor/exchange_account/state.py` | `open_exchange_account_position()` mints into an existing state |
| `tradeexecutor/strategy/position_internal_share_price.py` (buy branch) | New series on a deposit into a zero-priced position |
| `tradeexecutor/state/position.py` | Flow-neutral `profit_usd`; closed-position split and money-weighted closed % |
| `tradeexecutor/testing/lighter_core.py` | `FakeLighterCore(withdrawal_fee=...)`: credit the Safe the amount minus the fee, in both withdrawal paths (lines 247 and 302) |
| `tradeexecutor/testing/lighter_share_price_timelines.py` (new) | The timelines of the state unit tests as data, a driver that applies their events to a `State` through the executor's own code, and the invariant checks; shared by the unit tests and the manual chart script |
| `tradeexecutor/strategy/chart/standard/exchange_account.py` (new) | `exchange_account_share_price(input: ChartInput) -> Figure`: one exchange-account position's share price, units, value and USD profit over time, with flow markers |
| `scripts/lagoon/manual-lighter-share-price-charts.py` (new) | Manual chart inspection, below |
| `README-Lighter.md`, `docs/lagoon-lighter-deployment.md` | The two figures, under strategy accounting |
| `CHANGELOG.md` | One dated line |

Reporting only: no change to how flows are executed or booked, to the NAV or
to Lagoon settlement.

## Tests

Following the repository's pytest rules: few test functions with several
asserts each, a docstring with numbered steps repeated as comments, no test
classes, helpers in the test module or under `tradeexecutor/testing/`.

### Invariants checked after every event

Every test checks these after each event, not only at the end:

1. **A flow does not move the return.** A deposit, withdrawal or transfer
   leaves the share price unchanged, except a funding source's, which falls
   by its fee over its value.
2. **A flow does not move the USD profit**, except by the fee it pays.
3. **Units follow the flow.** A flow mints or burns `amount / share price`
   units; a revaluation mints and burns none.
4. **USD profit is earnings minus fees.** Each position's USD profit equals
   the gains and losses booked on it minus the fees it paid. Summed over the
   master and the sub-accounts, it equals the change in their combined value
   minus net deposits from the Safe, counting each withdrawal at what the
   Safe received.

### State unit tests

New module `tests/exchange_account/test_lighter_flow_neutral_profit.py`, with
no chain. It builds a `State` with a master and two sub-account positions
from `create_lighter_exchange_account_pairs()`, as
`tests/exchange_account/test_lighter_funding_trade.py` does, and drives each
event through the code the executor uses:

| Event | Driven through |
|-------|----------------|
| Opening, placeholder or real | `open_exchange_account_position()` |
| Deposit, withdrawal (with a Safe credit below the debit) | `create_exchange_account_transfer()`, `complete_exchange_account_transfer()` |
| Transfer between accounts, with a fee | `create_exchange_account_funding()`, `complete_exchange_account_funding()` |
| Gain or loss | `ExchangeAccountValuator` with a stub account-value function, as `test_exchange_account_share_price.py` does |
| Write-off and close | `close_exchange_account_position()` |

**One test, parametrised over the timelines.** The timelines, their events
and their expected values live in
`tradeexecutor/testing/lighter_share_price_timelines.py`, so the manual chart
script replays exactly what the test checks. The expected values below were
computed with the current share-price functions plus the proposed
`total_withdrawn`; the test asserts each row and the four invariants.

| Timeline | Events | Expected at the end |
|----------|--------|---------------------|
| A. Gain before an inflow | Deposit 10; revalue to 11; deposit 11; revalue to 24.2 | 20 units, price 1.21, return +21%, USD +3.2. The deposit minted 10 units at 1.10 and left the return at +10% |
| B. Inflow before a loss, partial exit | Deposit 10 and 10; revalue to 18; withdraw 9 | 10 units, price 0.90, return −10%, USD −2 before and after the withdrawal |
| C. Full exit and re-entry | Deposit 10; revalue to 9; withdraw 9; deposit 9; revalue to 9.9 | Units 10 → 0 → 10, price 0.99, return −1% (0.9 × 1.1), USD −0.1 |
| D. Funding chain, fee 0.5 per transfer | Master deposit 10; fund trend 4; trend revalued to 4.4; trend sends 2.2 to arb; arb revalued to 1.98; withdraw all three | Master −5%, USD −0.5; trend −2.5%, USD −0.1; arb −10%, USD −0.22. Sum −0.82 = +0.40 − 0.22 − 1.00 fees |
| E. Valuation-seeded opening | Placeholder 0, first valuation 5; deposit 5; revalue to 11 | 10 units, return +10%, USD +1 |
| F. Withdrawal fee | Deposit 10; withdraw 10, Safe credited 9.9 | Return 0%, USD −0.1 |
| G. Write-off and close | Deposit 10; revalue to 10.4; withdraw 10; close, writing off 0.4 | Closed: unrealised 0, realised USD 0, closed % 0 (design 3) |
| H. Total loss and re-funding | Deposit 10; revalue to 0; deposit 5; revalue to 5.5 | No crash; a new series of 5 units, return +10% since the loss, USD −9.5 (design 5) |
| I. Repeated allocation | Placeholder; valuation seeds 5; `open_exchange_account_position()` allocates 5 more | Allocation minted into the existing state: 10 units, `total_withdrawn` kept |

**One test for old states and bad paths:**

- a share-price state serialised without `total_withdrawn` loads as `None`,
  keeps the current formula and stays `None` through a valuation, a funding
  outflow and a withdrawal;
- a rejected repeat completion of a funding leaves both accounts' units and
  `total_withdrawn` unchanged, and a verified recovery of a started or
  failed funding counts the outflow once (extending the recovery case of
  `test_lighter_funding_trade.py`);
- after timeline G, `PositionStatistics` from `calculate_position_statistics()`
  carries the figures in `profit_usd` and `internal_profit_usd`.

### Anvil fork with the Typer CLI

New module `tests/lagoon/test_lagoon_lighter_flow_neutral_profit_e2e.py`. It
uses the shared Ethereum fork and the simulated Lighter Core
(`FakeLighterCore`) through the `prepare_tutorial()` and `run_cli()` helpers
of `tradeexecutor/testing/lighter_tutorial.py`, like
`test_lagoon_lighter_cli_manual_e2e.py`. Every Ethereum effect is real; the
fake sequencer charges its 0.5 USDC transfer fee. One test function deploys
once and walks one timeline across the three accounts:

| Step | Command or action | Timeline covered |
|------|-------------------|------------------|
| 1 | `lagoon-deploy-vault`, `init`, `correct-accounts --process-redemption`, subscribe with `lagoon-settle` | E on the master: the activation deposit seeds it |
| 2 | `perform-test-trade --pair "(ethereum, lighter, LIGHTER-trend, USDC)" --buy-only` | A Safe deposit to the master and a funding with a fee |
| 3 | Credit trend's account in the fake core; `correct-accounts` | A gain, booked by the correct-accounts sync path |
| 4 | Fund arb; credit its account in the fake core; `lagoon-settle`; fund arb again | A, with the gain booked by the settlement valuation path |
| 5 | `perform-test-trade --lighter-transfer trend:arb --buy-only` | D: transfer between sub-accounts, fee on trend |
| 6 | Debit arb's account; `correct-accounts`; `lighter-move-funds --lighter-account-index <arb>` for half its balance | B |
| 7 | Set trend's account to zero; `correct-accounts`; fund trend again | H: must not crash |
| 8 | `withdrawal_fee` set in the fake core; `lighter-move-funds` withdraws the master's free balance | F |
| 9 | Withdraw arb entirely, fund it again, credit it; `correct-accounts` | C |
| 10 | Withdraw every account; `lagoon-redeem` | Final vault check |

After every step the test reads the state file and asserts the four
invariants and each position's units, return % and USD profit. After step 10
the per-position USD profits sum to the gains and losses injected into the
fake core minus the fees the fake core charged, which also equals the
deployer's final USDC minus its starting USDC, before gas and within
Lagoon's share rounding.

The scripted-strategy flows through `start` are covered by extending
`verify_lighter_scripted_transfers()`, so
`tests/lagoon/test_lagoon_lighter_scripted_transfers_e2e.py` checks
invariant 4 for the `PositionManager` custody path and the
`ExchangeAccountValuator` valuation path.

### Manual chart inspection

Numbers can pass every assert and still look wrong over time: a spike, a gap,
units moving on a revaluation, a series that does not restart. The new script
`scripts/lagoon/manual-lighter-share-price-charts.py` renders the charts for
a person to inspect. It needs no chain, network or secrets:

```shell
poetry run python scripts/lagoon/manual-lighter-share-price-charts.py
```

It:

1. builds a `State` with a master account and one sub-account per timeline
   of the state unit tests, named after the timeline (`gain-before-inflow`,
   `inflow-before-loss`, `exit-and-reentry`, `valuation-seeded`,
   `withdrawal-fee`, `write-off-and-close`, `total-loss-and-refund`,
   `repeated-allocation`), with the funding chain of timeline D running from
   the master through two more sub-accounts;
2. replays every timeline's events with the shared driver, one event per
   simulated hour, and calls `update_statistics()` after each event, so the
   positions' `PositionStatistics` history has a point per event;
3. runs the four invariants and stops with the failing position and event if
   one breaks, before drawing anything;
4. renders `exchange_account_share_price()` for every position, and one
   overview chart with every position's share price, to PNG files with
   `render_plotly_figure_as_image_file()`;
5. prints a `tabulate` table of each position's final return %, USD profit,
   units and `total_withdrawn`, next to the expected values, and the path of
   every PNG.

Output goes to `OUTPUT_DIR`, default `/tmp/lighter-share-price-charts`.
`OPEN_BROWSER=true` also opens each chart with
`open_plotly_figure_in_browser()`.

What to look for in the PNGs, also listed in the script header:

- the share price is flat across every deposit, withdrawal and transfer
  marker, except a fee step down on a funding source;
- the share price moves only at revaluation markers;
- units step only at flow markers, never at a revaluation;
- the USD profit line moves only at revaluations and fees, and ends at the
  expected value;
- `total-loss-and-refund` drops to zero and restarts at the initial price on
  the re-funding;
- `write-off-and-close` ends at the close with no trailing points;
- no negative units, no gaps and no points before a position opened.

A smoke test in the unit module renders one position's chart to PNG bytes
and asserts they are not empty, so the chart and the script's rendering path
cannot break unnoticed. Inspecting the images remains manual.

### Regression runs

`tests/exchange_account/`, `tests/test_state.py -k share_price`,
`tests/hyperliquid/test_hypercore_position_profit_attribution.py` (HyperCore
vaults use the same share-price code), and the existing Lighter e2e tests
under `tests/lagoon/`. Run with `.local-test.env` sourced, the worktree
`PYTHONPATH`, and `-n auto` for more than five tests.

## Non-goals

- **Out-of-band transfers.** `correct-accounts` books a trader's own
  transfer between sub-accounts as profit on one and loss on the other.
  Treating it as a flow needs each sub-account's authenticated transfer
  history, and so its key from `LIGHTER_SUB_ACCOUNT_OPERATOR_RECORD_FILES`.
  A separate plan.
- **Withdrawal fees in the return %**, which needs the shortfall booked as a
  balance update before the sell.
- **Backfilling positions that predate the change** (design 4).
- **Spot and vault positions**, which have the same understatement after a
  partial sell.

## Open questions

1. Design 5 restarts the return % after a total loss. The alternative is to
   report −100% for the rest of the position's life, which is the
   time-weighted return but hides any later performance. The USD profit is
   the same either way.
2. GMX, Derive and CCXT positions share this code. New positions on those
   protocols will report the new USD figure; confirm no dashboard depends on
   the old one.

## Review history

Codex CLI, GPT 6.1 sol, 2026-10-29: three rounds, each requesting changes,
every finding verified against the code. Resolved in this version: the
counter must be carried forward rather than reset by updates; commands
outside the executor loop must not corrupt an unknown total (`None` is carried
forward); the withdrawal fee is a USD loss; closed positions need the
realised/unrealised split; and `open_exchange_account_position()` must not
replace an existing state. The other findings, including a fourth round,
concerned rebuilding and backfilling old states, which this version drops
(design 4); their caveats are kept there for the follow-up.

## Implementation status

Implemented on branch `lighter-subaccount-share-price`, 2026-10-29, as
designed, with the deviations and findings below.

### Verified by tests

- `tests/exchange_account/test_lighter_flow_neutral_profit.py`: the nine
  timelines, every invariant after every event and the final figures; an old
  state reloaded without the new key; a failed transfer recovered once and a
  repeated completion refused; the stored statistics; a PNG render.
- `tests/lagoon/test_lagoon_lighter_flow_neutral_profit_e2e.py`: the Typer
  CLI timeline on the fork with the fake Lighter Core. Final USD profits:
  master −2.5 (five transfer fees), trend −1.8, arb −0.38; their sum equals
  the deployer's USDC change.
- `verify_lighter_scripted_transfers()` now checks each account's USD profit
  is minus the fees it paid, in `test_lagoon_lighter_scripted_transfers_e2e.py`.
- Regression: `tests/exchange_account/`, `tests/test_state.py`, the HyperCore
  profit attribution and accounting tests, and every Lighter test under
  `tests/lagoon/` pass.
- `scripts/lagoon/manual-lighter-share-price-charts.py` renders twelve charts;
  they were inspected and led to step lines and charts starting at the
  account's first flow.

### Deviations from the plan

- **Valuations in the unit timelines** run through
  `ExchangeAccountSyncModel.sync_positions()`, the `correct-accounts` path,
  not `ExchangeAccountValuator`: the sync model takes the event's timestamp,
  so balance updates and statistics share the replay's clock in the charts.
  The valuator path is covered by the scripted-transfer and CLI fork tests.
- **States rebuilt from history** by `migrate_share_price_state()` keep
  `total_withdrawn` unknown, like states saved before the change.
- **No withdrawal fee in the CLI fork test.** `lighter-move-funds` refuses a
  withdrawal whose claimable amount differs from the request, on both the
  master's secure path and a sub-account's L1 path, so a short credit cannot
  happen through the CLI today. Timeline F covers the accounting through the
  state helpers, which accept a credit below the debit. The planned
  `FakeLighterCore(withdrawal_fee=...)` setting was dropped as unused.
- **Recovery** is tested in the new module rather than by extending
  `test_lighter_funding_trade.py`.

### Bugs found and fixed

- **Float residue read as a −100% return.** The share-price state tracks the
  value as a float while the books hold Decimals. Withdrawing the whole 1.7
  USDC of an account tracked at 1.7000000000000002 left 2e-16 shares, so a
  zero value read as a zero share price. A burn within
  `FULL_EXIT_TOLERANCE` (1e-9, relative) of the whole holding now burns all
  of it.
- **Float residue after a total loss.** A valuation to zero left 2e-16 of
  value, a share price of 1e-16, and the next deposit minted 1.6e16 shares
  instead of starting a new series. A valuation within the same tolerance of
  zero now values the position at exactly zero. Found by the CLI fork test.
- **`perform-test-trade --amount` booked a float's binary expansion.**
  `Decimal(amount)` on a float made `--amount 3.3` book and transfer
  3.29999999999999982236431605997495353221893310546875 USDC on the `--pair`,
  `--single-pair` and vault paths; they now convert through `str`, as
  `--lighter-transfer` did. Found by the CLI fork test.
