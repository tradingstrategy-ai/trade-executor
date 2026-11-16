# Lighter test-trade capital flows and the exchange-account share price

## Problem

The Lighter AI website reports a 95.1% loss on position #1, the Lighter master
account (`https://tradingstrategy.ai/strategies/lighter-ai/open-positions/1`).
The real loss is about 11%. The live state, read from
`https://lighter-ai.tradingstrategy.ai/state` on 2026-11-16 13:35 UTC, holds:

```
share_price_state:
  initial_share_price: 1.0
  current_share_price: 0.049338
  total_supply:        268,564.21
  cumulative_quantity: 13,250.35
  total_invested:      15,008.885773
  total_withdrawn:     null
latest stats.positions["1"] sample:
  internal_profit_pct: -0.9507
  internal_profit_usd: -255,313.85
```

Replaying the position's 3 trades and 505 balance updates in time order through
`tradeexecutor/strategy/position_internal_share_price.py` reproduces this state
exactly:

| UTC | Event | Share price |
|---|---|---|
| 2026-09-17 16:06 | Trade 1, 0 USD placeholder (`open_exchange_account_position`) | 1.0 |
| 2026-09-17 16:08 | Balance update 1, +18 USD: `ExchangeAccountValuator` seeds the state, 18 shares | 1.0 |
| 2026-09-17 16:37 | Balance update 4, −16.995365 USD: the test-trade withdrawal to the Safe, booked as a valuation change | 0.99974 → 0.0556 |
| 2026-09-19 | Trade 2, +3,000 USD `lighter-move-funds` deposit, mints at 0.0556 | 0.0556 |
| 2026-09-22 | Trade 3, +11,990.89 USD automatic cash management | 0.0559 |
| until now | Trading results | 0.0559 → 0.0493 |

With balance update 4 treated as a capital outflow, the same replay gives a
share price of 0.887851 (−11.21%) and a flow-neutral profit of −1,741.54 USD
(value 13,250.35 + withdrawn 16.995365 − invested 15,008.885773).

The `profit_usd` of −255k comes from the
`total_assets - total_supply * initial_share_price` fallback in
`TradingPosition.get_share_price_profit()`, used because `total_withdrawn` is
`null`. The state predates withdrawn-total tracking (#1695).

## Root cause

`lagoon-lighter-test-trade` (`tradeexecutor/cli/commands/lagoon_lighter_test_trade.py`)
moves real USDC between the Safe and Lighter without booking it:

- Step 2 calls `deposit_usdc_from_lagoon_safe_into_lighter()`, and step 3's
  `checkpoint_lagoon_nav()` runs `ExchangeAccountValuator`, which records the
  deposit as a `vault_flow` balance update ("Exchange account valuation").
- Steps 8–10 request the secure withdrawal, claim it to the Safe and run
  `checkpoint_lagoon_nav()` again. The valuator records the withdrawal as a
  negative valuation balance update, and
  `update_share_price_state_for_balance_update()` treats it as a loss, because
  the share supply does not change.

The other Lighter custody paths already book transfers. `lighter-move-funds`
uses `create_exchange_account_transfer()` and
`complete_exchange_account_transfer()` (trade flag `external_account_transfer`).
Automatic cash management uses `withdraw_from_lighter()` trades in
`exchange_account/lighter_strategy.py`. Sub-account funding uses
`apply_capital_outflow()` in `exchange_account/state.py`. In
`State.mark_trade_success()`, a successful transfer trade mints or burns
internal shares and adjusts reserves. So the test trade is the only path that
leaves capital flows as valuation changes.

No current command repairs this. `correct-history` edits fund-level
statistics and can prune position statistics before a cutoff, but recomputes
nothing. `backfill_share_price_state()` skips positions that already have a
state, has no CLI entry point, replays all trades before all balance updates,
and deliberately leaves `total_withdrawn` unknown.

## Goals

1. New test trades book their Safe ↔ Lighter deposit and withdrawal as transfer
   trades, so the internal share price measures trading results only.
2. A separate CLI command repairs an existing position's share-price state and
   its stored `stats.positions` samples. With no arguments, it finds the
   misbooked withdrawals from the test trade's journal; the operator can add
   or drop balance updates.
3. An integration test runs that command on a copy of the 2026-11-16 lighter-ai
   state and checks the corrected figures.

Not in scope: changing fund-level NAV, share price or treasury statistics,
which are correct; and automatic detection of misbooked flows in old states.

## 1. Prevent recurrence: book test-trade transfers

Change `lagoon_lighter_test_trade.py` to book its custody movements with the
transfer API that `lighter-move-funds` also uses:
`create_exchange_account_transfer()`, `State.start_execution()` and
`complete_exchange_account_transfer()`. The `lighter-move-funds` wrappers
are not moved. They import from the test-trade module, so sharing them would
need a new module and test patch changes, and the core calls are only a few
lines.

- Each transfer trade carries `other_data["lighter_test_trade_run"]`, the
  journal's `created_at`, so a resumed run finds the trade it booked.
- Baseline before the deposit: run `checkpoint_lagoon_nav()` once before
  creating the deposit trade. On a fresh state, `sync_initial()` creates the
  reserve position without importing the Safe balance, and
  `start_execution(underflow_check=True)` checks allocations against the
  recorded reserve. Any Lighter balance already there, such as an activation
  deposit, then seeds the share-price state at 1.0 as capital, and the test
  deposit mints shares on top of it.
- Deposit (step 2): create and start the deposit trade, which allocates the
  reserve and saves state. Pass a `tx_observer` to
  `deposit_usdc_from_lagoon_safe_into_lighter()` that saves each tx hash in
  the trade's `other_data` before broadcast. After
  `wait_for_lighter_collateral()`, check that Safe USDC fell by the amount,
  complete the trade, save state, then advance the journal with
  `deposit_trade_id` and `deposit_block`.
- Withdrawal (steps 8–9): after the journal records the request intent, find
  or create the withdrawal trade and save it, then request. A rejected request
  marks the trade failed with `outcome = "withdrawal_rejected"`. After the
  claim, or the relayer payment `_claim_unless_paid()` detects, complete the
  trade with the claimed amount, save state, then advance the journal with
  `withdrawal_trade_id` and `withdrawal_block`. A journal from before this
  change, resumed in `withdrawal_requested`, books the trade at that point.
- Order: complete each transfer before the following `checkpoint_lagoon_nav()`,
  so the valuator's balance update only carries fees and price movement.
- Recovery: state is the source of truth for the transfer.
  - A completed run deposit whose journal is still `created` skips the
    pre-deposit custody checks and only advances the journal.
  - A started deposit trade with no recorded tx hash was never broadcast, so
    it is reused.
  - A deposit trade with recorded hashes that never completed fails closed
    for the operator to inspect.
  - A completed withdrawal trade whose journal is still `withdrawal_requested`
    skips the claim.
- Treasury: `LagoonVaultSyncModel` treasury sync replaces the recorded reserve
  with Safe USDC read at a buffered block, and its claim protection only scans
  vault-settlement trades. Rather than change that shared code,
  `checkpoint_lagoon_nav()` takes `min_safe_block`: the block recorded after
  the deposit or claim. It waits until `get_safe_latest_block()` reaches that
  block before syncing. `lighter-move-funds` does not sync the treasury
  itself, so it is not affected within the command.

Tests in `tests/lagoon/test_lagoon_lighter_test_trade.py`. The old fake
deposit did not debit the Safe, and the test expected the round trip to
increase Safe cash. The custody-aware fake now moves USDC out of the Safe
(an impersonated Anvil account) and reports its tx hash through
`tx_observer`. The claim fake moves it back.

- The full flow ends with two successful `external_account_transfer` trades of
  the deposit and claimed amounts. `share_price_state.total_withdrawn` equals
  the claimed amount, and the share price is 1.0. No balance update is of
  withdrawal size. The recorded reserve equals Safe USDC at every treasury
  sync after the first, checked **before** the sync, so the sync cannot hide
  a booking error.
- Crash and resume: one run crashes after the deposit trade completes but
  before the journal reaches `deposited`. The next run crashes after the
  withdrawal trade completes but before `withdrawn`. The third run completes
  with one deposit, one withdrawal and one claim.
- The original resume after a Lighter failure still passes.
- Not covered by a test: the wait for the treasury snapshot block, because
  Anvil and unit-testing runs read the latest block.

The `lighter-move-funds` command and its tests are unchanged.

## 2. Repair command: `rebuild-exchange-account-share-price`

Shared logic goes into
`tradeexecutor/strategy/position_internal_share_price_repair.py`, which
imports the functions it replays. The Typer entry point goes into
`tradeexecutor/cli/commands/rebuild_exchange_account_share_price.py`, registered
in `cli/main.py`, and following the
`correct-history` structure.

The command runs with no arguments. Inside the service container,
`docker compose run --rm lighter-ai rebuild-exchange-account-share-price` finds
the state, the positions and the misbooked withdrawals on its own. Every
default can be overridden.

| Option | Default | Override use |
|---|---|---|
| `--id`, `--state-file`, `--strategy-file`, `--log-level`, `--unit-testing` | Shared options and their environment variables (`STATE_FILE`, `STRATEGY_FILE`, …). The state defaults to `state/<executor-id>.json`, as in other commands. | Run against a copy of the state. |
| `--position-id` (repeatable) | Every open exchange-account position. | Limit the run to some positions. |
| `--lighter-test-trade-journal` (repeatable path) | Every `<executor-id>.lighter-test-trade*.json` file in the state file's folder. This covers the test trade's default journal path and the new path an operator chooses for a rerun. | Use journals kept elsewhere. |
| `--capital-withdrawal-balance-update-id` (repeatable) | None. | Add a withdrawal the journal does not show. The position is the one that owns the balance update. |
| `--ignore-balance-update-id` (repeatable) | None. | Drop a detected withdrawal. |
| `--dry-run` | Off. | Print the report and write nothing. |
| `--auto-approve` (`AUTO_APPROVE`) | Off: print the report, then ask for confirmation. A run with no terminal refuses to write. | Run non-interactively. |

Only withdrawals are reclassified. In this incident the deposit correctly
seeded the state. A journal deposit that landed in a position that already had
a share-price state would be fake profit. The command reports that case and
refuses to fix it rather than guessing (see detection rule 4).

### Detecting misbooked withdrawals

The evidence is the test trade's own journal, so detection needs no RPC or
Lighter credentials. For each journal file:

1. Skip it unless `phase` is `withdrawn` or `complete` and it has
   `withdrawal_usdc`. A journal for another vault fails the run (compare
   `vault_address` with the state's Lagoon vault), because it points at a wrong
   state folder.
2. The position is the open Lighter exchange account whose
   `exchange_subaccount_id` equals the journal's `account_index`.
3. The withdrawal is that position's single `vault_flow` balance update with
   `quantity == -withdrawal_usdc`, compared as Decimals, and
   `block_mined_at` at or after `withdrawal_requested_at`. No match is reported
   and skipped. A balance update that already carries the marker counts as
   already repaired. More than one match fails the run and lists them. The
   operator then picks one with `--capital-withdrawal-balance-update-id` and
   `--ignore-balance-update-id`.
4. Skip the journal if the position already has a successful
   `external_account_transfer` withdrawal trade with the journal's
   `claim_tx_hash` or withdrawal request. Then the flow was booked properly (the
   fixed test trade in section 1). Report a journal deposit that did not seed
   the share-price state as unsupported, and fail the run.

The report also lists **suspects** without acting on them: any other valuation
balance update that alone moves the internal share price by more than 50%.
This flags misbooked flows from operations that leave no journal, for the
operator to confirm with `--capital-withdrawal-balance-update-id`. A real
liquidation also shows here, so a suspect is never applied automatically.

A position with no withdrawal to apply is still rebuilt when its stored
`total_withdrawn` is `null` and every balance update is a valuation update.
That fills in `total_withdrawn` and moves `profit_usd` off the
supply × initial fallback. The same checks apply. If nothing changes, the
command prints "nothing to repair" and exits 0 without writing.

### Replay

`replay_exchange_account_share_price(position, withdrawal_balance_update_ids)`
walks the position's successful trades and balance updates in the order the
live code applied them: by `executed_at` for trades and `block_mined_at` for
balance updates, trades first on equal timestamps, then by id. It uses only the
existing functions:

- trades: `create_share_price_state()` / `update_share_price_state()`, as in
  `State.mark_trade_success()`; a zero-quantity placeholder trade leaves no state,
  as in `open_exchange_account_position()`;
- the first balance update that makes the value positive while there is no
  state: `create_share_price_state_for_exchange_account()`, as in
  `ExchangeAccountValuator`;
- a balance update to apply as a withdrawal: `apply_capital_outflow()` with its
  absolute amount and the Decimal remaining quantity;
- other balance updates: `update_share_price_state_for_balance_update()`.

It returns the final state and a timeline with one entry per event: the
event, its timestamp, the position's cumulative quantity, and the state after
it.

`total_withdrawn` is the replay's own total (sell `executed_reserve` plus
applied withdrawals), starting from the creating functions' `0.0`. This is only
safe when every balance update is a valuation update or an applied withdrawal.
The command refuses a position with any other balance update (for example
sub-account funding fees or close write-offs), rather than repeating the
`migrate_share_price_state()` uncertainty.

Mark each applied balance update with
`other_data["share_price_capital_flow"] = "withdrawal"`. The replay also treats
any balance update already carrying that marker as a withdrawal. This makes a
second run idempotent. `migrate_share_price_state()` is left alone. It only
runs on positions without a state, and its trades-then-balance-updates order
could not use the marker correctly anyway.

### Checks before writing

1. The position exists, is an **open** exchange account, and owns every
   withdrawal to apply, whether detected or named. Closed positions are refused. Their
   `get_total_profit_percent()` is a money-weighted return from
   `calculate_closed_flow_neutral_return()`, not the share price, so the stats
   rewrite below would not match. Each withdrawal is a `vault_flow` balance update
   with a negative quantity.
2. **Faithfulness:** the replay without the new withdrawals (markers from an
   earlier run still apply) must reproduce the stored `share_price_state`:
   `current_share_price`, `total_supply`, `cumulative_quantity`, and
   `total_invested` within a relative tolerance of 1e-9. A mismatch means the
   replay is not the live history, and the command exits non-zero without
   writing.
3. The value and quantity of the position, its trades, balance updates
   (other than the marker), valuation updates, reserves and
   `state.portfolio.calculate_total_equity()` are unchanged. Assert this before
   saving.

### Statistics rewrite

Live samples (`calculate_position_statistics()`) read the position's current
quantity, share supply and capital totals when statistics run after a cycle's
execution. `last_valuation_at` only copies `last_pricing_at`, and transfers do
not advance it, so neither sample timestamp reliably identifies the accounting
state behind a sample. Match by content instead:

1. For each `PositionStatistics` sample of the position with a non-null
   `internal_share_price`, find the uncorrected replay entry whose
   `total_supply` equals the stored `internal_total_supply` and whose
   cumulative quantity equals the sample's `value`, with a relative tolerance
   of 1e-9. When several entries match, take the latest one at or before the
   sample's `calculated_at` plus the cycle's execution window. A sample with no
   match makes the command exit without writing, and it reports that sample.
2. Check every matched sample: its stored `internal_share_price`,
   `internal_profit_pct`, `internal_profit_usd`, `profitability` and
   `profit_usd` must equal values recomputed from the uncorrected entry with
   the live formulas, including the `total_supply * initial_share_price`
   fallback while `total_withdrawn` is unknown. This proves the alignment for
   every sample, not just the first and last.
3. Rewrite each sample from the **corrected** entry at the same event index:
   - `internal_share_price = value / total_supply` (or the state price at zero supply)
   - `internal_total_supply = total_supply`
   - `internal_profit_pct = internal_share_price / initial_share_price - 1`
   - `internal_profit_usd = value + total_withdrawn - total_invested`
   - `profitability = internal_profit_pct` and `profit_usd = internal_profit_usd`,
     which is what `get_total_profit_percent()` and `get_total_profit_usd()`
     return for an open exchange account.

Samples before the first applied withdrawal come out unchanged, apart from
`internal_profit_usd` and `profit_usd`. Those switch from the supply × initial
fallback to the flow-neutral figure, which is equal while no deposit has been
made at a share price other than 1.0. Assert that.

### Output and safety

- For each position, print the journals used, the withdrawals applied with
  their source (journal or option), ignored items and suspects. Then print the
  stored and rebuilt current share price, profit % and USD, `total_invested`,
  `total_withdrawn`, the first and last sample's figures, and the number of
  samples changed.
- `--dry-run` stops after the report. Otherwise ask for confirmation unless
  `--auto-approve` is set, as `repair` does. Then take a backup with
  `bootstrap.backup_state(state_file, backup_suffix="share-price-rebuild-backup")`
  and save through the store. All positions are checked before anything is
  written, so a failure on one leaves the file untouched.
- Production run (documented in the strategies operations guide, not
  automated): release the image; stop `lighter-ai`; back up
  `state/lighter-ai.json`; check that `state/lighter-ai.lighter-test-trade*.json`
  exists; run `docker compose run --rm lighter-ai
  rebuild-exchange-account-share-price --dry-run` and check that it picks
  balance update 4 on position 1; run it again without `--dry-run` and confirm;
  restart; check `/state`, `/status` and the position page. If the journal is
  missing, use `--capital-withdrawal-balance-update-id 4`.

## 3. Integration test on a copy of the lighter-ai state

Fixture: `tests/exchange_account/lighter_ai_share_price/lighter-ai.json.gz`, the
2026-11-16 13:35 UTC download of `https://lighter-ai.tradingstrategy.ai/state`
(2.6 MB raw, 112 KB gzipped; public API data with no credentials). Record its
source and capture time in the test module docstring. The test decompresses it
into `tmp_path` and runs the command through `tradeexecutor.cli.main.app`, as
`tests/mainnet_fork/correct_history/test_correct_history.py` does. It needs no
RPC, no fork and no strategy universe.

Journal fixture: `lighter-ai.lighter-test-trade.json`, copied from
`~/strategies/state/` on the production server. It holds addresses, the API
key index, balances and tx hashes; the private key is in the separate operator
record. Review it before committing. Until then the test can write a journal
with the same fields from the values in the state (account 748071, withdrawal
16.995365 USDC, requested before 2026-09-17 16:37 UTC). The real file is
better, because it also checks the field names and the amount the test trade
recorded. The test copies the journal next to the state in `tmp_path`, under
the default name for executor id `lighter-ai`.

`tests/exchange_account/test_lighter_ai_share_price_rebuild.py`:

1. **Baseline:** the loaded fixture has `current_share_price ≈ 0.049338` and
   `get_total_profit_percent() ≈ -0.9507`.
2. **Dry run, no other arguments:** exit code 0; the report names balance
   update 4 on position 1 from the journal; the file's SHA-256 is unchanged; no
   backup is written.
3. **Repair with defaults:** only `--auto-approve` (and the state file and
   unit-testing settings through the environment, as the `correct-history`
   test does); then
   - `current_share_price ≈ 0.887851`, `get_total_profit_percent() ≈ -0.11215`
   - `total_withdrawn == 16.995365`, `total_invested == 15008.885773`
   - `get_total_profit_usd() ≈ -1741.54`
   - position value 13,250.354528, quantity, trade and balance-update counts,
     and total equity equal the baseline
   - balance update 4 carries the marker
   - every one of the 587 samples matched and passed the uncorrected check
   - the first stats sample after 2026-09-17 16:37 UTC has
     `internal_share_price ≈ 0.99974` (it was 0.0556); the last has
     `internal_profit_pct ≈ -0.11215` and `internal_profit_usd ≈ -1741.54`;
     samples before then are unchanged; the sample count is unchanged
   - a `share-price-rebuild-backup` file exists
4. **Idempotence:** a second run with no arguments reports "nothing to
   repair", exits 0 and leaves the file byte-identical.
5. **Overrides:** without a journal, a run with defaults applies nothing and
   lists balance update 4 as a suspect. Adding
   `--capital-withdrawal-balance-update-id 4` gives the same result as step 3.
   With the journal, `--ignore-balance-update-id 4` applies nothing.
6. **Refusals:** each of these exits non-zero and leaves the file
   byte-identical:
   - a balance-update id not on an open exchange-account position
   - a positive balance update
   - a journal for another vault
   - a journal amount matching two balance updates (edit a copy of the fixture)
   - a stats sample altered so that it no longer matches
   - running without `--auto-approve` and without a terminal

## Review

Reviewed with Codex CLI 0.169.0 (`gpt-6.1-sol`, read-only) on 2026-11-16. All
seven findings were accepted and are reflected above:

- the reserve and Lighter baseline before the deposit
- crash windows around the state save and the journal
- the treasury snapshot taken before a claim
- sample alignment by content and a per-sample check
- dropping the migration marker change
- excluding closed positions
- the test patch targets and the custody-aware fake

After that review, the repair command was changed to run with no arguments:
it detects withdrawals from the test-trade journal and accepts overrides.

## Delivery

- Implement in a new `trade-executor` worktree from `master`. The main checkout
  has unrelated uncommitted work on `fix/lagoon-settlement-window-wait`.
- One PR, or two if the test-trade refactor grows: `fix:` for the test-trade
  booking and `feat:` for the repair command, with a `CHANGELOG.md` line.
- Run, each with the repository's `.local-test.env` and worktree `PYTHONPATH`:
  - `pytest tests/exchange_account/test_lighter_ai_share_price_rebuild.py`
  - `pytest tests/exchange_account/test_exchange_account_share_price.py`
  - `pytest tests/lagoon/test_lagoon_lighter_test_trade.py`
  - `pytest tests/lagoon/test_lighter_move_funds.py`
- After release, run the production repair as in section 2, and add the run
  procedure to `strategies/docs/deployment-and-manual-operations.md`.
