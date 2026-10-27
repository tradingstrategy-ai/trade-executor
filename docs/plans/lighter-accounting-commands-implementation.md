# Lighter sub-accounts: implementation plan for the accounting and test-trade commands

Implements items 1 to 3 of
[lighter-accounting-commands-compatibility.md](lighter-accounting-commands-compatibility.md):
`perform-test-trade`, `check-accounts` and `correct-accounts` for a Lagoon
vault with a Lighter master account and sub-accounts. Written 2026-10-27
against master `99cacee5f`, revised after a Codex (gpt-6-astra) review, a
clean-context subagent review and the maintainer's decisions recorded below. Every change reuses what `start`,
`lighter-move-funds` and the tests already have; no new abstractions, no new
commands, no new strategy parameters.

## Decisions

- **`perform-test-trade` does not checkpoint the state before a Lighter
  submission.** It is a single-use manual command run by an operator who
  watches it. If it times out after Lighter accepted a transfer, the state
  holds no trade for the moved funds; `check-accounts` then shows the
  difference on the account and the Safe, `correct-accounts` books it as a
  balance update, and the operator reruns the test trade. The same
  trade-off already applies to cross-chain test trades
  (`tradeexecutor/cli/commands/perform_test_trade.py:276`).
- **`correct-accounts` closes Lighter positions under Lighter's own
  close and dust rules, separate from the HyperCore ones.** HyperCore dust
  is redeemed and written off under its own thresholds in
  `tradeexecutor/ethereum/vault/hypercore_small_position_cleanup.py` and
  `tradeexecutor/strategy/dust.py`; none of that code, its constants or
  `close_position_with_empty_trade()` is reused. Lighter's rule follows its
  protocol: a sub-account cannot withdraw less than the 1 USDC minimum
  (`LIGHTER_MIN_MAINNET_USDC`), so equity below it is unwithdrawable dust.
  A position whose account the strategy no longer declares is closed when
  its equity is below that minimum, with the residual written off; an
  undeclared account at or above it is reported with the command that
  drains it. Declared accounts are never closed by `correct-accounts`; the
  strategy manages them. `check-accounts` only reports; it never changes
  the state.

## 1. `perform-test-trade`

Goal: `perform-test-trade --single-pair` on a Lighter strategy moves a small
amount of USDC Safe → Lighter → Safe through the normal execution pipeline,
booked as the same custody trades `start` makes, so an operator can verify a
fresh deployment end to end. `--all-pairs` does one round trip per account.

### Changes

1. **Build the universe with the state.**
   `perform_test_trade.py:232`: add `state=state` to `construct_universe()`,
   as `check-accounts` and `correct-accounts` do. Without it a sub-account
   strategy gets placeholder account indices.
2. **Wire the Lighter routing configuration.** Import
   `_create_lighter_routing_config()` from
   `tradeexecutor/cli/commands/start.py:68` (no move; nothing requires it),
   call it with the same inputs and pass the result as
   `lighter_routing_config=` to `create_execution_and_sync_model()`
   (`tradeexecutor/cli/bootstrap.py:214`). `perform-test-trade` gains the
   same four options `start` has. Move the three option declarations that
   `start.py:162-176` still defines inline (`lighter_operator_record_file`,
   `lighter_withdrawal_timeout`, `lighter_transfer_timeout`) into
   `shared_options.py` next to `lighter_sub_account_operator_record_files`
   so both commands declare them once. The helper already raises
   `ValueError` naming `LIGHTER_OPERATOR_RECORD_FILE` when the strategy sets
   `lighter_cash_management` and no record is given (`start.py:98`); that is
   the "not configured" error, nothing new is needed.
3. **An exchange-account branch in `make_test_trade()`**
   (`tradeexecutor/cli/testtrade.py:1024`). After the pair is resolved and
   before the reserve checks at `testtrade.py:1185-1193`, so that a
   `close_only` withdrawal works with an empty Safe, if
   `pair.is_exchange_account()`:
   - Require `pair.get_exchange_account_protocol() == LIGHTER_PROTOCOL`;
     raise with a clear message for any other protocol.
   - Build one `PositionManager` as the spot path does and run a round trip
     with the existing custody methods
     (`tradeexecutor/strategy/pandas_trader/position_manager.py`), one
     `execution_model.execute_trades()` call per trade. `execute_trades()`
     already selects the sequential path for Lighter
     (`tradeexecutor/ethereum/execution.py:855`) and settles a transfer
     between accounts off-chain (`:780`), and `runner.setup_routing()` gives
     the routing state the Lagoon vault a sub-account withdrawal needs
     (`tradeexecutor/ethereum/lagoon/execution.py:81`); no new execution
     code.
     - master pair: `deposit_to_lighter(amount)`; `withdraw_from_lighter("master", amount)`.
     - sub-account pair with id `x`: `deposit_to_lighter(amount)`;
       `transfer_lighter_funds("master", x, amount)`;
       `withdraw_from_lighter(x, amount)`. The withdrawal is the key-free L1
       path, so no sub-account key is needed.
   - Do not add `TradeFlag.test_trade`: `TradingPosition.is_test()` is
     true when any trade carries it (`tradeexecutor/state/position.py:427`),
     which would turn the vault's permanent master position into a test
     position for `close-position --all-test-trades`
     (`tradeexecutor/cli/close_position.py:204-212`) and hide it from
     `get_closed_positions_for_pair()` (`tradeexecutor/state/portfolio.py:354`).
     Pass `notes="perform-test-trade: ..."` to the custody methods instead;
     the trade summary already prints notes.
   - `buy_only` stops after the deposit (and the funding for a sub-account);
     `close_only` performs the withdrawal only. No short, no `test_short`.
   - **Fee headroom.** A transfer between accounts charges Lighter's fee on
     top of `amount` on the source (`position_manager.py:523`). Live, the
     fee is 0 for the same owner; the fake sequencer charges one. Document
     that the sub-account round trip needs the master's free collateral to
     cover `amount + fee`, and let the routing's own rejection surface it;
     no fee logic in the test trade.
   - `amount` defaults to the command's `--amount`; Lighter's 1 USDC minimum
     applies, so document that the default `1.0` is the minimum
     (`LIGHTER_MIN_MAINNET_USDC`).
4. **Pair selection.** In `perform_test_trade.py:252`, when
   `--single-pair` is given and the universe contains Lighter pairs, use
   `get_lighter_master_pair(universe.iterate_pairs())` instead of
   `universe.get_single_pair()`, which asserts one pair. `--all-pairs`
   already iterates every pair; each Lighter pair gets its own round trip.
   No new option.
5. **Trading Strategy client.** Replace `assert client is not None`
   (`perform_test_trade.py:198`) with a warning; a strategy whose universe
   needs market data fails at `construct_universe()` with the client's own
   error, a code-built Lighter universe does not.
6. **Document the no-checkpoint trade-off** from the decisions above in the
   command's docstring and in `README-Lighter.md`, next to the recovery
   steps.

### Not done

- No change to `lagoon-lighter-test-trade`; it stays master-only.
- No sub-account to sub-account transfer in the test trade; it needs the
  sub-account's key and the transfer tutorial already covers it.

### Test

One black-box test in `tests/lagoon/test_lagoon_lighter_test_trade_cli_e2e.py`,
structured like `tests/lagoon/test_lagoon_lighter_sub_account_corrections_e2e.py`:
the shared fixed Ethereum fork, `SIMULATE=true`, the `Tutorial` helpers of
`tradeexecutor/testing/lighter_tutorial.py` with
`strategies/test_only/lighter_scripted_transfers_strategy.py`, and
`run_cli()` for every command.

1. `deploy()`, `initialise_state()`.
2. Subscribe without running the strategy: `subscribe()` settles the
   Lagoon deposit with a `start` cycle (`lighter_tutorial.py:533`), which
   would run the script's first step and move the Safe cash to Lighter.
   Add `settle_with_start: bool = True` to `subscribe()`; with `False` it
   settles with `run_cli(["correct-accounts", "--process-redemption"])`,
   exactly as `initialise_state()` does (`:507`). The Safe then holds the
   subscription.
3. `perform-test-trade --single-pair --buy-only --amount 2`: one deposit
   trade on the master position, `success`, notes starting with
   "perform-test-trade"; the master now holds the 1 USDC activation deposit
   plus 2 USDC, which is the fee headroom the next step needs (the fake
   charges `SIMULATED_TRANSFER_FEE_USDC` per funding on the master,
   `lighter_core.py:252`, and rejects a funding only when
   `amount + fee` exceeds the free collateral, so without this step two
   sub-accounts would sit exactly on the boundary).
4. `perform-test-trade --all-pairs --amount 1`: the master gets another
   deposit and withdrawal; each sub-account position gets a funding and a
   withdrawal trade; the Safe's USDC is down by the net master deposit
   only, because every round trip withdraws what it deposited, while the
   master's Lighter equity is down by one fake transfer fee per
   sub-account, booked as fee balance updates on the master position by
   `complete_exchange_account_funding()` (`tradeexecutor/exchange_account/state.py:355-361`);
   `check-accounts` passes.
4b. `perform-test-trade --single-pair --close-only --amount 2`: one
   withdrawal trade on the master; the Safe's USDC equals its value before
   step 3, as the fake master withdrawal charges no fee
   (`lighter_core.py:234`).
5. Without `LIGHTER_OPERATOR_RECORD_FILE`: the command exits 1 with the
   `ValueError` from `_create_lighter_routing_config()` and writes no trade.

## 2. `check-accounts`

Goal: the command reports instead of crashing for a pending Lighter
transfer, and the account-set check compares what the Safe owns on Lighter
with what the strategy declares and what the state tracks, so that
`correct-accounts` has something to act on.

### Changes

1. **Report pending Lighter transfers as rows, keep raising for anything
   else.** In `tradeexecutor/cli/commands/check_accounts.py`, after loading
   the state, collect the trades of open and frozen positions with
   `trade.is_exchange_account_transfer_pending()` whose pair protocol is
   Lighter. For every other unfinished trade (`trade.is_unfinished()` or a
   non-Lighter pending transfer) raise `UncleanState` exactly as
   `state.check_if_clean()` does today, so nothing new is bypassed. Log one
   line per pending Lighter trade (trade id, position ticker, status, the
   `other_data` hashes `lighter_transfer_tx_hash`,
   `lighter_withdrawal_request_id`, `lighter_claim_tx_hash`) with the hint
   "run correct-accounts or repair". Add the keyword
   `check_clean: bool = True` to `_check_accounts()`
   (`tradeexecutor/strategy/account_correction.py:1296`) and pass
   `check_clean=not pending`; nothing else in that function depends on the
   call. Combine the result as `clean = clean and not pending` after the
   call, because `clean, df = _check_accounts(...)` at
   `check_accounts.py:206` would otherwise overwrite an earlier `False`.
2. **The account-set check knows declared accounts and tracked positions.**
   Keep `lighter_pairs` from positions for the reader selection and the
   mixed-protocol validation (`check_accounts.py:195-203`,
   `lighter.py:398`). Give `check_lighter_account_set()`
   (`tradeexecutor/exchange_account/lighter.py:91`) a second set,
   `declared`, the indices of
   `validate_lighter_exchange_account_pairs(universe.iterate_pairs())` (the
   universe is built with the state and already translated,
   `tradeexecutor/strategy/trading_strategy_universe.py:819`), and the
   per-account equity it can read from the same
   `fetch_lighter_accounts_equity()` call `read_lighter_equity()` uses. Run
   it whenever the sync model has a vault and either set is non-empty. The
   rule, each case with its own message:
   - an owned account with equity at or above `LIGHTER_DUST_USDC` and no
     open or frozen position is unclean: "run correct-accounts" (it
     reopens a declared account, and reports an undeclared one with the
     drain command);
   - a tracked position whose account is not declared is unclean: "run
     correct-accounts" (it closes the position when the equity is Lighter
     dust, below `LIGHTER_DUST_USDC`, and otherwise names the drain
     command);
   - an owned account that is neither declared nor tracked is unclean when
     its equity is at or above `LIGHTER_DUST_USDC`, and clean below it;
   - a declared account with equity below `LIGHTER_DUST_USDC` and no
     position is clean, so a closed or never-funded sub-account does not
     fail the check.
   `LIGHTER_DUST_USDC = LIGHTER_MIN_MAINNET_USDC` is declared in
   `tradeexecutor/exchange_account/lighter.py` with a comment saying why:
   it is the smallest amount a Lighter withdrawal can move.
3. **A Lighter read failure is unclean.** `check_lighter_account_set()`:
   return `(False, message)` on a request error and when Lighter lists no
   accounts for the Safe. The function is only called for Lagoon vaults, so
   a hot-wallet deployment is unaffected.
4. **Guard the hot wallet.** `check_accounts.py:115`: log the address only
   when `hot_wallet is not None`, as `correct_accounts.py:762` does.

## 3. `correct-accounts`

Goal: `--dry-run` previews what the real run does, frozen positions are
handled like open ones, and positions of accounts the strategy no longer
declares are cleaned up like HyperCore dust.

### Changes

1. **Dry run previews the reconciliation.**
   `correct_accounts.py:801`: call
   `reconcile_verified_lighter_transfers(..., mutate=True)` in both modes.
   A dry run never persists: it uses `SimulateStore` (`correct_accounts.py:775`),
   the early save at `:809` is guarded by `if not dry_run`, and the dry-run
   return at `:1306` precedes every later save (`:1352`, `:1400`, `:1420`,
   `:1460`). The preflight then sees the completed transfers and the
   printed corrections match what the real run books. Change the log line
   at `:810-814` to say "Reconciled N verified Lighter transfer(s)
   (dry run, not saved)" in dry-run mode, so the preview reads as what it
   is.
2. **Frozen positions.** `create_missing_exchange_account_positions()`
   (`account_correction.py:1018`): look the pair up among
   `state.portfolio.get_open_and_frozen_positions()` before creating a
   position. `ExchangeAccountSyncModel.sync_positions()`
   (`tradeexecutor/exchange_account/sync_model.py:155`): iterate
   `get_open_and_frozen_positions()` so a frozen position is revalued like
   an open one. This applies to every exchange-account protocol the sync
   model serves, which is the intended behaviour: `check-accounts` already
   values frozen positions, so the two commands then agree. The executor
   itself never freezes an exchange-account position: a failed custody
   transfer keeps it open (`tradeexecutor/state/freeze.py:36-42`), so a
   frozen one comes from manual repair. `PositionManager` and
   `ensure_lighter_positions()` look up open positions only
   (`position_manager.py:445,545`), so document in `README-Lighter.md`
   that a frozen Lighter position must be unfrozen with `repair` before
   `start` runs, or the next custody trade opens a second position for the
   account.
3. **Close positions of undeclared accounts under Lighter's own rules.**
   New module `tradeexecutor/exchange_account/lighter_cleanup.py`, Lighter
   only, with one public function
   `close_undeclared_lighter_positions(state, universe, equity, executed_at) -> list[TradingPosition]`
   called from `correct_accounts.py` right after the exchange-account sync
   has booked the live equity on every position (`:1080-1088`), next to
   (not inside) the first HyperCore dust call at `:1157` and before the
   preflight at `:1169` and the dry-run return at `:1306`, so a dry run
   previews the closes and the real run's preflight sees the closed
   positions. A frozen Lighter position is skipped with a warning "run
   repair first": `unfreeze_position()` (`tradeexecutor/state/repair.py:790-802`)
   only accepts repaired trades and asserts `can_be_closed()`, which is
   never true for an exchange-account position
   (`tradeexecutor/state/position.py:1438-1441`). For each open Lighter
   position whose pair is not in the universe:
   - equity below `LIGHTER_DUST_USDC`: close it with a new
     `close_exchange_account_position()` in
     `tradeexecutor/exchange_account/state.py`, the counterpart of
     `open_exchange_account_position()` (`state.py:380`): when the
     quantity is not zero it books the residual as one balance update of
     `-quantity` with cause `BalanceUpdateCause.correction` and the note
     "Lighter dust below the 1 USDC withdrawal minimum written off"; it
     then records a zero-quantity synthetic closing trade at price 1.0
     without custody flags, completed with `trade.mark_success(force=True)`
     as the opener does (`state.py:608-616`, because
     `State.mark_trade_success()` rejects a flagged exchange-account sell
     with zero reserve, `tradeexecutor/state/state.py:944-949`), and calls
     `state.portfolio.close_position(position, executed_at)`. The account
     stops being tracked and `check-accounts` is clean again;
   - equity at or above `LIGHTER_DUST_USDC`: leave it and log at error
     level "Lighter account `<index>` holds `<equity>` USDC but the
     strategy no longer declares it; drain it with
     `lighter-move-funds --lighter-account-index <index>` and rerun
     correct-accounts". The command still saves its other corrections and
     exits 1 on every exit path, including the dry-run return at `:1306`
     and the early successful exit at `:1375`.
   An owned account that is neither declared nor tracked gets the same
   error message at or above the dust threshold, from the account-set
   check below. No automatic withdrawal: a Lighter withdrawal takes about
   half an hour to become claimable and is an operator decision. Nothing
   from the HyperCore cleanup is called or imported.
4. **Run the account-set check.** Immediately after the cleanup, at the
   same place, for a Lagoon vault, call `check_lighter_account_set()` with
   the same arguments as `check-accounts` (change 2.2); remember the
   result and exit 1 with its message on every exit path, after
   `store.sync()` at `:1352` in the real run and after the preview in a
   dry run. This catches an owned account with funds that has no position
   and is not declared.
5. **Document** in `README-Lighter.md`: a closed position of a declared
   account is reopened at zero on the next `correct-accounts`; a position
   of an account the strategy stopped declaring is closed once drained;
   `--dry-run` previews the reconciliation without saving it.

## Tests for `check-accounts` and `correct-accounts`

Extend `tests/lagoon/test_lagoon_lighter_sub_account_corrections_e2e.py`
after its current step 5, so the scenarios run on the vault and the fake
sequencer it already has. The scripted strategy there moves fixed amounts
through five `start` cycles and ends with every account drained to the Safe
(`tradeexecutor/testing/lighter_scripted_transfers.py`), which gives the
closed-account scenarios for free; its two sub-accounts become undeclared
by pointing `STRATEGY_FILE` at `strategies/test_only/minimal_lighter_strategy.py`,
which declares only the master, with `LIGHTER_ACCOUNT_INDEX` set to the
master index the state records.

6. Interrupt the next scripted transfer the way
   `tests/lagoon/test_lagoon_lighter_multi_subaccount_e2e.py:309` does:
   `fake.execute_transfers = False`, `run_start()` raises `TimeoutError`,
   the state holds one pending funding with `lighter_transfer_tx_hash`.
   `check-accounts` exits 1, the output names the trade id and "pending",
   and the balance table is still printed. `fake.execute_pending_transfers()`,
   then `correct-accounts --dry-run` with the sub-account signer files:
   exits 0, logs "Reconciled 1 verified Lighter transfer(s) (dry run, not
   saved)", the state file is unchanged. `correct-accounts` without
   `--dry-run` completes it.
7. Restore `fake.execute_transfers = True` (as
   `test_lagoon_lighter_multi_subaccount_e2e.py:319` does) and run the
   remaining scripted cycles until every account is drained;
   `check-accounts` passes with three zero-equity positions.
8. Move one sub-account position from `open_positions` to
   `frozen_positions` in the saved state, credit its fake account, run
   `correct-accounts`: exactly one balance update on the frozen position,
   no new position for its account; `check-accounts` passes. Move the
   position back to `open_positions` in the state before step 9.
9. Switch `STRATEGY_FILE` to `minimal_lighter_strategy.py` with
   `LIGHTER_ACCOUNT_INDEX=<master>`. `check-accounts` exits 1 naming the
   funded, now undeclared sub-account. `correct-accounts` exits 1 with the
   `lighter-move-funds` instruction and closes the other, drained
   sub-account's position. Debit the fake account to 0.4 USDC, below the
   1 USDC withdrawal minimum, run `correct-accounts`: it writes the
   residual off as one `correction` balance update and closes the last
   position; `check-accounts` passes with the master alone.

New unit test in `tests/exchange_account/test_lighter_exchange_account.py`
for `check_lighter_account_set()` (none exists today): a mocked request
error returns unclean; each of the four rule cases in change 2.2 returns the
expected result from a mocked account list and equity snapshot. Unit test
in `tests/exchange_account/test_lighter_multi_account_positions.py` (extend
the existing test): `close_undeclared_lighter_positions()` on a state with
one declared, one undeclared dust and one undeclared funded position closes
only the dust position, with its write-off balance update, and reports the
funded one; `state.check_if_clean()` passes afterwards.

## Order and size

1. `check-accounts` changes 3 and 4, `correct-accounts` change 2: small and
   independent.
2. `check-accounts` change 2 with `correct-accounts` changes 3 and 4 (they
   share the account-set rule), then `check-accounts` change 1 and
   `correct-accounts` change 1.
3. `perform-test-trade`, the largest change, last.

Documentation corrections listed in the compatibility document are made
with the change that resolves each claim.
