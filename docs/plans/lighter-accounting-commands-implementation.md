# Lighter sub-accounts: implementation plan for the accounting and test-trade commands

Implements items 1 to 3 of
[lighter-accounting-commands-compatibility.md](lighter-accounting-commands-compatibility.md):
`perform-test-trade`, `check-accounts` and `correct-accounts` for a Lagoon
vault with a Lighter master account and sub-accounts. Written 2026-10-27
against master `99cacee5f` and revised after a Codex (gpt-6-astra) review.
Every change reuses what `start`, `lighter-move-funds` and the tests already
have; no new abstractions, no new commands, no new strategy parameters.

## 1. `perform-test-trade`

Goal: `perform-test-trade --single-pair` on a Lighter strategy moves a small
amount of USDC Safe → Lighter → Safe through the normal execution pipeline,
booked as the same custody trades `start` makes, so an operator can verify a
fresh deployment end to end. `--all-pairs` does one round trip per account.

### Changes

1. **Build the universe with the state.**
   `tradeexecutor/cli/commands/perform_test_trade.py:232`: add `state=state`
   to `construct_universe()`, as `check-accounts` and `correct-accounts`
   do. Without it a sub-account strategy gets placeholder account indices.
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
3. **Persist the state before every Lighter submission.** `perform-test-trade`
   saves only after the whole run (`perform_test_trade.py:363`), while
   `LighterRouting` checkpoints before it submits and after it records a
   transfer hash through `ExecutionModel.sync_state_before_broadcast()`
   (`tradeexecutor/strategy/execution_model.py:91`), which does nothing
   until a callback is set. Before the Lighter round trip, call
   `execution_model.set_pre_broadcast_state_sync_callback(lambda: store.sync(state))`
   (`execution_model.py:76`, as `tradeexecutor/strategy/runner.py:1132`
   does for `start`), except in `--simulate` mode. A timeout then leaves a
   pending custody trade in the saved state for `correct-accounts`, never
   moved funds without a record.
4. **An exchange-account branch in `make_test_trade()`**
   (`tradeexecutor/cli/testtrade.py:1024`). After the pair is resolved and
   before the spot logic, if `pair.is_exchange_account()`:
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
   - Add `TradeFlag.test_trade` to each trade's flags after creation,
     keeping the custody flags the methods set.
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
5. **Pair selection.** In `perform_test_trade.py:252`, when
   `--single-pair` is given and the universe contains Lighter pairs, use
   `get_lighter_master_pair(universe.iterate_pairs())` instead of
   `universe.get_single_pair()`, which asserts one pair. `--all-pairs`
   already iterates every pair; each Lighter pair gets its own round trip.
   No new option.
6. **Trading Strategy client.** Replace `assert client is not None`
   (`perform_test_trade.py:198`) with a warning; a strategy whose universe
   needs market data fails at `construct_universe()` with the client's own
   error, a code-built Lighter universe does not.

### Not done

- No change to `lagoon-lighter-test-trade`; it stays master-only.
- No sub-account to sub-account transfer in the test trade; it needs the
  sub-account's key and the transfer tutorial already covers it.

### Test

One black-box test in `tests/lagoon/test_lagoon_lighter_test_trade_cli_e2e.py`
on the simulated stack (`tradeexecutor/testing/lighter_simulation.py`,
`FakeLighterCore`), reusing the deploy and init phases of
`tradeexecutor/testing/lighter_tutorial.py` with
`strategies/test_only/lighter_multi_subaccount_strategy.py`:

1. Deploy a vault with two sub-accounts, `init`, `correct-accounts`.
2. Subscribe and settle with `correct-accounts --process-redemption`, not
   with `start`: the strategy keeps no Safe cash
   (`lighter_multi_subaccount_strategy.py:50`) and a `start` cycle would
   deposit the subscription before the test trade can.
3. `perform-test-trade --single-pair`: one deposit and one withdrawal trade
   on the master position, both `success`, both flagged `test_trade`; the
   Safe's USDC is unchanged, as the fake master withdrawal charges no fee
   (`tradeexecutor/testing/lighter_core.py:234`).
4. `perform-test-trade --all-pairs`: each sub-account position has a funding
   and a withdrawal trade; the Safe's USDC is unchanged, because each round
   trip deposits and withdraws the same amount, while the master's Lighter
   equity is down by one fake transfer fee per sub-account
   (`lighter_core.py:252`), booked as fee balance updates on the master
   position; `check-accounts` passes.
5. Without `LIGHTER_OPERATOR_RECORD_FILE`: the command exits 1 with the
   `ValueError` from `_create_lighter_routing_config()` and writes no trade.

## 2. `check-accounts`

Goal: the command reports instead of crashing for a pending Lighter
transfer, and the account-set check uses the accounts the deployment
declares, while a declared account that has funds but no active position
is still caught.

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
2. **The account-set check uses the declared accounts; the per-position
   checks stay as they are.** Keep `lighter_pairs` from positions for
   the reader selection and the mixed-protocol validation
   (`check_accounts.py:195-203`, `lighter.py:398`), so a residual Lighter
   position in a universe without Lighter pairs is still checked. For the
   account-set check pass
   `validate_lighter_exchange_account_pairs(universe.iterate_pairs())`
   (the universe is built with the state and already translated,
   `tradeexecutor/strategy/trading_strategy_universe.py:819`) and run it
   whenever the universe has Lighter pairs and the sync model has a vault.
   Extend `check_lighter_account_set()` (`lighter.py:91`) with the equity
   snapshot it can read from the same `fetch_lighter_accounts_equity()`
   call `read_lighter_equity()` uses: an owned account that is declared
   but has no open or frozen position passes only with zero equity; a
   closed sub-account position with funds left behind is therefore
   unclean, and one drained to zero is clean. Pass the open and frozen
   position indices as a second set for that rule.
3. **A Lighter read failure is unclean.** `check_lighter_account_set()`
   (`lighter.py:116-122`): return `(False, message)` on a request error and
   when Lighter lists no accounts for the Safe. The function is only called
   for Lagoon vaults, so a hot-wallet deployment is unaffected.
4. **Guard the hot wallet.** `check_accounts.py:115`: log the address only
   when `hot_wallet is not None`, as `correct_accounts.py:762` does.

### Test

Extend `tests/lagoon/test_lagoon_lighter_sub_account_corrections_e2e.py`
with two steps, no new test:

- Make a master to sub-account funding pending the way
  `tests/lagoon/test_lagoon_lighter_multi_subaccount_e2e.py:309` does
  (the fake sequencer holds the transfer record and the trade carries
  `lighter_transfer_tx_hash`), run `check-accounts`: exits 1, the output
  names the trade id and "pending", and the balance table is still printed.
- Reconcile it with `correct-accounts`, drain one sub-account through the
  fake sequencer and reconcile again, close that position in the state,
  run `check-accounts`: exits 0. Then put equity back on the drained
  account in the fake and run `check-accounts`: exits 1 naming the account.

Unit test in `tests/exchange_account/test_lighter_exchange_account.py`
(extend the existing `check_lighter_account_set` test): a mocked request
error returns unclean; a declared account without a position passes with
zero equity and fails with equity.

## 3. `correct-accounts`

Goal: `--dry-run` previews what the real run does, undeclared or missing
accounts are flagged, and frozen positions are handled like open ones.

### Changes

1. **Dry run previews the reconciliation.**
   `tradeexecutor/cli/commands/correct_accounts.py:801`: call
   `reconcile_verified_lighter_transfers(..., mutate=True)` in both modes.
   A dry run never persists: it uses `SimulateStore` (`correct_accounts.py:775`),
   the early save at `:809` is guarded by `if not dry_run`, and the dry-run
   return at `:1306` precedes every later save (`:1352`, `:1400`, `:1420`,
   `:1460`). The preflight then sees the completed transfers and the
   printed corrections match what the real run books.
2. **Flag owned accounts outside the universe.** After the exchange-account
   sync, for a Lagoon vault, call `check_lighter_account_set()` with the
   same universe-based pair list and position-index set as `check-accounts`
   (change 2.2). On a mismatch log the message at error level and exit 1 on
   every exit path, including the dry-run return at `:1306` and the early
   successful exit at `:1375`; ordinary corrections are still saved first,
   because they are right regardless, and the operator must declare or
   drain the extra account.
3. **Frozen positions.** `create_missing_exchange_account_positions()`
   (`account_correction.py:1018`): look the pair up among
   `state.portfolio.get_open_and_frozen_positions()` before creating a
   position. `ExchangeAccountSyncModel.sync_positions()`
   (`tradeexecutor/exchange_account/sync_model.py:155`): iterate
   `get_open_and_frozen_positions()` so a frozen position is revalued like
   an open one. This applies to every exchange-account protocol the sync
   model serves, which is the intended behaviour: `check-accounts` already
   values frozen positions, so the two commands then agree.
4. **Document** in `README-Lighter.md` that a closed exchange-account
   position is reopened at zero on the next `correct-accounts` while its
   account is declared, and that `--dry-run` previews the reconciliation
   without saving it.

### Test

Extend `tests/lagoon/test_lagoon_lighter_sub_account_corrections_e2e.py`
with three steps:

- With the pending funding from the `check-accounts` step above, run
  `correct-accounts --dry-run` with the operator record: exits 0, logs
  "Found 1 verified Lighter transfer(s)", the state file is unchanged.
- Add an extra sub-account to `FakeLighterCore` for the Safe without
  declaring it, run `correct-accounts`: the balance corrections are saved
  and the command exits 1 naming the account index.
- Freeze one sub-account position in the state, change its fake equity, run
  `correct-accounts`: exactly one balance update on the frozen position and
  no new position for its account.

## Order and size

1. `check-accounts` changes 2 to 4 and `correct-accounts` change 3: small
   and independent.
2. `check-accounts` change 1 and `correct-accounts` changes 1 and 2.
3. `perform-test-trade`, the largest change, last.

Documentation corrections listed in the compatibility document are made
with the change that resolves each claim.
