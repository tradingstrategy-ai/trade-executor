# Lighter sub-accounts: accounting and test-trade command compatibility

What `check-accounts`, `correct-accounts` and `perform-test-trade` do today
for a Lagoon vault whose Safe owns a Lighter master account and sub-accounts
(each its own exchange-account position, see
[README-Lighter.md](../../tradeexecutor/exchange_account/README-Lighter.md)),
and what still has to change. Written 2026-10-27 after PR #1683 merged, from
reading the code at `befacd195`; nothing here has been implemented yet.

## Summary of actions

| Priority | Command | Action |
|---|---|---|
| 1 | `perform-test-trade` | Add an exchange-account branch: deposit → withdraw round trip through `PositionManager`, pass `state=` to universe construction, and stop `--single-pair` / `--all-pairs` from hitting the router assert |
| 1 | `check-accounts` | Show a pending Lighter transfer as a row instead of raising `UncleanState` |
| 2 | `correct-accounts` | Make `--dry-run` preview instead of abort when a transfer is pending; run the owned-account set check and exit 1 on a mismatch |
| 2 | `correct-accounts` | Look up frozen positions before creating one, and sync frozen Lighter positions |
| 3 | `check-accounts` | Treat a Lighter API error as unclean for Lagoon vaults; take the expected accounts from the state, not only from open positions; accept a closed sub-account position with zero equity; guard `hot_wallet` being `None` |
| 3 | `lagoon-lighter-test-trade` | Decide whether it stays master-only, or revalues sub-account positions and gains a sub-account withdrawal; it overlaps with `lighter-move-funds` |
| 3 | Docs | Correct the claims listed at the end |

## `check-accounts`

### Works today

- Every open or frozen exchange-account position is compared with the
  account's live public Lighter equity, one row per account
  (`tradeexecutor/strategy/account_correction.py:1338-1384`, reader
  `tradeexecutor/exchange_account/lighter.py:464-499`). A difference sets the
  result unclean and the command exits 1; nothing is written to the state.
- In vault mode `check_lighter_account_set()` compares the accounts the Safe
  owns on Lighter with the accounts that have positions
  (`tradeexecutor/cli/commands/check_accounts.py:195-221`,
  `lighter.py:91-126`).
- `LIGHTER_OPERATOR_RECORD_FILE` and
  `LIGHTER_SUB_ACCOUNT_OPERATOR_RECORD_FILES` are optional; they only let the
  read-only reconcile read transfer history for an interrupted transfer
  between accounts (`check_accounts.py:132-142`). Without the sender's record
  the transfer is logged as pending
  (`tradeexecutor/ethereum/lighter/transfer_verification.py:261-267`).

### Gaps

1. **A pending Lighter transfer makes the command raise instead of report.**
   `_check_accounts()` calls `state.check_if_clean()`
   (`account_correction.py:1296`), which raises `UncleanState` for any pending
   exchange-account transfer (`tradeexecutor/state/state.py:1405`). The
   README tells the operator to inspect such a transfer with `check-accounts`.
   Fix: in `check-accounts`, skip `check_if_clean()` for Lighter-pending
   trades and print them as pending rows with the saved request or transfer
   hash.
2. **A Lighter API error passes the check.** An error or an empty account
   list returns clean (`lighter.py:116-122`). Fix: return unclean for a Lagoon
   Lighter vault, or add `--strict`.
3. **The account-set check needs at least one Lighter position.** The
   expected accounts come from open positions only
   (`check_accounts.py:195-203`); a vault whose positions have not been
   created yet reports clean. Fix: take the expected accounts from
   `state.sync.deployment.lighter_accounts` or the universe pairs.
4. **A closed sub-account position fails the check forever.** Closed
   positions are not in the tracked set but Lighter still lists the account.
   Fix: accept an owned account with zero equity and a closed position.
5. **`hot_wallet` can be `None`.** `check_accounts.py:115` reads
   `hot_wallet.address` without a guard; `correct_accounts.py:762` has one.

### Test coverage

Covered: master equity drift exits 1 (`tests/exchange_account/test_lighter_cli_accounts.py:139`,
hot-wallet mode, so without the account-set check); per-sub-account drift
naming both sub-accounts
(`tests/lagoon/test_lagoon_lighter_sub_account_corrections_e2e.py:142-148`);
an untracked owned sub-account fails
(`tests/lagoon/test_lagoon_lighter_multi_subaccount_e2e.py:266-270`).
Missing: a pending transfer, a Lighter API failure, a closed sub-account, a
vault with no Lighter position yet.

## `correct-accounts`

### Works today

- The universe is built with the state (`correct_accounts.py:927-933`), so
  the pairs carry the real master and sub-account indices.
  `create_missing_exchange_account_positions()` opens a zero position for
  every universe exchange-account pair without an open one
  (`account_correction.py:967-1046`).
- `ExchangeAccountSyncModel.sync_positions()` books one `vault_flow` balance
  update per position whose Lighter equity changed
  (`tradeexecutor/exchange_account/sync_model.py:126-262`); a Lighter API
  error raises rather than booking a stale value.
- The Safe reserve is corrected on-chain as before (`correct_accounts.py:1252-1266`).
- `reconcile_verified_lighter_transfers(mutate=not dry_run)` runs first
  (`correct_accounts.py:800-815`): it expires transfers that were never
  submitted and completes verified ones; an unverified one stops the command
  at the preflight (`account_correction.py:1666-1673`).
- `--process-redemption` runs the Lagoon treasury sync before the corrections
  (`correct_accounts.py:1211-1235`).

### Gaps

1. **`--dry-run` aborts where the real run succeeds.** With `mutate=False`
   the verified transfers stay pending and the preflight raises
   `UncleanState` (`correct_accounts.py:827`). Fix: exclude the trades the
   reconcile returned from the preflight in dry-run mode, and print what the
   real run would complete.
2. **Owned accounts outside the universe are neither fixed nor flagged.**
   A Safe-owned Lighter account missing from `lighter_accounts`, or present
   there but not declared in `Parameters.lighter_sub_accounts`, gets no
   position (`tradeexecutor/exchange_account/lighter_strategy.py:86-90`) and
   `check_lighter_account_set()` is not run here. Fix: run the account-set
   check after the sync and exit 1 on a mismatch.
3. **Frozen positions are duplicated and not synced.**
   `get_position_by_trading_pair()` ignores frozen positions
   (`tradeexecutor/state/portfolio.py:440-471`), so a frozen Lighter position
   gets a second open position for the same account, and
   `sync_positions()` iterates open positions only (`sync_model.py:155`)
   while `check-accounts` values frozen ones. Fix: look up with
   `get_open_and_frozen_positions()` before creating, and sync frozen Lighter
   positions too.
4. **Closed positions are reopened** on every run while the pair is in the
   universe (`account_correction.py:1018`). Probably intended; document it.
5. **No Lighter dust handling**; only HyperCore dust is closed
   (`correct_accounts.py:1338`). Zero-equity sub-accounts stay open, which
   suits the model.

### Test coverage

Covered: initial master and two sub-account positions via
`--process-redemption` (`test_lagoon_lighter_multi_subaccount_e2e.py:256-264`);
reconciling an interrupted master to sub-account funding (`:309-326`);
per-sub-account drift booked with the master untouched
(`test_lagoon_lighter_sub_account_corrections_e2e.py:151-165`);
single-account drift (`test_lighter_cli_accounts.py:171`); dry-run without a
pending transfer (`tests/lagoon/test_lighter_move_funds.py:367`).
Missing: dry-run with a pending transfer, a missing sender record, a frozen
or closed position, an owned account outside the universe.

## `perform-test-trade`

### Works today

Nothing for Lighter. `make_test_trade()` (`tradeexecutor/cli/testtrade.py:1024`)
knows spot, credit, vault and cross-chain pairs only.

### Gaps

1. **Wrong account indices for sub-account strategies.** The universe is
   constructed without `state=` (`tradeexecutor/cli/commands/perform_test_trade.py:232-238`),
   so `resolve_lighter_accounts()` uses placeholder indices
   (`lighter_strategy.py:94`) and the pairs do not match the state's
   positions. Fix: pass `state=state`.
2. **`--single-pair` fails** for master plus sub-accounts:
   `universe.get_single_pair()` (`perform_test_trade.py:253`) needs one pair.
3. **Any Lighter pair hits the router assert.** With an existing position the
   buy is skipped and `close_position()` runs; otherwise `open_spot()` runs
   (`testtrade.py:1252-1340`). Either trade lacks
   `TradeFlag.external_account_transfer` and fails at
   `tradeexecutor/strategy/generic/generic_router.py:132-133`; `--all-pairs`
   fails on the first Lighter pair. Fix: add an exchange-account branch to
   `make_test_trade()` that does a small round trip with
   `PositionManager.deposit_to_lighter()` and `withdraw_from_lighter(account)`
   (`tradeexecutor/strategy/pandas_trader/position_manager.py:456,482`),
   optionally `transfer_lighter_funds()` master to sub-account and back, and
   books the result like any other Lighter custody trade. The command then
   needs `LIGHTER_OPERATOR_RECORD_FILE` and the Lighter routing
   configuration, as `start` does. Until then, refuse Lighter pairs with a
   message pointing to `lagoon-lighter-test-trade`.
4. **`assert client is not None`** (`perform_test_trade.py:198`) demands a
   Trading Strategy API key although Lighter universes are built in code.

### `lagoon-lighter-test-trade`

- Does a Safe to master deposit, opens and closes an ETH/USD long on the
  master, then a secure withdrawal and the Safe claim, with a NAV checkpoint
  after each phase and a resumable journal
  (`tradeexecutor/cli/commands/lagoon_lighter_test_trade.py:906,1043,1058`).
- `--lighter-account-index` must be the master
  (`lagoon_lighter_test_trade.py:773-775`); only the master is traded.
  Sub-accounts are added to the posted NAV (`:875-884`) but their positions
  are not revalued (`:588-610`). There is no sub-account withdrawal and no
  master to sub-account funding.
- Overlaps with `lighter-move-funds`, which records
  `external_account_transfer` trades and supports a sub-account L1
  withdrawal with `--lighter-account-index <sub>`; the test-trade command
  calls the eth_defi helpers directly and records no custody trades.
- Covered by one test, master only, with the sequencer mocked
  (`tests/lagoon/test_lagoon_lighter_test_trade.py:127`), in the
  `warm_rpc_test_group` CI group.

## Documentation claims to correct

- "Inspect it with `check-accounts`" for an interrupted transfer
  (`README-Lighter.md:253`, `:461`; `docs/lagoon-lighter-deployment.md:217`):
  the command raises `UncleanState` today (gap 1 above).
- "`correct-accounts --dry-run` does not persist proposed corrections"
  (`README-Lighter.md:815-816`): true, but with a pending transfer it aborts
  instead of previewing.
- "opened at zero by `correct-accounts`" (`README-Lighter.md:181`): only for
  accounts declared in `Parameters` and resolved in the state; an undeclared
  owned account is silently left out.
- No document mentions `perform-test-trade` for Lighter; the tutorials use
  `lagoon-lighter-test-trade`, which matches the code.
