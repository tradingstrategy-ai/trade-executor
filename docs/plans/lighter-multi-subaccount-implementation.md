# Lighter multi-subaccount vaults: implementation plan

## Goal

Support deploying and operating a Lagoon vault whose Safe owns one Lighter
master account and several Lighter sub-accounts, in both
`web3-ethereum-defi` and `trade-executor`:

- `lagoon-deploy-vault` creates the master account, N sub-accounts and the
  API keys for each, and hands the Nautilus keys over in a protected file.
- `trade-executor` values all accounts in NAV, deposits from the Safe to the
  master account, withdraws from any account back to the Safe, and moves
  capital between the master account and sub-accounts from `decide_trades()`.
- A black-box Typer CLI integration test drives one vault with two
  sub-accounts through deposit, transfer in both directions and withdrawal,
  with Lighter Core mocked and everything on Ethereum mocked by an Anvil fork.
- A manual mainnet tutorial script repeats that lifecycle with real money and
  ends by refunding the deployer with all stablecoins and ETH.

This plan implements the analysis in
[`lighter-multi-subaccount-analysis.md`](./lighter-multi-subaccount-analysis.md).
Read that document first for the protocol facts; they are not repeated here.

Interpretation: "two vaults" in the request means two Lighter sub-accounts
under one Lagoon vault, because Lighter only allows key-only transfers between
accounts of the same L1 owner. Two separate Lagoon vaults cannot transfer to
each other without an L1 signature a Safe cannot produce.

## Scope and non-goals

In scope:

- Ethereum mainnet Lighter deployment only; Lighter EVM is handled by keeping
  every address and chain id behind `LighterDeployment` as the analysis
  recommends.
- Master account as the treasury hub, sub-accounts as Nautilus trading
  accounts. Each sub-account gets a Nautilus key (slot 4) and an executor
  key (slot 5).
- One new internal-transfer trade kind, routed through the existing
  `decide_trades()` → `TradeExecution` → `GenericRouting` → `LighterRouting`
  pipeline.
- A pure allocator policy that turns target weights into transfer decisions.
- CLI, operator record, deployment report and documentation updates.
- The two test deliverables above.

Out of scope:

- Lighter order routing inside `trade-executor`. Nautilus trades.
- Guard Solidity changes. The analysis shows none is mandatory. The optional
  account-index allowlist is deferred.
- Whitelisting `changePubKey` for the asset manager. Key registration remains
  a Safe-owner action done at deployment.
- Creating sub-accounts after deployment. A later version can add a
  governance command for it.
- Lighter EVM deployment, Robinhood Chain, Lighter public pools.
- Dynamic weight discovery from Nautilus. Weights are strategy parameters.

## Design decisions

### One TradingPosition per Lighter account

Every Lighter account, the master and each sub-account, is one
exchange-account `TradingPairIdentifier` and one open `TradingPosition` in
the portfolio. The position quantity is that account's USDC equity, kept in
sync by `ExchangeAccountSyncModel` exactly as today, so NAV, PnL, account
checks and reporting work per account.

Pair identity rules. `TradingPairIdentifier.__eq__` and `__hash__` compare
only `base.address`, `quote.address` and `fee`, and `get_identifier()` returns
`pool_address`. Today every Lighter pair shares `LIGHTER_L1_CONTRACT` as base
address and pool address with `internal_id = 1`, so two Lighter pairs compare
equal and every `p.pair == pair` lookup collides. Derive encodes only
`pool_address`, which is not enough, and no test has ever put two
exchange-account pairs of one protocol in one portfolio. The rules are:

- **Base asset address is unique per account.** It is a synthetic 20-byte
  address `0x4c49` ("LI") followed by the account index as 36 hex digits,
  so it is readable in logs and cannot collide with a real token or with
  another account. `pool_address` and `exchange_address` use the same value.
  The old `LIGHTER_L1_CONTRACT` address moves to `other_data["lighter_l1_contract"]`.
- **`internal_id` is unique and namespaced.** `internal_id =
  LIGHTER_EXCHANGE_ACCOUNT_INTERNAL_ID_OFFSET + account_index` with the offset
  at `10**15`, above every Trading Strategy dataset pair id and above the
  `2**48` Lighter index range, so a mixed universe with GMX or Derive pairs
  (both hard-coded to `internal_id = 1`) and dataset pairs keeps the
  duplicate-id assertion in `create_pair_universe_from_code` satisfied.
- **Ticker and description carry the account.** `get_ticker()` gives
  `LIGHTER-<index>-USDC` and `get_human_description()` gives
  `Lighter master account 748071` or `Lighter sub-account 281474976710600
  (trend)`, so `show-positions`, `check-accounts` rows and logs are
  distinguishable.
- `other_data` gains `lighter_account_role` (`master` or `sub_account`),
  `lighter_master_account_index` and an optional `lighter_strategy_label` for
  the Nautilus strategy name. `exchange_subaccount_id` keeps the account index
  for `get_exchange_account_id()`.
- Position creation is per pair: `open_exchange_account_position()` is called
  once for every pair whose position is missing, both by strategies and by
  `create_missing_exchange_account_positions()` in account correction. The
  "any exchange-account position exists" guard in `lighter-ai.py` is replaced
  by a per-pair check.
- The existing single-account identity is migrated once with
  `repair --migrate-lighter-pair-identity` for the current `lighter-ai`
  production state. It rewrites every pair and asset reference carrying the
  old identity across open, frozen and closed positions, their trades and
  balance updates, and the blacklist structures (`asset_blacklist`,
  `blacklisted_assets`, `blacklist_reason`) so a blacklist against the old
  base address keeps blocking the migrated account. It runs as an idempotent
  preview first, validates that every migrated pair exists in the updated
  universe, and refuses an ambiguous mapping instead of dropping a
  restriction. New deployments never produce the old identity.
- Derive keeps its current identity in this work; a follow-up can apply the
  same base-address rule if Derive ever needs two sub-accounts in one
  portfolio.

### Changes required for one position per account

The table lists every place that assumes one exchange-account position, with
the change this plan makes. Places that already iterate every position (NAV
summation, `get_position_by_id`, `find_position_for_trade`, `check_if_clean`,
`ExchangeAccountSyncModel`, `ExchangeAccountValuator`,
`_build_exchange_account_checks`, freeze rules, `reconcile_verified_lighter_transfers`,
trade ordering, `PositionManager.get_exchange_account_available_balance`) need
no change and are not listed.

| Location | Problem today | Change |
|----------|---------------|--------|
| `identifier.py` `__eq__`/`__hash__`, `get_identifier()` | All Lighter pairs equal | Solved by the unique base address; add a regression test that two Lighter pairs are unequal, hash differently and have different identifiers. |
| `portfolio.py` `get_position_by_trading_pair`, `get_open_position_for_pair`, `get_closed_positions_for_pair`, `get_frozen_position`, `get_equity_for_pair` | Match on `p.pair == pair` or `pool_address`, so the second account hits the first position | Solved by identity; covered by a test with two open Lighter positions. |
| `portfolio.py` `get_all_traded_pairs`, `get_single_pair` | Set de-dupe collapses N pairs to one | Solved by identity; `get_single_pair()` must now raise for a multi-account universe, which the test strategy no longer calls. |
| `Portfolio.create_trade` without `position=` | Attaches the second account's opening trade to the first position | `open_exchange_account_position()` passes the pair with unique identity; also assert the found position's pair identifier equals the requested one. |
| `trading_strategy_universe.py` duplicate `internal_id` assert; `dex_data_translation.py` pair cache; `alpha_model.py` signals keyed by `internal_id` | `internal_id = 1` for every account | Namespaced `internal_id`. |
| `lighter.py` `validate_lighter_exchange_account_pairs` | Raises on more than one pair | Keep the single-protocol rule, require exactly one `master` role, require unique account indices. Update `test_lighter_exchange_account.py`. |
| `ethereum_protocol_adapters.py` `_auto_discover_lighter` and `lighter.py` `create_lighter_vault_valuation_func` | Lagoon NAV = Safe + `pairs[0]` only; sub-account equity would be missing from posted NAV | Valuation function takes all account indices and sums one by-L1-address snapshot; fails closed if any account is missing. This is NAV-critical and is the first assertion in the e2e test. |
| `lighter_routing.py` `_prepare_deposit` | L1 deposit funds the master; verification waits on `trade.pair`'s account | Deposit trades are only accepted on the master pair; a sub-account funding is an internal transfer trade. |
| `lighter_routing.py` `_prepare_withdrawal` | Uses the single operator record's account index, not `trade.pair` | Withdrawals are only accepted on the master pair; the signer is looked up from `LighterOperatorRecords` by `trade.pair.get_exchange_account_id()` and must match. |
| `account_correction.py` `create_missing_exchange_account_positions` | Second pair is treated as existing | Solved by identity; the 1 USD reserve placeholder per auto-created position becomes N USD, reconciled by the Lagoon reserve sync and documented. |
| `account_correction.py` `_format_account_check_row` | Every Lighter row shows the same index, address and ticker | Print the account index and role from the pair; `holding_address` `lighter:<id>` is already set. |
| `exchange_account/valuation.py` `balance_update_id` | Per-position `max + 1` versus the portfolio-wide counter in the sync model | Use `portfolio.next_balance_update_id` in both places; `ExchangeAccountValuator._create_balance_update()` only receives the position today, so the valuator must be given the portfolio (or an id allocator callback) when constructed. |
| `state.py` `is_good_pair` blacklist | Keyed by base asset, so one blacklist hits all accounts | Solved by the unique base address. |
| `lighter_move_funds.py` `_get_lighter_position` | Requires exactly one position matching the operator index | `--lighter-account-index` selects the position; default is the master. |
| `lagoon_lighter_test_trade.py` | "Expected exactly one" Lighter position | Select the master position explicitly; the test trade stays a master-account test. |
| `lagoon_redeem.py` | Single `LIGHTER_ACCOUNT_INDEX` for the 5 USDC ignore rule | Read all account indices from state positions and apply the threshold to the sum. |
| `check_accounts.py` | Fails through the validator | Follows the validator change. |
| `strategies/lighter-ai.py`, `lighter_cash_management_strategy.py` | `any(is_exchange_account)` and `get_single_pair()` | Per-pair position creation; new multi-account strategy helper. |
| `exchange_account/state.py` `create_exchange_account_transfer`, `mark_trade_success`, reserve allocation | Model only Safe ↔ account moves | New internal transfer trade kind below. |
| `tests/exchange_account/` | No test with two exchange-account pairs in one universe | Add `test_lighter_multi_account_positions.py`: two pairs in one universe, two positions, sync updates both, NAV sums both, lookups by pair return the right position. |

### Custody flows

| Movement | Mechanism | Signer |
|----------|-----------|--------|
| Safe → master | L1 `approve` + `deposit(safe, USDC, Perps, amount)` through the guard (existing) | Asset manager hot wallet |
| master → sub | L2 `Transfer`, `transfer_same_master_account` | Executor master key (slot 4) |
| sub → master, sub → sub | L2 `Transfer` | Executor sub-account key (slot 5) |
| master → Safe | L2 secure `Withdraw` + Safe `withdrawPendingBalance` claim (existing) | Executor master key + hot wallet |
| sub → Safe (recovery only) | L1 `withdraw(subIndex, ...)` through the guard, then claim | Hot wallet only |

The normal withdrawal path to the Safe always goes through the master account:
sub → master L2 transfer first, then the existing master → Safe flow. The L1
`withdraw(subIndex)` path is implemented as a `lighter-move-funds` recovery
option only and is verified live by the manual tutorial.

### Internal transfer trade kind

- `TradeFlag.internal_account_transfer` marks a transfer between two Lighter
  accounts. It is created as a **pair** of trades sharing
  `other_data["lighter_transfer_id"]`: a negative-quantity trade on the source
  position and a positive-quantity trade on the destination position.
- No reserve allocation, no Safe transaction and no receipt. The fee reported
  by `transferFeeInfo` is recorded on the source trade as
  `lighter_transfer_fee_usdc` and realised as a negative balance update on the
  source position, so NAV and PnL stay honest.
- **Off-chain execution branch.** The current pipeline cannot carry such a
  trade: `State.start_execution()` has no branch for an exchange-account
  trade without the external-transfer flag and raises `NotImplementedError`,
  with the flag it moves reserve capital; `execute_trades()` asserts the
  external flag on every exchange-account trade before doing anything
  (`execution.py:834-837`); `_collect_signed_txs_by_chain()` and
  `broadcast()` assert signed transactions; and settlement is driven by
  receipts. The plan therefore adds:
  - `execute_trades()`: relax the line-834 assertion to accept either flag.
  - `State.start_execution()`: an `internal_account_transfer` branch that
    moves no reserve capital and only marks the trade started.
  - `_execute_trades_sequentially()`: a trade with the internal flag
    bypasses `_execute_trade_batch()` entirely. It is executed by
    `routing_model.setup_trades()` and settled immediately with
    `routing_model.settle_trade()`; no transaction collection, broadcast or
    receipt lookup. The loop `continue`s on any leg whose status is already
    `success` before `check_trade_before_execution()` and
    `start_execution()`, which is how the destination leg is skipped after
    the source leg completed both.
  - **Leg status model.** Source leg: `planned` → `started` → `broadcasted`
    when the L2 transfer is accepted by Lighter (so `is_unfinished()` and
    `check_if_clean()` already treat it as in flight) → `success` on the
    final `transfer_history` row. Destination leg: `planned` → `success`,
    completed by the source leg's settlement with `mark_success(force=True)`
    because it never broadcasts. `TradeExecution.mark_success()` asserts
    `broadcasted` unless forced (`trade.py:1677`).
  - **Execution sort keys** in `get_execution_sort_position()`. The runner
    re-sorts every cycle (`runner.py:1404-1412`), so the planner's list order
    is not what executes. Within one class later trade ids run first, so the
    keys must encode the custody order explicitly: sub → master source leg
    `-trade_id - 70_000_000` (before the existing exchange withdrawal bump of
    60,000,000), master → sub `trade_id + 20_000_000` and sub → sub
    `trade_id + 30_000_000` (after the existing deposit bump of 10,000,000).
    The destination leg takes its source leg's key plus one so the two legs
    are adjacent.
  - `State.mark_trade_success()`: the internal branch books no reserve
    adjustment, updates `share_price_state` on both positions as a capital
    flow out and in, and books the fee exactly once as a negative balance
    update on the source position.
  - The 1:1 check in `runner.py:1395-1399` applies to internal legs the same
    way as to external transfers: `planned_reserve > 0` and
    `planned_price == 1.0`, which `create_trade(quantity, reserve=None,
    assumed_price=1.0)` satisfies.
  - **One pending predicate per kind.** `is_internal_account_transfer_pending()`
    is a separate predicate; `is_external_account_transfer_pending()` is
    unchanged so `reconcile_verified_lighter_transfers()` and
    `lighter_move_funds._get_pending_transfer()` keep feeding only external
    transfers into `verify_lighter_transfer()`. `check_if_clean()` raises if
    either predicate is true for any trade.
  - Restart behaviour: a source leg with a recorded L2 tx hash but no
    completion is never re-submitted; `correct-accounts` and `repair` decide
    from `transfer_history` whether to complete or reject both legs.
- `LighterRouting.setup_trades()` quotes the fee, checks
  `amount + fee <= source free collateral`, signs and submits the transfer
  for the source trade, stores the L2 tx hash on both trades, checkpoints,
  and waits for `transfer_history` to show the row with final status.
- **Verification uses `transfer_history` as the authoritative evidence:**
  tx hash, source, destination, asset, amount, fee and final status. Equity
  observations on both accounts are recorded for reconciliation only, because
  Nautilus PnL, funding and trading fees can move either account while the
  transfer settles, so an equity-delta check would reject valid transfers.
- Execution order inside one cycle: all sub → master transfers, then master →
  Safe withdrawal, then Safe → master deposit, then master → sub transfers,
  then sub → sub transfers. The existing "withdrawals first" ordering is
  extended with this rule.
- A pending internal transfer makes the state unclean exactly like a pending
  external transfer. `start` aborts on it; `check-accounts` shows it;
  `correct-accounts` and `repair` reconcile it from `transfer_history` plus
  both accounts' public equity.

### Cash policy

One pure planner in `tradeexecutor/exchange_account/cash_manager.py`,
`LighterCashPlanner.plan(snapshot) -> list[CashMovement]`, built from the
existing `ExchangeCashManager` rules plus a sub-account allocator. It works on
projected balances in this order:

1. **Safe liquidity requirement first.** `required_safe = max(pending
   redemptions - pending deposits, 0) + safe buffer`. If the Safe is short,
   the shortfall must come out of Lighter: first from the master's free
   collateral, then from sub-accounts in descending free-collateral order,
   each retrieval sized as `amount` with `amount + fee <= free collateral`
   and `amount >= minimum`. The master → Safe withdrawal is sized from the
   master's **projected** collateral after those retrievals, so an empty
   master with funded sub-accounts still produces the withdrawal in the same
   cycle. If margin in use prevents retrieving enough, the planner returns
   what it can and a `shortfall_usdc` the strategy logs; Lagoon keeps the
   redemption deferred, as today.
2. **Deposit idle Safe cash** above `required_safe` to the master, as today.
3. **Allocate the remaining investable capital** (projected master free
   collateral plus every sub-account's free collateral, minus the master
   buffer) across sub-accounts by `target_weights`, moving only accounts
   outside `rebalance_band`, never below `minimum_transfer_usdc`, never
   margin in use, and always with the fee budgeted on top of the amount from
   the source's projected balance. Sub → sub moves are preferred over
   sub → master → sub where both legs would otherwise be needed.

Inputs: `LighterCashSnapshot(safe_usdc, pending_deposits_usdc,
pending_redemptions_usdc, master=(equity, free), sub_accounts={index:
(equity, free)}, transfer_fee_usdc, usdc_decimals)`. The planner returns the
movements in custody order for readability, but execution order is decided
by the sort keys above, not by list order. The planner is deterministic and
network free.

`decide_trades()` builds the snapshot from `PositionManager` and the Lagoon
queue, calls the planner once and turns each movement into the matching
trade kind.

### Credentials

- `LighterOperatorRecords` in `lighter_operator.py` replaces the two duplicate
  `LighterOperatorRecord` classes. It holds the master record plus
  `sub_accounts: dict[int, LighterSubAccountOperatorRecord]`, each with the
  executor key only. The Nautilus keys are never loaded by the executor.
- The deployment writes `2 + N` files below the record path: the public text
  record, the executor operator JSON (master + executor sub keys, mode 0600),
  and one `*.nautilus.<index>.json` per sub-account containing only that
  sub-account's slot-4 key (mode 0600). The hand-over file is the only
  artefact that leaves the executor host.
- Legacy single-account operator JSON stays loadable. `start` refuses to run a
  multi-account universe with a legacy record.

### Runtime configuration

| Variable | Command | Meaning |
|----------|---------|---------|
| `LIGHTER_SUB_ACCOUNT_COUNT` | `lagoon-deploy-vault` | Number of sub-accounts to create, default 0 |
| `LIGHTER_SUB_ACCOUNT_API_KEY_INDEX` | `lagoon-deploy-vault` | Nautilus slot, default 4 |
| `LIGHTER_EXECUTOR_API_KEY_INDEX` | `lagoon-deploy-vault` | Executor slot on sub-accounts, default 5 |
| `LIGHTER_ACCOUNT_INDEXES` | strategies | Comma-separated master then sub-account indices, replaces `LIGHTER_ACCOUNT_INDEX` for multi-account strategies |
| `LIGHTER_SUB_ACCOUNT_WEIGHTS` | strategies (test and tutorial only) | Comma-separated weights matching the sub-account order |
| `LIGHTER_TRANSFER_TIMEOUT` | `start`, `lighter-move-funds` | Seconds to wait for an L2 transfer, default 300 |
| `LIGHTER_OPERATOR_RECORD_FILE` | `start`, diagnostics | Unchanged, new shape |

Strategy parameters: `lighter_cash_management`, the existing buffer and
minimum parameters, plus `lighter_sub_account_weights: dict[int, Decimal]`
and `lighter_rebalance_band: Decimal`.

## Deliverables: web3-ethereum-defi

| Item | File | Notes |
|------|------|-------|
| Multi-account discovery | `eth_defi/lighter/api.py` | `fetch_lighter_accounts(session, l1_address) -> LighterAccounts(master_index, sub_account_indices)`, `wait_for_lighter_sub_accounts(session, l1_address, count, timeout)`. Keep `fetch_lighter_account_index()` as the master lookup. |
| Multi-account equity | `eth_defi/lighter/valuation.py` | `fetch_lighter_accounts_equity(session, l1_address) -> dict[int, LighterEquity]` from one response. |
| Transfer helpers | `eth_defi/lighter/transfer.py` (new) | `fetch_lighter_transfer_fee(session)`, `request_lighter_transfer(signer_client, to_account_index, amount, fee)`, `wait_for_lighter_transfer(session, account_index, tx_hash, auth_token_manager, timeout)`. SDK imported lazily. |
| Sub-account creation | `eth_defi/lighter/sub_account.py` (new) | `create_lighter_sub_accounts(signer_client, session, l1_address, count, timeout)` returns the new indices in creation order. |
| L1 withdraw helper | `eth_defi/lighter/lagoon.py` | `request_usdc_withdrawal_from_lighter_account(web3, hot_wallet, vault, account_index, amount, zk_lighter)` through `performCall`. |
| Batched key registration | `eth_defi/lighter/pubkey.py` | `build_change_pubkey_multisend(web3, entries, zk_lighter)` and `execute_change_pubkeys()` for a single-owner Safe. |
| Deployment config | `eth_defi/erc_4626/vault_protocol/lagoon/deployment.py` | `LagoonConfig.lighter_sub_account_count`, `lighter_sub_account_api_key_index`, `lighter_executor_api_key_index`. `LighterAccountSetup` gains `sub_accounts: tuple[LighterSubAccountSetup, ...]`. Activation creates the sub-accounts after the master key is live, registers 2 keys per sub-account in one Safe transaction, waits for each key, and seeds each sub-account with 1 USDC by L2 transfer so Nautilus can trade immediately. The activation deposit becomes `1 + N × (1 + transfer fee)` USDC instead of the fixed 1 USDC, debited from the 20 USDC bootstrap subscription, so the Safe retains `19 - N × (1 + fee)`; `_validate_lighter_api_key_deployment_config` rejects an `N` that does not fit the bootstrap budget. |
| Optional SDK extra | `pyproject.toml` | `lighter` extra pinning the same fork revision as trade-executor; every SDK import is lazy and guarded with a clear error. |
| Scripts | `scripts/lighter/lagoon-lighter-create-subaccounts.py`, `scripts/lighter/lagoon-lighter-change-pubkey.py` | Create sub-accounts with the master key; print a MultiSend Safe transaction for a list of keys. |
| Docs | `eth_defi/lighter/README-lighter-guard.md` | New section on sub-accounts and the key-free L1 withdraw path. |

## Deliverables: trade-executor

| Item | File | Notes |
|------|------|-------|
| Pair identity and positions | `tradeexecutor/exchange_account/lighter.py`, `tradeexecutor/state/identifier.py` | `create_lighter_exchange_account_pair(quote, account_index, role, master_account_index, strategy_label)` with the unique synthetic base address and namespaced `internal_id`; `create_lighter_exchange_account_pairs(quote, accounts)`; ticker and description per account. Validator keeps the single-protocol rule, drops the single-pair rule, requires exactly one master and unique indices. One `TradingPosition` per pair, created per pair by strategies and account correction. |
| NAV | `lighter.py`, `ethereum_protocol_adapters.py` | `create_lighter_vault_valuation_func(web3, safe_address, reserve_asset, account_indices, session)` sums one by-L1-address snapshot; fails closed if any index is missing. `_auto_discover_lighter` passes all indices. |
| Operator records | `tradeexecutor/exchange_account/lighter_operator.py` | `LighterOperatorRecords`, loader for the new and legacy shapes, one `SignerClient` per account, `request_lighter_transfer()` wrapper. Delete the duplicate in `lagoon_lighter_test_trade.py`. |
| Transfer state | `tradeexecutor/exchange_account/state.py`, `tradeexecutor/state/trade.py` | `TradeFlag.internal_account_transfer`, `create_internal_account_transfer(state, source_position, destination_position, amount, fee, ...)` returning both trades, `complete_internal_account_transfer()`, `is_internal_account_transfer_pending()`. |
| Routing | `tradeexecutor/ethereum/lighter/lighter_routing.py` | `LighterRoutingConfig.operator_records`, `transfer_timeout`. `setup_trades()` dispatches on the flag and the pair roles; refuses a Safe-facing trade on a sub-account pair; refuses a transfer without a signer for the source. |
| Verification | `tradeexecutor/ethereum/lighter/transfer_verification.py` | `verify_lighter_internal_transfer()`, extend `reconcile_verified_lighter_transfers()`. |
| Cash policy | `tradeexecutor/exchange_account/cash_manager.py` | `LighterCashSnapshot`, `LighterCashPlanner`, `CashMovement`; fee-aware projected balances; Safe liquidity requirement planned before allocation. |
| Strategy helper | `tradeexecutor/exchange_account/lighter_strategy.py` (new) | `decide_lighter_cash_management(input, master_pair, sub_pairs, parameters) -> list[TradeExecution]` wrapping both policy layers; used by the test strategy, the tutorial strategy and `lighter-ai.py`. |
| Execution pipeline | `tradeexecutor/state/state.py`, `tradeexecutor/ethereum/execution.py` | Relax the exchange-account flag assertion in `execute_trades()`; `start_execution()` internal branch with no reserve movement; sequential loop bypasses `_execute_trade_batch()` for internal legs and settles them directly, skipping legs already successful; leg status model with `mark_success(force=True)` for the destination; `mark_trade_success()` books both legs atomically, share-price capital flows and the fee once; `check_if_clean()` checks the new predicate. |
| Runner and trade | `tradeexecutor/strategy/runner.py`, `tradeexecutor/state/trade.py` | New sort keys in `get_execution_sort_position()`; `TradeFlag.internal_account_transfer`; `is_internal_account_transfer_pending()`; internal legs skip price structure and price impact like external transfers and are subject to the same 1:1 reserve check. |
| Startup | `tradeexecutor/cli/commands/start.py` | Abort on a pending internal transfer; assert signer coverage; new env vars. |
| Deployment CLI | `lagoon_deploy_vault.py`, `deploy_report.py`, `preflight_report.py` | New options; reports list the master and each sub-account with indices, key slots, public keys and tx hashes; Nautilus hand-over files. |
| Diagnostics | `lighter_move_funds.py`, `lagoon_lighter_test_trade.py`, `lagoon_redeem.py`, `check_accounts.py`, `correct_accounts.py`, `repair.py` | `--lighter-account-index`, transfer mode (`t`), L1 withdraw recovery mode, multi-account NAV in redeem, pair-identity migration in repair, internal-transfer reconciliation. |
| Strategies | `strategies/lighter-ai.py`, `strategies/test_only/lighter_multi_subaccount_strategy.py` (new) | Multi-account universe from `LIGHTER_ACCOUNT_INDEXES`; the test strategy reads weights from `LIGHTER_SUB_ACCOUNT_WEIGHTS` so tests and the tutorial can change them between cycles. |
| Test support | `tradeexecutor/testing/lighter_core.py` (new), `eth_defi/lighter/testing.py` | `FakeLighterCore`, see below, and `credit_lighter_pending_balance_on_anvil()`. Test helpers must live in the source tree. |
| Docs | `tradeexecutor/exchange_account/README-Lighter.md`, `docs/lagoon-lighter-deployment.md` | Multi-account custody model, deployment, hand-over, env vars, recovery. |
| Changelog | `CHANGELOG.md` | One line, dated. |

## Fake Lighter Core

Lighter's sequencer cannot run on an Anvil fork. The existing tests patch the
individual eth_defi read functions imported into trade-executor modules with
`mocker.patch` and keep every Ethereum transaction real. This plan keeps that
seam approach but replaces ad-hoc lambdas with one stateful object so all
seams agree with each other.

`tradeexecutor/testing/lighter_core.py`:

```python
@dataclass
class FakeLighterAccount:
    index: int
    role: str                      # "master" | "sub_account"
    collateral: Decimal
    margin_in_use: Decimal = Decimal(0)
    api_keys: dict[int, str]       # slot -> public key

class FakeLighterCore:
    """In-memory Lighter Core for one L1 owner."""
    master_index: int
    accounts: dict[int, FakeLighterAccount]
    transfer_fee_usdc: Decimal
    transfer_history: list[dict]
    withdraw_history: list[dict]
    withdrawal_delay_seconds: int = 0

    # read seams
    def fetch_accounts(self, l1_address) -> LighterAccounts
    def fetch_accounts_equity(self, l1_address) -> dict[int, LighterEquity]
    def fetch_total_equity(self, account_index) -> LighterEquity
    def fetch_transfer_fee(self) -> Decimal
    def transfer_history_rows(self, account_index) -> list[dict]
    def withdraw_history_rows(self, account_index) -> list[dict]
    def fetch_api_key(self, account_index, slot) -> str | None

    # write seams (what the SDK signer would do)
    def create_sub_account(self, master_index) -> int      # next index counting down from 2**48 - 2
    def register_api_key(self, account_index, slot, pubkey)
    def transfer(self, from_index, to_index, amount) -> str  # returns tx hash, charges fee
    def request_withdrawal(self, account_index, amount) -> str
    def observe_l1_deposit(self, amount)                      # credits master

    # L1 mirror for a withdrawal claim on the fork
    def credit_pending_balance_on_anvil(self, web3, safe_address, amount)
```

A pytest fixture `fake_lighter_core(mocker, web3_ethereum)` in
`tests/lagoon/conftest.py` patches every seam and returns the object. Two
seams do fork work because a black-box `start` or `lagoon-deploy-vault` run
cannot be interleaved by the test: the patched `wait_for_lighter_account`
calls `register_lighter_account_on_anvil()` for the Safe address it is given
(the Safe only exists after deployment, as `test_lagoon_lighter_deploy_e2e.py`
does today), and the patched `wait_for_lighter_collateral` and
`fetch_lighter_total_equity` observe a Safe → master deposit by scanning
`ZkLighter` `Deposit` events for the Safe since the last observed block and
crediting the master with the deposited amount. `observe_l1_deposit()` is
the internal method those seams call, not a test-driven step. The seams are the module-level
names each module imports, which is how every existing Lighter test mocks
today (there is no HTTP-level fake and no fake `SignerClient` in
trade-executor):

| Module | Names patched |
|--------|---------------|
| `eth_defi.erc_4626.vault_protocol.lagoon.deployment` | `create_lighter_session`, `wait_for_lighter_account`, `wait_for_lighter_collateral`, `wait_for_lighter_api_key`, new `create_lighter_sub_accounts`, `wait_for_lighter_sub_accounts`, `request_lighter_transfer` |
| `tradeexecutor.exchange_account.lighter` | `fetch_lighter_total_equity`, new `fetch_lighter_accounts_equity`, `fetch_lighter_accounts` |
| `tradeexecutor.ethereum.lighter.lighter_routing` | `fetch_lighter_total_equity`, `request_lighter_withdrawal`, `wait_for_lighter_withdrawal_claimable`, new `request_lighter_transfer`, `wait_for_lighter_transfer`, `fetch_lighter_transfer_fee` |
| `tradeexecutor.ethereum.lighter.transfer_verification` | `fetch_lighter_total_equity`, `wait_for_lighter_collateral`, new `fetch_lighter_transfer_history` |
| `tradeexecutor.cli.commands.lighter_move_funds`, `lagoon_redeem`, `correct_accounts` | The same readers and request wrappers they import |

The two duplicate withdrawal wrappers (`lighter_operator.py` and
`lagoon_lighter_test_trade.py`) are consolidated first so there is one set of
names to patch.

What stays real on the Anvil fork: Lagoon, Safe, guard and module deployment
through `deploy_automated_lagoon_vault(..., use_forge=True)`, USDC funding
and Lagoon subscription with `fund_lagoon_vault()`, Safe `approve` and
`deposit` calls through the guard, every `changePubKey` Safe transaction, the
`withdrawPendingBalance` claim calldata validated against the forked guard
with `eth_call`, state writes, NAV events, and every Typer command.

Only the master needs `register_lighter_account_on_anvil()`, because the L1
`changePubKey` and `withdraw` calls validate `msg.sender`'s master mapping
only; sub-account indices are checked by the sequencer, which the fake core
stands in for.

**Withdrawal claim on the fork.** `start` signs and broadcasts a real
`withdrawPendingBalance(safe, USDC, amount)` through the Safe module in
`LighterRouting._prepare_claim()`. The existing CLI tests patch
`claim_usdc_to_lagoon_safe_from_lighter` and transfer USDC from a whale, but
the router never calls that helper and a whale transfer does not create
Lighter's pending balance, so the broadcast claim would revert. The plan adds
`credit_lighter_pending_balance_on_anvil(web3, owner, asset_index, amount)`
to `eth_defi/lighter/testing.py` next to `register_lighter_account_on_anvil()`.
It writes `pendingAssetBalances[assetIndex][masterAccountIndex]` with
`anvil_setStorageAt`. The mapping is declared in `ExtendableStorage.sol`,
the last base of `ZkLighter`, behind a `uint256[420]` gap, so the base slot
is probed once against the forked contract and pinned as a constant the way
`LIGHTER_ADDRESS_TO_ACCOUNT_INDEX_SLOT = 19` was. The value is the packed
`PendingBalance{uint128 balanceToWithdraw; uint8 gasReserveValue}` with
`balanceToWithdraw` in Lighter ticks (`amount = baseAmount × tickSize`);
`withdrawPendingBalance` does not check the gas reserve, so the tick amount
in the low 128 bits suffices. The helper asserts
`getPendingBalance(owner, assetIndex)` reads back the amount. `FakeLighterCore.request_withdrawal()` calls it when the withdrawal
becomes claimable, so the real claim executes on the fork and the Safe's USDC
rises through the protocol. If the slot derivation proves fragile, the
fallback is a single seam on `LighterRouting._prepare_claim` that validates
the real calldata through the guard with `eth_call` and substitutes a whale
transfer, and the test then states explicitly that the claim execution is
simulated.

The test uses the session `anvil_fork_pool` fixture at
`ETHEREUM_MIDNIGHT_BLOCK` with `evm_snapshot_revert`, and carries the
`warm_rpc_test_group` and `xdist_group("fork:ethereum:midnight")` markers that
`tests/conftest.py` enforces.

Operator record shape written by the deployment and read by `start`:

```json
{
  "deployments": {
    "ethereum": {
      "vault_address": "0x...",
      "safe_address": "0x...",
      "module_address": "0x...",
      "lighter_account_setup": {
        "account_index": 748071,
        "api_key_index": 4,
        "private_key": "...",
        "sub_accounts": [
          {"account_index": 281474976710600, "executor_api_key_index": 5, "executor_private_key": "...", "nautilus_api_key_index": 4},
          {"account_index": 281474976710599, "executor_api_key_index": 5, "executor_private_key": "...", "nautilus_api_key_index": 4}
        ]
      }
    }
  }
}
```

The Nautilus private keys are not in this file. They are only in the
per-sub-account hand-over files.

## Tests

### eth-defi

- Mocked API: `fetch_lighter_accounts` with master plus two sub-accounts,
  by-L1-address equity, transfer fee, transfer history matching.
- Fork: batched `changePubKey` MultiSend from the Safe for one master and two
  sub-account indices succeeds; the guard accepts `withdraw(subIndex)` and
  the `performCall` reaches the protocol.
- FakeSignerClient: `create_lighter_sub_accounts` returns indices in order;
  `request_lighter_transfer` passes the exact fee and asset id; timeouts are
  bounded and log no secrets.
- Deployment unit test with the Lighter seams mocked: `LighterAccountSetup`
  serialises and loads two sub-accounts; the public report never contains a
  private key; the Nautilus hand-over files are mode 0600.

### trade-executor pure tests

- `LighterCashPlanner`: equal weights from an empty sub-account set; a weight
  change producing one sub → sub transfer; margin in use capping the amount;
  band and minimum suppression; deterministic ordering; an empty master with
  funded sub-accounts and a pending redemption producing sub → master
  retrievals and the master → Safe withdrawal in one plan; a non-zero fee
  with a fully allocated source never overspending (`amount + fee <= free`);
  a margin shortfall reported rather than planned.
- Transfer verification: a transfer whose `transfer_history` row is final
  completes even when both accounts' equity moved by Nautilus PnL during
  settlement.
- State: internal transfer trade pair creation, completion, pending detection,
  fee balance update, `check_if_clean` behaviour.
- Pair identity and positions: two Lighter pairs are unequal, hash
  differently and have distinct identifiers and `internal_id`s; two open
  positions coexist and every `Portfolio` pair lookup returns the right one;
  `open_exchange_account_position()` for the second pair opens a second
  position; the sync model updates both with unique balance update ids;
  `get_single_pair()` raises; the migration helper rewrites the legacy
  identity.

### trade-executor Typer black-box integration test

`tests/lagoon/test_lagoon_lighter_multi_subaccount_e2e.py`, Anvil Ethereum
fork, invoking the real app through `get_command(app).main(args=...)` with
the environment injected by `mocker.patch.dict("os.environ", ..., clear=True)`
as the existing Lighter CLI tests do. No command callback is called directly
and no state file is edited by the test.

Steps:

1. Deploy with `lagoon-deploy-vault --generate-lighter-api-key` and
   `LIGHTER_SUB_ACCOUNT_COUNT=2`. Assert the fake core has the master and two
   sub-accounts, two keys per sub-account, the operator JSON holds the master
   key and the two executor keys, the two hand-over files hold only the
   Nautilus keys, and no private key appears in stdout, logs, reports or the
   runtime artefact.
2. `init`, then `correct-accounts --process-redemption`. Assert three
   exchange-account positions, one per account, with distinct pair
   identities, tickers and `internal_id`s, each holding its own account's
   equity, and the bootstrap accounting (20 USDC subscribed, 1 USDC on the
   master plus the seeded sub-account balances). Assert `check-accounts`
   prints three distinguishable rows.
3. Subscribe 200 USDC through Lagoon and settle with `lagoon-settle`.
4. `start` with `RUN_SINGLE_CYCLE` and `LIGHTER_SUB_ACCOUNT_WEIGHTS=0.5,0.5`.
   Assert the cycle produced one real Safe → master deposit on the fork and
   two master → sub transfers in the fake core, the positions hold the
   expected quantities, the fee is booked, NAV equals Safe plus the sum of
   the three accounts, and `check-accounts` is clean.
5. `start` with weights `0.75,0.25`. Assert exactly one sub → sub transfer of
   the expected amount and no Safe transaction.
6. Mark margin in use on sub-account 1 in the fake core and run `start` with
   weights `0.25,0.75`. Assert the transfer is capped to free collateral.
7. Request a Lagoon redemption larger than the Safe balance while the master
   is empty. `start` must produce sub → master transfers followed by one
   master → Safe secure withdrawal in the same cycle; the fake core marks it
   claimable and credits the pending balance on the fork, the real Safe claim
   executes, and the next `lagoon-settle` settles the redemption. Assert NAV
   and reserve. Set a non-zero transfer fee in the fake core for this step
   and assert the fee is booked once on each source position.
8. Negative path in three invocations, with `LIGHTER_TRANSFER_TIMEOUT=5` so
   the wait fits the test budget. First, the fake core accepts a transfer but
   never finalises it; `start` submits it, checkpoints and raises
   `TimeoutError`, which the test catches with `pytest.raises` because
   `standalone_mode=False` propagates exceptions instead of exit codes.
   Second, `start` again must fail with the unclean-state error before any
   submission or accounting change, and the state file must be identical
   apart from timestamps. Third, finalise the transfer in the fake core, run
   `correct-accounts`, and assert both legs and the fee are applied exactly
   once.
9. Drain: set weights to `0,0`, request a redemption for the full vault, run
   `start` and `lagoon-settle`. The planner retrieves everything above the
   1 USDC transfer minimum from both sub-accounts and withdraws the master
   to the Safe. Assert the residuals: at most 1 USDC per sub-account plus
   fees, and the master below the 5 USDC rule.
10. `lagoon-redeem`. Assert zero deployer shares, bounded Safe dust, and that
    the redemption NAV ignores on-exchange equity below the existing 5 USDC
    threshold summed across all accounts rather than per account.

The test is marked slow and runs under `pytest -n auto` in CI with the other
Lagoon fork tests.

### Manual mainnet tutorial

`scripts/lagoon/manual-trade-executor-lighter-subaccounts.py`, following
`scripts/lagoon/manual-trade-executor-lighter.py`: in-process Typer commands
through the same `run_cli()` pattern, secrets excluded from every
representation, public assertions at every checkpoint, mainnet only, small
amounts.

Differences from the existing tutorial:

- A phase journal beside the run directory with `LIGHTER_RESUME_PHASE`, since
  two secure withdrawals of about 20 minutes each make an interrupted run
  likely. A resumed run re-verifies the completed phases from public data
  and never repeats a Lighter request or Safe claim.
- Capital movement is driven by real `start` single cycles with the test
  strategy and `LIGHTER_SUB_ACCOUNT_WEIGHTS`, so the live run exercises the
  same `decide_trades()` path as production. Direct eth_defi helpers are used
  only for the final sweep.

Phases:

1. Validate configuration, connect, record deployer USDC and ETH balances.
2. Deploy with two sub-accounts. Verify the public account indices and key
   slots through the Lighter API. Keep the hand-over files in the run
   directory; the tutorial never loads them.
3. `init`, `correct-accounts --process-redemption`, `check-accounts`.
4. Subscribe the configured USDC (default 40) with the
   `deposit-and-settle.py` flow and `lagoon-settle`.
5. `start` with weights `0.5,0.5`: Safe → master deposit, master → sub
   transfers. `lagoon-settle`, `show-valuation`, assert NAV against public
   equities.
6. `start` with weights `0.75,0.25`: one sub → sub transfer. Assert.
7. Optional Nautilus stand-in: with `LIGHTER_TUTORIAL_TRADE=true`, open and
   close a minimum ETH/USD position on sub-account 1 with its Nautilus key via
   the existing eth_defi order helpers, so the sub-account shows PnL and
   margin in use during one `start` cycle. Assert the allocator respected
   free collateral.
8. Set `lighter_safe_cash_buffer_usd` and
   `lighter_free_collateral_buffer_usd` to 0 for the remaining phases, set
   weights `0,0`, and request a Lagoon redemption for the vault value minus
   a configured 3 USDC that stays on sub-account 2 (the planner leaves it
   there because the redemption is sized to need everything else). One
   `start` cycle produces two sub → master transfers and the master → Safe
   secure withdrawal; `start` itself waits for claimability and claims, as
   it does in production. `lagoon-settle` settles the redemption.
9. Recovery demonstration on the retained balance: `lighter-move-funds
   --lighter-account-index <sub2> --l1-withdraw` requests the L1
   `withdraw(subIndex)` through the guard. Completion is checked key-free by
   polling `ZkLighter.getPendingBalance(safe, USDC)` on-chain until the
   amount appears, then the Safe claims it with `withdrawPendingBalance`.
   Assert the observed request transaction, the pending balance and the Safe
   receipt. Record the observed proof-settlement delay in the docs.
10. `lagoon-settle`, then `lagoon-redeem`: zero deployer shares, Safe dust
    bounded.
11. Refund the deployer. The tutorial deploys the vault with management and
    performance fee rates of zero, so no fee shares are minted to another
    recipient and the deployer holds the whole share supply. Before any
    sweep the script verifies that there are no pending or claimable deposit
    or redemption requests, that total share supply is zero, and reads the
    USDC balances of the Safe, the vault contract and the Silo; any
    outstanding claim is completed first. Lagoon settles redemptions by
    moving USDC into the vault contract and paying claims from there, so the
    vault balance is checked, not assumed. The deployer is the sole Safe
    owner, so the script then executes direct Safe transactions (not through
    the guard) that send any remaining USDC and any ETH held by the Safe to
    the deployer. This promotes `_sweep_safe_usdc_to_operator()` from
    eth_defi's `scripts/hyperliquid/manual-test-lagoon-account-mode-roundtrip.py`
    into a reusable `sweep_safe_to_owner(vault, owner, tokens,
    include_eth=True)` helper in `eth_defi/erc_4626/vault_protocol/lagoon/`.
    Rounding residue that Lagoon cannot return is reported, not swept. The
    script prints a reconciliation table: starting USDC and ETH, ending USDC
    and ETH, gas spent, Lighter transfer fees paid, trading PnL and fees from
    the optional phase 7, and Lighter dust left on-exchange (below the 1 USDC
    withdrawal minimum per account). The run passes only if every unexplained
    difference is zero.
12. Clear all private keys from memory and print the public artefact path.

Note on dust: Lighter enforces a 1 USDC minimum per secure withdrawal and per
transfer, and charges a transfer fee. Phase 8 drains both sub-accounts to the
master by L2 transfer and withdraws the master in one `start` cycle; phase 9
drains the retained balance on sub-account 2 through the L1 `withdraw` path.
That is two delayed withdrawals in total, which is what the journal resumes
across. Up to about 1 USDC per account can remain on Lighter and is reported,
not hidden.

### Focused verification

```shell
source .local-test.env && PYTHONPATH="$(pwd):$PYTHONPATH" poetry run pytest tests/exchange_account/test_lighter_sub_account_allocator.py --timeout=300
source .local-test.env && PYTHONPATH="$(pwd):$PYTHONPATH" poetry run pytest tests/exchange_account/test_lighter_internal_transfer_state.py --timeout=300
source .local-test.env && PYTHONPATH="$(pwd):$PYTHONPATH" poetry run pytest tests/exchange_account/test_lighter_multi_account_positions.py --timeout=300
source .local-test.env && PYTHONPATH="$(pwd):$PYTHONPATH" poetry run pytest tests/lagoon/test_lagoon_lighter_multi_subaccount_e2e.py --timeout=900
source .local-test.env && PYTHONPATH="$(pwd):$PYTHONPATH" poetry run pytest tests/lagoon/test_lagoon_lighter_deploy_e2e.py tests/lagoon/test_lagoon_lighter_cash_management.py tests/lagoon/test_lighter_move_funds.py -n auto --timeout=900
```

In eth_defi:

```shell
poetry run pytest tests/lighter/test_lighter_activation_api.py tests/lighter/test_lighter_transfer.py tests/lighter/test_lighter_sub_account.py --timeout=300
poetry run pytest tests/guard/test_guard_lighter_lagoon.py tests/lighter/test_lighter_change_pubkey.py --timeout=600
```

## Implementation sequence

1. eth-defi read helpers, transfer and sub-account helpers, batched
   `changePubKey`, optional extra, unit and fork tests. Release and bump the
   pin in trade-executor.
2. trade-executor pair identity, multi-account NAV, operator records,
   `FakeLighterCore`, migration. At this point a multi-account vault can be
   valued and Nautilus can trade it. Sub-accounts are created with the
   eth_defi script from step 1 and capital is moved with its transfer helper
   until the executor paths land in steps 4 and 5.
3. eth-defi deployment of sub-accounts and keys, hand-over files; trade-executor
   deployment CLI and reports; e2e test steps 1 to 3.
4. Internal transfer trade kind, routing, verification, reconciliation,
   allocator, strategy helper, start-up checks; e2e test steps 4 to 9.
5. Diagnostics commands, L1 withdraw recovery mode, docs, changelog.
6. Manual tutorial script; run it once live with the designated test
   deployer; record observed delays and fees back into the docs.

Each step is one pull request per repository, with the e2e test extended in
the same pull request as the behaviour it covers.

## Acceptance criteria

- `lagoon-deploy-vault` creates N sub-accounts with two keys each in one run
  and produces the hand-over files; no private key appears in any public
  output.
- A multi-account universe values NAV as Safe plus every account, verified by
  the e2e test and the live tutorial.
- `decide_trades()` with target weights produces deposits, transfers in both
  directions and withdrawals through the normal pipeline, with fees booked.
- An interrupted transfer blocks `start` and is reconciled by
  `correct-accounts` or `repair` without repeating a Lighter request.
- The e2e test passes in CI with every Safe, guard, Lagoon and USDC effect
  executed on the fork, Lighter Core faked, and the proof-backed pending
  balance supplied by the storage fixture (or, with the fallback seam, the
  claim explicitly marked as simulated).
- The live tutorial completes and refunds the deployer with every stablecoin
  and ETH not consumed by gas, Lighter fees or documented on-exchange dust.

## Open questions

1. Decided: the deployment seeds each sub-account with 1 USDC so the first
   Nautilus start is independent of the executor's cycle. Kept here for the
   record.
2. Should `lighter-ai.py` take weights as static parameters or read a file
   written by the Nautilus allocator? This plan assumes static parameters.
3. The e2e test needs the pinned `lighter-sdk` only for key generation and
   the native signer's public-key derivation; confirm the fake core can skip
   signature verification without importing the SDK in tests.
4. Should Derive's pair identity be fixed the same way in this work, so that
   the `exchange_account` kind has one identity rule for all protocols? This
   plan leaves Derive unchanged.
