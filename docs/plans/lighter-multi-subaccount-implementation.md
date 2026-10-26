# Lighter multi-subaccount vaults: implementation plan

## Goal

Support deploying and operating a Lagoon vault whose Safe owns one Lighter
master account and several Lighter sub-accounts, in both
`web3-ethereum-defi` and `trade-executor`:

- `lagoon-deploy-vault` creates the master account, N sub-accounts and one
  Nautilus API key per sub-account, and hands each key over in a protected
  file.
- `trade-executor` holds one `TradingPosition` per account, values all of
  them in NAV, deposits from the Safe to the master account, withdraws from
  any account back to the Safe, and funds sub-accounts from the master
  account, all from `decide_trades()`.
- A black-box Typer CLI integration test drives one vault with two
  sub-accounts through deposit, funding, withdrawal from a sub-account and
  rebalancing, with Lighter Core faked and everything on Ethereum executed on
  an Anvil fork.
- A manual mainnet tutorial script repeats that lifecycle with real money and
  ends by refunding the deployer with all stablecoins and ETH.

This plan implements the analysis in
[`lighter-multi-subaccount-analysis.md`](./lighter-multi-subaccount-analysis.md).
Read that document first for the protocol facts; they are not repeated here.

Interpretation: "two vaults" in the request means two Lighter sub-accounts
under one Lagoon vault, because Lighter only allows key-only transfers between
accounts of the same L1 owner. Two separate Lagoon vaults cannot transfer to
each other without an L1 signature a Safe cannot produce.

## Implementation status

Implemented on 2026-10-26 on branch `lighter-multi-subaccount` in both
repositories. The mainnet tutorial is written but has not been run live; the
four mainnet verifications below remain open until it is.

Verified by tests:

- `tests/lagoon/test_lagoon_lighter_multi_subaccount_e2e.py`: real
  `lagoon-deploy-vault` with two sub-accounts, `init`, `correct-accounts`,
  ten `start` cycles, `check-accounts`, `lighter-move-funds`, `lagoon-redeem`
  and the Safe sweep on the Ethereum fork, with only Lighter Core faked.
- `tests/exchange_account/test_lighter_multi_account_positions.py`,
  `test_lighter_sub_account_allocator.py` and `test_lighter_funding_trade.py`.
- eth-defi: `tests/lighter/test_lighter_sub_accounts.py` (including a live
  public-API read of a real owner's account tree),
  `tests/guard/test_guard_lighter_lagoon.py::test_guard_lighter_sub_account_withdraw_and_claim`
  and `tests/lagoon/test_lagoon_deployment_json.py`.

Deviations from the plan:

- The Lighter SDK is not an eth-defi extra. eth-defi keeps its existing
  decision not to depend on the SDK; `eth_defi.lighter.sdk.import_lighter_sdk()`
  imports it on use with a clear error.
- `ExchangeAccountValuator` keeps its per-position balance update ids.
  Instead, the writers that use the portfolio counter skip past the
  position's existing ids, which removes the collision without changing the
  valuator's interface.
- `lagoon-redeem` reads the accounts from `LIGHTER_ACCOUNT_INDEXES` rather
  than from state, matching how strategies are configured.
- Sub-account indices are classified by Lighter's `account_type` (0 master,
  1 sub-account, 2 and above pools), observed on mainnet; the index range is
  only a fallback.
- `check-accounts`, `correct-accounts` and `repair` accept
  `LIGHTER_OPERATOR_RECORD_FILE` to read transfer history for an interrupted
  funding transfer.
- `check-accounts` also fails when the Safe owns a Lighter account that has
  no position, because NAV would miss it.

Pre-existing bugs found and fixed:

- `create_execution_model()` never passed the Lighter routing configuration
  to `LagoonExecution`, so automatic Lighter withdrawals from `start` had no
  signer. The earlier cash-management test only exercised deposits.
- `dex_data_translation` dropped the Lighter `other_data` keys, including
  `lighter_deployment`, when translating pairs.
- `lighter-move-funds` revalued every exchange-account position with the
  moved account's equity.

## Scope and non-goals

In scope:

- Ethereum mainnet Lighter deployment only; Lighter EVM is handled by keeping
  every address and chain id behind `LighterDeployment` as the analysis
  recommends.
- Master account as the treasury hub, sub-accounts as Nautilus trading
  accounts. The executor holds only the master API key it already holds today.
- One new trade kind, master → sub-account funding, routed through the
  existing `decide_trades()` → `TradeExecution` → `GenericRouting` →
  `LighterRouting` pipeline. Every other movement reuses the existing
  deposit and withdrawal trade kinds.
- A small allocator next to the unchanged `ExchangeCashManager`; rebalancing
  converges over cycles.
- CLI, deployment report and documentation updates.
- The two test deliverables above.

Out of scope:

- Lighter order routing inside `trade-executor`. Nautilus trades.
- Guard Solidity changes. The analysis shows none is mandatory.
- Executor API keys on sub-accounts and direct sub → sub transfers. A
  sub-account is drained to the Safe with the key-free L1 `withdraw` path
  and refilled through the master. A later version can add sub-account keys
  if rebalancing latency becomes a problem.
- Whitelisting `changePubKey` for the asset manager. Key registration remains
  a Safe-owner action done at deployment.
- Creating sub-accounts after deployment; a state migration for the existing
  `lighter-ai` deployment (see below, none is needed).
- Lighter EVM deployment, Robinhood Chain, Lighter public pools.
- Dynamic weight discovery from Nautilus. Weights are strategy parameters.

## Design decisions

### One TradingPosition per Lighter account

Every Lighter account, the master and each sub-account, is one
exchange-account `TradingPairIdentifier` and one open `TradingPosition`. The
position quantity is that account's USDC equity, kept in sync by
`ExchangeAccountSyncModel` exactly as today, so NAV, PnL, account checks and
reporting work per account.

Pair identity. `TradingPairIdentifier.__eq__` and `__hash__` compare only
`base.address`, `quote.address` and `fee`, and `get_identifier()` returns
`pool_address`. Today every Lighter pair shares `LIGHTER_L1_CONTRACT` as base
and pool address with `internal_id = 1`, so two Lighter pairs compare equal
and every `p.pair == pair` lookup collides. Derive varies only `pool_address`,
which is not enough, and no test has ever put two exchange-account pairs of
one protocol in one portfolio. The rules are:

- **The master pair keeps its current identity.** Base address
  `LIGHTER_L1_CONTRACT`, `internal_id = 1`. The running `lighter-ai`
  production state therefore needs no migration.
- **Each sub-account pair gets a unique synthetic base address**:
  `0x4c49` ("LI") followed by the account index as 36 hex digits, readable in
  logs and unable to collide with a token or another account. `pool_address`
  and `exchange_address` use the same value, so identifier lookups also
  separate.
- **Sub-account `internal_id` is namespaced**: `10**15 + account_index`,
  above every Trading Strategy dataset pair id and above the `2**48` Lighter
  index range, so a mixed universe with GMX or Derive pairs (both
  `internal_id = 1`) keeps the duplicate-id assertion in
  `create_pair_universe_from_code` satisfied.
- **Ticker and description carry the account.** `LIGHTER-<index>-USDC` and
  `Lighter sub-account 281474976710600 (trend)`, so `show-positions`,
  `check-accounts` rows and logs are distinguishable.
- `other_data` gains `lighter_account_role` (`master` or `sub_account`),
  `lighter_master_account_index` and an optional `lighter_strategy_label`.
  `exchange_subaccount_id` keeps the account index for
  `get_exchange_account_id()`.
- Position creation is per pair: `open_exchange_account_position()` is called
  once for every pair whose position is missing, by strategies and by
  `create_missing_exchange_account_positions()`. The "any exchange-account
  position exists" guard in `lighter-ai.py` becomes a per-pair check.

### Changes required for one position per account

Places that already iterate every position (NAV summation,
`get_position_by_id`, `find_position_for_trade`, `check_if_clean`,
`ExchangeAccountSyncModel`, `ExchangeAccountValuator`,
`_build_exchange_account_checks`, freeze rules,
`reconcile_verified_lighter_transfers`, trade ordering,
`PositionManager.get_exchange_account_available_balance`) need no change.

| Location | Problem today | Change |
|----------|---------------|--------|
| `identifier.py` `__eq__`/`__hash__`, `get_identifier()` | All Lighter pairs equal | Solved by the sub-account base address; regression test that a master pair and a sub-account pair are unequal, hash differently and have different identifiers. |
| `portfolio.py` pair lookups (`get_position_by_trading_pair`, `get_open_position_for_pair`, `get_closed_positions_for_pair`, `get_frozen_position`, `get_equity_for_pair`, `get_all_traded_pairs`, `get_single_pair`) | Match on `p.pair == pair` or `pool_address` | Solved by identity; covered by a test with two open Lighter positions; `get_single_pair()` raises for a multi-account universe. |
| `Portfolio.create_trade` without `position=` | Second account's opening trade attaches to the first position | Solved by identity; `open_exchange_account_position()` asserts the found position's identifier equals the requested one. |
| `trading_strategy_universe.py` duplicate `internal_id` assert; `dex_data_translation.py` pair cache; `alpha_model.py` signals keyed by `internal_id` | `internal_id = 1` for every account | Namespaced sub-account `internal_id`. |
| `lighter.py` `validate_lighter_exchange_account_pairs` | Raises on more than one pair | Keep the single-protocol rule, require exactly one `master` and unique indices. Update `test_lighter_exchange_account.py`. |
| `ethereum_protocol_adapters.py` `_auto_discover_lighter`, `lighter.py` `create_lighter_vault_valuation_func` | Lagoon NAV = Safe + `pairs[0]` only | Valuation takes all account indices and sums one by-L1-address snapshot; fails closed if any account is missing. NAV-critical, first assertion in the e2e test. |
| `lighter_routing.py` `_prepare_deposit` | L1 deposit funds the master; verification waits on `trade.pair`'s account | Deposit trades are accepted on the master pair only. |
| `lighter_routing.py` `_prepare_withdrawal` | Uses the single operator record's account index, not `trade.pair` | Master pair: existing SDK secure withdrawal with the master key. Sub-account pair: L1 `withdraw(subIndex)` through the guard (below). Same trade kind, different request. |
| `account_correction.py` `create_missing_exchange_account_positions`, `_format_account_check_row` | Second pair treated as existing; identical row labels; a 1 USD reserve placeholder is deducted per auto-created position | Solved by identity; rows print index and role; Lighter positions are created with `reserve_amount=Decimal(0)` as the strategies already do, so empty sub-accounts open at zero and no reserve is moved. The sync model initialises share-price state from the first positive valuation. |
| `exchange_account/valuation.py` `balance_update_id` | Per-position `max + 1` versus the portfolio counter in the sync model | Use `portfolio.next_balance_update_id`; the valuator is given the portfolio when constructed. |
| `lighter_move_funds.py` `_get_lighter_position` | Requires exactly one position | `--lighter-account-index` selects the position; default master. |
| `lagoon_lighter_test_trade.py` | "Expected exactly one" Lighter position | Select the master position explicitly; stays a master-account test. |
| `lagoon_redeem.py` | Single `LIGHTER_ACCOUNT_INDEX` for the 5 USDC rule | Read all indices from state positions and apply the threshold to the sum. |
| `check_accounts.py` | Fails through the validator | Follows the validator change; add a check that the set of on-exchange accounts equals the set of positions. |
| `strategies/lighter-ai.py`, `lighter_cash_management_strategy.py` | `any(is_exchange_account)`, `get_single_pair()` | Per-pair position creation; shared strategy helper. |
| `tests/exchange_account/` | No test with two exchange-account pairs | `test_lighter_multi_account_positions.py`: two pairs in one universe, two positions, sync updates both, NAV sums both, lookups return the right position. |

### Custody flows

| Movement | Trade kind | Mechanism | Signer |
|----------|-----------|-----------|--------|
| Safe → master | Existing deposit (buy, external flag, master pair) | L1 `approve` + `deposit(safe, USDC, Perps, amount)` through the guard | Asset manager hot wallet |
| master → Safe | Existing withdrawal (sell, external flag, master pair) | L2 secure `Withdraw` with the master key, wait, Safe `withdrawPendingBalance` claim | Master key + hot wallet |
| sub → Safe | Existing withdrawal (sell, external flag, sub pair) | **New request**: L1 `withdraw(subIndex, USDC, Perps, amount)` through the guard, wait for the pending balance on-chain, same claim | Hot wallet only |
| master → sub | **New**: funding (buy, internal flag, sub pair) | L2 `Transfer` with the master key, `transfer_same_master_account` | Master key |

There is no sub → master and no sub → sub movement. Rebalancing from one
sub-account to another is sub → Safe in one cycle, then Safe → master and
master → sub in the next. That is about 20 minutes plus one cycle, which is
acceptable because only free collateral can move and allocation changes
between sub-strategies are infrequent.

The sub → Safe request is one Safe module transaction and needs no API key.
The pending balance is credited to the master's L1 address when Lighter's
batch is verified, so the wait polls `ZkLighter.getPendingBalance(safe,
USDC)` on-chain until the amount appears, then the existing claim runs. No
authenticated Lighter API call is involved. Only the request site in
`_prepare_withdrawal()` and the wait differ from the master withdrawal; the
trade, its evidence keys (`lighter_withdrawal_request_id` becomes the L1 tx
hash), the claim, reserve credit and reconciliation are unchanged.

### Master → sub funding trade

- `TradeFlag.internal_account_transfer` marks a buy on a sub-account position
  funded from the master account rather than from reserves. It is one trade
  with one ordinary lifecycle: `planned` → `started` → `broadcasted` when
  Lighter accepts the L2 transfer → `success` on the final `transfer_history`
  row. No forced status, no linked second trade.
- The master's decrease, amount plus fee, is booked at settlement as a
  `BalanceUpdate` on the master position with cause `vault_flow` and a note
  naming the trade, the same mechanism `ExchangeAccountSyncModel` uses for
  equity changes. The next sync then sees only PnL on the master.
- **Share-price treatment of that decrease.**
  `update_share_price_state_for_balance_update()` keeps share supply constant
  and treats the whole quantity change as performance, so booking
  `amount + fee` through it would show transferred principal as a master
  loss. `complete_exchange_account_funding()` therefore applies the principal
  as a capital outflow with a new small helper
  `apply_capital_outflow(state, amount, at)` in
  `position_internal_share_price.py` that burns shares at the current share
  price (the arithmetic of the existing sell branch of
  `update_share_price_state()`), and books only the fee through the
  balance-update path as an expense. The helper is idempotent per trade id
  so reconciliation cannot apply it twice.
- **Pipeline changes.** `execute_trades()` asserts the external flag on every
  exchange-account trade (`execution.py:834-837`); relax it to accept either
  flag. `State.start_execution()` gets an internal branch that moves no
  reserve capital. In `_execute_trades_sequentially()` an internal trade
  bypasses `_execute_trade_batch()`: `setup_trades()` submits the transfer,
  records the L2 tx hash and checkpoints, waits for `transfer_history`, and
  `settle_trade()` is called directly, with no transaction collection,
  broadcast or receipt. `mark_trade_success()` gets an internal branch that
  adjusts no reserves and updates `share_price_state` on the sub-account
  position as a capital inflow.
- **Sort key.** The runner re-sorts every cycle by
  `get_execution_sort_position()`. Withdrawals, master and sub alike, keep the
  existing exchange withdrawal key and run first. The deposit keeps its key.
  The funding trade gets `trade_id + 20_000_000`, after the deposit bump of
  10,000,000, so a cycle runs withdrawals, then the Safe → master deposit,
  then master → sub funding.
- The 1:1 check in `runner.py:1395-1399` applies to the funding trade as to
  external transfers.
- **One pending predicate.** `is_external_account_transfer_pending()` is
  generalised to cover both flags (renamed
  `is_exchange_account_transfer_pending()`, old name kept as a wrapper), so
  `check_if_clean()`, `start`, `check-accounts` and
  `reconcile_verified_lighter_transfers()` gate on it unchanged; the
  reconciliation dispatches on the flag to the right verifier. An interrupted
  funding transfer blocks `start`. `correct-accounts` and `repair` reconcile
  it from `transfer_history`: a final row completes the trade and books the
  master decrease. A missing row after the timeout leaves the trade pending
  and unverified, as `verify_lighter_transfer()` already does for incomplete
  evidence; only a definitive sequencer rejection before funds moved
  releases it, mirroring `outcome == "withdrawal_rejected"`. A timeout alone
  does not prove that no funds moved.
- Verification uses the `transfer_history` row as the authoritative evidence
  (tx hash, source, destination, amount, fee, final status). Equity
  observations are recorded for reconciliation only, because Nautilus PnL can
  move either account while the transfer settles.

### Cash policy

`ExchangeCashManager` is unchanged. It decides the Safe ↔ master movement
from the master's free collateral as today.

The policy is **phased: one kind of movement per cycle**, sized from
observed balances only. Trades are re-sorted by the runner and their sizing
inputs are the balances read at decision time, so a cycle must never plan a
movement that depends on another movement in the same cycle. The shared
helper `decide_lighter_cash_management(input, master_pair, sub_pairs,
parameters)` in `tradeexecutor/exchange_account/lighter_strategy.py` picks
the first applicable phase and returns its trades:

1. **Safe liquidity.** Call the unchanged `ExchangeCashManager.decide()` with
   `exchange_available_usdc` set to the aggregate withdrawable collateral
   (master free plus every sub-account's free collateral). If it returns a
   withdrawal, split it across accounts: master first (SDK secure
   withdrawal), then sub-accounts in descending free collateral (L1
   `withdraw`), each `>= minimum`. Return those withdrawal trades, built with
   the existing `create_exchange_account_transfer()`.
2. **Deposit idle Safe cash.** If the manager returned a deposit, return that
   one master deposit trade.
3. **Withdraw overweight sub-accounts.** With `investable = master free + sum
   of sub free` and `target = weight × investable`, for every sub-account
   above target by more than `rebalance_band`, return a withdrawal of the
   excess to the Safe. The next cycle's phase 2 deposits it to the master.
4. **Fund underweight sub-accounts** from the master's observed free
   collateral, decrementing a running budget by `amount + fee` for each,
   with `amount >= minimum`, in ascending order of current weight.

The pure arithmetic lives in one function in `cash_manager.py`,
`plan_sub_account_rebalance(master_free, sub_free, target_weights, fee,
minimum, band) -> list[tuple[int, Decimal]]` returning signed amounts per
account (negative: withdraw to Safe, positive: fund from master), used by
phases 3 and 4. No projected balances, no new snapshot or decision types
beyond that function. Convergence takes as many cycles as the Safe round trip
needs; every phase is idempotent so repeated cycles finish the job. The
master keeps `lighter_free_collateral_buffer_usd` as its own minimum.

### Credentials

- `LighterOperatorRecord` and `LIGHTER_OPERATOR_RECORD_FILE` are unchanged:
  the executor needs only the master key. The two duplicate copies of the
  record class and the withdrawal wrappers (`lighter_operator.py` and
  `lagoon_lighter_test_trade.py`) are consolidated into
  `lighter_operator.py` first.
- The deployment writes `2 + N` files below the record path: the public text
  record, the executor operator JSON (master key, mode 0600, same shape as
  today), and one `*.nautilus.<index>.json` per sub-account containing only
  that sub-account's slot-4 key (mode 0600). The hand-over files are the
  only artefacts that leave the executor host; the executor never loads them.

### Runtime configuration

| Variable | Command | Meaning |
|----------|---------|---------|
| `LIGHTER_SUB_ACCOUNT_COUNT` | `lagoon-deploy-vault` | Number of sub-accounts to create, default 0 |
| `LIGHTER_SUB_ACCOUNT_API_KEY_INDEX` | `lagoon-deploy-vault` | Nautilus key slot, default 4 |
| `LIGHTER_ACCOUNT_INDEXES` | strategies | Comma-separated master then sub-account indices from the public deployment report; replaces `LIGHTER_ACCOUNT_INDEX` for multi-account strategies |
| `LIGHTER_SUB_ACCOUNT_WEIGHTS` | strategies (test and tutorial only) | Comma-separated weights in sub-account order |
| `LIGHTER_TRANSFER_TIMEOUT` | `start`, `lighter-move-funds` | Seconds to wait for an L2 funding transfer, default 300 |
| `LIGHTER_WITHDRAWAL_TIMEOUT` | `start`, `lighter-move-funds` | Unchanged; also bounds the on-chain pending-balance wait for a sub-account withdrawal |

Strategy parameters: `lighter_cash_management` and the existing buffers and
minimum, plus `lighter_sub_account_weights: dict[int, Decimal]` and
`lighter_rebalance_band: Decimal`.

## Deliverables: web3-ethereum-defi

| Item | File | Notes |
|------|------|-------|
| Multi-account discovery | `eth_defi/lighter/api.py` | `fetch_lighter_accounts(session, l1_address) -> LighterAccounts(master_index, sub_account_indices)`, `wait_for_lighter_sub_accounts(session, l1_address, count, timeout)`. `fetch_lighter_account_index()` stays as the master lookup. |
| Multi-account equity | `eth_defi/lighter/valuation.py` | `fetch_lighter_accounts_equity(session, l1_address) -> dict[int, LighterEquity]` from one response. |
| Funding transfer helper | `eth_defi/lighter/transfer.py` (new) | `fetch_lighter_transfer_fee(session)`, `request_lighter_transfer(signer_client, to_account_index, amount, fee)`, `wait_for_lighter_transfer(session, account_index, tx_hash, auth_token_manager, timeout)`. SDK imported lazily. |
| Sub-account creation | `eth_defi/lighter/sub_account.py` (new) | `create_lighter_sub_accounts(signer_client, session, l1_address, count, timeout)` returns the new indices in creation order. |
| L1 withdraw helper | `eth_defi/lighter/lagoon.py` | `request_usdc_withdrawal_from_lighter_account(web3, hot_wallet, vault, account_index, amount, zk_lighter)` through `performCall`, and `wait_for_lighter_pending_balance(web3, zk_lighter, owner, asset_index, amount, timeout)` polling `getPendingBalance`. |
| Deployment config | `eth_defi/erc_4626/vault_protocol/lagoon/deployment.py` | `LagoonConfig.lighter_sub_account_count`, `lighter_sub_account_api_key_index`. `LighterAccountSetup` gains `sub_accounts: tuple[LighterSubAccountSetup, ...]` (index, key slot, public key, private key with `repr=False`). After the master key is live, activation creates the sub-accounts, registers one key per sub-account with sequential `execute_change_pubkey()` calls on the single-owner deployer Safe, and waits for each key. Sub-accounts are not seeded: the activation deposit stays at 1 USDC on the master and the Safe keeps 19 USDC as today, and the first `start` cycles fund the sub-accounts through the normal funding trade. Mainnet verification must confirm that `changePubKey` is accepted for an unfunded sub-account. |
| Optional SDK extra | `pyproject.toml` | `lighter` extra pinning the same fork revision as trade-executor; every SDK import is lazy with a clear error. |
| Scripts | `scripts/lighter/lagoon-lighter-create-subaccounts.py` | Create sub-accounts with the master key and print their indices; the existing `lagoon-lighter-change-pubkey.py` already takes any account index. |
| Docs | `eth_defi/lighter/README-lighter-guard.md` | New section on sub-accounts and the key-free L1 withdraw path. |

## Deliverables: trade-executor

| Item | File | Notes |
|------|------|-------|
| Pair identity and positions | `tradeexecutor/exchange_account/lighter.py`, `tradeexecutor/state/identifier.py` | `create_lighter_exchange_account_pair(quote, account_index, role, master_account_index, strategy_label)`; `create_lighter_exchange_account_pairs(quote, master_index, sub_indices)`; ticker and description per account; validator changes. |
| NAV | `lighter.py`, `ethereum_protocol_adapters.py` | `create_lighter_vault_valuation_func(web3, safe_address, reserve_asset, account_indices, session)`; `_auto_discover_lighter` passes all indices. |
| Operator record | `tradeexecutor/exchange_account/lighter_operator.py` | Consolidate the duplicate class and wrappers; no shape change. |
| Funding trade | `tradeexecutor/exchange_account/state.py`, `tradeexecutor/state/trade.py` | `TradeFlag.internal_account_transfer`, `create_exchange_account_funding(state, master_position, sub_position, amount, fee, ...)`, `complete_exchange_account_funding()` booking the master balance update, `is_internal_account_transfer_pending()`, sort key, `transfer_rejected` outcome. |
| Execution pipeline | `tradeexecutor/state/state.py`, `tradeexecutor/ethereum/execution.py`, `tradeexecutor/strategy/position_internal_share_price.py` | Relaxed flag assertion; `start_execution()` and `mark_trade_success()` internal branches; sequential loop bypass of the transaction batch for internal trades; `apply_capital_outflow()` for the master principal; the generalised pending predicate. |
| Routing | `tradeexecutor/ethereum/lighter/lighter_routing.py` | Deposits on the master pair only; `_prepare_withdrawal()` dispatches on the pair role (SDK request for master, L1 request plus on-chain wait for sub-accounts); `_prepare_funding()` for the new trade. |
| Verification | `tradeexecutor/ethereum/lighter/transfer_verification.py` | `verify_lighter_funding()`, extend `reconcile_verified_lighter_transfers()` for both kinds. |
| Cash policy | `tradeexecutor/exchange_account/cash_manager.py` | `plan_sub_account_rebalance()`; `ExchangeCashManager` untouched. |
| Strategy helper | `tradeexecutor/exchange_account/lighter_strategy.py` (new) | `decide_lighter_cash_management()` with the four phases. |
| Startup | `tradeexecutor/cli/commands/start.py` | Abort on a pending funding transfer; `LIGHTER_TRANSFER_TIMEOUT`. |
| Deployment CLI | `lagoon_deploy_vault.py`, `deploy_report.py`, `preflight_report.py` | New options; reports list the master and each sub-account with index, key slot, public key and tx hashes; Nautilus hand-over files. |
| Diagnostics | `lighter_move_funds.py`, `lagoon_lighter_test_trade.py`, `lagoon_redeem.py`, `check_accounts.py`, `correct_accounts.py`, `repair.py` | `--lighter-account-index` on `lighter-move-funds` with the L1 withdraw for sub-accounts and `_resume_withdrawal()` extended to resume an interrupted L1 withdrawal from the trade's recorded request hash and pending balance; master selection in the test trade; multi-account NAV in redeem; funding reconciliation; account-set check. This is the shared recovery path the tutorial relies on. |
| Strategies | `strategies/lighter-ai.py`, `strategies/test_only/lighter_multi_subaccount_strategy.py` (new) | Multi-account universe from `LIGHTER_ACCOUNT_INDEXES`; the test strategy reads weights from `LIGHTER_SUB_ACCOUNT_WEIGHTS` so tests and the tutorial can change them between cycles. |
| Test support | `tradeexecutor/testing/lighter_core.py` (new), `eth_defi/lighter/testing.py` | `FakeLighterCore` and `credit_lighter_pending_balance_on_anvil()`. Test helpers live in the source tree. |
| Docs | `tradeexecutor/exchange_account/README-Lighter.md`, `docs/lagoon-lighter-deployment.md` | Multi-account custody model, deployment, hand-over, env vars, recovery. |
| Changelog | `CHANGELOG.md` | One line, dated. |

## Fake Lighter Core

Lighter's sequencer cannot run on an Anvil fork. The existing tests patch the
eth_defi functions imported into trade-executor modules with `mocker.patch`
and keep every Ethereum transaction real. This plan keeps that seam approach
but replaces ad-hoc lambdas with one stateful object so all seams agree.

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
    finalise_transfers: bool = True   # False leaves a funding transfer pending

    # read seams
    def fetch_accounts(self, l1_address) -> LighterAccounts
    def fetch_accounts_equity(self, l1_address) -> dict[int, LighterEquity]
    def fetch_total_equity(self, account_index) -> LighterEquity
    def fetch_transfer_fee(self) -> Decimal
    def transfer_history_rows(self, account_index) -> list[dict]
    def withdraw_history_rows(self, account_index) -> list[dict]
    def fetch_api_key(self, account_index, slot) -> str | None

    # write seams (what the SDK signer or the sequencer would do)
    def create_sub_account(self, master_index) -> int      # counts down from 2**48 - 2
    def register_api_key(self, account_index, slot, pubkey)
    def transfer(self, from_index, to_index, amount) -> str  # charges fee
    def request_secure_withdrawal(self, account_index, amount) -> str
    def credit_master_from_deposit(self, amount)
    def settle_l1_withdrawal(self, web3, safe_address, account_index, amount)
```

A pytest fixture `fake_lighter_core(mocker, web3_ethereum)` in
`tests/lagoon/conftest.py` patches every seam and returns the object. The
seams are the module-level names each module imports:

| Module | Names patched |
|--------|---------------|
| `eth_defi.erc_4626.vault_protocol.lagoon.deployment` | `create_lighter_session`, `wait_for_lighter_account`, `wait_for_lighter_collateral`, `wait_for_lighter_api_key`, new `create_lighter_sub_accounts`, `wait_for_lighter_sub_accounts`, `request_lighter_transfer` |
| `tradeexecutor.exchange_account.lighter` | `fetch_lighter_total_equity`, new `fetch_lighter_accounts_equity` |
| `tradeexecutor.ethereum.lighter.lighter_routing` | `fetch_lighter_total_equity`, `request_lighter_withdrawal`, `wait_for_lighter_withdrawal_claimable`, new `request_lighter_transfer`, `wait_for_lighter_transfer`, `fetch_lighter_transfer_fee`, `wait_for_lighter_pending_balance` |
| `tradeexecutor.ethereum.lighter.transfer_verification` | `fetch_lighter_total_equity`, `wait_for_lighter_collateral`, new `fetch_lighter_transfer_history` |
| `tradeexecutor.cli.commands.lighter_move_funds`, `lagoon_redeem`, `correct_accounts` | The same readers and request wrappers they import |

Two seams do fork work, because a black-box `start` or `lagoon-deploy-vault`
run cannot be interleaved by the test:

- The patched `wait_for_lighter_account` calls
  `register_lighter_account_on_anvil()` for the Safe address it is given, as
  `test_lagoon_lighter_deploy_e2e.py` does today. Only the master needs it:
  the L1 `changePubKey` and `withdraw` calls validate `msg.sender`'s master
  mapping only, and the sequencer's sub-account ownership check is what the
  fake stands in for.
- The patched `wait_for_lighter_collateral` credits the master with the
  expected deposit, as `test_lagoon_lighter_cash_management.py` does today.

**Withdrawal claim on the fork.** `start` signs and broadcasts a real
`withdrawPendingBalance(safe, USDC, amount)` through the Safe module. A whale
transfer to the Safe does not create Lighter's pending balance, so the claim
would revert. `credit_lighter_pending_balance_on_anvil(web3, owner,
asset_index, amount)` in `eth_defi/lighter/testing.py` writes
`pendingAssetBalances[assetIndex][masterAccountIndex]` with
`anvil_setStorageAt`. The mapping is declared in `ExtendableStorage.sol`, the
last base of `ZkLighter`, behind a `uint256[420]` gap, so the base slot is
probed once against the forked contract and pinned as a constant the way
`LIGHTER_ADDRESS_TO_ACCOUNT_INDEX_SLOT = 19` was. The value is the packed
`PendingBalance{uint128 balanceToWithdraw; uint8 gasReserveValue}` with
`balanceToWithdraw` in Lighter ticks; `withdrawPendingBalance` does not check
the gas reserve. The helper asserts `getPendingBalance(owner, assetIndex)`
reads back the amount. The fake calls it from `settle_l1_withdrawal()` for
sub-account withdrawals and from the patched
`wait_for_lighter_withdrawal_claimable` for master withdrawals, so the real
claim executes on the fork. If the slot derivation proves fragile, the
fallback is one seam on `LighterRouting._prepare_claim` that validates the
real calldata through the guard with `eth_call` and substitutes a whale
transfer, with the test stating that the claim execution is simulated.

The test uses the session `anvil_fork_pool` fixture at
`ETHEREUM_MIDNIGHT_BLOCK` with `evm_snapshot_revert`, and carries the
`warm_rpc_test_group` and `xdist_group("fork:ethereum:midnight")` markers that
`tests/conftest.py` enforces.

## Tests

### eth-defi

- Mocked API: `fetch_lighter_accounts` with master plus two sub-accounts,
  by-L1-address equity, transfer fee, transfer history matching.
- Fork: the guard accepts `withdraw(subIndex)` through `performCall` and the
  protocol accepts it for a registered master; `wait_for_lighter_pending_balance`
  returns once the storage fixture credits the balance; `changePubKey` from
  the Safe for a sub-account index.
- FakeSignerClient: `create_lighter_sub_accounts` returns indices in order;
  `request_lighter_transfer` passes the exact fee and asset id; timeouts are
  bounded and log no secrets.
- Deployment unit test with the Lighter seams mocked: `LighterAccountSetup`
  serialises and loads two sub-accounts; the public report never contains a
  private key; the Nautilus hand-over files are mode 0600.

### trade-executor pure tests

- `plan_sub_account_rebalance()` and the phased helper: equal weights from
  empty sub-accounts; a weight change producing one withdrawal in one cycle
  and, with the Safe refilled, one funding trade in a later cycle; a Safe
  shortfall split master first then sub-accounts in descending free
  collateral and nothing else that cycle; margin in use capping; band and
  minimum suppression; the funding budget never exceeding observed master
  free collateral with fees; idempotence on a converged snapshot.
- Funding trade state: creation, the `broadcasted` transition, completion
  with the master balance update split into principal outflow and fee
  expense, idempotent re-application, pending detection through the shared
  predicate, `check_if_clean`, a timed-out transfer staying pending.
- Verification: a funding transfer whose `transfer_history` row is final
  completes even when both accounts' equity moved by Nautilus PnL.
- Pair identity and positions: a master pair and a sub-account pair are
  unequal, hash differently and have distinct identifiers and `internal_id`s;
  two open positions coexist and every `Portfolio` pair lookup returns the
  right one; `open_exchange_account_position()` for the second pair opens a
  second position; the sync model updates both with unique balance update
  ids; `get_single_pair()` raises.

### trade-executor Typer black-box integration test

`tests/lagoon/test_lagoon_lighter_multi_subaccount_e2e.py`, Anvil Ethereum
fork, invoking the real app through `get_command(app).main(args=...)` with
the environment injected by `mocker.patch.dict("os.environ", ..., clear=True)`
as the existing Lighter CLI tests do. No command callback is called directly
and no state file is edited by the test. `LIGHTER_WITHDRAWAL_TIMEOUT` and
`LIGHTER_TRANSFER_TIMEOUT` are set to a few seconds.

1. Deploy with `lagoon-deploy-vault --generate-lighter-api-key` and
   `LIGHTER_SUB_ACCOUNT_COUNT=2`. Assert the fake core has the master and two
   sub-accounts with one key each, the operator JSON holds only the master
   key, the two hand-over files hold only the Nautilus keys, and no private
   key appears in stdout, logs, reports or the runtime artefact.
2. `init`, then `correct-accounts --process-redemption`. Assert three
   exchange-account positions with distinct identities, tickers and
   `internal_id`s, the master holding 1 USDC and both sub-accounts opened
   at zero with no reserve moved, the bootstrap accounting (20 USDC
   subscribed, 19 USDC in the Safe), and three distinguishable
   `check-accounts` rows.
3. Subscribe 200 USDC through Lagoon and settle with `lagoon-settle`.
4. `start` with `RUN_SINGLE_CYCLE` and `LIGHTER_SUB_ACCOUNT_WEIGHTS=0.5,0.5`.
   Assert one real Safe → master deposit on the fork and nothing else in
   that cycle. `start` again: assert two funding trades in the fake core,
   the fee booked once per funding trade on the master as an expense and
   the principal as a capital outflow, NAV equal to Safe plus the three
   accounts, and `check-accounts` clean.
5. `start` with weights `0.75,0.25`. Assert one sub-account 2 → Safe
   withdrawal: a real `withdraw(subIndex)` through the guard on the fork,
   the pending balance credited by the fixture, the real claim executed, the
   reserve credited. `start` again: the Safe → master deposit. `start`
   again: the master → sub-account 1 funding trade. Assert the split.
6. Mark margin in use on sub-account 1 in the fake core and run `start` with
   weights `0.25,0.75`. Assert the withdrawal is capped to free collateral.
7. Request a Lagoon redemption larger than the Safe balance while the master
   is empty. `start` must produce sub-account withdrawals in descending
   free-collateral order and nothing else; `lagoon-settle` then settles the
   redemption. Assert NAV and reserve.
8. Negative path in three invocations. First, set `finalise_transfers=False`
   and run `start` with weights that need a funding trade; it submits the
   transfer, checkpoints and raises `TimeoutError`, caught with
   `pytest.raises` because `standalone_mode=False` propagates exceptions.
   Second, `start` again must fail with the unclean-state error before any
   submission or accounting change, and the state file must be identical
   apart from timestamps. Third, finalise in the fake core, run
   `correct-accounts`, and assert the trade and the master balance update are
   applied exactly once.
9. Drain: weights `0,0`, request a redemption for the full vault, `start`
   and `lagoon-settle`. Assert residuals of at most 1 USDC per sub-account
   and the master below the 5 USDC rule.
10. `lagoon-redeem`. Assert zero deployer shares, bounded Safe dust, and the
    5 USDC rule applied to the sum of all accounts.

The test is marked slow and runs under `pytest -n auto` in CI with the other
Lagoon fork tests.

### Manual mainnet tutorial

`scripts/lagoon/manual-trade-executor-lighter-subaccounts.py`, following
`scripts/lagoon/manual-trade-executor-lighter.py`: in-process Typer commands
through the same `run_cli()` pattern, secrets excluded from every
representation, public assertions at every checkpoint, mainnet only, small
amounts. The vault is deployed with management and performance fee rates of
zero, so the deployer holds the whole share supply.

Differences from the existing tutorial:

- A small progress manifest beside the run directory holding the deployment
  addresses, tutorial parameters and the last completed phase, so a rerun
  never redeploys or repeats a subscription. Custody recovery is not
  tutorial logic: trade records in executor state are authoritative for
  requests, hashes and claims, an interrupted `start` is recovered with the
  production path (`check-accounts`, then `correct-accounts` or
  `lighter-move-funds` resume), and the tutorial then continues from the
  manifest's phase.
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
5. `start` with weights `0.5,0.5`: Safe → master deposit. `start` again:
   two funding transfers. `lagoon-settle`, `show-valuation`, assert NAV
   against public equities.
6. `start` with weights `0.75,0.25`: sub-account 2 → Safe through the L1
   `withdraw` path, proving it live: assert the request transaction, the
   on-chain pending balance, and the Safe receipt; record the observed
   proof-settlement delay. `start` again: Safe → master deposit. `start`
   again: master → sub-account 1 funding. Assert the split.
7. Set both buffers to 0, weights `0,0`, and request a full Lagoon
   redemption. `start` produces the master → Safe secure withdrawal and the
   two sub-account → Safe L1 withdrawals; `start` waits and claims each, as
   in production. `lagoon-settle` settles the redemption.
8. `lagoon-redeem`: zero deployer shares, Safe dust bounded.
9. Refund the deployer. Before any sweep the script verifies there are no
   pending or claimable deposit or redemption requests, that total share
   supply is zero, and reads the USDC balances of the Safe, the vault
   contract and the Silo; any outstanding claim is completed first. Lagoon
   settles redemptions by moving USDC into the vault contract and paying
   claims from there, so the vault balance is checked, not assumed. The
   deployer is the sole Safe owner, so the script then executes direct Safe
   transactions (not through the guard) that send any remaining USDC and any
   ETH held by the Safe to the deployer, using
   `sweep_safe_to_owner(vault, owner, tokens, include_eth=True)` promoted
   from `_sweep_safe_usdc_to_operator()` in eth_defi's
   `scripts/hyperliquid/manual-test-lagoon-account-mode-roundtrip.py`.
   Rounding residue that Lagoon cannot return is reported, not swept. The
   script prints a reconciliation table: starting USDC and ETH, ending USDC
   and ETH, gas spent, Lighter transfer fees paid, and Lighter dust left
   on-exchange (below the 1 USDC withdrawal minimum per account). The run
   passes only if every unexplained difference is zero.
10. Clear the master private key from memory and print the public artefact
    path.

Note on dust: Lighter enforces a 1 USDC minimum per secure withdrawal and per
transfer, and charges a transfer fee. Up to about 1 USDC per account can
remain on Lighter and is reported, not hidden.

### Focused verification

```shell
source .local-test.env && PYTHONPATH="$(pwd):$PYTHONPATH" poetry run pytest tests/exchange_account/test_lighter_sub_account_allocator.py --timeout=300
source .local-test.env && PYTHONPATH="$(pwd):$PYTHONPATH" poetry run pytest tests/exchange_account/test_lighter_funding_trade_state.py --timeout=300
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

1. eth-defi read helpers, transfer and sub-account helpers, L1 withdraw and
   pending-balance helpers, storage fixture, optional extra, unit and fork
   tests. Release and bump the pin in trade-executor.
2. trade-executor pair identity, multi-account NAV, consolidated operator
   record, `FakeLighterCore`. At this point a multi-account vault can be
   valued and Nautilus can trade it; sub-accounts are created with the
   eth_defi script from step 1 and funded with its transfer helper.
3. eth-defi deployment of sub-accounts and keys, hand-over files;
   trade-executor deployment CLI and reports; e2e test steps 1 to 3.
4. Sub-account withdrawal request in routing, funding trade kind, pipeline
   branches, verification, reconciliation, phased cash helper, start-up
   checks, `lighter-move-funds` resume for the L1 path; e2e test steps 4 to
   10.
5. Diagnostics commands, docs, changelog.
6. Manual tutorial script; run it once live with the designated test
   deployer; record observed delays and fees back into the docs.

Each step is one pull request per repository, with the e2e test extended in
the same pull request as the behaviour it covers.

Mainnet verification needed before step 4 is finalised: one real
`CreateSubAccount` from a Safe-owned master, one `changePubKey(subIndex)` from
the Safe on a still unfunded sub-account, one master → sub L2 transfer, and
one L1 `withdraw(subIndex)` through the guard followed by the claim. The
tutorial covers all four.

## Acceptance criteria

- `lagoon-deploy-vault` creates N sub-accounts with one Nautilus key each in
  one run and produces the hand-over files; no private key appears in any
  public output; the executor's operator record is unchanged in shape.
- A multi-account universe holds one position per account and values NAV as
  Safe plus every account, verified by the e2e test and the live tutorial.
- `decide_trades()` with target weights produces deposits, funding transfers
  and withdrawals from any account through the normal pipeline, converging
  over cycles, with fees booked.
- An interrupted funding transfer or sub-account withdrawal blocks `start`
  and is reconciled or resumed by `correct-accounts`, `repair` or
  `lighter-move-funds` without repeating a Lighter request or Safe claim.
- The e2e test passes in CI with every Safe, guard, Lagoon and USDC effect
  executed on the fork, Lighter Core faked, and the proof-backed pending
  balance supplied by the storage fixture (or, with the fallback seam, the
  claim explicitly marked as simulated).
- The live tutorial completes and refunds the deployer with every stablecoin
  and ETH not consumed by gas, Lighter fees or documented on-exchange dust.

## Open questions

1. Should `lighter-ai.py` take weights as static parameters or read a file
   written by the Nautilus allocator? This plan assumes static parameters.
2. Is the sub → Safe → master → sub round trip, about 20 minutes plus one
   cycle, acceptable for production rebalancing? If not, the follow-up is an
   executor key per sub-account and a direct sub → master transfer; the
   funding trade and allocator designs extend to it without rework.
3. Should Derive's pair identity get the same base-address rule, so the
   `exchange_account` kind has one identity rule for all protocols? This plan
   leaves Derive unchanged.
