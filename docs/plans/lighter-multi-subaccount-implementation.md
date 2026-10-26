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

### Account roles and pair identity

- Every Lighter account becomes one exchange-account pair. The account index
  is encoded into `pool_address` and `exchange_address` as
  `0x` + 40 hex digits derived from the index, and `internal_id` is the
  account index, the same approach `derive.py` uses for sub-account ids.
- `other_data` gains `lighter_account_role` (`master` or `sub_account`),
  `lighter_master_account_index` and an optional `lighter_strategy_label` for
  the Nautilus strategy name. `exchange_subaccount_id` keeps the account index
  for compatibility with `get_exchange_account_id()`.
- The existing single-account identity (`LIGHTER_L1_CONTRACT` as pool address,
  `internal_id = 1`) is retained only through a one-off state migration in
  `repair --migrate-lighter-pair-identity` for the current `lighter-ai`
  deployment. New deployments never produce it.

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
- `LighterRouting.setup_trades()` signs and submits the transfer for the
  source trade, stores the L2 tx hash on both trades, checkpoints, and waits
  for `transfer_history` to show the row. `settle_trade()` verifies source and
  destination equity deltas with the same tolerance rules as deposits.
- Execution order inside one cycle: all sub → master transfers, then master →
  Safe withdrawal, then Safe → master deposit, then master → sub transfers,
  then sub → sub transfers. The existing "withdrawals first" ordering is
  extended with this rule.
- A pending internal transfer makes the state unclean exactly like a pending
  external transfer. `start` aborts on it; `check-accounts` shows it;
  `correct-accounts` and `repair` reconcile it from `transfer_history` plus
  both accounts' public equity.

### Cash policy

Two pure layers in `tradeexecutor/exchange_account/cash_manager.py`:

1. `ExchangeCashManager` is unchanged in behaviour and decides the Safe ↔
   master movement from the master account's free collateral.
2. `SubAccountAllocator.decide(snapshot)` takes
   `SubAccountSnapshot(master_free_usdc, accounts={index: (equity,
   free_collateral)}, target_weights, minimum_transfer_usdc,
   rebalance_band)` and returns an ordered list of
   `InternalTransferDecision(from_index, to_index, amount_usdc)`. It never
   moves margin in use, never moves below the minimum, and skips accounts
   within the band of their target.

`decide_trades()` runs layer 1, then layer 2 on the balances expected after
layer 1, and returns all resulting trades. The runner executes them in the
order above.

### Credentials

- `LighterOperatorRecords` in `lighter_operator.py` replaces the two duplicate
  `LighterOperatorRecord` classes. It holds the master record plus
  `sub_accounts: dict[int, LighterSubAccountOperatorRecord]`, each with the
  executor key only. The Nautilus keys are never loaded by the executor.
- The deployment writes three files below the record path: the public text
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
| `LIGHTER_TRANSFER_TIMEOUT` | `start` | Seconds to wait for an L2 transfer, default 300 |
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
| Deployment config | `eth_defi/erc_4626/vault_protocol/lagoon/deployment.py` | `LagoonConfig.lighter_sub_account_count`, `lighter_sub_account_api_key_index`, `lighter_executor_api_key_index`. `LighterAccountSetup` gains `sub_accounts: tuple[LighterSubAccountSetup, ...]`. Activation creates the sub-accounts after the master key is live, registers 2 keys per sub-account in one Safe transaction, waits for each key, and seeds each sub-account with 1 USDC by L2 transfer so Nautilus can trade immediately. |
| Optional SDK extra | `pyproject.toml` | `lighter` extra pinning the same fork revision as trade-executor; every SDK import is lazy and guarded with a clear error. |
| Scripts | `scripts/lighter/lagoon-lighter-create-subaccounts.py`, `scripts/lighter/lagoon-lighter-change-pubkey.py` | Create sub-accounts with the master key; print a MultiSend Safe transaction for a list of keys. |
| Docs | `eth_defi/lighter/README-lighter-guard.md` | New section on sub-accounts and the key-free L1 withdraw path. |

## Deliverables: trade-executor

| Item | File | Notes |
|------|------|-------|
| Pair identity | `tradeexecutor/exchange_account/lighter.py` | `create_lighter_exchange_account_pair(quote, account_index, role, master_account_index, strategy_label)`, `create_lighter_exchange_account_pairs(quote, accounts)`. Validator keeps the single-protocol rule, drops the single-pair rule, and requires exactly one master. |
| NAV | `lighter.py`, `ethereum_protocol_adapters.py` | `create_lighter_vault_valuation_func(web3, safe_address, reserve_asset, account_indices, session)` sums one by-L1-address snapshot; fails closed if any index is missing. `_auto_discover_lighter` passes all indices. |
| Operator records | `tradeexecutor/exchange_account/lighter_operator.py` | `LighterOperatorRecords`, loader for the new and legacy shapes, one `SignerClient` per account, `request_lighter_transfer()` wrapper. Delete the duplicate in `lagoon_lighter_test_trade.py`. |
| Transfer state | `tradeexecutor/exchange_account/state.py`, `tradeexecutor/state/trade.py` | `TradeFlag.internal_account_transfer`, `create_internal_account_transfer(state, source_position, destination_position, amount, fee, ...)` returning both trades, `complete_internal_account_transfer()`, `is_internal_account_transfer_pending()`. |
| Routing | `tradeexecutor/ethereum/lighter/lighter_routing.py` | `LighterRoutingConfig.operator_records`, `transfer_timeout`. `setup_trades()` dispatches on the flag and the pair roles; refuses a Safe-facing trade on a sub-account pair; refuses a transfer without a signer for the source. |
| Verification | `tradeexecutor/ethereum/lighter/transfer_verification.py` | `verify_lighter_internal_transfer()`, extend `reconcile_verified_lighter_transfers()`. |
| Cash policy | `tradeexecutor/exchange_account/cash_manager.py` | `SubAccountSnapshot`, `SubAccountAllocator`, `InternalTransferDecision`. |
| Strategy helper | `tradeexecutor/exchange_account/lighter_strategy.py` (new) | `decide_lighter_cash_management(input, master_pair, sub_pairs, parameters) -> list[TradeExecution]` wrapping both policy layers; used by the test strategy, the tutorial strategy and `lighter-ai.py`. |
| Runner | `tradeexecutor/strategy/runner.py`, `tradeexecutor/state/trade.py` | Execution ordering rule; internal transfers skip price structure, price impact and the 1:1 reserve assertion the same way external transfers do. |
| Startup | `tradeexecutor/cli/commands/start.py` | Abort on a pending internal transfer; assert signer coverage; new env vars. |
| Deployment CLI | `lagoon_deploy_vault.py`, `deploy_report.py`, `preflight_report.py` | New options; reports list the master and each sub-account with indices, key slots, public keys and tx hashes; Nautilus hand-over files. |
| Diagnostics | `lighter_move_funds.py`, `lagoon_lighter_test_trade.py`, `lagoon_redeem.py`, `check_accounts.py`, `correct_accounts.py`, `repair.py` | `--lighter-account-index`, transfer mode (`t`), L1 withdraw recovery mode, multi-account NAV in redeem, pair-identity migration in repair, internal-transfer reconciliation. |
| Strategies | `strategies/lighter-ai.py`, `strategies/test_only/lighter_multi_subaccount_strategy.py` (new) | Multi-account universe from `LIGHTER_ACCOUNT_INDEXES`; the test strategy reads weights from `LIGHTER_SUB_ACCOUNT_WEIGHTS` so tests and the tutorial can change them between cycles. |
| Test support | `tradeexecutor/testing/lighter_core.py` (new) | `FakeLighterCore`, see below. Test helpers must live in the source tree. |
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

A pytest fixture `fake_lighter_core(mocker)` in `tests/lagoon/conftest.py`
patches every seam and returns the object. The seams are the module-level
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
stands in for. The claim's L1 effect is mirrored by a USDC whale transfer to
the Safe, exactly as `test_lighter_move_funds.py` and
`test_lagoon_lighter_test_trade.py` do, because a static fork cannot produce a
proof-backed pending balance. The test uses the session `anvil_fork_pool`
fixture at `ETHEREUM_MIDNIGHT_BLOCK` with `evm_snapshot_revert`, and carries
the `warm_rpc_test_group` and `xdist_group("fork:ethereum:midnight")` markers
that `tests/conftest.py` enforces.

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

- `SubAccountAllocator`: equal weights from an empty sub-account set; a weight
  change producing one sub → sub transfer; margin in use capping the amount;
  band and minimum suppression; deterministic ordering.
- State: internal transfer trade pair creation, completion, pending detection,
  fee balance update, `check_if_clean` behaviour.
- Pair identity: distinct identities per account index, migration helper.

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
   exchange-account positions with distinct pair identities and the
   bootstrap accounting (20 USDC subscribed, 1 USDC on the master plus the
   seeded sub-account balances).
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
7. Request a Lagoon redemption larger than the Safe balance. `start` must
   produce sub → master transfers followed by one master → Safe secure
   withdrawal; the fake core marks it claimable, the Safe claim calldata is
   validated against the guard, the L1 effect is mirrored, and the next
   `lagoon-settle` settles the redemption. Assert NAV and reserve.
8. Negative path: make the fake core leave a transfer pending and run
   `start`; assert a non-zero exit, no new trades and no state mutation. Then
   complete it in the fake core, run `correct-accounts`, and assert the trade
   is reconciled.
9. `lagoon-redeem` with all accounts near empty. Assert zero deployer shares,
   bounded Safe dust, and that the redemption NAV ignores on-exchange equity
   below the existing 5 USDC threshold summed across all accounts rather than
   per account.

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
8. Set weights `0,0` and request a full Lagoon redemption. `start` produces
   sub → master transfers and the master → Safe secure withdrawal; wait for
   claimability with the auth-token manager; the claim lands in the Safe.
   `lagoon-settle` settles the redemption.
9. Recovery demonstration: `lighter-move-funds --account-index <sub1>
   --l1-withdraw` for any residue above 1 USDC left on a sub-account, proving
   the key-free guard path live.
10. `lagoon-redeem`: zero deployer shares, Safe dust bounded.
11. Refund the deployer. The deployer is the sole Safe owner, so the script
    executes direct Safe transactions (not through the guard) that send any
    remaining USDC and any ETH held by the Safe to the deployer. This
    promotes `_sweep_safe_usdc_to_operator()` from eth_defi's
    `scripts/hyperliquid/manual-test-lagoon-account-mode-roundtrip.py` into a
    reusable `sweep_safe_to_owner(vault, owner, tokens, include_eth=True)`
    helper in `eth_defi/erc_4626/vault_protocol/lagoon/`. The Lagoon vault
    contract itself never holds assets after full redemption, so no vault
    sweep is needed; the script asserts its USDC and ETH balances are zero.
    It then prints a reconciliation table: starting USDC and ETH, ending USDC
    and ETH, gas spent, Lighter transfer fees paid, and Lighter dust left
    on-exchange (below the 1 USDC withdrawal minimum per account). The run
    passes only if every unexplained difference is zero.
12. Clear all private keys from memory and print the public artefact path.

Note on dust: Lighter enforces a 1 USDC minimum per secure withdrawal and per
transfer, and charges a transfer fee. The tutorial sweeps sub-accounts to the
master with transfers of everything above 1 USDC, then withdraws the master.
Up to about 1 USDC per account can remain on Lighter and is reported, not
hidden.

### Focused verification

```shell
source .local-test.env && PYTHONPATH="$(pwd):$PYTHONPATH" poetry run pytest tests/exchange_account/test_lighter_sub_account_allocator.py --timeout=300
source .local-test.env && PYTHONPATH="$(pwd):$PYTHONPATH" poetry run pytest tests/exchange_account/test_lighter_internal_transfer_state.py --timeout=300
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
   valued and Nautilus can trade it; capital moves manually.
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
- The e2e test passes in CI with all Ethereum effects real on the fork and
  only Lighter Core faked.
- The live tutorial completes and refunds the deployer with every stablecoin
  and ETH not consumed by gas, Lighter fees or documented on-exchange dust.

## Open questions

1. Should the deployment seed each sub-account with 1 USDC, or leave seeding
   to the first `start` cycle? Seeding at deployment keeps the first Nautilus
   start independent of the executor's cycle.
2. Should `lighter-ai.py` take weights as static parameters or read a file
   written by the Nautilus allocator? This plan assumes static parameters.
3. The e2e test needs the pinned `lighter-sdk` only for key generation and
   the native signer's public-key derivation; confirm the fake core can skip
   signature verification without importing the SDK in tests.
