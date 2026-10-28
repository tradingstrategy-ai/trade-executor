# Lighter exchange account integration

This document describes how to deploy and monitor a Safe-owned Lighter account
with a Lagoon vault on Ethereum. It follows the same external exchange-account
model as Derive and GMX: capital leaves the Safe for an external venue, while
trade-executor synchronises that venue's equity into strategy state and Lagoon
NAV.

The initial integration provides deployment, custody and valuation. Lighter
orders are submitted by a separate execution tool, not by trade-executor.

## Supported functionality

| Capability | Support |
|------------|---------|
| Safe-owned Lighter account activation | `lagoon-deploy-vault` |
| Delegated API-key generation and registration | `lagoon-deploy-vault` |
| Public total-equity valuation | Yes, without an API key |
| Lagoon NAV posting | `lagoon-settle` |
| Manual Safe/Lighter custody diagnostic | `lighter-move-funds` |
| Custody test trade through the normal pipeline | `perform-test-trade --single-pair` or `--all-pairs` |
| Read-only accounting comparison | `check-accounts` |
| Accounting correction, PnL and verified-transfer reconciliation | `correct-accounts` |
| Interrupted-transfer reconciliation | `correct-accounts` or `repair` |
| Sub-accounts with their own API keys | `lagoon-deploy-vault` with `STRATEGY_FILE` declaring `Parameters.lighter_sub_accounts` |
| Automatic master to sub-account funding and rebalancing | `start` with a multi-account strategy |
| Perpetual order execution inside trade-executor | No; use a separate execution tool |

## Value and custody model

The strategy universe contains one synthetic Lighter exchange-account pair
per Lighter account, see [Sub-accounts](#sub-accounts). Its value is
Lighter's public `total_asset_value`, including collateral and unrealised
profit or loss. For a single account the Lagoon NAV is:

```text
Safe native-USDC balance
+ public Lighter total equity
+ pending Lagoon settlement value
```

The Safe remains the Lighter account owner. Its guarded module can deposit USDC
to that account and claim withdrawals back to the Safe. Trading uses a delegated
Lighter API key registered in a numbered key slot.

## Sub-accounts

A vault can split its Lighter capital between the Safe-owned master account
and sub-accounts, each traded by its own external process such as a Nautilus
strategy. Each account is a separate executor position, and the Lagoon NAV
values all of them from one public snapshot:

```text
Safe native-USDC balance
+ master account equity + every sub-account's equity
+ pending Lagoon settlement value
```

The master account is the treasury hub. By default the executor holds only
the master API key; each sub-account key goes to that sub-account's trader.
Capital moves as follows:

| Movement | Mechanism | Signer |
|----------|-----------|--------|
| Safe to master | L1 deposit through the guard | Asset manager |
| Master to Safe | Secure withdrawal, then the Safe claim | Master API key, asset manager |
| Master to sub-account | L2 same-owner transfer | Master API key |
| Sub-account to Safe | L1 `withdraw(subAccountIndex)` through the guard, then the Safe claim | Asset manager only |
| Sub-account to sub-account or master | L2 same-owner transfer | The sending sub-account's key: its trader, or the executor when given the key with `LIGHTER_SUB_ACCOUNT_OPERATOR_RECORD_FILES` |

The automatic cash policy never signs with a sub-account key. It
rebalances by withdrawing an overweight sub-account to the Safe in one
cycle, depositing it to the master in the next and then funding the
underweight sub-account. A trader may transfer between sub-accounts with its
own key; the next `start` cycle revalues both positions from Lighter.

Mainnet observations from 2026-10-26:

- A master secure withdrawal takes about 20 minutes. A sub-account L1
  withdrawal became claimable after about 29 minutes. The
  `LIGHTER_WITHDRAWAL_TIMEOUT` default of `start` and `perform-test-trade`
  is therefore one hour.
- Lighter charged no fee for same-owner transfers. The authenticated
  `transferFeeInfo` quotes 0 USDC between accounts of the same owner (master
  to sub-account and sub-account to sub-account) and 1 USDC to a different
  owner; the public, unauthenticated endpoint quotes 1 USDC regardless. The
  funding trade reads the authenticated quote when it submits.
- Minimums: 1 USDC for a transfer, a secure withdrawal and a direct deposit;
  4 USDC for a fast withdrawal, which a Safe owner cannot use anyway.
- Lighter documents that a transfer to a different owner also needs an L1
  wallet signature, which a Safe cannot give. This was not tested.

### Deploy with sub-accounts

Deploy with `STRATEGY_FILE` instead of `DENOMINATION_ASSET`; one sub-account
is created per `SubAccount` the strategy declares:

```shell
export GENERATE_LIGHTER_API_KEY=true
export LIGHTER_API_KEY_INDEX=4
export LIGHTER_SUB_ACCOUNT_API_KEY_INDEX=4
export STRATEGY_FILE=strategies/my-lighter-strategy.py  # declares the sub-accounts

trade-executor lagoon-deploy-vault
```

After the master key is registered, deployment creates the sub-accounts with
the master key and registers one API key on each with a Safe `changePubKey`
transaction. Sub-accounts start empty; `start` funds them once the vault has
operating capital. The number of sub-accounts depends on the Lighter account
tier: Standard allows 4, Plus 16 and Premium 64.

Deployment writes these records next to `VAULT_RECORD_FILE`, beside the
Markdown report:

| File | Contents | Audience |
|------|----------|----------|
| `vault-record.txt` | Public metadata, including every sub-account index and public key | Operators |
| `vault-record.json` | Master API key and public sub-account metadata, mode 0600 | The executor only |
| `vault-record.nautilus.<id>.json` | One sub-account's API key, mode 0600 | The trader of that sub-account |

The executor reads a hand-over file only when an operator passes it in
`LIGHTER_SUB_ACCOUNT_OPERATOR_RECORD_FILES`, and no sub-account key appears
in any report.

### Saving keys before they are registered

`lagoon-deploy-vault` writes the master and sub-account private keys to
their records only after the deployment returns. If it fails after a key has
been registered on Lighter, that key exists nowhere else and the account
needs a new key through a Safe `changePubKey`. Pass
`--lighter-key-backup-dir` (`LIGHTER_KEY_BACKUP_DIR`) to save each key to a
mode-0600 file in that directory as soon as it is generated, before it is
registered:

```shell
trade-executor lagoon-deploy-vault --generate-lighter-api-key \
    --lighter-key-backup-dir /secure/path/key-backup ...
```

The files are deleted once every record is written and kept if the
deployment fails. Each is created exclusively, so a retried deployment never
overwrites an earlier attempt's keys. The Python tutorials predate the option
and still save the keys themselves.

### Optional: let the executor sign for sub-accounts

An operator may also give the executor a sub-account's hand-over file with
`LIGHTER_SUB_ACCOUNT_OPERATOR_RECORD_FILES`, a comma-separated list of
paths. A strategy can then send a transfer from that sub-account directly,
to the master or to another sub-account, with
`PositionManager.transfer_lighter_funds(source, destination, amount)`; it is
booked exactly like a master to sub-account funding, with the amount and
fee as balance updates on the source position. The automatic cash policy
still rebalances through the Safe. Pass the same
variable to `check-accounts`, `correct-accounts` and `repair`, which need the
sending account's record to read its transfer history. Each hand-over file
also carries the Safe module address, so it loads as an operator record.

### Strategy setup for sub-accounts

A strategy declares the sub-strategies it runs as `SubAccount` values in
`Parameters`: a stable id, a name and the account's target weight. See
[strategy parameters](../../docs/strategy-parameters.md) for what belongs
in `Parameters` and what does not. No Lighter account
index appears in the strategy. The id is the key the strategy moves funds
with; once a deployment has created the account for an id, keep the id
unchanged in the strategy and in the state.

```python
from tradeexecutor.exchange_account.lighter_strategy import create_lighter_account_universe
from tradeexecutor.exchange_account.sub_account import SubAccount

class Parameters:
    lighter_cash_management = True
    lighter_sub_accounts = [
        SubAccount(id="trend", name="Trend following"),
        SubAccount(id="arb", name="Funding-rate arbitrage"),
    ]

def create_trading_universe(input):
    return create_lighter_account_universe(input, Parameters.lighter_sub_accounts)
```

The resolution of ids to Lighter accounts lives in the state, not in the
strategy. `lagoon-deploy-vault` reads the declared sub-accounts from the
strategy module, creates one Lighter sub-account per declaration, names each
hand-over key file after its id and records the id in the deployment
artefact. `init` copies the master and sub-account indices into
`state.sync.deployment.lighter_accounts`, and an id's index never changes
afterwards. Universe construction receives the state and builds one
exchange-account pair per account with the real indices; without a state,
at deployment or in a universe check, it uses placeholders. Each account is
its own trading position, opened at zero by `correct-accounts` or by the
first custody method that touches it, and
`tradeexecutor.exchange_account.sub_account` maps positions to ids and ids
to positions through the state. The single-account `strategies/lighter-ai.py`
predates this and builds its one pair from `LIGHTER_ACCOUNT_INDEX`; it is
unchanged apart from its `ExchangeType.lighter`.

`decide_trades()` moves funds between the accounts by id with the
`PositionManager` custody methods.
Each call creates one trade that `start` executes: a Safe deposit is a buy
on the master position, a withdrawal is a sell on the account's position,
and a transfer between accounts is a buy on the destination with the amount
and fee booked on the source. A transfer sent from a sub-account needs that
sub-account's key, see the previous section.

```python
def decide_trades(input):
    position_manager = input.get_position_manager()
    if position_manager.is_lighter_transfer_pending():
        return []
    return [
        position_manager.deposit_to_lighter(),                              # all idle Safe cash to the master
        position_manager.transfer_lighter_funds("master", "trend", Decimal(100)),
        position_manager.transfer_lighter_funds("trend", "arb", Decimal(25)),  # signed with trend's key
        position_manager.withdraw_from_lighter("arb"),                      # arb's free collateral to the Safe
    ]
```

`get_lighter_balance(account)` and `get_lighter_free_collateral(account)`
read an account's tracked equity and its free collateral on Lighter.
[`strategies/test_only/lighter_scripted_transfers_strategy.py`](../../strategies/test_only/lighter_scripted_transfers_strategy.py)
is the complete example; the transfer-accounting tutorial and its black-box
test run it.

For automatic cash management, give each `SubAccount` its target share of
the capital and use `decide_lighter_cash_management()` from
`tradeexecutor.exchange_account.lighter_strategy`:

```python
from tradeexecutor.exchange_account.lighter_strategy import decide_lighter_cash_management

class Parameters:
    lighter_cash_management = True
    lighter_sub_accounts = [SubAccount(id="trend", name="Trend following", weight=0.5), SubAccount(id="arb", name="Funding-rate arbitrage", weight=0.5)]
    lighter_safe_cash_buffer_usd = Decimal(20)
    lighter_free_collateral_buffer_usd = Decimal(0)

def decide_trades(input):
    return decide_lighter_cash_management(input)
```

Optional parameters: `lighter_min_transfer_usd` (defaults to Lighter's 1 USDC
minimum), `lighter_rebalance_band_usd` (default 0) and
`lighter_transfer_fee_usd` (default 0, as Lighter charges nothing for
transfers between accounts of one owner).

Each cycle moves one kind of capital, sized from observed balances: first
Safe liquidity for pending redemptions (master first, then the sub-accounts
with the most free collateral), then idle Safe cash to the master, then
overweight sub-accounts to the Safe, then underweight sub-accounts from the
master. Only free collateral moves; margin in use stays.
`lighter_transfer_fee_usd` is the fee the planner reserves on the master for
each funding transfer; the fee actually charged is booked from Lighter's
transfer history. See
[`lighter_multi_subaccount_strategy.py`](../../strategies/test_only/lighter_multi_subaccount_strategy.py).

Set `LIGHTER_TRANSFER_TIMEOUT` for `start` or `perform-test-trade` to bound
the wait for a transfer between accounts to appear in Lighter's transfer
history.

### Recovery

An interrupted transfer makes the state unclean, and `start` refuses to run.
`check-accounts` lists each pending Lighter transfer with its trade id,
status and saved hashes, exits 1 and still prints the balance table. Then:

- **Transfer between accounts:** `correct-accounts` or `repair` reconciles
  it from Lighter's transfer history; pass the sending account's record
  (`LIGHTER_OPERATOR_RECORD_FILE` for the master,
  `LIGHTER_SUB_ACCOUNT_OPERATOR_RECORD_FILES` for a sub-account) so they can
  read it. A timeout alone never releases a transfer, because it does not
  prove that no funds moved. A transfer interrupted before its submission
  was checkpointed is expired; one whose submission started without a saved
  hash stays pending until you check the sender's transfer history manually.
- **Withdrawal waiting to become claimable:** run `lighter-move-funds
  --lighter-account-index <index>` for the account. It resumes the saved
  request, waits for it, claims it into the Safe and completes the
  accounting, for automatic and manual withdrawals alike, without sending a
  second request.
- **Withdrawal whose Safe claim was broadcast:** `correct-accounts` or
  `repair` reconciles it from the claim receipt.
- **Transfer the halted cycle never started:** `correct-accounts` or
  `repair` expires it. Execution starts each transfer just before submitting
  it, so a still-planned transfer moved no funds.

`lighter-move-funds --lighter-account-index <index>` withdraws from a
sub-account with the key-free L1 path. Deposits always go to the master.

### Tutorials

Both tutorials share
[`tradeexecutor/testing/lighter_tutorial.py`](../testing/lighter_tutorial.py):
deployment with key backups, state initialisation, subscription,
redemption, the final sweep and a reconciliation table of the deployer's
balances. The fork and fake sequencer live in
[`lighter_simulation.py`](../testing/lighter_simulation.py), and both
tutorials end with the trade table of
[`lighter_report.py`](../testing/lighter_report.py): every trade with its
description, status, notes and Ethereum and Lighter hashes. Live runs
resume from `progress.json` in their run directory; simulated runs always
start fresh.

[`scripts/lagoon/manual-trade-executor-lighter-transfers.py`](../../scripts/lagoon/manual-trade-executor-lighter-transfers.py)
is the executable specification of the transfer accounting. Its strategy,
[`strategies/test_only/lighter_scripted_transfers_strategy.py`](../../strategies/test_only/lighter_scripted_transfers_strategy.py),
declares its sub-accounts in `Parameters` and emits one fixed step per
`start` cycle with the `PositionManager` custody methods: deposit the Safe cash to
the master, fund both sub-accounts, move capital from one sub-account to the
other and from a sub-account to the master with the sub-account keys, then
withdraw everything. After winding the vault up it verifies the state with
`verify_lighter_scripted_transfers()`: every movement is one successful
trade between the right positions, fixed amounts match, each funding's fee
and principal are balance updates on the source position, no position holds
USDC, and inflows, revaluations, outflows and fees add up. With
`SIMULATE=true` it runs on an Anvil fork with the fake sequencer in about two
minutes and with a transfer fee, so the fee accounting is checked;
`tests/lagoon/test_lagoon_lighter_scripted_transfers_e2e.py` runs it that way
in CI. Without it, it runs live for 10 USDC plus gas. It passed live on
2026-10-26: Lighter charged no transfer fee, the sub-account-signed transfers
went through, every USDC came back to the deployer and the run took 66
minutes, most of it the three withdrawals. The same strategy module deploys
and runs on a fork or on mainnet, because the account indices come from the
state.

[`scripts/lagoon/manual-trade-executor-lighter-subaccounts.py`](../../scripts/lagoon/manual-trade-executor-lighter-subaccounts.py)
is live only. It deploys a vault with two sub-accounts and funds them with
the production cash policy at 50/50 weights, acts as the trader for one
sub-account to sub-account transfer, lets the policy rebalance through the
Safe, then drains, redeems and sweeps everything back to the deployer. It
needs 5 USDC plus gas: the 1 USDC bootstrap and a 4 USDC subscription.

The 2026-10-26 run verified sub-account creation with the master key,
sub-account `changePubKey` through the Safe, master to sub-account transfer,
the key-free L1 `withdraw(subIndex)`, master secure withdrawal, full drain
and redemption. The deployer got every USDC back with no Lighter fees, for
about 0.011 ETH of gas. Its trader step, a sub-account to sub-account
transfer signed with the sub-account's own key, was added afterwards; the
transfer tutorial has since verified such transfers live.

## Deploy a Safe-owned Lighter account

### Deployer funds

Before running the command, fund the deployer/initial asset-manager address
with the following native Ethereum assets:

| Asset | Amount | Why it is needed |
|-------|--------|------------------|
| ETH | A non-zero balance sufficient for current mainnet gas | Deploys the Lagoon contracts and Safe, then sends the Lighter activation transactions. Gas prices and the deployment path vary, so there is no safe fixed ETH amount. |
| Native Ethereum USDC | **1 USDC** for activation | The command subscribes 1 USDC through Lagoon and deposits all of it to activate Lighter, the direct Ethereum deposit minimum. The Safe holds no USDC afterwards; add operating capital with a normal Lagoon subscription. |

Use the canonical Ethereum USDC contract, not bridged USDC:
`0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48`. The Lighter API key does not
exist before deployment and must not be funded separately: the command creates
it for the Safe-owned Lighter account.

Set the normal Lagoon deployment variables and enable Lighter API-key creation:

| Variable | Required | Description |
|----------|----------|-------------|
| `JSON_RPC_ETHEREUM` | Yes | Ethereum mainnet RPC URL |
| `PRIVATE_KEY` | Yes | Funded Ethereum deployer and initial asset-manager key |
| `DENOMINATION_ASSET` | Yes | Native Ethereum USDC address |
| `VAULT_RECORD_FILE` | Yes | Path for the public text record and protected JSON operator record |
| `GENERATE_LIGHTER_API_KEY` | Yes | Set to `true` to activate Lighter and register a new API key |
| `LIGHTER_API_KEY_INDEX` | No | Lighter API-key slot; defaults to `4` |
| `EXECUTOR_ID` | Recommended | Stable name for the public runtime deployment artefact |
| `STATE_FILE` | Recommended | Places that runtime artefact next to executor state |

```shell
export JSON_RPC_ETHEREUM="https://..."
export PRIVATE_KEY="0x..."
export DENOMINATION_ASSET="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
export VAULT_RECORD_FILE="/secure/path/lighter-vault-record.txt"
export GENERATE_LIGHTER_API_KEY=true
export LIGHTER_API_KEY_INDEX=4
export EXECUTOR_ID="lighter-vault"
export STATE_FILE="state/lighter-vault.json"

trade-executor lagoon-deploy-vault
```

This is a live Ethereum operation. It spends gas, makes the accounted 1 USDC
Lagoon subscription and deposits all of it to Lighter. It cannot run with `SIMULATE=true`, because an Anvil
fork cannot create an account in Lighter's live sequencer state.

The deployment command writes several outputs with different audiences:

| Output | Contents |
|--------|----------|
| Text record | Public vault and Lighter account metadata |
| Markdown report | Public deployment and guard report |
| Executor deployment artefact | Public runtime metadata when `EXECUTOR_ID` is configured |
| Paired `vault-record.json` | Public metadata and the generated Lighter private key |

Only the paired JSON operator record contains the private key. It is created
with mode `0600`. Move it to a secret manager immediately. Do not put it in
strategy configuration, command-line arguments, logs, support tickets or chat.
Public outputs use an explicit metadata allowlist, so unknown SDK fields are
not copied to them.
A failed deployment can lose a key it already registered unless
`--lighter-key-backup-dir` is set; see
[Saving keys before they are registered](#saving-keys-before-they-are-registered).

## Example: deploy and test a Lighter vault

This is the operator flow for a small mainnet test. It assumes that the
`trade-executor` console command is installed and that the deployer has the
funds listed above. Use funds that can tolerate execution fees and price
movement.

### 1. Pair a static exchange-account strategy module

Pair the vault with a strategy that declares one synthetic Lighter
exchange-account pair. It identifies the public Lighter account for valuation;
it does not hold an API key and does not submit a perp order. This is the same
external-account shape used by the [Derive](./README-Derive.md#strategy-setup)
and [GMX](../../strategies/test_only/minimal_gmx_strategy.py) examples.

```python
from tradeexecutor.exchange_account.lighter import create_lighter_exchange_account_pair

lighter_pair = create_lighter_exchange_account_pair(
    quote=usdc,
    account_index=LIGHTER_ACCOUNT_INDEX,
)
```

Use [`strategies/test_only/minimal_lighter_strategy.py`](../../strategies/test_only/minimal_lighter_strategy.py)
as the complete, passive module template. It builds the one-pair universe and
returns no executor trades. Set `LIGHTER_ACCOUNT_INDEX` to the public index in
the deployment report. Keep normal strategy logic separate from the external
execution tool until trade-executor itself supports Lighter order routing.

### Automatic Safe cash management

For an alpha model that should park idle Safe USDC on Lighter, use the same
synthetic pair with `lighter_cash_management=True`. The strategy's
`decide_trades()` builds an `ExchangeCashSnapshot` from `PositionManager` and
the Lagoon queue, then returns a `TradeExecution` when `ExchangeCashManager`
selects a movement. The custody movement therefore follows the ordinary
approval, checkpoint, router and settlement pipeline; it is not a second
transfer subsystem. See
[`strategies/test_only/lighter_cash_management_strategy.py`](../../strategies/test_only/lighter_cash_management_strategy.py)
for a small complete example.

The relevant parameters are:

```python
class Parameters:
    lighter_cash_management = True
    lighter_safe_cash_buffer_usd = Decimal("20")
    lighter_free_collateral_buffer_usd = Decimal("0")
```

This single-account module keeps a module-level `LIGHTER_ACCOUNT_INDEX`, the
layout `strategies/lighter-ai.py` uses. New strategies declare their
sub-accounts in `Parameters` and take every account index from the state, as
[Strategy setup for sub-accounts](#strategy-setup-for-sub-accounts) and
[`docs/strategy-parameters.md`](../../docs/strategy-parameters.md) describe.
Runtime secrets and withdrawal settings are parsed by the CLI and passed to
routing explicitly.

The manager deposits only Safe cash above the configured buffer. When Lagoon
has pending redemptions and the Safe is short, it withdraws available Lighter
collateral. Pending Lagoon deposits offset matching redemptions, so the Safe
target is `max(pending redemptions - pending deposits, 0) + buffer`. A
withdrawal normally waits about 20 minutes, and the executor
intentionally blocks for it with a one-hour safety timeout. No later trades or
NAV/account checks run while the transfer is in flight. If the process stops,
the next `start` aborts before any new trade. Reconcile verified evidence with
`correct-accounts` or `repair`; it never repeats a Lighter request or Safe
claim. A claim already broadcast to Ethereum remains unclean until that
evidence is recorded. Run read-only `check-accounts` first to inspect this
state before choosing either correcting command. Keep
`LIGHTER_OPERATOR_RECORD_FILE=/secure/lighter/lighter-vault.json` available to
`start` for the owner-only delegated key record. The key itself is never
stored in strategy parameters or executor state.

When automatic cash management is enabled, Lagoon defers redemption settlement
if the Safe and pending Silo deposits cannot cover the redemption. The next
strategy decision can then return the Lighter withdrawal trade before NAV is
posted. This is deliberately synchronous and conservative: a long withdrawal
delay is preferable to publishing a temporary low NAV or starting trades
against expected proceeds. If Lighter has insufficient free collateral, this
deferral remains in place until an operator restores liquidity; do not bypass
it by posting an unsupported lower NAV. `lighter-move-funds` remains the
stopped-executor diagnostic and recovery command.

Use `input.get_position_manager().get_current_cash()` as the `safe_usdc`
argument. It is the latest treasury-synchronised Safe reserve; a strategy must
not make a direct Web3 balance read from `decide_trades()`.

### 2. Deploy through the Typer CLI

Choose a protected directory for the operator record. Do not use a directory
that is committed, copied to executor state, or published as deployment
output.

```shell
export JSON_RPC_ETHEREUM="https://..."
export PRIVATE_KEY="0x..."  # Funded deployer and initial asset manager
export DENOMINATION_ASSET="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
export VAULT_RECORD_FILE="/secure/lighter/lighter-vault.txt"
export GENERATE_LIGHTER_API_KEY=true
export LIGHTER_API_KEY_INDEX=4
export EXECUTOR_ID="lighter-vault"
export STATE_FILE="state/lighter-vault.json"

trade-executor lagoon-deploy-vault
```

Record the vault addresses and `account_index` from the public text/Markdown
report. The generated Lighter private key is written only to the paired
`/secure/lighter/lighter-vault.json` operator record, with mode `0600`; it is
never written to the text record, Markdown report, executor deployment
artefact, strategy module or logs. Back up that JSON file in the organisation's
secret manager without printing it to a terminal or chat.

### Custody test trade with `perform-test-trade`

`perform-test-trade` verifies a deployment's custody path with the same
trades `start` makes:

```shell
export LIGHTER_OPERATOR_RECORD_FILE="/secure/path/vault-record.json"

# Safe -> master -> Safe
trade-executor perform-test-trade --single-pair --amount 1

# Every account; a sub-account is funded through the master and withdrawn with the key-free L1 path
trade-executor perform-test-trade --all-pairs --amount 1

# One sub-account, by its id; --buy-only leaves the funding there
trade-executor perform-test-trade --pair "(ethereum, lighter, LIGHTER-trend, USDC)" --buy-only --amount 2

# Between two accounts, signed with the source's key; back again unless --buy-only
export LIGHTER_SUB_ACCOUNT_OPERATOR_RECORD_FILES="/secure/path/vault-record.nautilus.trend.json,/secure/path/vault-record.nautilus.arb.json"
trade-executor perform-test-trade --lighter-transfer trend:arb --buy-only --amount 1
```

A Lighter account's pair is `LIGHTER-ACCOUNT` for the master and
`LIGHTER-<id>` for a sub-account, on the `lighter` exchange.
`--lighter-transfer SOURCE:DESTINATION` takes `master` or sub-account ids.
It checks before the first leg that the executor holds the signer of every
leg, the source and, without `--buy-only`, the destination, so a missing key
never strands funds halfway. `--lighter-transfer master:trend` also completes
a sub-account funding that stopped after its Safe deposit.

[`manual-lighter-subaccounts-cli.sh`](../../scripts/lagoon/manual-lighter-subaccounts-cli.sh)
runs the whole lifecycle with these commands, from deployment to the final
redemption; see [the deployment guide](../../docs/lagoon-lighter-deployment.md#test-from-the-command-line).

The operator record is required for any Lighter vault, whether or not the
strategy sets `lighter_cash_management`; without it the command refuses
before moving anything. `--amount` defaults to 1 USDC, Lighter's minimum.
`--buy-only` leaves the deposit (and the funding of a sub-account) in place.
A transfer between accounts charges Lighter's fee on top of the amount on
the master; it is zero between accounts of one owner, but the master's free
collateral must cover it if one is charged. Withdrawals run one after
another and each waits up to `LIGHTER_WITHDRAWAL_TIMEOUT` for Lighter to make
it claimable, so expect about half an hour per account live. Outside
`--simulate`, `--all-pairs` accepts fewer than ten pairs.

The command saves the state only when it finishes, and never with
`--simulate`. That is an accepted trade-off for a manual one-off check. If
it is interrupted after Lighter accepted a deposit or a transfer between
accounts, `check-accounts` shows the difference and `correct-accounts`
books it before the test trade is rerun. If it is interrupted during a
withdrawal, the USDC waits in the Safe's pending balance on the ZkLighter
contract and no saved trade tracks it, so `lighter-move-funds` cannot
resume it: once it is claimable, claim it into the Safe with
`eth_defi.lighter.lagoon.claim_usdc_to_lagoon_safe_from_lighter()`, and
only then run `correct-accounts`. Run neither `correct-accounts` nor a NAV
settlement in between, or the in-transit amount is booked as a loss.

### 3. Perform a resumable ETH/USD test trade

For an existing Lighter-enabled Lagoon vault, use the dedicated Typer command.
It reads the delegated key only from the mode-`0600` operator record produced
by deployment, makes a bounded ETH/USD long round trip, and posts Lagoon NAV
at each completed phase. It never places the key in executor state, reports or
command-line arguments.

Run the complete operator flow in this order while the normal executor is
stopped:

1. `lagoon-deploy-vault --generate-lighter-api-key` creates the Safe-owned
   Lighter account. Its accounted 1 USDC Lagoon subscription moves entirely
   to Lighter as the activation deposit, so the Safe is empty.
2. `trade-executor init`, then `trade-executor correct-accounts --process-redemption`,
   recovers the original Lagoon settlement from the deployment artefact and
   creates the 1 USDC public Lighter exchange-account position.
3. `scripts/lagoon/deposit-and-settle.py 19` subscribes 19 USDC of test
   capital through Lagoon and settles it into the Safe.
4. `trade-executor lagoon-lighter-test-trade --lighter-test-deposit-usdc 19` moves the 19 USDC
   to Lighter, opens and closes the ETH/USD perpetual, claims the withdrawal,
   and posts NAV at each phase.

Never substitute `lagoon-first-deposit` for steps 2 or 3: that command is only for a
vault with no NAV and no shares, while Lighter activation has already created
one share.

```shell
source .local-test.env
export JSON_RPC_ETHEREUM="https://..."
export PRIVATE_KEY="0x..."  # Funded Safe owner / asset manager; do not echo it
export EXECUTOR_ID="lighter-vault"
export STRATEGY_FILE="strategies/test_only/minimal_lighter_strategy.py"
export STATE_FILE="state/lighter-vault.json"
export VAULT_ADDRESS="0x..."
export VAULT_ADAPTER_ADDRESS="0x..."
export ASSET_MANAGEMENT_MODE="lagoon"
export LIGHTER_ACCOUNT_INDEX="..."  # Public index from the deployment report
export LIGHTER_OPERATOR_RECORD_FILE="/secure/lighter/lighter-vault.json"
# Fresh Lighter deployment: 1 USDC is already on Lighter, so add 19 USDC.
export LIGHTER_TEST_DEPOSIT_USDC="19"
# export LIGHTER_TEST_POSITION_USDC="..."

trade-executor init
trade-executor correct-accounts --process-redemption
poetry run python scripts/lagoon/deposit-and-settle.py 19
trade-executor lagoon-lighter-test-trade --lighter-test-deposit-usdc 19
```

`LIGHTER_TEST_DEPOSIT_USDC` defaults to 20 USDC for an arbitrary existing
vault. A fresh Lighter deployment should override it to 19 USDC; together with
the initial 1 USDC Lighter collateral this makes 20 USDC total. Lighter's
direct Ethereum contract deposit has a 1 USDC minimum. The live ETH/USD market metadata
inspected on 2026-09-16 required both 0.005 ETH and 10 USDC of notional; at the
then-current price, 0.005 ETH was about 12 USDC. The 20 USDC total therefore
clears the zk deposit minimum and gives the automatically sized ETH perpetual
test order modest collateral headroom. The command fetches the market limits
at runtime because Lighter may change them. Ensure the Safe has the selected
deposit amount available: 19 USDC for a fresh deployment after the
`deposit-and-settle.py` subscription, or 20 USDC by default.

See Lighter's
[deposit documentation](https://apidocs.lighter.xyz/docs/deposits-transfers-and-withdrawals)
and live
[ETH market metadata](https://mainnet.zklighter.elliot.ai/api/v1/orderBookDetails?market_id=0).

If the Safe does not have enough USDC for the test, stop the executor and fund
the vault through Lagoon's investor deposit lifecycle. Do not transfer USDC
directly to the Safe, because that bypasses Lagoon share and investor-flow
accounting. From a source checkout, the reusable script inherits the normal
executor environment:

```shell
poetry run python scripts/lagoon/deposit-and-settle.py 20
```

The production Docker image does not include Poetry. From the strategies
Compose deployment, run:

```shell
docker compose run --rm \
  --entrypoint /usr/local/bin/python \
  lighter-ai \
  scripts/lagoon/deposit-and-settle.py 20
```

`PRIVATE_KEY` must identify the funded deployer and authorised Lagoon asset
manager. It needs the requested native USDC plus ETH for gas. The script
approves and requests the deposit, invokes the real `lagoon-settle` command to
post NAV, settle the complete investor queue and update executor state, and
then claims the deployer's vault shares. A rerun with the same amount resumes
an exact pending or claimable deposit instead of submitting it twice. After a
successful run, both the Safe balance and executor reserve include the new
capital. `lagoon-settle` settles the vault's eligible investor queue, so do not
run the script while another investor's request needs separate handling.

Lighter activation subscribes 1 USDC to Lagoon, then moves all of it from
the Safe to Lighter. A correctly deployed vault therefore starts with an empty
Safe and a zero executor reserve, and `deposit-and-settle.py` adds its first
operating capital. `lagoon-first-deposit` intentionally rejects the vault
because that command is only for a vault with no NAV or shares. Until the first
subscription settles, the Safe holds no cash to pay a Lagoon redemption.

### Yubi deployment secret mapping

The Lighter AI Yubi deployment convention maps the protected host directory
`~/secrets/lighter` to `/secure-lighter` in the manual-command container. Its
delegated-key record is conventionally
`~/secrets/lighter/lighter-ai-vault-info.json`. `lagoon-lighter-test-trade`,
`lighter-move-funds`, `perform-test-trade` and a `start` with
`lighter_cash_management` need this record; `check-accounts`,
`correct-accounts` and `repair` take it only to reconcile an interrupted
transfer between accounts:

```shell
export LIGHTER_OPERATOR_RECORD_FILE="/secure-lighter/lighter-ai-vault-info.json"
```

This path is not a credential, but the JSON file is mode `0600` and contains
the delegated private key. Do not place the key itself in an environment
variable, strategy file, state file or Compose configuration.

The command writes an owner-only, non-secret recovery journal beside the state
file. Re-run the same command after a failed public observation or order call;
it resumes only completed, proven phases and refuses ambiguous mutations. If a
withdrawal request failed before Lighter accepted it, confirm no matching
withdrawal-history row exists, then change the journal phase from
`withdrawal_requested` to `closed` before retrying. Stop the normal strategy
executor for the duration of this manual operator test.

`scripts/lagoon/manual-trade-executor-lighter.py` remains a standalone
end-to-end tutorial which deploys and redeems its own temporary vault.

For a vault already deployed outside the tutorial, use the same static strategy
and run a Lagoon settlement after each completed Lighter movement:

```shell
trade-executor lagoon-settle
trade-executor check-accounts
```

`lagoon-settle` synchronises both sides of the custody boundary: it reads the
current Lighter equity and reconciles the executor reserve to the Safe's actual
USDC balance before posting NAV. After a completed `lighter-move-funds`
operation, `check-accounts` should already be clean. Use
`correct-accounts` only for residual PnL or an interrupted/out-of-band flow;
see [External transfer accounting](#external-transfer-accounting).

Do not run a NAV cycle while a deposit, order fill or withdrawal is still in
flight. Wait for Lighter's public account response and the Safe balance to
reflect the completed movement first.

### Secure withdrawals and Safe custody

Lighter's `withdrawalDelay` endpoint reports the current secure-withdrawal
delay in seconds. It is dynamic and only an operator estimate: wait for the
specific `withdraw_history` item to become `claimable` before the Safe sends
the L1 claim. `lagoon-lighter-test-trade` logs this value immediately before it
submits its secure withdrawal; `lighter-move-funds` does the same. The reported
delay is not a deadline. In the live manual test, claimability lagged the
reported value, so configure the timeout with headroom and rely on history
status rather than elapsed time.

Lighter fast USDC withdrawals require the L1 account's Ethereum EOA private
key. The Lighter owner in this integration is a Lagoon Safe contract, not an
EOA, so it has no such key. Fast withdrawal is therefore unsupported here;
use a secure withdrawal followed by the Safe-gated L1 claim. Never substitute
the deployer or asset-manager key: it does not own the Safe's Lighter account.

While a secure withdrawal is pending, Lighter may show zero account equity
before the Safe receives the claimed USDC. Pause normal execution,
`correct-accounts` and Lagoon NAV settlement during this in-transit window so
AlphaModel sizing cannot act on a temporarily understated reserve or NAV.

### Manual custody diagnostic

`lighter-move-funds` is an operator diagnostic and error-recovery command, not
an everyday capital-management path. Stop the executor before using it. It
shows Safe USDC, tracked reserve, Lighter collateral, equity, available balance,
margin requirements, allocated margin, gross notional, position-record count
and unrealised PnL. It then asks for a deposit (`d`) or withdrawal (`w`), an
amount in USDC and a final confirmation which defaults to no.

```shell
export EXECUTOR_ID="lighter-vault"
export STRATEGY_FILE="strategies/test_only/minimal_lighter_strategy.py"
export STATE_FILE="state/lighter-vault.json"
export JSON_RPC_ETHEREUM="https://..."
export PRIVATE_KEY="0x..."
export ASSET_MANAGEMENT_MODE="lagoon"
export VAULT_ADDRESS="0x..."
export VAULT_ADAPTER_ADDRESS="0x..."
export LIGHTER_ACCOUNT_INDEX="..."
export LIGHTER_OPERATOR_RECORD_FILE="/secure-lighter/lighter-ai-vault-info.json"
# Optional: defaults are 900 seconds for deposits and 3600 seconds for withdrawals.
export LIGHTER_DEPOSIT_TIMEOUT="900"
export LIGHTER_WITHDRAW_TIMEOUT="3600"

trade-executor lighter-move-funds
```

The command records a successful movement as one flagged synthetic trade on
the moved account's position, `LIGHTER-ACCOUNT` for the master or
`LIGHTER-<id>` for a sub-account, and changes the Safe reserve by the same
USDC amount. `check-accounts` therefore verifies both sides of a completed movement,
while `correct-accounts` repairs residual PnL or out-of-band drift without
rewriting this transfer record.

Secure withdrawals save their request before waiting and resume after a
restart without blindly submitting a second request. The command claims only
the matching `claimable` history row and records the observed Safe USDC credit;
the credit can be lower than the requested amount because of protocol fees. A
definitive API rejection is retained as a failed transfer for operator review.
An ambiguous transport failure remains pending and is never resubmitted
automatically. Keep unrelated Safe transfers stopped until the withdrawal
finishes, because they can obscure the saved pre-withdrawal balance used for
recovery. If an ambiguous request never appears in withdrawal history, keep
the executor stopped and investigate the saved transfer instead of resubmitting
it. A deposit interrupted after its physical movement but before state
persistence needs operator investigation; it is never replayed automatically.

## Strategy setup

Use the public account index from the deployment report in the strategy
universe:

```python
from tradeexecutor.exchange_account.lighter import create_lighter_exchange_account_pair

lighter_pair = create_lighter_exchange_account_pair(
    quote=usdc,
    account_index=123,
)
```

The strategy may contain only one external exchange-account protocol. The
current Lighter adapter supports the canonical Ethereum deployment and native
USDC. It does not need the Lighter private key because valuation uses the
unauthenticated public API.

See `strategies/test_only/minimal_lighter_strategy.py` for a complete static
universe.

Runtime commands need only public Lighter metadata:

| Variable | Required | Description |
|----------|----------|-------------|
| `STRATEGY_FILE` | Yes | Strategy containing the synthetic Lighter pair |
| `LIGHTER_ACCOUNT_INDEX` | Yes for the example strategy | Public account index written by deployment |
| `STATE_FILE` | Yes | Executor state containing the exchange-account position |
| `JSON_RPC_ETHEREUM` | Yes | Ethereum RPC used for Safe and reserve reads |
| `ASSET_MANAGEMENT_MODE` | Yes | Executor custody mode, normally `lagoon` for a deployed vault |
| `PRIVATE_KEY` | Yes | Asset-manager Ethereum key used for signed maintenance commands; never the Lighter API key |
| `VAULT_ADDRESS` | Lagoon mode | Lagoon vault address |
| `VAULT_ADAPTER_ADDRESS` | Lagoon mode | Trading strategy module address |

Do not configure the generated Lighter private key for `init` or
`lagoon-settle`; they use public account equity or executor state only.
`check-accounts`, `correct-accounts` and `repair` take the operator records
only to read transfer history when reconciling an interrupted transfer
between accounts.

## Accounting command lifecycle

Configure the deployed vault, state file, public Lighter account index and
strategy file, then initialise the external-account position:

```shell
export STRATEGY_FILE="strategies/test_only/minimal_lighter_strategy.py"
export LIGHTER_ACCOUNT_INDEX=123
export STATE_FILE="state/lighter.json"

trade-executor init
trade-executor correct-accounts
```

Run `correct-accounts` after the initial collateral deposit is visible on
Lighter. It creates or updates the external-account balance from public equity.

Use `check-accounts` for a read-only comparison. It exits successfully when
the tracked quantity matches public Lighter equity and exits with an accounting
mismatch when they differ. For a Lagoon vault it also exits 1 while a Lighter
transfer is pending or when the account sets below differ. It does not write
a balance update:

```shell
trade-executor check-accounts
```

Use `correct-accounts` to accept the observed Lighter equity into state. A
change is recorded as an exchange-account `BalanceUpdate`, anchored to the
current Ethereum block for audit even though the Lighter observation itself is
current rather than historical. The synchroniser cannot yet distinguish
trading PnL from a deposit or withdrawal solely from the account-value change:

```shell
trade-executor correct-accounts
trade-executor check-accounts
```

`correct-accounts --dry-run` previews the run: it reconciles verified
transfers and applies the Lighter close rule below in memory, prints the
corrections and saves nothing.

### Accounts the Safe owns, the strategy declares and the state tracks

For a Lagoon vault both commands compare three sets: the Lighter accounts the
Safe owns, the accounts the strategy declares (its universe) and the accounts
with an open or frozen position. Lighter cannot withdraw less than its
1 USDC minimum, so equity below `LIGHTER_DUST_USDC` (1 USDC) is
unwithdrawable dust. The check fails when:

- an owned account holds at least 1 USDC but has no position: for a
  declared account `correct-accounts` opens it; an undeclared one is
  outside the vault's accounting and needs the operator;
- a position tracks an account the strategy no longer declares;
- a tracked or declared account is not owned by the Safe, or Lighter cannot
  be read.

A declared account below 1 USDC without a position passes, so a drained
sub-account does not fail the check. A closed position of a declared account
is reopened at zero by the next `correct-accounts`.

When the strategy stops declaring a sub-account, `correct-accounts` closes
its position under Lighter's own rule, separate from the HyperCore dust
rules: below 1 USDC the residual is written off as one `correction` balance
update and the position is closed; at or above it the command logs the
`lighter-move-funds --lighter-account-index <index>` command that drains the
account, saves its other corrections and exits 1 until the account is
drained. The posted NAV applies the same rule: an owned account outside the
universe aborts the NAV only when it holds 1 USDC or more.

The executor never freezes a Lighter position; a failed custody transfer
keeps it open. A position frozen by a manual state edit is still revalued by
`correct-accounts`, but `start` looks up open positions only, so move it back
to the open positions by hand before the next `start`, or the next custody
trade opens a second position for the account. `repair` cannot do this: it
unfreezes only positions with a repaired trade.

### External transfer accounting

Completed `lighter-move-funds` deposits and withdrawals have a durable
executor record with direction, amounts and public request or transaction
identifiers. The delegated key and bearer tokens are never written to state,
reports or command output. Strategy-driven and out-of-band movements remain
reconciliation cases.

`correct-accounts` corrects the on-chain Lagoon Safe reserve even when the
sole trading position is a Lighter exchange account. It still reads Lighter
equity through the public API rather than treating the synthetic
`LIGHTER-ACCOUNT` asset as an ERC-20 balance. `correct-accounts --dry-run`
does not persist proposed corrections.

An unfinished secure withdrawal deliberately makes executor state unclean, so
normal execution and account correction stop rather than booking its temporary
Lighter debit as a loss. Resume `lighter-move-funds` and let it claim or detect
the completed withdrawal first.

`repair` first reconciles an interrupted Lighter transfer when its public
Ethereum and Lighter evidence is complete, then repairs other interrupted
executor trades and transactions. A transfer without complete evidence remains
unclean rather than being generically refunded. A healthy Lighter
exchange-account position and its spoofed successful opening trade are left
unchanged:

```shell
trade-executor repair --auto-approve
```

Run Lagoon settlement after deposits, trading and withdrawals to post the
combined Safe and Lighter NAV:

```shell
trade-executor lagoon-settle
```

The Lighter endpoint reports current state and cannot be pinned to the Ethereum
block used for the Safe balance. A secure withdrawal is asynchronous: its live
`withdrawalDelay` value is dynamic, and only `withdraw_history=claimable`
permits the Safe's L1 claim. Pause normal execution while collateral is in
transit and synchronise only after both sides show the completed movement.

The integration fails closed:

- Negative, malformed or non-finite equity aborts account correction and NAV
  settlement before state is mutated.
- A public API failure does not reuse a stale external-account value for a new
  NAV.
- Negative equity is never accepted, even temporarily.

## Python API and module structure

The main integration points are deliberately small:

- `cash_manager.py` contains the pure Safe/exchange cash-allocation policy and
  the sub-account rebalance planner.
- `lighter_strategy.py` contains `decide_lighter_cash_management()`, the
  phased cash policy for a master and its sub-accounts.
- `lighter.py` creates the synthetic pairs, validates public equity and provides
  account, free-collateral and Lagoon NAV readers.
- `lighter_routing.py` prepares Safe-owned deposits and withdrawal claims,
  submits L2 transfers between accounts and key-free sub-account withdrawals
  in the normal execution pipeline.
- `transfer_verification.py` checks public Ethereum and Lighter evidence before
  an interrupted custody transfer is reconciled.
- `sync_model.py` converts a changed external-account value into a
  `BalanceUpdate` without creating an EVM trade.
- `utils.py` dispatches `correct-accounts` to the Lighter public reader.
- `ethereum_protocol_adapters.py` discovers the Lighter pair at runtime and
  wires both exchange-account and Lagoon NAV valuation.

For direct integration code, create the pair with
`create_lighter_exchange_account_pair()` and the reader with
`create_lighter_account_value_func()`. Most strategies should rely on runtime
auto-discovery rather than constructing the reader themselves.

## Authenticated SDK calls

The Lighter SDK creates short-lived bearer tokens from the delegated API key.
`eth_defi.lighter.sdk.LighterAuthTokenManager` keeps a token in memory,
refreshes it shortly before expiry and retries one HTTP 401 with a newly signed
token. Its default lifetime is ten minutes with a 30-second refresh margin.

The manager accepts any asynchronous SDK operation that receives the token, but
the caller must decide whether replay is safe. Use it for idempotent reads such
as withdrawal-history polling. Do not wrap withdrawal or order submission
unless replay safety is guaranteed. Neither tokens nor SDK exception payloads
are written to logs.

## Live tutorial

`scripts/lagoon/manual-trade-executor-lighter.py` is an executable Python
tutorial for the full mainnet lifecycle. It:

1. deploys and activates a temporary Lagoon vault with `lagoon-deploy-vault`;
2. subscribes, settles and deposits USDC to Lighter;
3. creates the exchange-account position with `correct-accounts` and checks NAV
   with `lagoon-settle`;
4. opens and closes a small ETH/USD perpetual long;
5. checks NAV before, during and after the position;
6. submits one secure withdrawal and polls it with rotating auth tokens;
7. claims USDC to the Safe and redeems all shares to the deployer.

The script asserts Safe, Lighter, share and deployer balances at every material
boundary. Its auth-token lifetime defaults to 30 seconds so a normal withdrawal
wait demonstrates rotation; production code should use the manager defaults.
The SDK is a locked trade-executor dependency. Its Git commit contains a native
signer, so any pin update requires the release owner's supply-chain review. Do
not override the pin with a separate `pip install`:

```shell
source .local-test.env
poetry install
poetry run python scripts/lagoon/manual-trade-executor-lighter.py
```

The tutorial needs `JSON_RPC_ETHEREUM` and `LIGHTER_TEST_PRIVATE_KEY`. Optional
controls include `LIGHTER_DEPOSIT_USDC`, `LIGHTER_POSITION_USDC`,
`LIGHTER_MAX_SLIPPAGE`, `LIGHTER_DEPOSIT_TIMEOUT`, `LIGHTER_WITHDRAW_TIMEOUT`,
`LIGHTER_AUTH_TOKEN_TIMEOUT` and `LIGHTER_RUN_DIR`.

Unlike the minimal single-chain deployment example above, the tutorial passes
`STRATEGY_FILE` to exercise the strategy-file deployment path and keeps its
state and runtime deployment artefact inside `LIGHTER_RUN_DIR`.

The withdrawal request itself is sent exactly once. Only the idempotent history
poll is retried or repeated. The generated API key stays in the protected
operator record and memory, and the tutorial verifies it is absent from public
artefacts.

For sub-accounts, see the [sub-account tutorials](#tutorials).

## Troubleshooting

### `check-accounts` reports a Lighter mismatch

Confirm that `LIGHTER_ACCOUNT_INDEX` points to the Safe-owned account, or that
`state.sync.deployment.lighter_accounts` records the master and every
sub-account the strategy declares, and that no deposit, withdrawal or order
settlement is still in flight. If Lighter
equity differs, run `correct-accounts` only after the public account view is
stable. If the Safe reserve differs by a completed deposit or withdrawal
amount, first establish that no transfer is still pending, then run
`correct-accounts`; Lagoon Lighter strategies reconcile both the public Lighter
equity and on-chain Safe reserve. Run `lagoon-settle` separately when the vault
needs a fresh posted NAV. Repeat `check-accounts` afterwards.

The account-set check has its own messages:

- *Could not read Lighter accounts* or *Lighter lists no accounts for
  `<safe>`*: check the network and that `VAULT_ADDRESS` is the vault whose
  Safe owns the accounts. Both fail the check; they used to pass.
- *declared Lighter accounts … hold funds but have no position*: run
  `correct-accounts`, which opens them.
- *positions track Lighter accounts … the strategy does not declare*: run
  `correct-accounts`; it closes a position once its account holds less than
  1 USDC, and otherwise names the `lighter-move-funds` command that drains it.
- *Lighter accounts … hold funds but are neither declared by the strategy nor
  tracked*: the account is outside the vault's accounting. Declare it in the
  strategy, or withdraw it to the Safe manually.

### `UnauthorizedException` during an authenticated SDK call

Lighter bearer tokens are short-lived. Use `LighterAuthTokenManager` for safe,
idempotent polling calls so it refreshes near expiry and retries one HTTP 401.
Do not blindly retry order or withdrawal submission because a timed-out request
may already have been accepted.

### Negative or malformed equity

Do not edit state to bypass the invariant. Verify the public account index and
Lighter API health. NAV posting and account correction intentionally stop
before mutation when the public value is negative, non-finite or malformed.

## See also

- [Lagoon Lighter deployment guide](../../docs/lagoon-lighter-deployment.md)
- [Derive exchange account integration](./README-Derive.md)
- [Lighter guard architecture](../../deps/web3-ethereum-defi/eth_defi/lighter/README-lighter-guard.md)
- [Lighter API authentication](https://apidocs.lighter.xyz/docs/authentication)
