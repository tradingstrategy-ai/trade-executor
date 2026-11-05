# Lagoon Lighter deployment and test trade

This guide deploys a Safe-owned Lighter account for a Lagoon vault and runs the
supported mainnet test lifecycle. `trade-executor` owns deployment, accounting
and Lagoon NAV posting; Lighter's SDK submits the ETH/USD perp order because
the executor does not yet route Lighter orders.

## What the deployer must hold

Fund the deployer/initial asset-manager address on Ethereum mainnet before the
deployment:

| Asset | Amount | Purpose |
|-------|--------|---------|
| ETH | Non-zero and sufficient for current mainnet gas | Lagoon/Safe deployment and Lighter activation transactions; gas varies. |
| Native Ethereum USDC | **1 USDC** for activation | Deployment subscribes 1 USDC and transfers all of it to Lighter, the direct Ethereum deposit minimum. The Safe holds no USDC afterwards. |

Use native Ethereum USDC at
`0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48`; bridged USDC is not suitable.

## Deploy with the Typer CLI

`lagoon-deploy-vault` creates the Lighter account owned by a fresh Ethereum
Lagoon Safe and registers its delegated API key:

```shell
export JSON_RPC_ETHEREUM="https://..."
export PRIVATE_KEY="0x..."
export DENOMINATION_ASSET="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
export VAULT_RECORD_FILE="/secure/path/vault-record.txt"
export GENERATE_LIGHTER_API_KEY=true
export LIGHTER_API_KEY_INDEX=4
export EXECUTOR_ID="lighter-vault"
export STATE_FILE="state/lighter-vault.json"

trade-executor lagoon-deploy-vault
```

To apply a 5,000 USDC daily automated-settlement budget, also set:

```shell
export LAGOON_MAX_SETTLEMENT_AMOUNT="5000"
export LAGOON_SETTLEMENT_WINDOW="86400"
```

This is a cumulative gross budget across the fixed 24-hour window, not a
one-settlement-per-day cooldown. The 1 USDC Lighter bootstrap subscription
therefore consumes 1 USDC of the first window, leaving 4,999 USDC available
for normal automated settlement.

The activation makes a Lagoon-accounted 1 USDC subscription and transfers all
of it to Lighter as the fixed 1 USDC activation deposit, leaving the Safe
empty. Add operating capital with a normal Lagoon subscription afterwards. It waits for
Lighter public state before executing `changePubKey`. It is Ethereum-only and
cannot be simulated on a private Anvil fork. Enabling it therefore spends real
gas and USDC.

The configured record path receives the public text record and a paired JSON
operator record. The JSON file is created with mode `0600` and contains the
generated private key only for an enabled activation. Back it up in the secret
manager immediately; never paste it into tickets, chat or diagnostics. The
Markdown report, text record and `state/{executor-id}.deployment.json` runtime
artefact contain the account index, API-key index, public key, transaction
hashes and observed collateral, but never the private key.

If `VAULT_RECORD_FILE` is `/secure/path/vault-record.txt`, the private key is
in the paired `/secure/path/vault-record.json` operator record. Do not print,
copy into a strategy module, pass on a command line, or add it to executor
state. Move the mode-`0600` JSON record to the secret manager after deployment.

Deployment writes the private records only after the whole command returns.
If it fails after a key has been registered on Lighter, that key is lost and
the account needs a new key through a Safe `changePubKey`, unless
`--lighter-key-backup-dir` is set: it saves each key to a mode-0600 file
before registration and deletes the files once the records are written.

### Sub-accounts

To give external traders their own Lighter sub-accounts, declare them in the
strategy's `Parameters.lighter_sub_accounts` as `SubAccount` values and pass
the strategy with `STRATEGY_FILE` instead of `DENOMINATION_ASSET` (optionally with
`LIGHTER_SUB_ACCOUNT_API_KEY_INDEX`, default 4). Deployment creates one
sub-account per declaration with the master key, registers one key on each
through the Safe and records each sub-account's id in the deployment
artefact; `init` then records the account indices in the state. Each sub-account's API key is written to its own mode-0600
`vault-record.nautilus.<id>.json` hand-over file; the executor record keeps
only the master key. Sub-accounts start empty.

A Standard Lighter account gets four sub-accounts. A master holding no funds was
refused a third with `21129 too many sub accounts`, which Lighter does not document.
Deployment keeps its 1 USDC activation deposit on the master while it creates them,
which was enough on mainnet on 2026-11-05, so do not lower it or withdraw it first.
To add a sub-account later, fund the master first. See the
[sub-account limits](https://github.com/tradingstrategy-ai/web3-ethereum-defi/blob/master/eth_defi/lighter/README-lighter-guard.md#limits).

Lighter allows 2 `L2CreateSubAccount` requests per minute per L1 address, and a
rejected request counts too, so
deployment waits for the limit: four sub-accounts take at least a minute. Use an
image later than `v1475` that includes eth-defi's `LighterSDKSession`. `v1475` and
earlier stop with HTTP 429 on the third sub-account.

If a deployment stops during Lighter activation, do not rerun it for the same
vault. Creating sub-accounts is not idempotent and a rerun deploys new contracts.
Find out how far it got before moving anything, because the funds can be in the
silo, the Safe, a Lighter account or a pending withdrawal. Owners are only added
after activation, so check that the deployer is still the Safe's sole owner with
threshold 1. If the funds are on Lighter:

1. Request each account's balance with the key-free L1 `withdraw`, through the
   module.
2. Wait about 30 minutes, then claim the pending balance to the Safe. Check the Safe
   first, because Lighter's relayer may already have paid it without a claim.
3. Transfer the Safe's USDC to the deployer with `execTransaction`, simulating it
   first.

Check that every account is empty, keep the key backup directory until the
replacement vault is live, and then deploy a new vault.

The executor funds sub-accounts from the master with same-owner L2
transfers, which Lighter did not charge for on 2026-10-26, and withdraws
them to the Safe with the key-free L1 `withdraw(subIndex)`, which took about
29 minutes on mainnet. A trader may transfer between sub-accounts with its
own key; the executor only revalues the positions, unless it is given that
key with `LIGHTER_SUB_ACCOUNT_OPERATOR_RECORD_FILES`. A strategy declares
its sub-accounts in `Parameters.lighter_sub_accounts` as `SubAccount` values
with stable ids, never Lighter account indices, and moves funds between them
by id with the `PositionManager` custody methods, see
[README-Lighter.md](../tradeexecutor/exchange_account/README-Lighter.md#strategy-setup-for-sub-accounts).

The live tutorial
[`manual-trade-executor-lighter-subaccounts.py`](../scripts/lagoon/manual-trade-executor-lighter-subaccounts.py)
needs 5 USDC plus gas: the 1 USDC bootstrap and a 4 USDC subscription. Its
deployment saves each generated key with `--lighter-key-backup-dir` in the
run directory until the records are written.
[`manual-trade-executor-lighter-transfers.py`](../scripts/lagoon/manual-trade-executor-lighter-transfers.py)
books every deposit, transfer and withdrawal as its own trade with a scripted
strategy, including transfers signed by the sub-account keys given to
`start` with `LIGHTER_SUB_ACCOUNT_OPERATOR_RECORD_FILES`, and verifies the
state afterwards; with `SIMULATE=true` it runs on an Anvil fork and the fake
Lighter sequencer. See
[README-Lighter.md](../tradeexecutor/exchange_account/README-Lighter.md#sub-accounts)
for the custody model, fees, strategy setup, recovery and the verified
mainnet results.

## Test from the command line

[`scripts/lagoon/manual-lighter-subaccounts-cli.sh`](../scripts/lagoon/manual-lighter-subaccounts-cli.sh)
is a mainnet test of a vault with a Lighter master account and two
sub-accounts that uses only `trade-executor` commands. Read it as the
runbook of such a deployment: every step explains what it does to the vault,
then runs plain commands. It deploys the vault, subscribes test capital,
funds each sub-account and moves USDC between them with
`perform-test-trade`, then withdraws every account, redeems all shares and
returns the capital to the deployer.

Between the funding and the withdrawals it stresses the accounting
commands. With capital on every account, `correct-accounts` must book
nothing, `lagoon-settle` must post a NAV equal to the Safe's and the Lighter
accounts' balances and to the state's total equity, and `check-wallet` must
pass. Then it changes the vault twice outside the executor: a trader moves
1 USDC between the sub-accounts with its own key, and the deployer sends
1 USDC straight to the Safe. After each, `check-accounts` must fail,
`correct-accounts` must book the difference and `lagoon-settle` must post
the NAV that follows.

| Variable | Required | Use |
|----------|----------|-----|
| `JSON_RPC_ETHEREUM` | Yes | Ethereum mainnet archive RPC |
| `LIGHTER_TEST_PRIVATE_KEY` | Yes | Funded deployer; the sole Safe owner and asset manager |
| `ETHERSCAN_API_KEY` | Yes | Contract verification at deployment |
| `TRADING_STRATEGY_API_KEY` | Yes | The Trading Strategy client every command creates |
| `SUBSCRIPTION_USDC` | No | Test capital in whole USDC, default 10 |
| `LIGHTER_RUN_DIR` | No | An earlier run's directory, to resume it |
| `HEARTBEAT_SECONDS` | No | Progress interval while a command runs, default 60 |

The script unsets every other `JSON_RPC_*` variable before it runs a command:
`lagoon-deploy-vault --strategy-file` connects to each one it finds and
refuses to deploy when they differ from the chains the strategy declares.

The deployer needs `1 + SUBSCRIPTION_USDC + 1` native USDC and ETH for the
deployment, its contract verification and about 30 further transactions;
the run of 2026-10-29 spent 0.0066 ETH of gas. The USDC comes back apart from any Lighter transfer fee and dust below the
1 USDC withdrawal minimum. A run takes about two hours, most of it the three
Lighter withdrawals. The script prints a timestamped banner per step, every
command before it runs and how long it took, a heartbeat while a command
runs, and the balances after every money step, all also written to
`run.log` in the run directory. A rerun with `LIGHTER_RUN_DIR` skips the
completed steps and checks the books and the balances before any money
step, so money is never moved twice. When an interruption left a state the
script cannot resolve itself, such as a transfer that moved but was not
saved, a withdrawal whose request was not recorded, or a deployment that
did not complete, it stops and prints what to do; commands to run by hand
start with `source <run directory>/env.sh`, which loads the run's public
configuration. The actions outside the executor, such
as the investor's deposit request and share claim, are subcommands of
`python -m tradeexecutor.testing.lighter_manual`.

`tests/lagoon/test_lagoon_lighter_cli_manual_e2e.py` runs the same commands
in the same order on an Ethereum fork with a simulated Lighter sequencer.

It passed live on 2026-10-28 in 1 hour 55 minutes: deployment took 15
minutes, the two sub-account L1 withdrawals 28 and 27 minutes and the
master's secure withdrawal 23 minutes. Lighter charged no transfer fee, the
deployer got all 11 USDC back, and the run spent 0.0074 ETH of gas. With
the accounting steps it passed again on 2026-10-29 in 2 hours 4 minutes:
`correct-accounts` booked nothing on clean books, `check-accounts` reported
both out-of-band changes, `correct-accounts` booked them, and every NAV
`lagoon-settle` posted matched the balances and the books. The deployer got
all 12 USDC back and the run spent 0.0066 ETH of gas.

## Arbitrum hub with Lighter public pools

A vault whose hub is not Ethereum can hold Lighter public pools through its
Ethereum satellite Safe. A pool is a vault pair on synthetic dataset chain
9998 (`lighter-pool-<index>`), created with
`tradeexecutor.ethereum.lighter.lighter_pool.create_lighter_pool_pair()`.
Its custody is the Ethereum Safe's Lighter master account (chain 1) and its
API transactions are signed for Lighter's chain 304. Strategies, funding and
deployment keep these three identities separate; pools are not
exchange-account positions.

Deploy from the strategy file with `--generate-lighter-api-key`. With a
non-Ethereum hub, `lagoon-deploy-vault` runs eth-defi's coordinated activation:
the deployer subscribes 20 USDC on the hub, 1 USDC is bridged to Ethereum
over CCTP and deposited to Lighter, and the API key is registered through the
satellite Safe. The command writes the owner-only operator record beside the
vault record, with the Ethereum Safe, its module and the hub vault. The
coordinator's activation report holds the generated key before registration;
it is kept mode 0600 beside the record and deleted once the record exists.
The universe needs `JSON_RPC_ETHEREUM`; sub-accounts are not created on this
path.

Pass the operator record with `LIGHTER_OPERATOR_RECORD_FILE` to `start` and
`perform-test-trade`. Pool trades run in
[`lighter_pool_routing.py`](../tradeexecutor/ethereum/lighter/lighter_pool_routing.py):
a buy deposits USDC from the Safe, waits for the Lighter credit and mints;
a sell burns, makes a secure withdrawal, waits until it is claimable and
claims it into the Safe. The trade is marked pending at its first
checkpoint, before any money moves, and each phase and each signed operation
is saved on the trade before submission; the normal settlement sweep resumes
a pending trade after a crash or an expired wait, and a second pool trade is
refused while one is unresolved. A crash before the first checkpoint leaves
a started trade with nothing moved, for `repair`. The account is dedicated
to the executor. A buy's amount is its total investment budget: it consumes
accounted idle collateral first and deposits only the shortfall from the Safe.
For example, a 5 USDC buy with the 1 USDC activation balance needs 4 USDC from
the Safe. A sell withdraws all collateral, separating returned idle cash from
the pool's realised profit.

Pool positions are valued at quantity × USDC per raw share from current pool
state, net of the operator's estimated fee on positive profit; during a
pending exit the burned shares are valued at the collateral they released.
Account equity is never added to that value, because whether Lighter's
account equity includes pool holdings is unverified.
Uninvested collateral is recorded separately in the executor's portfolio and
included in NAV, including before the first purchase. It is not spendable Safe
cash or available CCTP funding. The public Ethereum satellite deployment
metadata identifies the account; valuation does not require its private key.
The treasury refreshes the account and pool holdings together before posting
NAV. An ownership mismatch, failed read, changing snapshot or unresolved pool
operation refuses publication rather than using incomplete value. Resume the
saved operation before retrying settlement. `show-valuation` displays the saved
idle-collateral observation and its timestamp without making network reads.

The 20 USDC bootstrap normally leaves approximately 18.99 USDC in the hub Safe
and 1 USDC on Lighter. Any unused bridge-fee allowance remains in the Ethereum
Safe: the first treasury valuation imports its verified balance into the existing
bridge cash position as an accounting correction. Only the actual CCTP fee is a
loss. The no-fee simulation therefore reconciles to 18.99 + 0.01 + 1 = 20 USDC.
Do not also book the activation USDC as bridge cash. Pool purchase
cost includes the consumed idle collateral, so investing it does not create
profit. If no pool holding remains but idle collateral is needed to fund an
investor redemption, settlement defers while trading continues: recover it through the pool
purchase/exit lifecycle. Idle dust is measured, not automatically written off.
`check-accounts` and `correct-accounts` support one chain only; reconcile a
multichain vault with `check-wallet`, the state and the NAV.

[`scripts/lagoon/manual-multi-vault-cli.sh`](../scripts/lagoon/manual-multi-vault-cli.sh)
is the runbook: it deploys an Arbitrum vault and opens and closes a Lighter
pool, Atoma and Upshift position. `--targets lighter,upshift` finishes the
two positions without a long lock in one sitting. Atoma takes three runs of
`--targets atoma`: deposit, request and claim. New shares stay locked until the
current 72-hour epoch ends, and a withdrawal requested in epoch N can settle only
after epoch N+1 ends, so the claim becomes possible six to nine days after the
deposit at the earliest, once Atoma's operator has settled that epoch. The last
run claims and redeems the investor.
With `SIMULATE=true` the script runs the same commands on Arbitrum and Ethereum
forks with the fake Lighter sequencer and forged CCTP attestations. See the
[implementation plan](plans/arbitrum-multi-vault-live-plan.md) for minimum
funding.

The first funded run, on 3 November 2026, used vault
[`0xCF68aF3B…`](https://arbiscan.io/address/0xCF68aF3BDFC38868F473a3177be174bFD9B36e99)
and Lighter account 766589:

| Step | Observed result |
|---|---|
| Deployment with coordinated activation | 20 minutes |
| pmalt round trip | 5.999515 USDC minted 8,093 raw shares; the burn released 5.999693 USDC. With 0.000485 USDC left in the account, 6.000178 USDC was withdrawn |
| Secure withdrawal | 44 minutes from request to claim |
| CCTP, each direction | about 15–20 minutes |
| Upshift Core USDC round trip | 5 USDC in, 4.999999 USDC out |
| Atoma Index deposit | 100 USDC minted 89.82658215276 shares, locked for epoch 452 |

Plan on these when operating such a vault:

- Lighter's `entry_timestamp` is in Unix milliseconds, and its withdrawal history
  ends with an empty page that repeats the end cursor, not with an empty cursor.
- If `lagoon-settle` fails with a Hypersync `RuntimeError: inner receiver`, unset
  `HYPERSYNC_API_KEY` to use the RPC log reader and resume.
- In the 3 November run, the 1 USDC activation deposit was outside NAV until a
  pool buy invested it. The first subscription settled at a vault value of
  18.99 USDC for 20 shares. The idle-collateral accounting above corrects this
  for new settlements; it does not repair shares already issued by that run.
- Atoma upgrades its proxies often, and eth-defi refuses an unreviewed
  implementation for exits as well as deposits. Check the implementation before
  opening a position; see
  [eth-defi issue 1721](https://github.com/tradingstrategy-ai/web3-ethereum-defi/issues/1721).

The process-level strategy-cycle tests are in
[`test_multi_vault_start_e2e.py`](../tests/lagoon/test_multi_vault_start_e2e.py).
They invoke the registered `start` command with normal durable state on the
existing dual forks, including provider refusal and restart after a submitted
mint. The simulation wrapper advances UTC with monotonic elapsed time from the
fork clock; it does not add production clock controls. These tests establish
simulated execution and accounting, not live sequencer or operator availability.
Pending pool operations defer portfolio decisions, NAV publication and statistics
until the existing settlement sweep resolves them. They do not terminate `start`.
This also defers stop-loss triggers on other positions until portfolio valuation
is complete. A pending pool order expires later planned pool orders in the same
batch without allocating their capital; orders on other venues may continue.
Expired top-ups do not block later account valuation.

`lagoon-redeem` is a terminal operation for pool deployments: it requires the
executor state and public satellite metadata, refreshes the account, and refuses
remaining pool collateral, positions, pending operations or satellite cash. Close
the positions and use `reclaim-satellites` for residual satellite USDC first.
The command then uses actual hub Safe cash as NAV, without the legacy
exchange-account dust exclusion. A completed claim with an empty global queue
does not trigger another settlement, including after a restart.

## Pair a compatible strategy module

Pair the vault with a passive exchange-account strategy. It declares the public
Lighter account index and native-USDC quote asset so the executor can value the
external account, but it contains no Lighter private key and produces no Lighter
order. This follows the same external-account strategy pattern as
[Derive](../tradeexecutor/exchange_account/README-Derive.md#strategy-setup)
and [GMX](../strategies/test_only/minimal_gmx_strategy.py).

```python
from tradeexecutor.exchange_account.lighter import create_lighter_exchange_account_pair

lighter_pair = create_lighter_exchange_account_pair(
    quote=universe.get_reserve_asset(),
    account_index=123,
)
```

[`strategies/test_only/minimal_lighter_strategy.py`](../strategies/test_only/minimal_lighter_strategy.py)
is the complete module template. Set `LIGHTER_ACCOUNT_INDEX` from the public
deployment report when running it. The API key remains only in the protected
operator record.

## Automatic exchange cash management

An alpha model can return an automatic Safe/Lighter custody transfer from
`decide_trades()` by setting `lighter_cash_management=True`. It builds an
`ExchangeCashSnapshot` from `PositionManager` and the Lagoon queue, asks the
pure `ExchangeCashManager` for one decision, then returns the resulting
`TradeExecution`. The transfer uses the normal pipeline, including reserve
allocation, checkpointing, `GenericRouting`, Lagoon Safe transactions and
state accounting. Do not call the manual `lighter-move-funds` command from a
running executor.

Use the small
[`lighter_cash_management_strategy.py`](../strategies/test_only/lighter_cash_management_strategy.py)
as a reference. Its policy retains a 20 USDC Safe buffer, keeps the configured
free Lighter collateral buffer, ignores transfers below 1 USDC and uses a
one-hour withdrawal timeout configured at the `start` CLI boundary:

```python
class Parameters:
    lighter_cash_management = True
    lighter_safe_cash_buffer_usd = Decimal("20")
    lighter_free_collateral_buffer_usd = Decimal("0")
```

That single-account module keeps a module-level `LIGHTER_ACCOUNT_INDEX`, as
`strategies/lighter-ai.py` does. A strategy with sub-accounts declares them
in `Parameters` and takes every account index from the state through
`create_lighter_account_universe()`; see `docs/strategy-parameters.md`.
Runtime secrets and withdrawal settings are parsed by the CLI and passed to
routing explicitly.

Use `PositionManager.get_current_cash()` and
`PositionManager.get_exchange_account_available_balance(pair)` in
`decide_trades()`. They expose the latest treasury-synchronised Safe reserve
and the public Lighter free collateral without a direct Web3 read in the
strategy.

Deposits move only idle Safe cash above the buffer. If pending redemptions
need liquidity, pending Lagoon deposits offset that need before the manager
creates a Lighter withdrawal. The Safe target is
`max(pending redemptions - pending deposits, 0) + buffer`. Secure
withdrawals normally take about 20 minutes, so the live executor waits
synchronously and starts no later trades while the request is pending. The
one-hour timeout is a safety limit, not the expected delay. A restart
aborts safely on an unfinished transfer; reconcile verified evidence through
`correct-accounts` or `repair` before restarting. Set
`LIGHTER_OPERATOR_RECORD_FILE` to the mode-`0600` deployment JSON so the
executor can sign the authenticated withdrawal request; never put its private
key in the strategy, environment value, state file or logs.

With automatic management enabled, Lagoon skips same-cycle NAV posting and
redemption settlement when Safe plus pending Silo deposits cannot cover the
redemption. The following strategy decision can return the withdrawal through
the normal pipeline. After the claim succeeds, the next cycle synchronises the
Safe balance and settles the queue. This conservative behaviour avoids a
temporary low NAV and avoids trading against money that has not arrived.

## Run the ETH/USD test lifecycle

For an already deployed Lighter-enabled vault, the Typer command is the
supported small real-money test. It uses the deployment's protected operator
record to deposit from the Safe, open and close ETH/USD, claim the withdrawal,
and synchronise Lagoon NAV throughout. It trades on the master only; to test
the custody path of every account, including sub-accounts, use
`perform-test-trade --single-pair` or `--all-pairs` as described in
[README-Lighter.md](../tradeexecutor/exchange_account/README-Lighter.md#custody-test-trade-with-perform-test-trade).

The operational sequence is:

1. Deploy with `GENERATE_LIGHTER_API_KEY=true`. The deployment subscribes 1
   USDC to Lagoon and transfers all of it from the Safe to Lighter, leaving the
   Safe empty. The vault is no longer eligible for `lagoon-first-deposit`.
2. Run `init`, then `correct-accounts --process-redemption`. The runtime
   deployment artefact supplies the original vault block, so the command records
   the 1 USDC Lagoon investor flow and the public 1 USDC Lighter exchange
   account without a repair or transfer script.
3. Run `scripts/lagoon/deposit-and-settle.py 19` to subscribe 19 USDC of test
   capital through Lagoon and settle it into the Safe.
4. Run `lagoon-lighter-test-trade --lighter-test-deposit-usdc 19` to make the bounded Lighter deposit, ETH/USD
   round trip and secure withdrawal back to the Safe.

For automated Safe cash management, a secure withdrawal writes its public
Lighter request ID to executor state before waiting. A later `start` aborts on
that unfinished transfer rather than risking a duplicate withdrawal request.
Use `check-accounts` to inspect recorded evidence, then
`correct-accounts` or `repair` to reconcile a transfer whose public Ethereum
and Lighter evidence is complete. A Safe claim already broadcast to Ethereum
also requires this reconciliation before normal execution can continue.
If Lighter cannot release enough free collateral, Lagoon keeps the redemption
deferred until an operator restores liquidity rather than posting a reduced NAV.

```shell
source .local-test.env
export JSON_RPC_ETHEREUM="https://..."
export PRIVATE_KEY="0x..."  # Funded Safe owner / asset manager; never print it
export EXECUTOR_ID="lighter-vault"
export STRATEGY_FILE="strategies/test_only/minimal_lighter_strategy.py"
export STATE_FILE="state/lighter-vault.json"
export VAULT_ADDRESS="0x..."
export VAULT_ADAPTER_ADDRESS="0x..."
export ASSET_MANAGEMENT_MODE="lagoon"
export LIGHTER_ACCOUNT_INDEX="..."  # Public index in deployment report
export LIGHTER_OPERATOR_RECORD_FILE="/secure/lighter/lighter-vault.json"
# Fresh Lighter deployment: 1 USDC is already on Lighter, so add 19 USDC.
export LIGHTER_TEST_DEPOSIT_USDC="19"
# export LIGHTER_TEST_POSITION_USDC="..."

trade-executor init
trade-executor correct-accounts --process-redemption
poetry run python scripts/lagoon/deposit-and-settle.py 19
trade-executor lagoon-lighter-test-trade --lighter-test-deposit-usdc 19
```

The generic test deposit defaults to 20 USDC. For a fresh Lighter-enabled
deployment, pass 19 USDC because its initial 1 USDC Lighter collateral brings
the test to 20 USDC total. The direct Ethereum Lighter contract has a 1 USDC
minimum deposit. Live ETH/USD metadata inspected on 2026-09-16 required
both 0.005 ETH and 10 USDC of notional; 0.005 ETH was about 12 USDC at the time.
Thus 20 USDC clears the zk deposit minimum and leaves modest collateral
headroom for the automatically sized ETH perpetual order. Market limits are
fetched at runtime and may change. The Safe must hold at least the selected
deposit amount, so a fresh deployment first subscribes it with
`deposit-and-settle.py`.

To add further capital to an existing vault, stop the executor and run the
reusable Lagoon subscription script. From a source checkout use:

```shell
poetry run python scripts/lagoon/deposit-and-settle.py 20
```

The production Docker image is intentionally slim and does not include Poetry.
From the strategies Compose deployment, use its installed Python interpreter:

```shell
docker compose run --rm \
  --entrypoint /usr/local/bin/python \
  lighter-ai \
  scripts/lagoon/deposit-and-settle.py 20
```

The script uses `PRIVATE_KEY` as both depositor and authorised asset manager.
It requires the same `JSON_RPC_ETHEREUM`, `EXECUTOR_ID`, `STRATEGY_FILE`,
`STATE_FILE`, `VAULT_ADDRESS`, `VAULT_ADAPTER_ADDRESS` and
`ASSET_MANAGEMENT_MODE=lagoon` environment as the executor. It never transfers
USDC directly to the Safe. Instead, it requests a Lagoon deposit, runs the real
`lagoon-settle` vault sync so the Safe and executor reserve receive the capital,
and claims the deployer's shares. A rerun only resumes an exact matching pending
or claimable deposit; it refuses a different amount. `lagoon-settle` settles
the vault's eligible investor queue, so do not run this operator script while
another investor's request needs separate handling.

A Lighter-enabled deployment subscribes 1 USDC to Lagoon and transfers all
of it to activate the Safe-owned Lighter account, leaving the Safe empty. Use
this script for the first operating capital; it accepts the zero-Safe
baseline. Until that subscription settles, the Safe holds no cash to pay a
Lagoon redemption. `lagoon-first-deposit` remains reserved for a vault with no
NAV or shares.

The Lighter AI Yubi deployment convention maps `~/secrets/lighter` to
`/secure-lighter` inside the manual-command container. Its mode-`0600`
operator record is conventionally
`~/secrets/lighter/lighter-ai-vault-info.json`; use
`LIGHTER_OPERATOR_RECORD_FILE=/secure-lighter/lighter-ai-vault-info.json` for
the manual command. The path is public configuration, but the JSON contents are
secret and must never be copied into an environment variable or a report.

The command runs the following sequence against the configured vault:

1. deposits Safe USDC to Lighter and posts a Lagoon NAV checkpoint;
2. opens a small ETH/USD long and posts another checkpoint;
3. closes the long with a reduce-only order and posts another checkpoint;
4. requests and claims the Lighter withdrawal to the Safe; and
5. posts the final Lagoon NAV checkpoint.

It verifies public Lighter equity and Lagoon NAV are never negative and writes
an owner-only recovery journal beside the state file. Re-run the same command
after a failure only after the operator has reviewed any ambiguous journal
phase. If a withdrawal request was not accepted, confirm its absence from
withdrawal history and change `withdrawal_requested` back to `closed` in the
journal before retrying. Stop the normal strategy executor while this manual
lifecycle runs.

The standalone
`scripts/lagoon/manual-trade-executor-lighter.py` tutorial still deploys and
redeems an isolated vault. The Lighter SDK is already pinned in this project's
Poetry dependencies, so run `poetry install` rather than a manual SDK install.

For an already deployed vault, run the Typer accounting cycle only after a
deposit, fill or withdrawal has completed on both Lighter and the Safe:

```shell
trade-executor correct-accounts
trade-executor check-accounts
trade-executor lagoon-settle
```

Never perform this cycle while a collateral movement or order settlement is in
flight.

### Secure withdrawals and Safe custody

Before requesting a secure withdrawal, query Lighter's dynamic
`withdrawalDelay` value in seconds. `lagoon-lighter-test-trade` does this and
logs the current delay immediately before submitting its withdrawal request.
The value is an operator estimate, not a completion guarantee: wait until the
matching `withdraw_history` row becomes `claimable`, then claim through the
Lagoon Safe module.

Fast USDC withdrawals are not supported for this vault shape. Lighter requires
the L1 account's Ethereum EOA private key for a fast withdrawal, but the
account owner is a Safe contract and has no EOA private key. Do not use the
deployer or asset-manager key as a substitute: neither owns the Safe's Lighter
account. Use the secure withdrawal and later Safe-gated L1 claim instead.

During the delay, Lighter can show zero account equity while USDC has not yet
arrived in the Safe. Pause normal strategy execution, `correct-accounts` and
Lagoon NAV settlement until the claim has completed; otherwise AlphaModel
sizing can see a temporarily understated reserve and NAV.

## Strategy accounting

The pair is an external exchange-account position. Runtime valuation uses
Lighter's unauthenticated public API, so the executor never needs the API
private key. The Lagoon NAV formula is:

```text
Safe USDC balance at the treasury-sync block
+ Lighter total_asset_value
+ pending Lagoon settlement value
```

The Safe balance is reconciled separately and is not included in the external
account position. Stop the executor while collateral is moving between the
Safe and Lighter, including Lighter's dynamic secure-withdrawal delay, and
resume only after the public account response and Safe balance reflect the
settlement. A negative or malformed Lighter equity response aborts the cycle;
it is never cached, written to state or posted as NAV.

Each Lighter account's position reports a return and a USD profit that do
not depend on the capital moved in and out of it: flows mint and burn
internal shares, and the USD profit counts the capital withdrawn and
invested. See
[Profit per account](../tradeexecutor/exchange_account/README-Lighter.md#profit-per-account).
