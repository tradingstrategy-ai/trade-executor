# Lagoon Lighter sub-account manual test from the command line

## Objective

Add a live Ethereum mainnet shell script,
`scripts/lagoon/manual-lighter-subaccounts-cli.sh`, that an operator can read
as documentation of what a Lighter vault deployment needs, and run to test the
whole custody lifecycle with real money and only `trade-executor` CLI
commands:

1. deploy a Lagoon vault whose Safe owns a Lighter master account and two
   sub-accounts, with an API key on each;
2. check the deployer is funded;
3. subscribe the deployer's USDC to the vault, settle it and claim the shares;
4. move USDC to the first sub-account with `perform-test-trade`;
5. move USDC to the second sub-account with `perform-test-trade`;
6. move USDC between the two sub-accounts with `perform-test-trade`;
7. unwind: withdraw every Lighter account to the Safe, redeem all shares and
   sweep the Safe, so the deployer gets all capital back apart from Lighter
   dust and gas.

The existing Python tutorials (`manual-trade-executor-lighter-subaccounts.py`
and `manual-trade-executor-lighter-transfers.py`) drive the same accounts
through `start` cycles and in-process Typer calls. This script is the
operator's view: every action is a shell command, and the few actions outside
the executor CLI go through one small testing module called from the shell.

## Prerequisites and inputs

The script runs from the repository root with `poetry run`. It does not
source `.local-test.env`; the operator exports the variables below first, as
the `~/code/strategies` deployment scripts expect.

| Variable | Required | Use |
|----------|----------|-----|
| `JSON_RPC_ETHEREUM` | Yes | Ethereum mainnet archive RPC; `lagoon-deploy-vault` and every later command. Several space-separated URLs are allowed. |
| `LIGHTER_TEST_PRIVATE_KEY` | Yes | Funded deployer, sole Safe owner and asset manager. Exported as `PRIVATE_KEY` for every command; never printed. The existing tutorials and `.local-test.env` already use this name. |
| `ETHERSCAN_API_KEY` | Yes | Contract verification during `lagoon-deploy-vault`, as in `deploy-lighter-ai.sh`. |
| `TRADING_STRATEGY_API_KEY` | Yes | The Trading Strategy client every command creates. The strategy has no market data, but the commands are only tested with a client. |
| `LIGHTER_RUN_DIR` | No | Resume an earlier run: state, artefacts, records and step markers live here. Default `~/.tradingstrategy/examples/lighter-cli-<timestamp>`. |
| `SUBSCRIPTION_USDC` | No | Test capital subscribed after deployment. Default `10`. |
| `LIGHTER_API_KEY_INDEX` | No | API key slot on the master and each sub-account. Default `4`. |

The operator asked for `LIGHTER_TEST_API_KEY`; the repository and the test
environment name the deployer key `LIGHTER_TEST_PRIVATE_KEY`, and the script
uses that name. `LIGHTER_TEST_ACCOUNT_API_KEY` in the test environment is the
key of an existing shared Lighter account and plays no part here: the
deployment generates fresh keys for the new Safe-owned accounts.

The deployer must hold:

| Asset | Amount | Why |
|-------|--------|-----|
| ETH | A full vault deployment with verification plus about 20 transactions; about 0.011 ETH in the comparable 2026-10-26 tutorial run | Safe, vault and guard deployment, Lighter activation, three `changePubKey` calls, deposits, L1 withdrawals, settlement and redemption. |
| Native Ethereum USDC | `1 + SUBSCRIPTION_USDC`, 11 by default | 1 USDC is the fixed Lighter activation subscription made by the deployment; the rest is the test capital. |

`USDC` means native Ethereum USDC, `0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48`.

Tools: `poetry`, `jq` (reads the deployment artefact), the `trade-executor`
console script through `poetry run`.

## File layout of a run

Everything a run produces lives under `$RUN_DIR`, created with mode 0700:

| Path | Written by | Contents |
|------|------------|----------|
| `state/lighter-cli.json` | `init` and every state-changing command | Executor state. `STATE_FILE` points here. |
| `state/lighter-cli.deployment.json` | `lagoon-deploy-vault` | Public deployment artefact. The executor resolves it as a sibling of `STATE_FILE`, so it must stay beside the state file. |
| `vault-record.txt` | `lagoon-deploy-vault` | Public deployment report. |
| `vault-record.json` | `lagoon-deploy-vault` | Master API key, mode 0600. `LIGHTER_OPERATOR_RECORD_FILE`. |
| `vault-record.nautilus.trend.json`, `vault-record.nautilus.arb.json` | `lagoon-deploy-vault` | One sub-account key each, mode 0600. `LIGHTER_SUB_ACCOUNT_OPERATOR_RECORD_FILES`. |
| `key-backup/` | `lagoon-deploy-vault` with the new backup option | Each generated key as soon as it exists; emptied on success. |
| `start.env` | step 0 | Deployer's USDC and ETH before the run, for the final reconciliation. |
| `deployment.env` | step 2 | Vault, module and account indices read from the artefact with `jq`. |
| `step-N.done` | the script | Completed-step markers. |

## Strategy module

`strategies/test_only/lighter_multi_subaccount_strategy.py` is used as is.
It declares two sub-accounts, `trend` and `arb`, with 50/50 weights and the
automatic cash policy. The script never runs `start`, so the policy does not
act; only the module's `Parameters.lighter_sub_accounts` and its universe
matter:

- `lagoon-deploy-vault --strategy-file` reads the declared sub-accounts and
  creates one Lighter sub-account per declaration;
- `init` records each id's Lighter account index in the state;
- `perform-test-trade` builds the universe from the state, one pair per
  account, with no market data. The pairs are named `LIGHTER-ACCOUNT` for the
  master and `LIGHTER-<id>` for a sub-account, on the `lighter` exchange.

No modification is needed. `lighter_scripted_transfers_strategy.py` would
also deploy the same accounts, but its `decide_trades()` runs a fixed
transfer script if anyone runs `start` against the vault, so it is the wrong
module to leave configured in an operator's environment.

## Code changes needed before the script can run

### 1. `lagoon-settle`: build the universe from the state

`lagoon-settle` constructs the trading universe without passing the loaded
state (`tradeexecutor/cli/commands/lagoon_settle.py`, the
`construct_universe()` call). For a strategy with sub-accounts the universe
then gets placeholder account indices, and the Lighter NAV reader rejects the
real, funded master as an account the strategy does not declare
(`tradeexecutor/exchange_account/lighter.py`, the ownership check in the
equity reader). The Python tutorials avoid the command and settle a
subscription with `correct-accounts --process-redemption` instead.

Fix the command: pass the state to universe construction, as
`perform-test-trade` does, so the universe resolves the deployed indices.
Cover it with a simulate-mode test that settles a subscription for a
two-sub-account vault. Until the fix lands, the script's step 4 can use
`correct-accounts --process-redemption`, which the tutorials have run live.

### 2. `perform-test-trade`: transfer between two accounts

Step 7 moves USDC from `trend` to `arb`. `PositionManager.transfer_lighter_funds(source, destination, amount)`
already books this as one trade, signed with the source's key when the
source's hand-over file is in `LIGHTER_SUB_ACCOUNT_OPERATOR_RECORD_FILES`,
which `perform-test-trade` already accepts. Add:

```text
--lighter-transfer SOURCE:DESTINATION
                        Move --amount USDC from one Lighter account to another
                        and, unless --buy-only, back again.
```

Implementation next to `_make_exchange_account_test_trade()` in
`tradeexecutor/cli/testtrade.py`: validate both ids against the universe's
Lighter pairs (`master` is a valid source, signed with the operator record,
which is how a half-done funding step is completed, see "Resuming a run
safely"), then execute `transfer_lighter_funds(source, destination,
amount)`; without `--buy-only` execute the reverse transfer. Before either
leg, check that every signer the whole round trip needs is configured: the
source's record, and the destination's record too when the funds come back.
The routing layer checks only the sender of each individual transfer, so a
missing destination key would otherwise fail after the first leg has moved
the funds. The option is mutually exclusive with `--pair`, `--single-pair`,
`--all-pairs` and `--all-vaults`. Lighter charges no fee between accounts of
one owner; the fee actually charged is booked from Lighter's transfer history
as for any transfer.

Funding one sub-account needs no new option: `--pair "(ethereum, lighter,
LIGHTER-trend, USDC)"` resolves the sub-account's pair through the
universe's exchange and pair lookup, and `--buy-only` already deposits Safe
to master, transfers master to the sub-account and skips the withdrawal. The
simulate-mode test below exercises this `--pair` form so a regression in the
lookup is caught.

### 3. `lagoon-deploy-vault`: save keys as they are generated

Deployment writes the master and sub-account private keys only after the
deployment helper returns. The Python tutorials patch the key generator to
save each key to a mode-0600 temporary file first; a shell script cannot.
Add:

```text
--lighter-key-backup-dir DIR    (env LIGHTER_KEY_BACKUP_DIR)
```

When set, each generated key is written to `DIR/lighter-api-key-<n>-<public
key prefix>.json`, created exclusively with mode 0600 before the key is
registered on Lighter, so a retry can never overwrite earlier recovery
material. The files are removed only after every record has been written;
on failure they stay. This moves the tutorials' `backup_generated_keys()`
into the command, where production deployments can use it too. A lost key
needs a Safe `changePubKey` to recover and does not lose Safe-owned funds,
but a mainnet script that can be rerun should not depend on that recovery.

### 4. `lighter-move-funds`: stdin answers

The unwind withdraws each account with `lighter-move-funds`, which asks in
order for the direction, the amount and a confirmation through
`typer.prompt()`, after resuming any pending withdrawal it finds. The script
answers over stdin:

```shell
printf 'w\n%s\ny\n' "$amount" | poetry run trade-executor lighter-move-funds
```

This works and needs no change. Its timeout variable is
`LIGHTER_WITHDRAW_TIMEOUT`; `perform-test-trade` reads
`LIGHTER_WITHDRAWAL_TIMEOUT`. The script exports both. Non-interactive
options for the command are optional scope.

## Interactive checkpoints

Three commands prompt when `UNIT_TESTING` is not set, and the script runs
without it so that backups and prompts behave as in production:

- `lagoon-deploy-vault` asks once to confirm the deployment, including the
  Lighter account and key creation.
- `correct-accounts` prints each correction and requires an exact `y`; it
  does not prompt when there is nothing to correct.
- `lighter-move-funds` asks for the direction, the amount and a
  confirmation, unless it resumes a saved withdrawal, which does not prompt.

The script answers through `run --stdin` and prints the answers with the
command. Every other command gets `/dev/null` as standard input, so an
unexpected prompt fails instead of hanging.

## Testing module called from the shell

New module `tradeexecutor/testing/lighter_manual.py`, run as
`poetry run python -m tradeexecutor.testing.lighter_manual <command>`. It
holds only what no CLI command does, reuses the helpers in
`tradeexecutor/testing/lighter_tutorial.py` where they do not need a
`Tutorial` object, and reads its configuration from the same environment
variables as the CLI commands (`JSON_RPC_ETHEREUM`, `PRIVATE_KEY`,
`VAULT_ADDRESS`, `VAULT_ADAPTER_ADDRESS`, `STATE_FILE`). It never prints a
private key or an RPC URL.

| Command | Does | Replaces |
|---------|------|----------|
| `check-funds --usdc N` | Checks the deployer's ETH is non-zero and its native USDC is at least `N`; prints both and writes `start.env`. Exit 1 if short. | The funds check in `prepare_tutorial()`. |
| `request-deposit AMOUNT` | Approves and calls `requestDeposit` from the deployer, unless a request is pending, a deposit is claimable or the deployer already holds shares. Prints one word, `requested`, `pending`, `claimable` or `held`, so the script knows whether to settle. | The onchain half of `scripts/lagoon/deposit-and-settle.py`, made safe to rerun. |
| `claim-shares` | Calls `finalise_deposit` once `maxDeposit(deployer) > 0`; a no-op when the shares are already held. Prints the share balance. | The claim in `deposit-and-settle.py`. |
| `balances [--json]` | Prints the deployer's USDC and ETH, the Safe's USDC and every Safe-owned Lighter account's equity and free collateral from the public API. | `fetch_lighter_balances()` and the log lines in `run_start()`. |
| `expect-balances safe=6 master=1 trend=1 arb=3 [--tolerance 0.01]` | Exits 0 when the observed Safe USDC and Lighter equities match, 1 otherwise. `arb<=1` accepts any balance up to the bound, for accounts left below the withdrawal minimum. Used after every money step as the assertion and before every money step as the resume guard. | The balance assertions in the tutorials. |
| `withdrawal-plan ACCOUNT_INDEX` | Prints `resume` when the state has a withdrawal in flight on the account, `withdraw <amount>` with its free USDC, or `skip` below Lighter's 1 USDC withdrawal minimum. Equity alone cannot decide: a submitted withdrawal already reads zero. | Reading observed balances instead of planned numbers. |
| `redeem-status` | Prints `redeem` while the deployer holds shares or has a redemption in progress, `done` otherwise, so a rerun of step 11 does not call `lagoon-redeem` with nothing to redeem. | The checks in `redeem_all_shares()`. |
| `sweep-safe` | Checks no shares are outstanding and nothing is claimable, then moves the Safe's USDC and ETH to the deployer with `sweep_safe_to_owner()`. | `sweep_safe()`. |
| `reconcile` | Explains the deployer's USDC and ETH change since `start.env`: fees from the state, Lighter dust, gas; fails on an unexplained difference above 2 raw units. | `reconcile_deployer_balances()`. |

`lagoon-settle` settles the deposit and `lagoon-redeem` requests, settles and
claims the redemption, so neither needs Python.

## Resuming a run safely

`perform-test-trade` saves the state only when it finishes, and the
deployer's deposit request stops being pending once settled. Completed-step
markers alone would therefore repeat a money movement after an interruption:
a rerun of a step whose command moved funds but never saved. The script
guards the money-moving steps with observed balances and idempotent
commands instead, and it never moves money in a state it cannot name.

**Recovering by hand.** The script writes `env.sh` in the run directory
with the run's public configuration. Every recovery instruction it prints
starts with `source $RUN_DIR/env.sh`, so a command run by hand acts on this
run and not on whatever the operator's shell held. Secrets are not stored:
the file refers to `LIGHTER_TEST_PRIVATE_KEY`, `JSON_RPC_ETHEREUM` and
`TRADING_STRATEGY_API_KEY` in the operator's shell.

**Deployment, step 1.** The deployment spends gas and the 1 USDC activation
deposit before it generates any key, so neither key files nor records can
show that an attempt was made. The script writes `deploy-started` before
running `lagoon-deploy-vault`. A rerun skips the step when the deployment
artefact exists, and stops when only the marker exists: the operator
recovers or abandons that deployment and starts a new run directory.

**Subscription, step 4.** All three commands are safe to rerun and run on
every pass: `request-deposit` records its request in a marker file and does
not request again; `lagoon-settle` runs only when `request-deposit` reports
`requested` or `pending`; `claim-shares` is a no-op when nothing is
claimable. An interruption between settlement and the claim is completed by
the rerun.

**Funding and transfer, steps 5 to 7.** Each `perform-test-trade` call is
`--buy-only`, so no withdrawal can be interrupted. Before the command:

1. `check-accounts` must pass. When it fails, the script names the state
   before it stops: the step's target balances mean its money moved but was
   not saved, and halfway balances mean a funding stopped after its Safe
   deposit. Either way the instruction is to book the movement with
   `correct-accounts` and rerun.
2. `expect-balances` with the step's **target** balances. If they already
   hold, the step is marked done and skipped: its money already moved.
3. Otherwise `expect-balances` with the step's **starting** balances must
   hold.
4. With halfway balances, `(8, 3, 0, 0)` for step 5, the deposit is booked
   and only the transfer is missing: the script prints
   `perform-test-trade --lighter-transfer master:trend --buy-only --amount 2`
   for the operator to run, and the next rerun finds the target balances.
5. Any other state stops the script with the balance table.

After the command, `check-accounts` and `expect-balances` with the target
balances must pass before the marker is written.

**Withdrawals, steps 8 to 10.** `lighter-move-funds` saves a withdrawal as
a pending trade before it submits the request and records the request id
after, and `check-accounts` fails while it is pending, so the
`check-accounts` precondition does not apply here. `withdrawal-plan` decides:

- a pending withdrawal with a recorded request id: `resume`, and
  `lighter-move-funds` waits, claims and books it without a second request;
- a pending withdrawal without one: `unconfirmed`, and the script stops.
  Only the Safe's transactions (sub-account L1 withdraw) or the master's
  Lighter withdrawal history can tell whether it was sent, and README-Lighter
  "Recovery" applies;
- otherwise `withdraw <amount>` with the account's free USDC, or `skip`
  below the 1 USDC minimum.

Each withdrawal is its own step with its own marker and reads its own
amount, so a second withdrawal never repeats a completed one and never
reuses another account's amount. A skipped account's balance is left to
`lagoon-redeem`, which leaves Lighter balances totalling under 5 USDC out
of the NAV.

## Test steps

Amounts are chosen against Lighter's limits: 1 USDC minimum transfer and
withdrawal, 1 USDC minimum direct Ethereum deposit. Expected balances assume
a zero fee between accounts of one owner, which the live tutorial observed on
2026-10-26; `expect-balances` tolerates 0.01 USDC. Withdrawal amounts are
read from observed balances, never from the plan, so a fee that does appear
cannot leave an account below its withdrawal minimum. The deployment's
activation leaves 1 USDC on the master and the Safe empty.

| # | Step | Command or action | Expected balances afterwards (Safe, master, trend, arb) |
|---|------|-------------------|---------------------------------------------------------|
| 0 | Check prerequisites | Shell: required variables set, `jq` and `poetry` found, `SIMULATE` not `true`; `lighter_manual check-funds --usdc 11`, which also requires 0.02 ETH by default and writes `start.env`. | Deployer has 0.02 ETH and 11 USDC or more. |
| 1 | Deploy the vault with two keyed sub-accounts | `printf 'y\ny\n' \| trade-executor lagoon-deploy-vault --strategy-file strategies/test_only/lighter_multi_subaccount_strategy.py --generate-lighter-api-key --lighter-api-key-index 4 --lighter-sub-account-api-key-index 4 --vault-record-file "$RUN_DIR/vault-record.txt" --lighter-key-backup-dir "$RUN_DIR/key-backup" --fund-name "Lighter CLI test" --fund-symbol LCLI --multisig-owners "$DEPLOYER" --any-asset --performance-fee 0 --management-fee 0 --etherscan-api-key "$ETHERSCAN_API_KEY"` with `EXECUTOR_ID=lighter-cli`, `STATE_FILE=$RUN_DIR/state/lighter-cli.json`, `PRIVATE_KEY`, `JSON_RPC_ETHEREUM`, `SIMULATE=false`. | 0, 1, 0, 0. Records and the artefact written as in the layout table. |
| 2 | Read the deployment | `jq` on `$RUN_DIR/state/lighter-cli.deployment.json` writes `deployment.env`: `VAULT_ADDRESS`, `VAULT_ADAPTER_ADDRESS` (module), `LIGHTER_MASTER_INDEX`, `LIGHTER_TREND_INDEX`, `LIGHTER_ARB_INDEX`; the script sources it and exports `ASSET_MANAGEMENT_MODE=lagoon`, `LIGHTER_OPERATOR_RECORD_FILE`, `LIGHTER_SUB_ACCOUNT_OPERATOR_RECORD_FILES`. | Every later command has its configuration. |
| 3 | Initialise state and account for the activation | `trade-executor init`; `printf 'y\n' \| trade-executor correct-accounts --process-redemption`; `trade-executor check-accounts`. | 0, 1, 0, 0. State has one position per account; the 1 USDC activation flow is booked. |
| 4 | Subscribe test capital and claim the shares | `lighter_manual request-deposit 10`; if it printed `requested` or `pending`, `trade-executor lagoon-settle` (after change 1; until then `printf 'y\n' \| trade-executor correct-accounts --process-redemption`); `lighter_manual claim-shares`; `trade-executor check-accounts`. | 10, 1, 0, 0. Deployer holds the shares. |
| 5 | Fund the first sub-account | `trade-executor perform-test-trade --pair "(ethereum, lighter, LIGHTER-trend, USDC)" --buy-only --amount 2`. | 8, 1, 2, 0. Two trades: a master deposit and a `trend` funding. |
| 6 | Fund the second sub-account | `trade-executor perform-test-trade --pair "(ethereum, lighter, LIGHTER-arb, USDC)" --buy-only --amount 2`. | 6, 1, 2, 2. |
| 7 | Transfer between sub-accounts | `trade-executor perform-test-trade --lighter-transfer trend:arb --buy-only --amount 1`, with both hand-over files in `LIGHTER_SUB_ACCOUNT_OPERATOR_RECORD_FILES`. | 6, 1, 1, 3. One trade on `arb` with the amount booked from `trend`. |
| 8 | Withdraw `arb` to the Safe | `plan=$(lighter_manual withdrawal-plan "$LIGHTER_ARB_INDEX")`; `printf 'w\n%s\ny\n' "$amount" \| trade-executor lighter-move-funds` with `LIGHTER_ACCOUNT_INDEX=$LIGHTER_ARB_INDEX`. The L1 `withdraw(subIndex)` sends the given amount, the whole observed balance here; it took about 29 minutes live. | 9, 1, 1, 0. |
| 9 | Withdraw `trend` to the Safe | Same, reading `trend`'s own amount, with `LIGHTER_ACCOUNT_INDEX=$LIGHTER_TREND_INDEX`. | 10, 1, 0, 0. |
| 10 | Withdraw the master to the Safe | `plan=$(lighter_manual withdrawal-plan "$LIGHTER_MASTER_INDEX")`; `printf 'w\n%s\ny\n' "$amount" \| trade-executor lighter-move-funds` without `LIGHTER_ACCOUNT_INDEX`. The secure withdrawal requests the given amount and leaves nothing behind; it took about 20 minutes live. | 11, 0, 0, 0. An account `withdrawal-plan` skipped keeps its sub-minimum balance and its target becomes `<=1`. |
| 11 | Redeem all shares | `trade-executor lagoon-redeem` requests redemption of every deployer share, posts NAV, settles and claims. With `STATE_FILE` set it finds every recorded Lighter account itself and writes off aggregate Lighter equity below 5 USDC from the NAV; `--lighter-account-index` is not needed. `trade-executor check-accounts`. | Deployer holds 11 USDC minus fees and dust; vault total supply 0; Safe 0. |
| 12 | Sweep and reconcile | `lighter_manual sweep-safe`; `lighter_manual reconcile`. | Safe holds nothing; unexplained USDC at most 2 raw units; gas spent printed. |

Steps 4 to 10 use the guards in "Resuming a run safely", and steps 5 to 10
are followed by `check-accounts`; the table omits those for brevity.

Steps 8 to 10 are the slow part: about an hour and twenty minutes of
waiting, in which the commands poll with their own timeouts
(`LIGHTER_WITHDRAW_TIMEOUT`). The script prints the time each withdrawal
started so an operator can estimate the wait. The three withdrawals run one
after another.

Alternatives for steps 8 to 10: `perform-test-trade --all-pairs --amount 1`
without `--buy-only` round-trips every account, but it deposits first and
withdraws only its own amount, so it does not drain the balances left by
steps 5 to 7. One or more `start` cycles after requesting the redemption
would drain every account through the strategy's cash policy, which is what
`redeem_all_shares()` does in the Python tutorial; it is not used here
because the script is meant to show the operator commands, and a `start`
cycle also executes whatever else the strategy decides.

## Script structure

`scripts/lagoon/manual-lighter-subaccounts-cli.sh`, following
`~/code/strategies/deploy/deploy-lighter-ai.sh`:

- `#!/bin/bash` and `set -euo pipefail`; a header comment stating what the
  script does, what it costs, that it is mainnet only, which commands prompt,
  and how to resume.
- A **prerequisites** block: `: "${JSON_RPC_ETHEREUM:?...}"` for each
  required variable, `SIMULATE` refused unless `false`, `command -v jq`.
- A **configuration** block with every exported variable the commands read,
  each with a one-line comment saying which command uses it and why:
  `EXECUTOR_ID`, `STATE_FILE`, `STRATEGY_FILE`, `PRIVATE_KEY` (from
  `LIGHTER_TEST_PRIVATE_KEY`), `ASSET_MANAGEMENT_MODE`, `VAULT_RECORD_FILE`,
  `LIGHTER_KEY_BACKUP_DIR`, `LIGHTER_OPERATOR_RECORD_FILE`,
  `LIGHTER_SUB_ACCOUNT_OPERATOR_RECORD_FILES`, `LIGHTER_WITHDRAWAL_TIMEOUT`,
  `LIGHTER_WITHDRAW_TIMEOUT`, `LIGHTER_TRANSFER_TIMEOUT`, `LOG_LEVEL`.
  `UNIT_TESTING` stays unset.
- A small set of helpers, described in "Observability" below: `log`,
  `step`, `done_step`, `run`, `guard_money_step` and the failure trap.
- Numbered sections 0 to 12 as in the table, each a short comment paragraph
  explaining what the step does to the vault and why, then the commands.
  Every command is a `run poetry run trade-executor ...` or
  `run poetry run python -m tradeexecutor.testing.lighter_manual ...` line,
  with options spelled out rather than hidden in variables, so the script
  reads as the deployment runbook.
- No loops and no inline Python.

Private material stays in `$RUN_DIR` with mode 0700 and the mode-0600
records; the script never echoes `PRIVATE_KEY`, the RPC URL or a record's
contents. `set -x` is not used. `ETHERSCAN_API_KEY` reaches
`lagoon-deploy-vault` through its environment variable rather than
`--etherscan-api-key`, so echoing the command line does not print it.

## Observability

The run takes well over an hour, most of it waiting on Lighter. At any
moment the operator must be able to tell what the script is doing, how long
it has been doing it, how long that normally takes, and, after a failure,
what failed and how to resume.

**One log of the whole run.** The script sends everything it and the
commands print to the terminal and to `$RUN_DIR/run.log` (`exec > >(tee -a
...) 2>&1`), and prints the log path at the start. Each run appends a
header with the start time, the git commit and the run directory, so a
resumed run's log reads as one history.

**Timestamped messages.** `log "message"` prints `[HH:MM:SS UTC] message`.
Every line the script prints itself goes through it.

**Step banners.** `step N "title"` prints a banner with the step number out
of 12, the title and the time, or `step N already done, skipping` when its
marker exists. `done_step N` prints `step N done in Xm Ys` and writes the
marker. The banner of a slow step also prints its typical duration, for
example `Lighter L1 sub-account withdrawal: typically 25 to 35 minutes`.

**Every command echoed and timed.** `run <command...>` prints `→ <command
line>` before running it and `✓ finished in Xs` or `✗ failed with exit code
N after Xs` afterwards. Secrets are never on a command line, so the echo is
safe. Executor commands run with `LOG_LEVEL=info`, so their own progress
lines, such as a transaction broadcast, a Lighter transfer submitted or a
withdrawal waiting to become claimable, appear in the log as they happen.

**Heartbeat during long waits.** While a command runs, `run` starts a
background heartbeat that prints `… still running <command name> (Nm
elapsed)` every 60 seconds and stops when the command exits. A long
withdrawal therefore shows signs of life even when the command itself is
quiet, and a stuck command is visible as a heartbeat without any command
output between beats. The heartbeat interval is `HEARTBEAT_SECONDS`,
default 60.

**Balances after every money step.** Steps 3 to 12 end with
`lighter_manual balances`, a table of the deployer's USDC and ETH, the Safe's
USDC and each Lighter account's equity and free collateral, labelled
`master`, `trend` and `arb`. The guard prints the expected and observed
values side by side when they differ.

**Where to look on chain.** After step 2 the script prints the vault, Safe
and module addresses with Etherscan links and the three Lighter account
indices with their Lighter explorer links, and saves them in
`deployment.env`.

**Failure report.** A trap on `ERR` and on interruption prints the step
that failed, the last command, the time spent in the step, the log path,
the `check-accounts` hint for steps 5 to 7, and the exact command to resume:
the same script with `LIGHTER_RUN_DIR` set to this run directory.

**Python module.** `lighter_manual` logs at INFO with timestamps through
the same logging setup as the tutorials, which keeps credential-bearing
provider logs at WARNING. Every command prints one result line the script
can parse and a human-readable summary.

## Tests

- A black-box test for `--lighter-transfer` and the `--pair` sub-account
  form in `tests/lagoon/test_lagoon_lighter_test_trade_cli_e2e.py`, in
  simulate mode on the shared Ethereum fork with the fake sequencer: fund
  `trend` with `--pair ... --buy-only`, transfer `trend:arb` with
  `--lighter-transfer --buy-only`, assert the balances and the booked
  trades, and check the command refuses `--lighter-transfer` before moving
  anything when the source's record is missing, and the round trip when the
  destination's record is missing.
- A simulate-mode test that `lagoon-settle` settles a subscription for the
  two-sub-account vault after change 1.
- A test that `lagoon-deploy-vault` writes one mode-0600 file per generated
  key under `LIGHTER_KEY_BACKUP_DIR`, refuses to overwrite an existing one,
  and removes them on success.
- `lighter_manual`: a unit test for argument parsing, `expect-balances`
  comparison and the `reconcile` arithmetic; every subcommand runs
  in-process, so it sees the fake sequencer, against the fork in the
  simulate-mode test above, including a second `request-deposit` after
  settlement to show it does not subscribe again.
- The shell script itself runs only on mainnet. `bash -n` and `shellcheck`
  in a unit test keep it parseable; a `SIMULATE=true` mode is out of scope,
  because `lagoon-deploy-vault` cannot activate Lighter on a fork.

## Documentation

- `docs/lagoon-lighter-deployment.md`: a "Test from the command line"
  section pointing at the script and listing the inputs table.
- `tradeexecutor/exchange_account/README-Lighter.md`, custody test trade
  section: the `--pair` sub-account form and `--lighter-transfer`, with one
  example each; the key backup option in the deployment section.
- The script's own header comment is the primary documentation.

## Implementation status

Implemented on 2026-10-28:

- `scripts/lagoon/manual-lighter-subaccounts-cli.sh` and
  `tradeexecutor/testing/lighter_manual.py`, with the observability above.
- `lagoon-settle` builds its universe from the state.
- `perform-test-trade --lighter-transfer SOURCE:DESTINATION`.
- `lagoon-deploy-vault --lighter-key-backup-dir`, through a new
  `lighter_api_key_callback` in eth_defi's `deploy_automated_lagoon_vault()`
  and `LagoonConfig`, called with each key before registration.
- `tests/lagoon/test_lagoon_lighter_cli_manual_e2e.py` runs the script's
  command sequence on the fork with the fake sequencer, including the
  unwind; without the `lagoon-settle` fix it fails with the funded master
  "not in the strategy universe". `tests/lagoon/test_lighter_manual_cli_units.py`
  covers parsing, the key backup and the script's syntax.

Simplifications made with it:

- The Python tutorials deploy with `--lighter-key-backup-dir` instead of
  patching eth_defi's key generator, which removed their own backup helpers.
- The Safe sweep, the deployer reconciliation and the Lighter equity read
  live once, in `lighter_manual`; the tutorials call them.
- `perform-test-trade`'s account test trade and transfer test trade share
  one executor for a single custody trade.

Deviations from the plan:

- `lighter-equity` became `withdrawal-plan`, and `redeem-status` was added,
  so that a withdrawal in flight is resumed rather than skipped and a
  finished redemption is not requested again.
- The unwind checks each account's own balance after its withdrawal rather
  than the Safe's total, because the withdrawn amounts are read from Lighter.
- `request-deposit` records the request in a marker file and reports
  `requested`, `pending`, `claimable` or `settled`: the deployment's
  activation subscription can leave the deployer holding shares, so share
  ownership cannot tell whether this run already subscribed.
- The script has six helpers and the heartbeat loop, not three helpers and
  no loops, to provide the observability.
- Step 1 does not deploy again when the deployment artefact exists, or
  when the master's key record exists without it: an interruption after a
  successful deployment would otherwise deploy a second vault.
- `TRADING_STRATEGY_API_KEY` is required, as every test of these commands
  runs with a Trading Strategy client.

## Review history

Reviewed by Codex with the GPT 6 Astra model on 2026-10-28. Its findings,
all verified against the code, changed the plan:

- `lagoon-settle` built the universe without the state, so sub-account
  vaults could not settle (change 1 added).
- Step markers alone could repeat money movements after an interruption;
  the balance-guarded resume replaced them.
- A `--lighter-transfer` round trip must preflight both signers.
- `lagoon-deploy-vault` and `correct-accounts` prompt; the script answers
  them explicitly.
- The deployment artefact is a sibling of `STATE_FILE`; the file layout now
  says so.
- `--pair "(ethereum, lighter, LIGHTER-trend, USDC)"` resolves a sub-account
  pair, so no `--lighter-account` option is needed.
- `lagoon-redeem` writes off aggregate Lighter equity below 5 USDC, not
  per-account dust, and needs no `--lighter-account-index` with `STATE_FILE`.
- Withdrawal amounts come from observed balances, so a transfer fee cannot
  leave an account below its minimum.
- The key backup option moved into scope.

A second pass on the revised plan found three gaps in the resume design,
fixed above: a pending withdrawal fails `check-accounts`, so withdrawal
steps resume through `lighter-move-funds` before any guard; step 4 could
skip the share claim, so its commands are idempotent and run on every pass;
a funding step interrupted between its deposit and its transfer had no
recovery, so `--lighter-transfer master:<id>` completes it. It also fixed
the unwind rows: each withdrawal reads its own amount, withdrawals send the
given amount in full, and the sub-minimum case uses an explicit `<=` target.

An implementation review by Codex with the GPT 6.1 Sol model on
2026-10-28 found four gaps, all fixed: an interruption before any key was
generated left no evidence of a deployment attempt, now the
`deploy-started` marker; a withdrawal saved before its request was recorded
was resumed as if submitted, now reported as `unconfirmed`; an interrupted
funding stopped at `check-accounts` without saying which state it was in,
now diagnosed; and recovery commands lacked the run's environment, now
`env.sh`. It also led to a minimum ETH check (`lagoon-redeem` refuses below
0.001 ETH), to unsetting an inherited `LIGHTER_ACCOUNT_INDEX`, to correcting
this plan's prompt list, and to fork coverage of all three withdrawals,
including the master's secure withdrawal through `lighter-move-funds`.

## Open questions

1. Should `lagoon-settle` be fixed in this work or in its own pull request,
   with the script using `correct-accounts --process-redemption` meanwhile?
2. Is `expect-balances` with a 0.01 USDC tolerance the right guard, or
   should the resume guard compare booked trades in the state instead?
