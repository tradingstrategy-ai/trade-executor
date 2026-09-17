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
| Native Ethereum USDC | **20 USDC** for initial capital | Deployment subscribes 20 USDC, transfers 1 USDC to Lighter and leaves 19 USDC in the Safe. |

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
one-settlement-per-day cooldown. The 20 USDC Lighter bootstrap subscription
therefore consumes 20 USDC of the first window, leaving 4,980 USDC available
for normal automated settlement.

The activation makes a Lagoon-accounted 20 USDC subscription, transfers its
fixed 1 USDC Lighter deposit and leaves 19 USDC in the Safe. It waits for
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

## Run the ETH/USD test lifecycle

For an already deployed Lighter-enabled vault, the Typer command is the
supported small real-money test. It uses the deployment's protected operator
record to deposit from the Safe, open and close ETH/USD, claim the withdrawal,
and synchronise Lagoon NAV throughout.

The operational sequence is:

1. Deploy with `GENERATE_LIGHTER_API_KEY=true`. The deployment subscribes 20
   USDC to Lagoon, transfers 1 USDC from the Safe to Lighter and leaves 19 USDC
   in the Safe. The vault is no longer eligible for `lagoon-first-deposit`.
2. Run `init`, then `correct-accounts --process-redemption`. The runtime
   deployment artefact supplies the original vault block, so the command records
   the 20 USDC Lagoon investor flow and the public 1 USDC Lighter exchange
   account without a repair or transfer script.
3. Run `lagoon-lighter-test-trade --lighter-test-deposit-usdc 19` to make the bounded Lighter deposit, ETH/USD
   round trip and secure withdrawal back to the Safe.

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
deposit amount.

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

A Lighter-enabled deployment subscribes 20 USDC to Lagoon, transfers 1 USDC
to activate the Safe-owned Lighter account and leaves 19 USDC in the Safe.
Thus a correctly deployed vault has an executor reserve before its first
normal trade. `lagoon-first-deposit` remains reserved for a vault with no NAV
or shares. The deposit-and-settle script also accepts a zero-Safe baseline only
to recover a legacy one-USDC Lighter activation deployment.

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
