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
| ETH | Non-zero and sufficient for current mainnet gas | Lagoon/Safe deployment and Lighter activation transactions. A fixed ETH amount would be misleading because gas varies. |
| Native Ethereum USDC | **1 USDC** plus the desired collateral and test-order amount | The fixed, accounted Lighter activation deposit is 1 USDC. All subscription and trading collateral is additional. |

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

The activation spends the fixed, Lagoon-accounted 1 USDC deposit and waits for
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

The executable Python tutorial is the supported small real-money test. It uses
Typer CLI commands whenever possible and Lighter's SDK only for the operations
that Anvil and the executor cannot simulate: the ETH/USD long and secure
withdrawal.

```shell
source .local-test.env
export LIGHTER_TEST_PRIVATE_KEY="0x..."  # Funded deployer; never print it
export JSON_RPC_ETHEREUM="https://..."
export LIGHTER_DEPOSIT_USDC="..."        # Collateral in addition to 1 USDC activation
export LIGHTER_POSITION_USDC="..."       # Small ETH/USD long notional

poetry run pip install lighter-sdk==1.1.2
poetry run python scripts/lagoon/manual-trade-executor-lighter.py
```

The tutorial deploys its own isolated vault, then:

1. subscribes and deposits USDC to Lighter;
2. runs `correct-accounts` and `lagoon-settle` and asserts the first NAV;
3. opens a small ETH/USD long, repeats the Typer NAV sync and checks balances;
4. closes the long, repeats the Typer NAV sync and checks balances;
5. withdraws collateral to the Safe, then redeems all shares to the deployer.

It asserts the Safe, Lighter, share and deployer balances at every material
boundary, including that Lighter equity is never negative. The selected
collateral target is at least the requested order notional plus a 5 USDC buffer
and never below Lighter's direct-deposit minimum.

For an already deployed vault, run the Typer accounting cycle only after a
deposit, fill or withdrawal has completed on both Lighter and the Safe:

```shell
trade-executor correct-accounts
trade-executor check-accounts
trade-executor lagoon-settle
```

Never perform this cycle while a collateral movement or order settlement is in
flight.

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
Safe and Lighter, and resume only after the public account response reflects
the settlement. A negative or malformed Lighter equity response aborts the
cycle; it is never cached, written to state or posted as NAV.
