# Lighter exchange account integration

This document describes how to deploy and monitor a Safe-owned Lighter account
with a Lagoon vault on Ethereum. It follows the same external exchange-account
model as Derive and GMX: capital leaves the Safe for an external venue, while
trade-executor synchronises that venue's equity into strategy state and Lagoon
NAV.

The initial integration provides deployment, custody and valuation. Lighter
orders are submitted by a separate execution tool, not by trade-executor.

## Value and custody model

The strategy universe contains one synthetic Lighter exchange-account pair.
Its value is Lighter's public `total_asset_value`, including collateral and
unrealised profit or loss. The Lagoon NAV is:

```text
Safe native-USDC balance
+ public Lighter total equity
+ pending Lagoon settlement value
```

The Safe remains the Lighter account owner. Its guarded module can deposit USDC
to that account and claim withdrawals back to the Safe. Trading uses a delegated
Lighter API key registered in a numbered key slot.

## Deploy a Safe-owned Lighter account

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

This is a live Ethereum operation. It spends gas and makes the accounted 1
USDC activation deposit. It cannot run with `SIMULATE=true`, because an Anvil
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

## Initialise and synchronise accounting

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
Run Lagoon settlement after later deposits, trading and withdrawals:

```shell
trade-executor lagoon-settle
```

The Lighter endpoint reports current state and cannot be pinned to the Ethereum
block used for the Safe balance. Pause normal execution while collateral is in
transit and synchronise only after both sides show the completed movement.

The integration fails closed:

- Negative, malformed or non-finite equity aborts account correction and NAV
  settlement before state is mutated.
- A public API failure does not reuse a stale external-account value for a new
  NAV.
- Negative equity is never accepted, even temporarily.

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
3. checks NAV with `correct-accounts` and `lagoon-settle`;
4. opens and closes a small ETH/USD perpetual long;
5. checks NAV before, during and after the position;
6. submits one secure withdrawal and polls it with rotating auth tokens;
7. claims USDC to the Safe and redeems all shares to the deployer.

The script asserts Safe, Lighter, share and deployer balances at every material
boundary. Its auth-token lifetime defaults to 30 seconds so a normal withdrawal
wait demonstrates rotation; production code should use the manager defaults.
The SDK is an optional tutorial dependency and is not installed by
trade-executor itself. Install the version verified by this tutorial before the
run:

```shell
source .local-test.env
poetry run pip install lighter-sdk==1.1.2
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

## See also

- [Lagoon Lighter deployment guide](../../docs/lagoon-lighter-deployment.md)
- [Derive exchange account integration](./README-Derive.md)
- [Lighter guard architecture](../../deps/web3-ethereum-defi/eth_defi/lighter/README-lighter-guard.md)
- [Lighter API authentication](https://apidocs.lighter.xyz/docs/authentication)
