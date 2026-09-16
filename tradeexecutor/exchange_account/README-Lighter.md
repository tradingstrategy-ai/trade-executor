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
| Read-only accounting comparison | `check-accounts` |
| Accounting correction and PnL balance updates | `correct-accounts` |
| Generic interrupted-trade repair | `repair` |
| Perpetual order execution inside trade-executor | No; use a separate execution tool |

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

### Deployer funds

Before running the command, fund the deployer/initial asset-manager address
with the following native Ethereum assets:

| Asset | Amount | Why it is needed |
|-------|--------|------------------|
| ETH | A non-zero balance sufficient for current mainnet gas | Deploys the Lagoon contracts and Safe, then sends the Lighter activation transactions. Gas prices and the deployment path vary, so there is no safe fixed ETH amount. |
| Native Ethereum USDC | **1 USDC**, plus the collateral and test-trade amount you intend to use | The command makes an accounted, fixed 1 USDC Lighter activation deposit. Later subscription and Lighter collateral are additional funds. |

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

### 3. Perform a resumable ETH/USD test trade

For an existing Lighter-enabled Lagoon vault, use the dedicated Typer command.
It reads the delegated key only from the mode-`0600` operator record produced
by deployment, makes a bounded ETH/USD long round trip, and posts Lagoon NAV
at each completed phase. It never places the key in executor state, reports or
command-line arguments.

```shell
source .local-test.env
export JSON_RPC_ETHEREUM="https://..."
export PRIVATE_KEY="0x..."  # Funded Safe owner / asset manager; do not echo it
export EXECUTOR_ID="lighter-vault"
export STRATEGY_FILE="strategies/test_only/minimal_lighter_strategy.py"
export STATE_FILE="state/lighter-vault.json"
export VAULT_ADDRESS="0x..."
export VAULT_ADAPTER_ADDRESS="0x..."
export LIGHTER_ACCOUNT_INDEX="..."  # Public index from the deployment report
export LIGHTER_OPERATOR_RECORD_FILE="/secure/lighter/lighter-vault.json"
export LIGHTER_TEST_DEPOSIT_USDC="..."
export LIGHTER_TEST_POSITION_USDC="..."

trade-executor lagoon-lighter-test-trade
```

### Yubi deployment secret mapping

The Lighter AI Yubi deployment convention maps the protected host directory
`~/secrets/lighter` to `/secure-lighter` in the manual-command container. Its
delegated-key record is conventionally
`~/secrets/lighter/lighter-ai-vault-info.json`. Only
`lagoon-lighter-test-trade` and `lighter-move-funds` need this record:

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
the `LIGHTER-ACCOUNT` position and changes the Safe reserve by the same USDC
amount. `check-accounts` therefore verifies both sides of a completed movement,
while `correct-accounts` repairs residual PnL or out-of-band drift without
rewriting this transfer record.

Secure withdrawals save their request before waiting and resume after a
restart without blindly submitting a second request. The command claims only
the matching `claimable` history row and requires its Safe USDC credit to match
exactly. A definitive API rejection is recorded as a failed trade, allowing a
fresh attempt. An ambiguous transport failure remains pending and is never
resubmitted automatically. Keep unrelated Safe transfers stopped until the
withdrawal finishes, because they can obscure the saved pre-withdrawal balance
used for recovery. If an ambiguous request never appears in withdrawal history,
keep the executor stopped and investigate the saved transfer instead of using
`repair` or resubmitting it. A deposit interrupted after its physical movement
but before state persistence is deliberately recovered with `correct-accounts`;
it is never replayed automatically.

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

Do not configure the generated Lighter private key for `init`,
`check-accounts`, `correct-accounts`, `repair` or `lagoon-settle`. These
commands either use public account equity or operate only on executor state.

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
mismatch when they differ. It does not write a balance update:

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

`repair` has a different purpose. It repairs interrupted executor trades and
transactions; it does not query Lighter equity or perform account correction.
A healthy Lighter exchange-account position and its spoofed successful opening
trade are left unchanged:

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

- `lighter.py` creates the synthetic pair, validates public equity and provides
  account and Lagoon NAV value functions.
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

## Troubleshooting

### `check-accounts` reports a Lighter mismatch

Confirm that `LIGHTER_ACCOUNT_INDEX` points to the Safe-owned account and that
no deposit, withdrawal or order settlement is still in flight. If Lighter
equity differs, run `correct-accounts` only after the public account view is
stable. If the Safe reserve differs by a completed deposit or withdrawal
amount, first establish that no transfer is still pending, then run
`correct-accounts`; Lagoon Lighter strategies reconcile both the public Lighter
equity and on-chain Safe reserve. Run `lagoon-settle` separately when the vault
needs a fresh posted NAV. Repeat `check-accounts` afterwards.

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
