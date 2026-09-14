# Lagoon Lighter API-key deployment and NAV syncing

## Objective

Expose the Lighter account activation and API-key registration added by
[`web3-ethereum-defi` PR #1566](https://github.com/tradingstrategy-ai/web3-ethereum-defi/pull/1566)
through `trade-executor lagoon-deploy-vault`, then represent the resulting
Lighter account as an external exchange account whose equity is included in
Lagoon NAV.

When explicitly enabled, a fresh Ethereum Lagoon deployment must:

1. configure the canonical Lighter contracts in the Lagoon guard;
2. make the accounted activation deposit through Lagoon;
3. create and register the requested Lighter API key before final Safe
   ownership is configured;
4. verify the account, collateral and registered public key through Lighter's
   public API; and
5. persist the generated private key in the operator's JSON deployment record;
   and
6. expose the public account index for Derive/GMX-style exchange-account and
   NAV syncing.

All console output, process logs, exceptions, text and Markdown reports, and
the runtime state-sibling deployment artefact must remain secret-free. The
operator JSON is the only command output allowed to contain the Lighter API
private key.

## Implementation status

This is the historical implementation plan for the feature. The deployment,
public reporting, external exchange account, never-negative equity invariant,
Lagoon NAV wiring and Anvil-backed Typer deployment coverage are implemented.

Later tutorial work introduced one deliberate scope clarification: the
official Lighter SDK remains absent from trade-executor runtime dependencies,
but the separately executed manual tutorial imports the optional SDK for order
and withdrawal signing. Runtime valuation and all Typer NAV commands still use
only the unauthenticated public API. The implemented invariant helper is named
`validate_lighter_account_value()` rather than the provisional name used later
in this plan.

The implementation is based on trade-executor `f6d08e24` and the merged
eth-defi Lighter support at `f7a21f5c`, with the shared multichain
prepare-and-validate helper carried in the worktree's follow-up submodule
revision.

## Upstream contract

Use the upstream feature rather than reproducing its cryptography, polling or
Safe ceremony in trade-executor:

- `LagoonConfig.generate_lighter_api_key` opts into activation;
- `LagoonConfig.lighter_api_key_index` selects the slot and defaults to
  `MIN_API_KEY_INDEX`;
- `LagoonConfig.lighter_deployment` receives
  `LighterDeployment.create_ethereum()`;
- `LagoonAutomatedDeployment.lighter_account_setup` returns the account index,
  key index, private and public keys, activation amount, transaction hashes and
  observed collateral;
- `fetch_lighter_total_equity()` reads a Lighter account through the public API
  and returns collateral, unrealised PnL and total asset value without an API
  private key;
- `get_deployment_data()` and `pformat()` are always redacted;
- `as_json_friendly_dict(include_secrets=False)` is redacted by default; and
- `write_json_file(..., include_secrets=True)` demonstrates the required
  mode-`0600`, exclusive-create handling for a secret-bearing report.

Bump `deps/web3-ethereum-defi` to `f7a21f5c` or a later master commit containing
it. Do not add the Lighter SDK to trade-executor: key generation and deployment
do not need it.

Before consuming the multichain path, extend eth-defi's
`deploy_multichain_lagoon_vault()` with a synchronous prepare-and-validate
phase. It must run the canonical per-chain deployment validation, including all
source Lighter checks, for every configuration before submitting any worker to
the thread pool. If the existing validation is private to the single-chain
deployment function, expose it as one public upstream helper and call it from
both entry points. Do not copy its individual manager, topology, settlement,
balance or Lighter rules into trade-executor.

## Scope and non-goals

In scope:

- the single-chain `lagoon-deploy-vault` path;
- the strategy-file path when its source Lagoon vault is on Ethereum;
- the pre-flight, text, JSON, Markdown and runtime deployment reports;
- a Lighter external-exchange-account position, public account valuation and
  Lagoon NAV syncing following the existing Derive/GMX architecture; and
- Ethereum-mainnet-fork Typer black-box deployment and NAV tests, plus focused
  serialisation and secret-leak regression coverage.

Out of scope:

- placing or cancelling Lighter orders in trade-executor;
- loading the Lighter API private key into trade-executor (the separate trading
  service consumes it through secret management);
- automating later collateral deposits or withdrawals while the executor is
  running;
- importing the official Lighter SDK;
- activating an existing vault, a guard-only deployment or a satellite Safe;
- treating fork-only Lighter state as visible to Lighter's public API; and
- a recovery or key-rotation state machine for an interrupted ceremony.

## Command interface and validation

Add these Typer options to
`tradeexecutor/cli/commands/lagoon_deploy_vault.py`:

```python
generate_lighter_api_key: bool = Option(
    False,
    envvar="GENERATE_LIGHTER_API_KEY",
    help="Activate the Safe-owned Lighter account and register a trading API key.",
)
lighter_api_key_index: int = Option(
    MIN_API_KEY_INDEX,
    envvar="LIGHTER_API_KEY_INDEX",
    help="Lighter API-key slot to register during a fresh Ethereum deployment.",
)
```

The generation flag implies Lighter guard whitelisting. Do not introduce a
second CLI switch which can activate the account without the guard or configure
the guard without the activation ceremony.

When generation is enabled, reject `--simulate` at the top of the command,
before `create_web3_config()` can launch its managed Anvil process. A fork
cannot create a Lighter account visible to the public API. Let eth-defi's
pre-transaction validation remain the single authority for the fresh-deployment
topology, Ethereum chain, native USDC underlying, manager roles, API-key index,
settlement cap and deployer balance. Do not duplicate those rules in the CLI.

When generation is disabled, `LIGHTER_API_KEY_INDEX` is unused and must not
change existing deployment behaviour. This avoids a shared environment file
breaking unrelated Lagoon deployments.

Extend `log_deployment_preflight_report()` with public-only Lighter fields:

- activation enabled or disabled;
- the API-key index;
- the canonical Lighter contract address;
- the fixed activation amount; and
- the path of the private JSON record.

The confirmation prompt should state that a Lighter account and key will be
created and that the private key will be saved to the JSON record. It must not
show the generated key, which does not exist yet.

## Deployment wiring

### Single-chain path

When generation is enabled, create
`LighterDeployment.create_ethereum()` and pass it, the generation flag and the
key index to `deploy_automated_lagoon_vault()`. Leave all three values at their
backwards-compatible defaults when disabled.

Do not call any Lighter helper directly from the CLI. A deployment is successful
only when eth-defi returns a non-null `lighter_account_setup` whose private key
is present. Treat a missing setup or missing key after an enabled deployment as
a hard failure before reporting `All ok.`.

### Strategy-file path

Thread the two CLI values into `_deploy_multichain()`. After
`translate_trading_universe_to_lagoon_config()` returns the ordinary chain
configs, have `_deploy_multichain()` set `lighter_deployment`,
`generate_lighter_api_key` and `lighter_api_key_index` only on the Ethereum
source-vault `LagoonConfig`. Do not add Lighter-specific arguments or branches
to the general universe translator. Satellite configurations must retain
`generate_lighter_api_key=False`.

When generation is enabled, reject a non-Ethereum source vault after the source
chain is resolved but before calling `deploy_multichain_lagoon_vault()`. The
upstream multichain entry point must then synchronously validate every prepared
configuration before it starts its thread pool. This guarantees that invalid
manager roles, guard-only/existing-vault topology, settlement cap, deployer
balance or Lighter configuration cannot race with satellite deployment
transactions. Disabled deployments retain their current chain support.

Add the following public configuration fields to
`_serialise_lagoon_config()` so deployment diagnostics are complete:

- canonical Lighter contract addresses;
- `generate_lighter_api_key`; and
- `lighter_api_key_index`.

These are configuration and public-key-slot metadata, not secrets. Never pass
`LighterAccountSetup` to `_serialise_simple_dataclass()` because its dataclass
contains the private key.

## Output and secret boundary

The command currently produces several related outputs. Give each one an
explicit security contract:

| Output | Lighter public metadata | Lighter private key | File handling |
|---|---:|---:|---|
| Console and logs | Yes | Never | Existing logging |
| `VAULT_RECORD_FILE` text record | Yes | Never | Existing path |
| Paired `VAULT_RECORD_FILE.with_suffix(".json")` operator record | Yes | **Yes, only when generated** | Mode `0600`; exclusive when generated |
| `deployment-report.md` | Yes | Never | Existing path |
| `state/{executor-id}.deployment.json` runtime artefact | Yes | Never | Existing path |

The paired JSON already is the command's authoritative machine-readable
operator record. Keep its existing keys backwards compatible and add one nested
`lighter_account_setup` object. For a successful enabled deployment it contains:

```json
{
  "account_index": 123,
  "api_key_index": 4,
  "private_key": "<secret>",
  "public_key": "0x...",
  "activation_amount": "1",
  "deposit_tx_hash": "0x...",
  "change_pubkey_tx_hash": "0x...",
  "observed_collateral": "1"
}
```

Obtain this object through the upstream explicit serialiser rather than reading
or formatting the private-key field in generic reporting code. In the
multichain schema, place it at
`deployments[<ethereum-source>]["lighter_account_setup"]`; satellite entries
are null. In a deployment without Lighter activation the field may be null or
absent according to the existing backwards-compatibility convention.

Resolve the output paths once at command start. `VAULT_RECORD_FILE` is documented
as a text path, but existing callers also pass a `.json` path and rely on the
current write-then-overwrite behaviour. Preserve their final JSON location
without briefly placing text in the secret target:

- for a `.json` value, use that exact path for the private operator JSON and use
  its `.txt` sibling for the human record;
- otherwise use the configured path for text and its `.json` sibling for the
  private operator JSON; and
- use the resolved JSON path consistently in pre-flight checks, confirmation
  text and the success log.

Build two distinct JSON payloads at the report boundary:

1. a public payload using `include_secrets=False`, used by the runtime sibling
   artefact and any human formatting; and
2. an operator payload using `include_secrets=True`, used only by the paired
   JSON writer.

Before either payload reaches `json.dumps()`, normalise every upstream summary
value with the existing `_serialise_artifact_value()`. In particular,
`get_deployment_data()` currently exposes `Lighter collateral` as a `Decimal`,
which the command's plain JSON writers cannot serialise. Apply the conversion
to the single-chain payload, each multichain `deployment_data` entry and the
runtime state-sibling payload. As part of the eth-defi revision bump, also
change that public summary field to a decimal string at its source, but retain
the trade-executor normalisation as the report-boundary guarantee.

Do not build one secret-bearing dictionary and attempt to redact copies later.
This prevents a future early return, debug log or additional output writer from
receiving the private payload accidentally. Give helper parameters names such
as `public_json_payload` and `private_json_payload`; avoid an ambiguous
`json_payload` once secrets are introduced.

Replace the generic `open(..., "wt")` path for the operator JSON with a small
private writer modelled on `LagoonAutomatedDeployment.write_json_file()`:

- serialise fully before opening the output;
- when generation is enabled, create with
  `os.open(..., O_WRONLY | O_CREAT | O_EXCL, 0o600)`;
- otherwise preserve replacement semantics while setting the resulting mode to
  `0600`;
- write and close without printing the payload;
- refuse to overwrite an existing report when Lighter key generation is
  enabled; and
- when generation is enabled, verify the target path is absent and its parent
  is writable during pre-flight, so a successful on-chain ceremony is not
  followed by a predictable report failure.

Always give the paired JSON mode `0600`, but preserve the existing overwrite
behaviour for deployments without Lighter key generation. Exclusive-create is
needed only when the report contains an irreplaceable generated key; applying
it to ordinary deployments would be an unrelated breaking change. Preserve the
current simulation rule: simulated deployments do not write the text or paired
JSON artefacts.

The runtime state-sibling artefact must use only the public payload. It exists
for module and vault discovery and must be safe to copy into diagnostics,
containers and support bundles. The live NAV reader needs only the public
account index and must never load a private key. A separate Lighter trading
service may consume the operator JSON through the deployment's secret-management
mechanism, but that is not part of this implementation.

## External exchange account and NAV syncing

### Strategy-universe representation

Add `tradeexecutor/exchange_account/lighter.py` following the Derive and GMX
modules. Provide `create_lighter_exchange_account_pair()` and
`has_lighter_exchange_account_pairs()`.

The pair is a `TradingPairKind.exchange_account` with a synthetic
`LIGHTER-ACCOUNT` base asset and these public `other_data` fields:

```python
{
    "exchange_protocol": "lighter",
    "exchange_subaccount_id": account_index,
    "exchange_is_testnet": False,
    "lighter_deployment": "ethereum",
}
```

Use `exchange_subaccount_id` for the Lighter account index so the existing
`get_exchange_account_id()` and exchange-account state machinery work without
special cases. Derive the synthetic asset identity from the canonical Ethereum
Lighter deployment rather than inventing a configurable address.

The strategy universe remains the authority for whether Lighter accounting is
enabled, as it is for GMX. A strategy creates the pair with the public
`account_index` emitted by `lagoon-deploy-vault`; the public text, Markdown,
operator JSON and runtime deployment artefact must all expose this value. Do
not infer a live Lighter position merely from the key-generation flag: an
operator may deploy the integration before funding a strategy, and an older
vault may have a Lighter account created by another process.

Support one Lighter account per strategy in this first version. If the universe
contains Lighter, define every non-Lighter exchange-account protocol as
incompatible and fail clearly if there is more than one Lighter pair or any
other exchange-account protocol. Run this validation once before
`_auto_discover_gmx()` or `_auto_discover_lighter()` reaches its existing
`account_value_func is not None` early return. This prevents a preconfigured
Derive/GMX function or GMX vault valuation from silently omitting Lighter
equity. Do not add aggregation or a general multi-exchange orchestration layer.

### Public Lighter account valuation

Add `create_lighter_account_value_func()` using eth-defi's
`fetch_lighter_total_equity(session, account_index)`. The returned function
must:

- validate that it received a Lighter exchange-account pair;
- obtain the account index through `get_exchange_account_id()`;
- create one unauthenticated session in the factory and reuse it for all
  valuations, mirroring the lifetime of GMX's reused Web3 connection;
- return `LighterEquity.get_total()` / `total_asset_value`, including Lighter
  collateral and unrealised PnL;
- enforce `total_asset_value >= 0` before returning it;
- exclude free USDC held by the Lagoon Safe, which treasury sync accounts for
  separately;
- accept the common optional `block_identifier` keyword for compatibility but
  document that the public Lighter API is a current observation and cannot be
  pinned to an Ethereum block; and
- use an unauthenticated HTTP session. It must have no parameter, environment
  lookup or file reader for the Lighter API private key.

Wire protocol `lighter` into
`tradeexecutor.exchange_account.utils.create_exchange_account_value_func()` so
`correct-accounts` can create and reconcile the position through the existing
`ExchangeAccountSyncModel`. Add Lighter universe auto-discovery to
`EthereumPairConfigurator`, parallel to `_auto_discover_gmx()`, so normal
`start` runs construct the value function from public metadata without new
secret-bearing CLI options.

Keep the existing accounting model unchanged:

- the open exchange-account position quantity is the total Lighter account
  value in USD;
- `ExchangeAccountValuator` updates its `BalanceUpdate` and `ValuationUpdate`
  records during revaluation; and
- `correct-accounts` may create a missing position from the strategy-universe
  pair and sync it using the same public value function.

Make non-negative Lighter equity a hard runtime invariant shared by the
account and vault value functions. After every upstream response, validate the
`Decimal` total before it can reach a balance update, valuation update or NAV
calculation. Raise a fixed, secret-free `ValueError` containing only the public
account index when the value is negative; do not use Python `assert`, which can
be disabled. Zero is valid. Consequently, a negative Lighter value must never
be written to state or posted to Lagoon, even if the public API returns one.

The Ethereum block stored on a valuation event is an observation anchor for
the cycle, not proof that the Lighter API value was historical at that block.
State this in the function documentation and tests rather than adding a new
valuation-result type.

### Lagoon NAV calculation

Add `create_lighter_vault_valuation_func()` alongside the account function.
Like the GMX-specific Lagoon function, it must calculate the custody and
external-account base valuation directly instead of combining a fresh external
position with a potentially stale portfolio reserve:

```text
Custom base valuation = Safe USDC balance at the treasury-sync block
                      + current Lighter total_asset_value

Lagoon posted NAV = custom base valuation
                  + pending vault settlement value
```

Read the Safe's native USDC balance from Anvil/mainnet at the
`block_number` supplied by `LagoonVaultSyncModel`. Read the Lighter component
through the public API for the account index discovered from the sole Lighter
pair. Do not include the Safe balance in the exchange-account position itself,
or it will be double counted.

Keep Safe USDC and Lighter equity as `Decimal` while summing them, then return
`float(safe_usdc + lighter_equity)` at the existing custom-valuation interface
boundary. `LagoonVaultSyncModel.calculate_valuation()` adds a pending-settlement
`float`; returning `Decimal` from the custom function would otherwise raise a
`TypeError`.

Have `_auto_discover_lighter()` set both `account_value_func` and
`vault_valuation_func`. The existing runner then installs the latter as
`LagoonVaultSyncModel.calculate_valuation_func`. Treasury sync continues to
reconcile the portfolio reserve for state/history, while the custom function
is the authoritative base value used for Lagoon posting. Preserve
`LagoonVaultSyncModel.calculate_valuation()` adding pending settlement value
after calling the custom function.

Fail closed on a Lighter API timeout, malformed response or negative/otherwise
invalid equity: propagate the error and do not post a cached Lagoon NAV. Rely
on eth-defi's existing returned-account-index check and let its mismatch error
propagate rather than duplicating it. Logs may contain the public account index
and a sanitised error class/message, but never request headers, environment
values or the private operator record.

Unlike GMX's fully on-chain valuation, the Safe balance and Lighter API value
cannot form an atomic block snapshot. Keep the first version operationally
simple: the executor/NAV poster must be stopped while collateral is being
deposited to or withdrawn from Lighter and may restart only after the public
API reflects settlement. Document this limitation in the strategy runbook and
do not attempt to infer in-flight transfers from balance changes in this
feature.

## Human-readable reports and logging

Extend the single-chain and multichain Markdown metadata in
`tradeexecutor/ethereum/lagoon/deploy_report.py` with the account index, API-key
index, public key, observed collateral and the two public transaction hashes.
The existing single-chain `print_deployment_report()` receives only
Safe/module/Web3 inputs, so give it an optional `public_lighter_metadata`
parameter and prepend that metadata to the guard report. Build this dictionary
in the CLI from `deploy_info.as_json_friendly_dict(include_secrets=False)` and
pass only the redacted dictionary—not `deploy_info` or
`LighterAccountSetup`—across the report API boundary. The multichain report uses
the corresponding redacted dictionary from its per-chain public payload. Use
`get_deployment_data()` for the existing public summary where convenient and
augment it from the explicit redacted serialiser where the summary does not
expose transaction hashes. The text record uses the same explicitly public
data.

After saving, log only a message such as:

```text
Saved private Lighter deployment data to /path/to/vault-record.json (mode 0600)
```

Apply these invariants throughout the command:

- never interpolate `private_key`, `lighter_account_setup` or the private JSON
  payload into a logger call, exception, assertion message or Typer output;
- never print or return the private payload from a helper used to format human
  reports;
- never include it in the guard report, configuration snapshot, transaction
  diagnostics or signed transaction data;
- keep `repr=False` on the upstream private-key field and do not use
  `dataclasses.asdict()` on that object;
- do not log environment dictionaries or process arguments; and
- review the runtime valuation path to ensure it has no operator-record file
  reader or private-key environment lookup; and
- use a fixed sentinel private key in tests and scan every captured output for
  both the full sentinel and distinctive substrings.

## Failure behaviour

The upstream ceremony is linear and fail-fast. Preserve that model:

- no report is called successful until Lighter's API confirms the registered
  public key;
- do not write a JSON record containing a private key before registration is
  confirmed;
- a failure must not log the generated private key or include it in exception
  context;
- do not catch an upstream failure merely to dump the deployment object; and
- if persistence fails after successful registration, stop with an actionable
  error naming only the intended report path and permissions.

Document that the upstream flow has no automatic recovery record. Operators
must inspect on-chain and public Lighter state before retrying an interrupted
deployment, especially if activation or `changePubKey` may already have
succeeded.

## Tests

Keep the cryptographic vectors, REST polling and full ceremony ordering in
eth-defi's tests. Add trade-executor coverage for its own wiring and output
boundary.

### CLI and configuration tests

- Assert both new CLI defaults and environment variable names.
- Assert disabled generation leaves existing deployment kwargs unchanged.
- Assert enabled single-chain deployment passes the canonical
  `LighterDeployment`, generation flag and chosen index upstream.
- Assert the strategy-file path enables activation only on the Ethereum source
  config and leaves every satellite disabled.
- In eth-defi, assert all per-chain configs are prepared and validated before
  the first thread-pool submission. Cover invalid manager roles,
  guard-only/existing-vault topology, settlement cap, deployer balance and
  Lighter configuration; in every case assert the worker/deployment call count
  remains zero on every chain.
- Assert the trade-executor multichain path rejects a non-Ethereum source before
  entering the upstream multichain function.
- Assert simulation is rejected by the CLI when generation is enabled.
- Assert a configured key index has no effect when generation is disabled.
- Assert pre-flight logging reports only public Lighter configuration.

### Report tests

Use an unmistakable fixed secret such as
`lighter-private-key-DO-NOT-LOG-7f91` and assert:

- the paired operator JSON contains it exactly at
  `lighter_account_setup.private_key` for single-chain output;
- multichain output contains it only under the Ethereum source deployment;
- the operator JSON mode is exactly `0600`;
- an existing operator JSON is not overwritten and deployment is not started
  when an enabled Lighter deployment detects the collision during pre-flight;
- a non-Lighter deployment retains the existing operator-JSON overwrite
  behaviour;
- console stdout/stderr, captured logs, text record, Markdown report and runtime
  state-sibling JSON contain neither the complete sentinel nor its distinctive
  substring `DO-NOT-LOG-7f91`;
- those public outputs contain the account index, key index, public key,
  collateral and transaction hashes;
- the single-chain Markdown API receives only the explicitly redacted Lighter
  metadata dictionary and renders it ahead of the guard report;
- redacted serialisation contains neither the complete sentinel nor its
  distinctive substring;
- single-chain and multichain public/private payloads containing upstream
  `Decimal` collateral values round-trip through the real JSON writers, with
  the collateral represented as a decimal string;
- a disabled deployment preserves the current JSON shape and report content
  apart from the documented optional field; and
- a failure before upstream key verification writes no secret-bearing report
  and emits no secret.

### External-account and NAV tests

Add focused tests for the Lighter exchange-account adapter:

- pair creation stores protocol `lighter`, the public account index and the
  canonical Ethereum deployment identity, and universe detection ignores
  unrelated pairs;
- the account value function passes the pair's account index to
  `fetch_lighter_total_equity()` and returns canonical total asset value;
- Safe USDC is excluded from the exchange-account position;
- the standard valuator and `ExchangeAccountSyncModel` create/update the
  USD-denominated position and preserve balance/valuation history;
- `correct-accounts` discovers a missing Lighter position and uses the public
  account reader without requesting a private key;
- auto-discovery wires both Lighter functions into the Ethereum pair
  configurator and the runner installs the vault function on
  `LagoonVaultSyncModel`;
- a Lighter universe mixed with any other exchange-account protocol fails
  before either GMX or Lighter auto-discovery can retain an existing value
  function;
- the vault value function returns `float` and a focused unit test exercises
  addition of a non-zero pending-settlement `float`;
- zero total equity is accepted;
- negative total equity from the public API raises before either the account or
  vault function returns, creates no balance/valuation update and leaves the
  last non-negative state unchanged;
- the negative-equity NAV test asserts no Lagoon valuation transaction is
  broadcast; and
- an API error, including eth-defi's account-index mismatch, aborts valuation
  instead of reusing a cached value.

Add a Lagoon NAV integration test on an externally managed Ethereum-mainnet
Anvil fork. Use a real Lagoon vault/Safe, real native USDC balances, real
treasury reconciliation and a real NAV-update transaction. Create the Lighter
pair from a deterministic public account index and mock only the public
Lighter API response that cannot observe fork-local state. Return distinct
collateral and unrealised-PnL values so the test proves it uses total asset
value rather than collateral alone.

The test should:

1. place a known native USDC balance in the Safe through the normal Lagoon
   funding/settlement lifecycle;
2. create or synchronise the Lighter exchange-account position through the
   production adapter;
3. run the normal revaluation and Lagoon treasury/NAV-sync path;
4. assert the state position equals the mocked Lighter total asset value;
5. assert the reconciled reserve equals the Safe's actual on-chain USDC
   balance;
6. ensure pending settlement value is zero at the assertion point, then assert
   the NAV posted by the real Lagoon transaction equals Safe USDC plus Lighter
   total asset value exactly once; and
7. assert the public API mock received the account index from the strategy
   pair.

Do not mock `ExchangeAccountValuator`, `ExchangeAccountSyncModel`,
`EthereumPairConfigurator`, `LagoonVaultSyncModel`, ERC-20 balance reads,
Lagoon contract calls or transaction broadcasting. The mandatory negative
test may stub the public API to return negative equity and must assert that no
NAV-update transaction is broadcast. There is no need to emulate Lighter's
matching engine, positions or L2 state in Anvil.

### Typer black-box integration test

Add a test alongside the existing Lagoon deploy CLI integration tests which
invokes the real Typer entry point:

```python
cli = get_command(app)
cli.main(args=["lagoon-deploy-vault"], standalone_mode=False)
```

Run it against an externally managed Ethereum-mainnet Anvil fork by supplying
its URL as `JSON_RPC_ETHEREUM`, with `UNIT_TESTING=true` to skip the interactive
confirmation. Do not set `SIMULATE=true`: the command correctly rejects that
combination, while an external fork still lets the test exercise the normal
deployment path.

The test must keep these operations real:

1. deploy the Lagoon vault, Safe and Lighter-enabled guard;
2. fund the deployer with fork ETH and native Ethereum USDC;
3. perform the accounted Lagoon subscription, valuation, settlement and claim
   used for activation funding;
4. execute the guarded USDC deposit into the canonical Lighter L1 contract;
5. generate the ECgFp5 key locally;
6. execute the Safe `changePubKey` transaction against the forked canonical
   Lighter contract; and
7. run the command's real text, private JSON, Markdown and runtime-artefact
   writers.

Mock only the boundary which an Anvil fork cannot reproduce: Lighter's sequencer
registration and public API observation of the fork-only account, collateral
and registered public key. Patch the session factory and three wait/poll names
as imported into
`eth_defi.erc_4626.vault_protocol.lagoon.deployment`; patching their definitions
in `eth_defi.lighter.api` would not affect the already-bound names. The fake for
the account-registration wait must call upstream's
`register_lighter_account_on_anvil()` for the Safe address it receives before
returning the deterministic account index. This storage forge represents the
single sequencer action Anvil cannot perform: without it, the canonical Lighter
contract's `addressToAccountIndex` lookup is empty and the Safe's real
`changePubKey` call reverts.

Return deterministic collateral from the remaining public API fakes and accept
the generated public key. Do not mock `deploy_automated_lagoon_vault()`, Lagoon
funding, `deposit_usdc_from_lagoon_safe_into_lighter()`, local key generation,
`execute_change_pubkey()` or any report builder/writer. The deposit and
`changePubKey` transactions must still execute against the canonical forked
Lighter L1 contract.

After the CLI returns, assert the L1 transaction receipts succeeded, the
private operator JSON contains the generated key and is mode `0600`, and all
public outputs omit the key read back from that JSON while retaining the public
account/key metadata. Also assert the mocked API boundary received the Safe
address, account index and API-key index, and that its public-key argument
matches the public key read back from the paired operator JSON.

Use the public account index read from the command's runtime artefact to create
a `LIGHTER-ACCOUNT` pair and assert it matches the index in the public reports.
Keep full NAV-cycle assertions in the companion Lagoon NAV integration test so
the deployment black-box test remains focused and failures are diagnosable.

Keep direct report-builder unit tests for the fixed sentinel leak checks. The
fork test must make no live Lighter HTTP request, but otherwise follows the same
black-box pattern as the existing Lagoon deployment tests. Give it the
repository-standard extended timeout for a contract deployment integration
test.

## Documentation and release notes

Update the command docstring/example with the Ethereum-only invocation and the
output split. Explicitly tell operators to back up the paired JSON in their
secret store immediately, never paste it into tickets or chat, and use the
public text/Markdown or runtime artefact for support.

Document how a strategy adds `create_lighter_exchange_account_pair()` with the
public account index from the deployment report, and that no API key is needed
for valuation. Document the NAV formula and require operators to stop the
executor while collateral is in transit between the Safe and Lighter, resuming
only after the public account response reflects settlement.

Add a `CHANGELOG.md` entry for the command flag, the accounted 1 USDC activation
deposit, private JSON report, external exchange-account representation and
Lagoon NAV syncing. State that enabling the option spends real Ethereum gas and
USDC and can wait for Lighter public-state propagation.

## Acceptance criteria

- A fresh Ethereum `lagoon-deploy-vault` run with the flag enabled returns only
  after the Safe-owned Lighter account and requested public key are visible.
- The paired operator JSON contains the generated private key, is created as
  mode `0600` and is never overwritten for an enabled Lighter deployment.
- Every human-readable output and the runtime deployment artefact contains only
  public Lighter metadata.
- A strategy can create a Lighter exchange-account pair from the reported
  public account index, and normal revaluation records its total account value
  without loading the signing key.
- Lagoon's custom base valuation is real Safe USDC plus current Lighter total
  asset value, with neither reserve double counting nor cached API fallback;
  the custom function returns `float` and the sync model then adds pending
  settlement value as it does today.
- Negative Lighter equity is never returned by an account/vault value function,
  recorded in state or posted to Lagoon; it always aborts the cycle before any
  mutation or valuation transaction. Zero equity remains valid.
- A universe containing Lighter and any other exchange-account protocol fails
  before value-function auto-discovery.
- Disabled deployments retain their existing transaction flow and reports.
- All multichain configurations pass synchronous upstream validation before
  any deployment worker is submitted, so unsupported topology or invalid
  Lighter configuration fails before any chain sends a transaction.
- No Lighter SDK dependency or duplicate key-generation implementation is added
  to trade-executor.
- The Typer black-box test exercises the real forked deployment and Lighter L1
  transactions, forging only the sequencer-owned account registration and
  mocking only public Lighter API observations unavailable for fork-only state.
- The Lagoon NAV integration test exercises real Safe balances, treasury sync
  and NAV posting on Anvil, mocking only Lighter's external API/L2 observation.
