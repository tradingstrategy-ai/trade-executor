# Lagoon Lighter manual trade and redemption tutorial

## Objective

Add a live Ethereum-mainnet Python tutorial at
`scripts/lagoon/manual-trade-executor-lighter.py`, following the structure of
`scripts/lagoon/manual-trade-executor-gmx.py`, that exercises the complete
single-owner Lagoon/Lighter lifecycle:

1. deploy a Lagoon vault, activate its Safe-owned Lighter account and register
   an API key with `lagoon-deploy-vault`;
2. initialise trade-executor state and its Derive/GMX-style Lighter external
   exchange-account position;
3. subscribe additional USDC to Lagoon and deposit it from the Safe to Lighter;
4. post and verify NAV;
5. open a small ETH-USD perpetual long with the generated Lighter API key;
6. post and verify NAV;
7. close the exact observed ETH-USD long, using a reduce-only order;
8. post and verify NAV;
9. securely withdraw all withdrawable USDC from Lighter to the same Lagoon Safe;
10. post and verify the final Safe-only NAV; and
11. redeem all deployer-owned Lagoon shares so all redeemable vault assets are
    credited back to the deployer.

## Implementation status and deviations

This file records the design considered before the live tutorial was written.
The mainnet lifecycle, Typer NAV checkpoints, balance assertions, API-key
secret boundary, 30-second token-rotation demonstration and final redemption
are implemented in `scripts/lagoon/manual-trade-executor-lighter.py`.

The smallest safe implementation differs from the initial plan in these ways:

- recovery checkpoints and `LIGHTER_RESUME_PHASE` are not implemented;
  withdrawal submission remains separate from the retryable history poll, but
  an interrupted run requires manual operator recovery;
- automated coverage is split into a real Typer deployment black-box test and
  a fork-backed Lagoon NAV test instead of driving the complete tutorial in one
  test; only Lighter sequencer observations are mocked in the deployment test;
- withdrawal history in Lighter SDK 1.1.2 does not expose the submitted L2
  transaction hash, so the tutorial matches the amount and fails immediately
  if more than one history row is ambiguous;
- a small direct Safe liquidity top-up may be needed before final redemption
  because an empty-queue NAV-only settlement does not update the settled
  `convertToAssets()` value after trading fees; the full redemption returns
  this top-up to the deployer; and
- SDK signing remains tutorial-local while the generic auth-token manager and
  Safe claim helper live in eth-defi. The runtime negative-equity helper is
  `validate_lighter_account_value()`.

This is a real-money manual test. Lighter does not provide a production-like
test environment for the Safe-owned Ethereum custody flow, so the tutorial must
not imply that an Anvil run can prove the sequencer, fills, zk-proof or live
withdrawal path.

The implementation builds on:

- [`web3-ethereum-defi` PR #1566](https://github.com/tradingstrategy-ai/web3-ethereum-defi/pull/1566),
  which supplies Lagoon activation, API-key registration, public Lighter reads
  and the existing one-shot ETH long tutorial;
- [Lighter API-key documentation](https://apidocs.lighter.xyz/docs/api-keys),
  including reserved key slots, per-key nonces and secure-withdrawal authority;
- [Lighter deposit and withdrawal documentation](https://apidocs.lighter.xyz/docs/deposits-transfers-and-withdrawals);
  and
- the official SDK's
  [`withdraw_normal.py`](https://github.com/elliottech/lighter-python/blob/main/examples/transfers/withdraw_normal.py)
  secure-withdrawal example.

## Scope and non-goals

In scope:

- one Ethereum-mainnet Lagoon vault owned, managed and funded by the test
  deployer;
- a Python tutorial that invokes real Typer commands in process with the same
  `run_cli()` pattern as the GMX and Hyperliquid manual scripts;
- the existing Lighter external exchange-account representation and public NAV
  reader;
- small ETH-USD market open and reduce-only close orders;
- a secure Lighter withdrawal back to the Safe, followed by Lagoon redemption
  to the deployer;
- public, explicit assertions at every checkpoint; and
- an Anvil-backed Typer black-box integration test which mocks only Lighter
  sequencer/SDK/proof effects that a fork cannot reproduce.

Out of scope:

- a general Lighter execution model or routing integration in trade-executor;
- recording individual Lighter orders as trade-executor spot/perpetual
  positions—the external account remains one aggregate USD-valued position;
- fast withdrawal or transfer to an arbitrary address, both of which require
  the Ethereum key and weaken the Safe-only custody boundary;
- automated API-key rotation or revocation;
- a general-purpose Lagoon deposit command or a new hierarchy of Lighter Typer
  commands solely for this tutorial;
- multi-investor accounting; and
- hiding trading loss: redemption returns current net assets, not necessarily
  the original USDC principal.

## Existing functionality and gaps

| Test step | Existing reusable path | Gap / planned treatment |
|---|---|---|
| Deploy and register key | `lagoon-deploy-vault --generate-lighter-api-key`, implemented over eth-defi PR #1566 | No gap. Use the paired mode-`0600` operator JSON as the only private-key source. |
| Initialise executor | `init` with `strategies/test_only/minimal_lighter_strategy.py` | The script must set `LIGHTER_ACCOUNT_INDEX` from public deployment metadata before constructing the strategy universe. |
| Create/sync external account | `correct-accounts` can create the missing exchange-account position and value it through the public API | Use the real Typer command after `init`; do not construct the state position directly. |
| Additional Lagoon subscription | `LagoonVault.request_deposit()` / `finalise_deposit()` and `lagoon-settle` | `lagoon-first-deposit` cannot be reused because API-key activation has already created supply and consumed the first accounted deposit. Do not use `fund_lagoon_vault()` here because it also settles outside trade-executor. Request and claim with the existing eth-defi vault methods, but let the real Typer NAV command observe and settle the pending subscription. |
| Safe-to-Lighter deposit | `deposit_usdc_from_lagoon_safe_into_lighter()` | No Typer command exists. Import this existing eth-defi helper. |
| NAV sync | `lagoon-settle`, Lighter exchange-account valuation and Safe-plus-Lighter custom vault valuation | No machine-readable `lagoon-settle` report exists. Run the real command, then verify state, public Lighter data and the emitted `NewTotalAssetsUpdated` event in the tutorial. Do not add a reporting command just for the tutorial. |
| Human valuation display | `show-valuation` | No gap. Run after every `lagoon-settle`; programmatic assertions still read state directly. |
| ETH long open/close | Logic exists only inside eth-defi's `lagoon-lighter-trade-example.py` and uses the separately installed official SDK | Promote the small sizing, signed-order and position-polling pieces to an importable eth-defi helper, or minimally make that existing code importable. Keep the SDK optional and use the same immutable revision/install workaround documented by the eth-defi tutorial. |
| Secure withdrawal request | Official SDK `SignerClient.withdraw()` can sign with the registered API key and targets the account's original L1 owner | Add a narrow reusable eth-defi secure-withdrawal request/poll helper. It must use the API key only; never request the deployer's Ethereum private key for an L2 withdrawal. |
| Withdrawal finality and L1 claim | Lighter withdrawal history reports a secure withdrawal as `claimable`; the guard already permits `withdrawPendingBalance(Safe, USDC, amount)` | Add a narrow `eth_defi.lighter.lagoon` Safe-claim helper using `TradingStrategyModuleV0.performCall()`. Keep request and claim as distinct phases because zk-proof finality is asynchronous and a run may be resumed at the claim step. |
| Return assets to deployer | `lagoon-redeem` requests, settles and finalises all shares held by the asset-manager/deployer | No new sweep/backdoor. Run the Typer command and assert zero deployer shares plus only an explicitly bounded USDC rounding remainder in the Safe. |
| Live simulation | Ethereum Anvil can execute Lagoon, Safe, guard, USDC and the Lighter L1 deposit call | Anvil cannot create sequencer account state, fill orders or produce a new zk-proof. Mock only those boundaries in automated coverage. The tutorial itself remains explicitly live-only. |

The two material package gaps are therefore an importable order helper and a
secure-withdrawal request/claim helper. Lack of protocol-specific Typer commands
is not itself a blocker and should not be solved as part of this manual-test
work.

## Tutorial interface and files

Create:

- `scripts/lagoon/manual-trade-executor-lighter.py`—the live operator tutorial;
- importable Lighter order/withdrawal helpers under
  `deps/web3-ethereum-defi/eth_defi/lighter/` only where the current example
  code cannot already be imported cleanly; and
- focused upstream and trade-executor tests described below.

Use environment variables consistent with the existing examples:

| Variable | Purpose |
|---|---|
| `JSON_RPC_ETHEREUM` | Ethereum mainnet RPC; required. |
| `LIGHTER_TEST_PRIVATE_KEY` | Funded deployer/Safe owner/asset manager; required and never logged. |
| `LIGHTER_DEPOSIT_USDC` | Target total Lighter collateral; default from the conservative ETH order size plus a buffer and never below the current direct-deposit minimum. |
| `LIGHTER_POSITION_USDC` | Optional small ETH position notional override. |
| `LIGHTER_MAX_SLIPPAGE` | Open and close maximum slippage, default 2%. |
| `LIGHTER_API_KEY_INDEX` | API-key slot, default `MIN_API_KEY_INDEX`; reject reserved slots through existing validation. |
| `LIGHTER_DEPOSIT_TIMEOUT` | Bounded wait for L1-to-Lighter credit, with a live-appropriate default. |
| `LIGHTER_WITHDRAW_TIMEOUT` | Bounded wait for the secure withdrawal to become claimable. |
| `LIGHTER_RESUME_PHASE` | Optional explicit `claim` resume; no automatic inference or general workflow engine. |
| `LIGHTER_RUN_DIR` | Persistent new directory for state and reports; default to a timestamped directory below `~/.tradingstrategy/examples`. |
| `TRADING_STRATEGY_API_KEY` | Optional for a code-defined universe. |
| `ETHERSCAN_API_KEY` | Optional deployment verification. |

The script must reject `SIMULATE=true` rather than accidentally combining a
forked Ethereum state with live Lighter public state. It must also preflight:

- Ethereum chain id 1;
- enough ETH for deployment, custody calls and redemption;
- enough native Ethereum USDC for the selected collateral;
- a new run directory and deployment-record path;
- the pinned optional Lighter SDK revision used by the existing eth-defi
  tutorial; and
- an ETH perpetual market whose live precision and minimums can support the
  requested order.

Lighter's current documentation contains both a 1 USDC direct-contract minimum
and a 5 USD Ethereum deposit table. Do not guess which one the service will
enforce. Keep the core constant aligned with the direct contract integration,
but make the live tutorial's default additional deposit at least 5 USDC and
fail on current market/API minimums before broadcasting.

## Secret boundary

The generated Lighter API private key is allowed only in memory and in the
paired operator deployment JSON created mode `0600` by
`lagoon-deploy-vault`. Apply all of these rules:

- create `LIGHTER_RUN_DIR` mode `0700` and use a fresh, persistent path; do not
  use `TemporaryDirectory`, because an interrupted live run must not lose its
  only copy of the key;
- parse the operator JSON directly in Python and validate that it is a regular
  file owned by the current user with no group/other permission bits before
  reading it;
- never put the Lighter API private key in Typer arguments, environment
  variables, checkpoint data, command logs, exception strings or raw SDK
  object representations;
- pass only public `LIGHTER_ACCOUNT_INDEX` and ordinary deployment addresses to
  the strategy/CLI environment;
- do not log a full `SignerClient`, signed transaction, SDK response, operator
  JSON, or caught exception whose payload has not been reduced to an approved
  public error code/message;
- implement that reduction as one small wrapper which exposes only the
  exception class and an allow-listed public status/code, including failures
  during `SignerClient` construction;
- do not enable shell/Python tracing or dump the process environment;
- close SDK/API sessions in `finally` blocks and drop the in-memory secret
  reference after the final signed request; and
- scan captured output and every non-operator artefact for both the full secret
  sentinel and a distinctive substring in automated tests.

The Ethereum deployer private key follows the existing manual-script pattern:
it is placed only in the patched in-process command environment, never in CLI
arguments or logged environment dumps.

## Command-first tutorial flow

Use a small `run_cli(args, env)` wrapper copied from the existing Lagoon manual
scripts. It may preserve harmless host variables needed by the process, but it
must log only the command name/options and never the patched environment.

### 1. Deploy and retain the key safely

Invoke the actual Typer app:

```text
lagoon-deploy-vault --generate-lighter-api-key
```

Pass configuration through the patched environment, including Ethereum RPC,
deployer key, `VAULT_RECORD_FILE`, zero tutorial fees, Safe owner and API-key
index. Read these values from the resulting operator JSON:

- vault, Safe and trading strategy module addresses;
- deployment block;
- public Lighter account index and API-key index; and
- the Lighter API private key, held only in a local variable.

Assert that the public text, Markdown and runtime deployment JSON contain the
account index but not the private key. Print only the public addresses, account
index, API-key slot and report path.

### 2. Initialise state and the external account

Construct the common command environment using
`minimal_lighter_strategy.py`, `ASSET_MANAGEMENT_MODE=lagoon`, the emitted vault
and module addresses, state/cache paths, and the public
`LIGHTER_ACCOUNT_INDEX`.

Run:

```text
init
correct-accounts
```

Use the same non-interactive manual-test convention as the other Lagoon scripts
for `correct-accounts`. Reload the state and assert exactly one open exchange
account position with protocol `lighter` and the expected account index. Do not
open this synthetic position with direct state APIs.

### 3. Add accounted capital and deposit it to Lighter

Resolve a conservative trade/collateral plan from current ETH market details.
The deployment activation has already subscribed and deposited its fixed
minimum. For any additional amount:

1. fetch current Safe USDC plus public Lighter total equity;
2. assert every component and total are finite and non-negative;
3. approve the vault and call the existing eth-defi
   `LagoonVault.request_deposit()` as the deployer, without posting a valuation
   or settling in direct Python;
4. run `lagoon-settle` so trade-executor observes the pending subscription,
   posts the active pre-subscription NAV, settles it and records the reserve
   inflow;
5. call `LagoonVault.finalise_deposit()` to claim the settled shares, then
   assert the share and Safe USDC deltas at raw-token precision; and
6. call `deposit_usdc_from_lagoon_safe_into_lighter()` for exactly the newly
   settled Safe balance, then wait with `LIGHTER_DEPOSIT_TIMEOUT` until public
   Lighter collateral reflects it.

The subscription settlement has different event semantics from an ordinary
no-queue NAV post: its `NewTotalAssetsUpdated` value excludes the pending
deposit, while settlement then moves that deposit into the active Safe. Verify
the request/settlement/share and Safe deltas separately at this point. Apply
the shared Safe-plus-Lighter NAV checkpoint only after the subsequent Lighter
deposit and no-queue `lagoon-settle`.

Never fund the Safe with an unaccounted direct USDC transfer. Record the
deployer's USDC balance immediately before the subscription so final recovery
can be measured honestly.

### 4. NAV checkpoint after deposit

Run:

```text
lagoon-settle
show-valuation
```

Then call the shared checkpoint assertion described below with expected ETH
position `flat`. This proves the executor exchange-account position, aggregate
portfolio value and posted Lagoon NAV all include the deposited Lighter equity.

### 5. Open ETH-USD long and sync NAV

Use the importable eth-defi helper around the official SDK to submit an IOC
market buy with bounded slippage. Use a unique client-order index and wait on
the public account endpoint until the signed ETH position is strictly positive.
Do not accept order submission alone as proof of a fill.

Before submitting, require the public pre-state to be flat with no active ETH
orders. If an earlier run may already have submitted an order, abort with
recovery instructions rather than choosing a new client-order index and
possibly doubling exposure.

Run `lagoon-settle` and `show-valuation`, then checkpoint with expected ETH
position `long`. The checkpoint must use Lighter's canonical total asset value,
which includes unrealised PnL, rather than collateral alone.

### 6. Close the exact long and sync NAV

Read the actually filled positive ETH size, convert it using the live market
precision and submit an ask with `reduce_only=True`. Never reuse the requested
open amount: a partial fill must not turn the account short. Wait until the
public account reports exactly flat, then run `lagoon-settle`,
`show-valuation`, and the flat checkpoint.

Before withdrawal, also assert there are no active ETH orders. If a close fails
or the position is not flat, stop and print secret-free recovery instructions;
do not continue to withdrawal or Lagoon redemption.

### 7. Securely withdraw to the Lagoon Safe

Use secure withdrawal only:

1. fetch `available_balance` after the account is flat and no orders remain;
2. quantise down to USDC precision using the verified withdrawal minimum/fee
   behaviour and leave only unavoidable sub-minimum dust;
3. submit `SignerClient.withdraw()` with the generated API key, USDC asset and
   perps route;
4. persist only its public transaction hash/sequence, requested amount and
   phase in one mode-`0600` checkpoint file;
5. poll authenticated withdrawal history until that exact request becomes
   `claimable`, using bounded retries and a long operator-configurable timeout;
   and
6. take the claimable raw amount from the exactly matched history entry—not
   from the original request—and call the new eth-defi Safe-claim helper to
   execute `withdrawPendingBalance(safe, USDC_ASSET_INDEX, claimable_raw_amount)`
   through `TradingStrategyModuleV0.performCall()`.

The request and claim helpers must be separately callable so an operator can
resume at the claim step without submitting a duplicate withdrawal. Match the
request by its public transaction identifier, account, asset, route and amount;
never choose “the latest withdrawal” without validation. Resume is deliberately
small: `LIGHTER_RESUME_PHASE=claim` loads the one checkpoint and operator JSON,
refuses to call `withdraw()` again, recreates only the auth needed to poll the
exact request, and proceeds to claim. A normal run must refuse to submit a new
withdrawal when that checkpoint already exists.

Once the withdrawal request is accepted, some value is in flight and is in
neither current Lighter equity nor Safe USDC. Do not run `lagoon-settle` or the
full NAV checkpoint between request and claim, including on resume. Immediately
before claim, use only a reduced safety check: finite non-negative Lighter
equity, flat ETH position, no active orders, and the exact matched request in
`claimable` state.

After the claim, assert the Safe USDC increase equals the history-derived
claimable raw amount and the Lighter account is flat with zero or documented
sub-minimum dust. Run one more `lagoon-settle` and `show-valuation` checkpoint
to prove NAV moved from the external account to the Safe without double
counting or going negative.

### 8. Redeem to the deployer

Run the existing:

```text
lagoon-redeem
```

This command redeems all shares held by the asset-manager/deployer and updates
state. Do not add an asset-manager “sweep Safe” call, which would bypass Lagoon
share accounting.

Assert:

- deployer Lagoon share balance is zero;
- the deployer's USDC increase equals the command's redeemed amount at raw
  precision;
- the Safe remainder is no larger than the bound computed from the deployed
  Lagoon vault's conversion functions and remaining total supply (do not
  hard-code one raw unit); and
- Lighter remains flat and its remaining equity is non-negative and no more
  than the documented withdrawal dust.

Report the initial subscription, final redeemed amount, realised round-trip
PnL/fees, remaining Safe dust and remaining Lighter dust separately. Do not say
the original principal was recovered if trading or protocol fees made the net
redemption smaller.

## Checkpoint assertion

Implement one small tutorial-local read-only helper and call it after every
ordinary NAV sync and before every irreversible step except the in-flight
withdrawal claim, which uses the reduced check defined above. It should:

1. fetch the public Lighter account and canonical total equity;
2. assert total equity is finite and **never negative**;
3. assert the expected ETH position state (`flat` or positive `long`), and
   reject a short at every phase;
4. reload trade-executor state and assert the Lighter exchange-account position
   quantity equals the public total equity within explicit decimal/raw-unit
   tolerance;
5. assert the state reserve quantity equals Safe USDC at the settlement block
   within raw-token tolerance;
6. assert every reserve, exchange-account and total portfolio equity component
   is finite and non-negative;
7. fetch Safe USDC at the settlement block;
8. inspect the latest `NewTotalAssetsUpdated` event emitted by the just-finished
   `lagoon-settle`; and
9. assert posted NAV equals Safe USDC plus Lighter total equity, with no double
   counting and no stale pre-command event accepted.

Capture the block before each `lagoon-settle` and require the matching event in
that block range. Because the public Lighter read is not block-pinnable, allow a
small documented raw-USDC tolerance for movement between the command's read and
the verification read; never turn that tolerance into permission for a
negative value. A negative or malformed equity response must abort before the
next order, withdrawal, NAV transaction or redemption.

## Minimal eth-defi additions

Keep the upstream additions narrow and reuse the already reviewed tutorial
logic:

1. Extract the order-sizing and public-position helpers from
   `scripts/lagoon/lagoon-lighter-trade-example.py` into one importable Lighter
   module. Expose separate `open` and `close` functions so NAV can be checked
   between them. Update the old tutorial to import those functions, preventing
   two implementations from drifting.
2. Add a secure-withdrawal request helper around the pinned official SDK's
   `SignerClient.withdraw()`, plus an authenticated withdrawal-history poller
   that returns a small public result object. Do not introduce the SDK as a
   mandatory eth-defi or trade-executor dependency.
3. Extend `eth_defi.lighter.lagoon` with a Safe-only claim helper that builds
   the canonical ZkLighter ABI call and broadcasts it through the existing
   module/guard path. Reuse `broadcast_tx()` and token conversion; reject a
   receiver other than `vault.safe_address`.
4. Update `README-lighter-guard.md` and the existing eth-defi trading tutorial
   to document the request/claim distinction and the exact pinned SDK API that
   was verified.

Retain and test trade-executor's command-level negative-equity guard in
`tradeexecutor.exchange_account.lighter._validate_lighter_equity()`. The
tutorial-local check is defence in depth: `lagoon-settle` itself must reject a
negative, non-finite or malformed Lighter value before state mutation or NAV
broadcast.

Before coding the withdrawal helper, run a focused compatibility spike against
the exact SDK revision already pinned by the eth-defi tutorial. Confirm the
`withdraw()` amount units, returned transaction identifier, auth-token creation,
withdrawal-history schema/status values, current withdrawal minimum/fee, and
whether claim amount is preserved at raw USDC precision. Treat a mismatch as a
plan update, not a reason to add generic SDK abstractions.

## Automated test coverage

### eth-defi unit and fork tests

Add focused tests for:

- market minimum/precision sizing and invalid inputs;
- open waits for a positive observed fill;
- close uses the observed fill, asks, and is reduce-only;
- a short/zero position is rejected by the close helper;
- secure withdrawal uses USDC/perps, the configured API-key slot and no
  Ethereum key;
- withdrawal-history matching ignores unrelated requests and recognises only
  the exact claimable request;
- timeout/retry behaviour is bounded and secret-free;
- the Safe claim encodes the expected receiver, asset index and exact raw
  amount; and
- the real forked guard accepts the encoded claim to the Safe and rejects any
  other receiver.

Mock the new order/withdrawal helper boundary in trade-executor tests rather
than importing or patching the optional SDK classes there. Mark the focused
eth-defi SDK tests skip-if-missing so the SDK remains an optional tutorial
dependency.

### Typer black-box lifecycle test

Add a test alongside `tests/lagoon/test_lagoon_lighter_deploy_e2e.py` that runs
the tutorial orchestration against an external Ethereum Anvil fork and invokes
the actual Typer app through Click/Typer's command entry point, as the other
deploy tests do. It must not call command callback functions directly.

Keep real in the test:

- Lagoon/Safe/guard deployment;
- native Ethereum USDC funding and Lagoon subscription/settlement;
- Safe `approve` and Lighter L1 `deposit` transactions;
- `init`, `correct-accounts`, every `lagoon-settle`, `show-valuation`, and
  `lagoon-redeem` command;
- strategy state writes and exchange-account balance updates;
- NAV event emission and raw-value checks; and
- all feasible guard/calldata validation.

Mock only the parts an Anvil fork cannot simulate:

- Lighter sequencer account creation/public state advancement;
- API-key registration observation;
- official SDK order signing/submission and fills;
- authenticated withdrawal-history/proof advancement to `claimable`; and
- the proof-created L1 pending balance/claim effect. Because a static Anvil fork
  cannot create the new proof-backed pending balance, validate the exact real
  claim calldata against the forked guard with `eth_call`, then use
  `fund_erc20_on_anvil()` to credit the Safe as the explicit mocked Lighter
  claim effect. Do not claim that `withdrawPendingBalance()` executed
  end-to-end, mock Lagoon redemption, or directly edit trade-executor state.

Drive mocked public snapshots through the real lifecycle:

```text
activation -> funded/flat -> long with PnL -> flat -> withdrawn to Safe
```

At each snapshot assert the expected state position, Safe reserve, emitted NAV
and total portfolio equity. Include a negative-equity snapshot and prove it
causes a non-zero command result with no state mutation, no NAV event and no
subsequent order/withdraw/redeem call.

Use a sentinel API private key and assert it appears only in the mode-`0600`
operator JSON, never in captured stdout/stderr, `caplog`, exceptions, state,
runtime deployment JSON, Markdown/text reports, checkpoint data, or any other
file below the run directory.

### Live manual acceptance

The live tutorial passes only when:

- every Typer command returns successfully;
- the four NAV checkpoints (funded, long, flat, withdrawn-to-Safe) match their
  Safe-plus-Lighter components;
- equity is never negative at any observation or stored checkpoint;
- the ETH position is positive only during the intended long phase and is flat
  before withdrawal;
- the secure withdrawal is matched and claimed to the Safe;
- all deployer shares are redeemed and all redeemable USDC is returned; and
- the generated Lighter API private key is absent from all public output.

## Implementation order

1. Verify the pinned SDK withdrawal API/schema with the focused spike.
2. Extract the existing eth-defi order helpers and add the narrow secure
   withdrawal/Safe claim helpers with unit and fork tests.
3. Implement the Python tutorial using existing Typer commands for deployment,
   state, NAV and redemption.
4. Add the checkpoint assertions and failure/recovery messages.
5. Add the Anvil-backed Typer black-box lifecycle test, mocking only the
   sequencer/SDK/proof seams.
6. Run focused eth-defi and trade-executor suites, then run the tutorial live
   once with the designated funded Lighter test deployer and retain the
   secret-bearing run directory securely.
