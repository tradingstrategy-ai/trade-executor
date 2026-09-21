# Automated Lighter cash management

> **Superseded implementation notes:** The first draft's generic receipt
> reconciliation, strategy-owned Lighter session, timeout strategy parameter,
> and pending-transfer resumption approach were replaced by
> [the simplification plan](lighter-cash-management-simplification.md). Keep
> this document as historical context only.

## Goal

Automatically move idle Lagoon Safe USDC to its Lighter account and return
enough Lighter collateral to the Safe before Lagoon redemptions settle. A
Lighter withdrawal normally takes about 20 minutes to become claimable. The
executor may block synchronously during this wait; this is intentional and is
documented as part of the live strategy operation.

Every movement must be a first-class `TradeExecution` in the existing pipeline:

```text
decide_trades()
  -> post_process_trade_decision()
  -> approval
  -> prepare_sorted_trades()
  -> checkpoint
  -> execution_model.execute_trades()
  -> GenericRouting
  -> LighterRouting
  -> success, failure, or pending continuation
```

Do not add a custody runner, journal or execution subsystem beside this
pipeline.

## Scope

The first version supports one Ethereum Lagoon vault and one Safe-owned Lighter
account.

- `ExchangeCashManager` is exchange-neutral decision arithmetic so another
  exchange can reuse it later.
- Execution is Lighter-specific through `LighterRouting`; do not build a
  generic multi-exchange execution framework now.
- Leave the existing synthetic exchange-account position bootstrap unchanged.
  It is state initialisation, not a custody transfer.
- Keep automatic cash management disabled by default and disabled in
  backtests. Moving custody does not change strategy equity, so no simulated
  transfer executor is needed.
- `lighter-move-funds` remains a stopped-executor diagnostic and recovery
  command, but uses the same router and state transitions as automation.

Never place the Lighter private key, auth token, signed request, or raw
authenticated SDK response in logs, exceptions, state, reports, or frontend
metadata.

## Decision policy

Add a pure `ExchangeCashManager`. It receives immutable `Decimal` inputs and
returns no action, a deposit amount, or a withdrawal amount. It does not access
state, Web3, Lighter credentials, or the network.

Inputs:

- Safe USDC from the preceding treasury sync;
- Lighter total equity and available balance;
- pending Lagoon redemptions;
- configured Safe cash buffer;
- configured Lighter free-collateral buffer; and
- minimum transfer amount.

The desired Safe balance is:

```text
pending Lagoon redemptions + Safe cash buffer
```

Policy:

1. If an exchange-account transfer is unfinished, do nothing.
2. If the Safe is below the desired balance, withdraw the shortfall, capped at:

   ```text
   Lighter available balance - free-collateral buffer
   ```

3. Otherwise deposit Safe cash above the desired balance. The policy receives
   the latest treasury-synchronised Safe reserve and does not make a direct
   on-chain balance read from `decide_trades()`.
4. Never count pending Lagoon Silo deposits as deployable Safe cash.
5. Never deposit cash reserved for redemptions or the Safe buffer.
6. Ignore transfers below the configured minimum.
7. Reject non-finite or negative inputs or results.
8. Revalidate available collateral immediately before submission. Abort rather
   than silently resize an already checkpointed trade.

The current Lighter strategy does not place directional orders, so do not add
future same-cycle margin-commitment machinery in this change.

Configuration:

- `lighter_cash_management`, disabled by default;
- `lighter_safe_cash_buffer_usd`;
- `lighter_free_collateral_buffer_usd`; and
- `lighter_min_transfer_usd`; and
- `lighter_withdrawal_timeout`, defaulting to 30 minutes.

Validate these at startup. Add `LIGHTER_OPERATOR_RECORD_FILE` to the shared
`start` options and require its mode-0600 operator record when live automation
is enabled.

Document the operational behaviour in the Lighter README and strategy runbook:
withdrawals normally take about 20 minutes, the executor process intentionally
waits during this period, no later trades are started, and the 30-minute
timeout is a safety limit rather than the expected withdrawal duration. Tell
operators to allow the process to finish or rerun it after a timeout; rerunning
continues a checkpointed public request.

## Trade and accounting lifecycle

When the manager chooses an action, `decide_trades()` creates one flagged
exchange-account transfer and returns it. Every custody `TradeExecution`
created by `decide_trades()` must be returned exactly once.

Update `create_exchange_account_transfer()`:

- deposit: positive quantity and `planned_reserve=amount`;
- withdrawal: negative quantity and no Safe reserve allocation;
- fixed assumed price of 1 USDC; and
- `TradeFlag.external_account_transfer` required.

Automatic transfers created by `decide_trades()` also carry one explicit
`TradeFlag.automatic_exchange_account_transfer`. Diagnostic
`lighter-move-funds` transfers do not. Use this single discriminator for
startup resume and cleanliness handling; do not infer automation from notes or
add a separate lifecycle state machine.

Add narrow existing-pipeline support:

- `post_process_trade_decision()` validates a positive transfer value and 1:1
  price, but does not require an AMM price structure or price-impact check.
- `get_execution_sort_position()` checks the transfer flag before
  `is_reduce()`, which does not support exchange-account pairs.
- Withdrawals sort before ordinary buys. Deposits sort after ordinary buys and
  before the final idle-credit sweep.
- `collect_post_execution_data()` skips AMM price collection for the flagged
  transfer.
- Unflagged exchange-account trades remain rejected.

Extend the standard state transitions:

- `State.start_execution()` allocates deposit reserves once through the normal
  reserve-allocation helper. Withdrawals allocate no Safe reserve.
- Deposit success consumes that existing allocation without another debit.
- Withdrawal success credits the exact verified Safe claim once.
- Both directions update the exchange-account internal share-price state so
  principal is a capital flow, not PnL.
- Update `TradingPosition.can_be_closed()` so an exchange-account position is
  not closed solely because its quantity reaches zero. A full withdrawal leaves
  the position in `open_positions` for future deposits and valuation sync.
- Exempt flagged exchange-account transfers in
  `freeze_position_on_failed_trade()` and the sequential failure path. A failed
  custody operation records a failed trade but never moves the persistent
  exchange-account position to `frozen_positions`.
- A definitely unsubmitted deposit releases its allocation. If a Safe debit is
  possible or uncertain, use the existing
  `retain_reserve_allocation_on_failure` mechanism until recovery proves
  otherwise.

Refactor `record_exchange_account_transfer()` into a compatibility wrapper over
these state transitions or remove it after all callers migrate. Protocol code
must not adjust reserves or call `trade.mark_success()` directly.

## Lighter router

Add `LighterRouting` and `LighterRoutingState`. Teach the existing protocol
configurator and router matching to produce a Lighter-only
`ProtocolRoutingConfig`, selected from the pair's exchange-account protocol
metadata. Do not attach `LighterRouting` to the shared `exchange_account`
configuration used by GMX and Derive; their existing `routing_model=None`
behaviour remains unchanged.

`GenericRouting` then handles Lighter like its other configured protocols:

- create the Lighter routing state;
- select `LighterRouting` for the exchange-account pair;
- set `trade.route`;
- call `setup_trades()` and `settle_trade()`; and
- require sequential execution.

Remove broad exchange-account assertions from `GenericRouting` and
`EthereumExecution` only for a flagged trade with a configured router.

### Deposit

`LighterRouting.setup_trades()` prepares and attaches both Safe transactions:

1. USDC approval; and
2. Lighter deposit.

Store both as normal `trade.blockchain_transactions` before broadcast. The
existing pre-broadcast callback checkpoints them. Normal broadcasting and
`settle_trade()` confirm the receipts, verify the Lighter collateral increase,
and call `State.mark_trade_success()`.

Refactor eth-defi helpers where needed so they prepare or return both
transactions instead of hiding the approval transaction.

### Withdrawal

A secure withdrawal starts with an authenticated Lighter SDK request and has no
EVM transaction yet. It is intentionally handled synchronously because a
withdrawal normally takes about 20 minutes.

`LighterRouting.setup_trades()`:

1. checkpoints the started trade and custody baselines;
2. submits the withdrawal request once;
3. stores only its public request ID;
4. marks the trade broadcasted; and
5. checkpoints again.

The same routing call then polls Lighter claimability until the withdrawal is
claimable or `lighter_withdrawal_timeout` expires. Polling must log progress at
info level without logging credentials or authenticated responses. While this
call is waiting, sequential execution cannot start later trades. Do not add a
new execution status or a recurring cycle-defer protocol for this delay.

When claimable, it prepares and checkpoints the claim transaction, broadcasts
it using the existing transaction builder, verifies the exact Safe credit and
Lighter debit, and completes the normal success transition. A timeout leaves
the checkpointed public request ID and started/broadcasted trade in state and
raises an actionable error telling the operator to rerun the executor. The
retry path must resume the existing request and must never submit a second
withdrawal.

If the process is restarted while a flagged automatic withdrawal is pending,
initialise routing first and let `LighterRouting` resume the same request by
polling and claiming it synchronously before normal cleanliness, treasury or
account checks. This is the only startup recovery special case. Manual
transfers, sibling unfinished trades and automatic transfers without a public
request ID remain unclean and fail closed.

The no-resize rule applies before the authenticated withdrawal submission only.
Lighter may return a final claimable amount different from the requested amount
because of precision or fee handling. At completion, populate `executed_*`
from the authoritative Lighter withdrawal result and measured Safe balance
increase, not from `planned_quantity`. If the verified Lighter debit and Safe
credit differ, record the residual explicitly as a protocol fee or realised
PnL; never hide it as transfer principal or leave the trade pending solely
because the final amount differs from the plan.

## Restart behaviour

Use existing planned, started, broadcasted, success and failed statuses. Do not
add a Lighter phase state machine.

- Deposits retain their deterministic Ethereum transaction hashes in
  `trade.blockchain_transactions`.
- Withdrawals retain the public Lighter request ID in `trade.other_data` and
  the eventual claim transaction in `trade.blockchain_transactions`.
- Claimability is read from Lighter and is not persisted as another phase.
- A known transaction hash or request ID resumes from its authoritative status.
- A restarted planned, started, or broadcasted trade without a public
  identifier remains fail-closed for stopped-executor recovery. Do not infer a
  match from amount, time, or balance deltas and never resubmit automatically.
- A deposit in started or ambiguous state keeps its reserve allocation.
- Only a newly returned and checkpointed trade in the current
  `decide_trades()` result may make its first submission.

No second journal is needed.

## Valuation and withdrawal blocking

Lighter may debit one custody side before crediting the other. During this
window, revaluation would create a false PnL `BalanceUpdate` and could post a
reduced Lagoon NAV.

Do not expose the intermediate Lighter withdrawal state to normal strategy
cycles. Sequential execution blocks inside the routed withdrawal until the
claim succeeds or the configured timeout expires, so no later planned trade,
valuation, account check or statistics update can observe an incomplete
transfer.

Before the withdrawal trade is created, the Lagoon liquidity check below must
prevent this cycle from posting NAV based on Lighter equity which is about to
leave the account. Keep the last fully verified on-chain valuation unchanged.
After a successful claim, finish the current execution without publishing a
second NAV in the same cycle; the next normal cycle syncs the Safe balance,
posts fresh NAV and settles the redemption queue.

At startup, first create the universe, `GenericRouting` and routing state, then
resume an automatic pending transfer synchronously before cleanliness, treasury,
valuation or account work. Keep the default `State.check_if_clean()` behaviour
unchanged; only the exact recoverable automatic transfer is handled by the
startup recovery path. Any sibling unfinished trade, every manual pending
transfer and an automatic transfer without a public transaction/request
identifier remain unclean.

Leave the existing asynchronous-vault resolver in its current post-treasury
position; changing its ordering is outside this feature.

## Lagoon redemption settlement

Before Lagoon submits a settlement transaction, calculate:

```text
settlement liquidity = Safe USDC + pending underlying deposits in the Silo
```

When Lighter automation is enabled and required redemption assets exceed this
liquidity:

- do not post NAV or submit settlement in this cycle, even before the
  withdrawal trade exists. Publishing NAV and then submitting a withdrawal in
  the same cycle creates a stale-high on-chain valuation;
- keep the redemption queue pending;
- update `state.sync.treasury.pending_redemptions`;
- log the Safe shortfall; and
- let `decide_trades()` create the Lighter withdrawal.

Thread `lighter_cash_management` explicitly into `LagoonVaultSyncModel`, which
cannot currently read strategy parameters. Apply the liquidity check before an
otherwise-permitted GuardV0 settlement transaction. Keep liquidity deferral
distinct from GuardV0 amount/window deferral and from an empty queue.

Return enough liquidity-deferral information for `StrategyRunner.tick()` to
continue only to the normal `decide_trades()` and execution pipeline needed to
create the withdrawal. Skip revaluation, NAV, statistics and unrelated strategy
work in this special cycle. If `decide_trades()` does not return the expected
single automatic withdrawal, end the cycle without posting NAV or attempting
settlement. The execution pipeline then waits synchronously for the claim. A
successful claim is reflected by the next normal cycle, which posts fresh NAV
and settles the queue.

When automation is disabled, retain existing Lagoon behaviour. After a claim
completes, the normal treasury sync posts NAV and settles the queue.

## Implementation tasks

1. Add the pure `ExchangeCashManager` and Lighter strategy parameters.
2. Extend exchange-transfer creation and standard start/success/failure
   accounting, including open-zero exchange positions, authoritative executed
   withdrawal amounts and no freezing for custody failures.
3. Add the automatic-transfer flag, Lighter-only route registration,
   `LighterRouting`, transaction preparation, synchronous withdrawal polling
   and restart recovery.
4. Pass every custody trade from `decide_trades()` through normal
   post-processing, approval, sorting, checkpointing and execution.
5. Initialise routing before startup recovery and add the pre-valuation
   liquidity check. Do not add recurring pending-transfer checks to unrelated
   strategy, trigger or account paths.
6. Add Lagoon Safe/Silo liquidity deferral before same-cycle NAV posting while
   still allowing the normal `decide_trades()` withdrawal path to run.
7. Thread `LIGHTER_OPERATOR_RECORD_FILE` through `start` and routing-state
   creation without logging or serialising its private key.
8. Move `lighter-move-funds` onto the same router and state transitions without
   constructing a full strategy runner.
9. Update `tradeexecutor/exchange_account/README-Lighter.md` and
   `deploy/README-lighter-ai.md` with configuration, delayed-withdrawal and
   stopped-executor recovery instructions.

## Correctness invariants

1. Safe reserve plus Lighter account value is unchanged by a completed
   transfer, apart from explicit protocol fees or separately recorded PnL.
2. No reserve, equity, available balance or free collateral becomes negative.
3. Transfer principal never creates a PnL `BalanceUpdate`.
4. Reserve accounting debits or credits each transfer exactly once.
5. At most one unfinished Lighter transfer exists.
6. Every irreversible operation follows a state checkpoint sufficient for
   fail-closed recovery.
7. A transfer that is pending or will be submitted in the current redemption
   cycle never causes NAV posting, account correction, valuation, statistics,
   or another transfer.
8. A pending withdrawal never executes later trades against expected proceeds.
9. A full withdrawal leaves the Lighter position open at zero.
10. Every Ethereum transaction is attached to the trade; only public Lighter
    identifiers are stored in `other_data`.
11. No path logs or serialises Lighter credentials or authenticated payloads.

## Tests

Follow the repository convention of a small number of tests with several
assertions.

### Policy

One parameterised test covers deposit, full and partial withdrawal, no-op,
minimum amount, an existing pending transfer, buffers and invalid inputs.

### State and routing

Use two focused tests:

1. Happy path for both directions through post-processing, approval, sorting,
   start, `LighterRouting` and success. Assert exact reserve changes,
   transaction/request tracking, principal-neutral share-price accounting, no
   principal PnL, authoritative claim amounts which differ from the plan, and an
   open zero-quantity position after full withdrawal.
2. Failure/restart path covering an unflagged exchange trade, definite failure,
   ambiguous deposit allocation retention, no-ID fail-closed recovery, and a
   withdrawal timeout stopping later trades without freezing the position or
   resubmitting. Include one automatic pending transfer plus an
   unrelated/manual unfinished trade to prove startup recovery remains scoped.

### Black-box integration

Add `tests/lagoon/test_lagoon_lighter_cash_management.py`, following the
structure of the existing fixed-fork black-box tests in
`test_lagoon_lighter_deploy_e2e.py`, `test_lagoon_lighter_test_trade.py` and
`test_lighter_move_funds.py`:

- use `AnvilForkPool`, `ETHEREUM_MIDNIGHT_BLOCK`, `evm_snapshot_revert()` and
  the unlocked native-USDC whale;
- connect with `create_multi_provider_web3()` and deploy and fund a real Lagoon
  vault with native USDC and
  `lighter_deployment=LighterDeployment.create_ethereum()` so its guard has the
  canonical Lighter contract and asset index whitelisted; do not rely on
  `any_asset=True` for Lighter permissions;
- add a dedicated strategy under `strategies/test_only/`; do not change
  `minimal_lighter_strategy.py`, which is shared by existing tests. Its first
  cycle opens the synthetic Lighter position with the existing spoofed helper
  and does not return that bootstrap trade. Later cycles return only the
  transfer produced by the real `ExchangeCashManager`. Enable
  `lighter_cash_management` and set `lighter_safe_cash_buffer_usd=20` explicitly
  in this strategy;
- invoke the real Typer `init`, single-cycle `start`, `check-accounts` and
  `correct-accounts --dry-run` commands against one persisted state file;
- set `RUN_SINGLE_CYCLE=true`, `UNIT_TESTING=true`, `CACHE_PATH` and
  `TRADING_STRATEGY_API_KEY` for every `start` invocation. Skip the test unless
  both `JSON_RPC_ETHEREUM` and `TRADING_STRATEGY_API_KEY` are available;
- create the same private mode-0600 operator JSON used by the existing Lighter
  CLI tests and pass it as `LIGHTER_OPERATOR_RECORD_FILE`;
- use one test function with a numbered docstring and matching numbered body
  comments, several assertions per lifecycle phase, the shared
  `fork:ethereum:midnight` xdist group and an extended pytest timeout.

Keep the following parts real on Anvil:

- Lagoon vault, Safe, strategy module and guard configuration;
- investor deposit, share issue, redemption request, NAV posting, settlement
  and final investor claim;
- native USDC balances and transfers;
- Safe approval and canonical Lighter L1 deposit calls;
- `decide_trades()` through approval, sorting, `GenericRouting`,
  `LighterRouting`, `LagoonExecution` and normal state accounting;
- state checkpoints, process-style state reloads, Ethereum broadcasts,
  receipts and deposit transaction attachment to `TradeExecution`.

Mock only behaviour which the Ethereum fork cannot reproduce:

- Lighter public equity/account reads;
- the authenticated SDK withdrawal request;
- withdrawal-history and claimability reads;
- the proof-backed Lighter L1 claim effect which a static fork cannot create.

Use a small mutable `FakeLighterSequencer` fixture with equity, pending amount,
public request id, claimable state and call counters. Patch each module where it
imports the low-level Lighter session/API functions used by routing, automatic
pair configuration, NAV valuation, `check-accounts` and `correct-accounts`.
Cover public equity, account-by-index, collateral waiting, withdrawal request
and withdrawal-history/claimability reads without allowing a real HTTP call.
Do not patch `ExchangeCashManager`, either router, `LagoonExecution`, state
accounting or Lagoon synchronisation. A withdrawal request returns one stable
public id and the fake must fail the test if production submits it twice.

The Safe-to-Lighter deposit remains a real call to the forked canonical L1
contract. The mocked collateral waiter may update fake sequencer equity only
after observing the successful deposit transaction.

Do not depend on an unproven Lighter pending-balance storage forge. A static
fork cannot reproduce the zk proof-backed state needed for a successful
`withdrawPendingBalance()` call. Follow the established Lighter Anvil pattern:

1. Let production routing build the real
   `withdrawPendingBalance(safe, USDC asset index, amount)` calldata.
2. Validate that calldata and the Safe receiver through the real guard using
   `guard.functions.validateCall(...).call()`. Do not `eth_call` the module's
   `performCall()`, which would execute the unavailable pending-balance claim
   and revert on the static fork.
3. Mock only the unavailable Lighter claim effect by crediting the Safe on
   Anvil with native USDC, using the unlocked whale or
   `fund_erc20_on_anvil()`.
4. Let production routing verify the Safe balance delta and complete the
   existing withdrawal trade through normal state accounting.

Do not claim that the black-box test executed the Lighter claim transaction or
attach the Anvil materialisation operation as a production claim receipt. Cover
claim transaction construction and attachment separately in the focused
`LighterRouting` test, and leave proof-backed end-to-end claim broadcasting to
the manual mainnet smoke test.

The main black-box test runs this lifecycle:

1. Fund the real Lagoon vault with 100 USDC and run `init`.
2. Run one `start` cycle. `decide_trades()` deposits 80 USDC through the normal
   pipeline and retains the configured 20 USDC Safe buffer. Verify both EVM
   receipts, state accounting, Safe balance, fake Lighter equity and clean
   `check-accounts`. Match the current CLI contract by catching
   `SystemExit(0)` from a clean check.
3. Request redemption of all investor shares through the real vault and run
   another cycle. Verify Lagoon settlement defers for insufficient Safe/Silo
   liquidity, the previously posted NAV remains unchanged, and exactly one
   80 USDC Lighter withdrawal is requested. Make the fake withdrawal become
   claimable after a few mocked polls, validate the real claim calldata through
   the guard and materialise only the unavailable Lighter token effect on
   Anvil. Assert that this same `start` invocation waits, claims it, and does
   not run later trades while waiting.
4. Run the next normal cycle and verify exact-once reserve credit, fresh NAV
   sync and redemption settlement.
5. Finalise the investor redemption explicitly from the depositor account;
   `start` settles the queue but does not claim investor assets.
6. Run real `check-accounts` and `correct-accounts --dry-run`, reload state and
   verify it is clean.

Use phase-specific assertions instead of repeating the full invariant list
after every command. During the withdrawal wait assert that no later trade or
second request is started and that the pre-withdrawal NAV remains unchanged.
At the final boundary assert:

- Safe balance, executor reserve, Lighter equity, available collateral and
  exchange-account quantity never became negative;
- Safe plus Lighter plus any explicitly pending withdrawal conserved principal;
- exactly one deposit and one withdrawal trade exist and the withdrawal SDK
  request count is one across restarts;
- transfer principal created no PnL `BalanceUpdate` and reserve accounting was
  applied exactly once;
- the successful deposit receipts are attached to the deposit trade, while only
  the public Lighter request id is persisted for the mocked withdrawal;
- no private key or authenticated payload appears in state, logs or captured
  CLI output; and
- the full withdrawal leaves one open zero-quantity Lighter position and the
  final state passes `State.check_if_clean()`.

Add one compact fixed-fork restart test for the ambiguous no-public-id case.
Make the SDK double accept a withdrawal and fail before returning its id. After
reloading the persisted state, the transfer remains fail-closed, retains its
accounting allocation and never submits automatically.

Add one compact fixed-fork restart test for the public-id timeout case. Make
the fake claimability stay false until the synchronous wait reaches its
timeout, then reload the state and verify the same request ID is claimed on the
next run without another withdrawal submission.

Keep unrelated regressions in focused unit/state tests: Lighter-only route
selection must leave GMX and Derive on their existing routing behaviour; the
asynchronous-vault resolver ordering must remain unchanged; manual pending
transfers remain unclean; and the existing full-withdrawal exchange-account
test continues to leave an open zero-quantity position. Route selection should
construct independent Lighter and non-Lighter pairs rather than a mixed-protocol
Lighter universe, which is intentionally rejected. Do not duplicate these
checks in the fixed-fork lifecycle file.

## Delivery order

1. Pure policy and state accounting.
2. Lighter router and manual command migration.
3. Strategy decision, synchronous withdrawal polling and restart recovery.
4. Lagoon settlement coordination, documentation, and Anvil integration test.
5. Enable only after a manual mainnet smoke test covers deposit, delayed
   withdrawal, restart, NAV and redemption settlement.
