# Simplify Lighter cash management

## Goal

Simplify the first Lighter automatic cash-management implementation without
changing its core execution model:

```text
decide_trades()
  -> TradeExecution
  -> normal approval and execution pipeline
  -> GenericRouting
  -> LighterRouting
  -> normal trade accounting
```

Keep the implementation fail-closed. Startup must detect an unfinished
exchange-account transfer and abort before universe warm-up or any transaction.
It must not resume a Lighter request, claim a withdrawal, rebroadcast, repair,
or otherwise mutate the transfer.

The cleanup must correct the identified accounting and recovery issues while
removing unused or duplicated wiring. Do not add a second custody journal,
background worker, transfer scheduler, generic recovery framework, or new trade
status.

## Scope

The first version continues to support one Ethereum Lagoon Safe and one Lighter
exchange account. The design may use exchange-neutral names for genuinely
shared cash-policy concepts, but execution and recovery remain explicitly
Lighter-specific until another exchange needs them.

Preserve these existing decisions:

- `decide_trades()` creates and returns the custody `TradeExecution`;
- custody transfers pass through the existing router and execution pipeline;
- a Lighter withdrawal may synchronously wait up to the configured timeout;
- private operator-key material is loaded only at the Typer CLI boundary and is
  never written to strategy state or logs;
- the strategy obtains Safe cash through `PositionManager`, not a direct Web3
  call; and
- Anvil handles Ethereum transactions in integration tests, while only the
  unavailable Lighter API and sequencer behaviour is mocked.

## Target ownership

Use the following narrow ownership boundaries.

### Cash policy

`tradeexecutor.exchange_account.cash_manager` owns deterministic transfer
arithmetic only. It must not import state, Web3, Lighter, Typer, or routing
code.

Make `ExchangeCashManager` a frozen, slotted dataclass containing the static
strategy policy:

- Safe cash buffer;
- exchange free-collateral buffer; and
- minimum transfer amount.

Validate these fields in `__post_init__()`. Remove the current stateless class
shape and its artificial classmethod validation.

Rename `ExchangeCashManagementInput` to a name describing an observed balance
snapshot, for example `ExchangeCashSnapshot`. It contains only dynamic values:

- Safe USDC;
- exchange available USDC;
- pending Lagoon deposits; and
- pending Lagoon redemptions.

Keep the existing small decision object and its `deposit`, `withdraw`, or no-op
result. Do not introduce a command hierarchy or polymorphic policy interface.

The desired pre-settlement Safe balance is:

```text
max(pending redemptions - pending deposits, 0) + Safe cash buffer
```

Pending Silo deposits are not deployable idle cash, but they do offset the cash
that the Safe needs to receive from Lighter for the same Lagoon settlement.
This formula must match the liquidity check in `LagoonVaultSyncModel`.

### State accounting

`tradeexecutor.exchange_account.state` owns only generic state transitions:

- create a flagged external-account transfer trade;
- allocate reserve through `State.start_execution()` when any automatic or
  manual deposit starts;
- complete a verified transfer using caller-supplied executed amounts; and
- enforce non-negative reserve and exchange-account quantities.

Move policy evaluation out of this module. Remove Web3 receipt inspection from
this module. It must not decide that an exchange operation completed.

Remove `TradeFlag.automatic_exchange_account_transfer` and the `automatic`
argument to `create_exchange_account_transfer()`. No production code consumes
the flag. `TradeFlag.external_account_transfer` is sufficient to select the
normal custody-transfer behaviour.

Keep generic repair from refunding a failed external deposit whose Safe debit
is uncertain. Add an explicit guard in the generic `repair_trade()` entry point
so future callers cannot bypass the existing fail-closed selector and restore
possibly spent reserves.

Remove the separate manual-transfer reserve mutation from
`record_exchange_account_transfer()` and rename the remaining operation to
`complete_exchange_account_transfer()`. Change `lighter-move-funds` to call
`State.start_execution()` before it submits a deposit or withdrawal and persist
that state checkpoint before the external action. After recording the submitted
transaction hash, move the trade to `broadcasted` and checkpoint again before
waiting for final evidence. A manual deposit therefore has the same
`reserve_currency_allocated` bookkeeping as a routed deposit. The shared
completion helper must consume that existing allocation and must never debit
reserves a second time. Remove
`mark_exchange_account_transfer_broadcasted()` if direct lifecycle calls make
it redundant after this conversion.

### Lighter observations and routing

`tradeexecutor.exchange_account.lighter` owns Lighter public balance validation
and creation of the public available-balance reader. It must not mutate
strategy state or execute recovery.

Avoid a module-global `LIGHTER_SESSION` in the reference strategy. Reuse the
public Lighter reader created by the runtime adapter through the normal pricing
context. Extend `ExchangeAccountPricingModel` with one optional,
protocol-supplied available-balance callable and expose
`get_exchange_account_available_balance(pair)` through `GenericPricing` and
`PositionManager`. `GenericPricing` routes the pair to its existing
exchange-account pricing model. Do not introduce a second generic account
snapshot type, add a Lighter-specific method to `PositionManager`, expose the
authenticated operator record to strategy code, or add Lighter credentials to
`StrategyParameters`.

The strategy assembles `ExchangeCashSnapshot` from existing narrow values:

- `PositionManager.get_current_cash()`;
- `PositionManager.get_exchange_account_available_balance(pair)`;
- `state.sync.treasury.pending_deposits`; and
- `state.sync.treasury.pending_redemptions`.

It passes that snapshot to the pure `ExchangeCashManager`, then passes the
decision to the generic exchange-account trade-creation helper. Refactor
`create_lighter_cash_management_transfer()` so it no longer creates a session
or performs an API call; remove it if this explicit composition makes the
helper redundant.

Use one public Lighter session per `EthereumPairConfigurator`/adapter setup and
pass it to account valuation, Lagoon NAV valuation, routing, and the available
balance reader. Do not independently create sessions in auto-discovery,
`create_lighter_adapter()`, and the strategy module.

`LighterRouting` owns immutable routing configuration and the public session.
Do not copy the operator record and session into `LighterRoutingState`.
`LighterRoutingState` should contain only per-cycle execution objects such as
the transaction builder, token cache, and Lagoon vault. Add one clear helper
that raises an actionable error when a transfer is prepared without a Lagoon
vault, instead of allowing an optional vault to fail with `AttributeError`.

### Runtime configuration

Keep operator-record parsing in `start.py` and continue passing the parsed
`LighterRoutingConfig` explicitly through bootstrap and routing setup. Do not
restore any environment-variable handoff below the Typer command.

Move `lighter_withdrawal_timeout` out of `StrategyParameters`. It is an
execution timeout, not strategy behaviour. Keep it solely as the Typer option
and `LighterRoutingConfig` field, with the existing 30-minute default.
Delete the strategy-parameter fallback in `start.py`, the policy-layer timeout
validation, and their tests. Validate the CLI value when constructing
`LighterRoutingConfig`.

Move the default timeout constant out of the pure cash-policy module and next
to Lighter routing/operator execution. Cash allocation arithmetic must not know
how long an exchange API call may block.

### Lagoon settlement deferral

Rename `defer_lighter_redemption_without_liquidity` to the protocol-neutral
`defer_redemption_without_reserve_liquidity`. The check itself only compares
Lagoon queue liquidity and does not access Lighter.

Generalise the associated warning as well: describe an external exchange
account and account-correction commands without naming Lighter or
`lighter-move-funds` in the generic Lagoon sync model.

Enable it when automatic external-account cash management is active. Keep the
current behaviour: record the observed pending queues, skip NAV posting and
settlement, and allow the following strategy decision to return a withdrawal
trade. Do not introduce callbacks from the Lagoon sync model to Lighter.

## Recovery and command behaviour

Remove the generic `reconcile_completed_external_account_transfers()` receipt
shortcut. EVM receipt success alone is not proof that a Lighter transfer was
fully accounted:

- a deposit also needs the expected Lighter collateral credit; and
- a withdrawal needs the actual Safe credit, which may differ from the planned
  amount because of protocol fees.

Extract one small Lighter-specific read-only verifier under
`tradeexecutor.ethereum.lighter`. `LighterRouting.settle_trade()`,
`check-accounts`, `correct-accounts`, and `repair` all use this same verifier.
It returns either an incomplete result or a verified execution result
containing the executed account quantity, Safe reserve amount, and public
metadata updates. It never mutates state. Normal routing and the two correcting
commands pass a verified result to the same generic state-transition helper;
`check-accounts` only reports it.

The inspection rules are:

1. A transfer without a complete set of recorded transaction hashes remains
   unfinished. Report its trade status, direction, public withdrawal request
   ID when present, and whether a claim transaction exists. Do not submit or
   sign anything.
2. A transaction without a successful receipt remains unfinished or failed as
   appropriate. Do not rebroadcast it through generic repair.
3. A deposit with successful approval/deposit receipts is complete only after
   the public Lighter API reports collateral greater than or equal to the
   persisted pre-deposit collateral plus the persisted deposit amount. The
   collateral observation is evidence, not the executed transfer amount.
4. A withdrawal with a successful claim receipt is complete only after reading
   the Safe balance at the receipt block and calculating the positive credit
   from the persisted pre-claim baseline.
5. For a deposit, use the persisted deposit amount as both positive
   `executed_amount` and `executed_reserve`. Do not turn a larger observed
   collateral delta caused by unrelated account activity into transfer value;
   normal exchange-account synchronisation handles any residual valuation
   change.
6. For a withdrawal, use the negative persisted `lighter_claimable_usdc` as
   `executed_amount`, the measured Safe credit as `executed_reserve`, and their
   difference as the protocol fee. Do not infer the custody quantity from a
   later total-equity or collateral observation that may include trading PnL.
7. After verification, allow the accounting helper to complete a
   `broadcasted`, `started`, or `failed` transfer through an explicit recovery
   transition. Scope `force=True` to this helper and clear `failed_at` only
   after conclusive verification, because `TradeExecution.get_status()` gives
   the failure marker precedence over `executed_at`. Do not weaken normal
   `TradeExecution.mark_success()` lifecycle checks globally.
8. The completion helper requires an existing deposit reserve allocation from
   `State.start_execution()`, consumes it exactly once, credits withdrawal
   reserves exactly once, and retains the existing non-negative reserve and
   exchange-account position invariants.

Keep one canonical set of public metadata keys for automatic and
`lighter-move-funds` transfers. Extract constants or a small dataclass only if
it removes the current duplicate key names. Never persist the operator private
key, auth token, signed request, or raw authenticated response.

Command responsibilities:

- `check-accounts` is read-only. It reports a completed-but-unreconciled or
  incomplete transfer and exits non-zero where appropriate, but never saves
  state.
- `correct-accounts` may apply a verified Lighter transfer result before normal
  balance correction and then save state.
- `repair` may apply the same verified result before generic repair. It must
  leave an incomplete Lighter request untouched.
- `start` performs the cleanliness check before universe warm-up and aborts on
  every unfinished external transfer. It never invokes recovery.
- `lighter-move-funds` remains the stopped-executor diagnostic command. Align
  its lifecycle and public metadata with routed transfers, including
  `State.start_execution()` and durable checkpoints, but do not make it an
  automatic startup path or route its protocol action through the live
  strategy loop.

## Implementation sequence

1. Refactor `ExchangeCashManager` into policy plus snapshot and add pending
   deposits to the formula.
2. Add the narrow available-balance delegation to exchange-account pricing,
   `GenericPricing`, and `PositionManager`. Update the reference strategy to
   assemble the pure cash snapshot and remove the module-global session.
3. Reuse one public Lighter session across configurator valuation and routing.
4. Remove the unused automatic-transfer flag and move policy invocation out of
   `exchange_account/state.py`.
5. Simplify `LighterRoutingState`, make Lagoon vault requirements explicit,
   and move the withdrawal timeout fully to CLI routing configuration.
6. Generalise the Lagoon liquidity-deferral variable name without changing its
   settlement behaviour.
7. Extract verification from `LighterRouting.settle_trade()`, replace generic
   receipt reconciliation with it, and wire the shared result to normal
   routing, read-only `check-accounts`, and mutating `correct-accounts` and
   `repair` paths.
8. Make `check-accounts` report-only and move the startup cleanliness check
   before warm-up and the live-unit-test early return. This deliberately makes
   live-unit-test mode fail closed when its state contains an unfinished
   external transfer.
9. Convert `lighter-move-funds` to the shared start/allocation/completion
   lifecycle, align its metadata, and update Lighter documentation.

## Tests

Follow the existing tests instead of creating a new fixture hierarchy.

### Pure policy tests

Extend `tests/exchange_account/test_exchange_cash_manager.py`:

1. Deposit only Safe cash above the buffer and net redemption need.
2. Count pending Lagoon deposits against pending redemptions.
3. Withdraw only free Lighter collateral when open positions tie up the rest.
4. Preserve the existing partial-withdrawal test where the redemption cannot be
   fully funded.
5. Return no transfer while another external-account transfer is unfinished.
6. Reject negative and non-finite policy and snapshot values.

Remove timeout-policy assertions and the automatic-transfer flag assertion
when those obsolete interfaces are deleted.

Use a small number of tests with several assertions, following `AGENTS.md` test
docstring and numbered-comment conventions.

### State and recovery tests

Extend `tests/exchange_account/test_exchange_account_transfer.py`:

1. Automatic and `lighter-move-funds` deposits both use
   `State.start_execution()`, hold one reserve allocation, and complete without
   a second reserve debit.
2. A mined deposit is not reconciled until Lighter collateral is observed.
3. A mined withdrawal credits the actual measured Safe delta and records any
   protocol fee; portfolio NAV decreases by exactly that fee rather than being
   asserted to remain unchanged.
4. Verified recovery works from `broadcasted`, `started`, and `failed` states,
   clears a verified failed marker, and leaves the final status successful.
5. A request ID without a claim transaction remains unclean and causes no API
   request, transaction signing, rebroadcast, reserve refund, or state write.
6. Generic `repair_trade()` refuses an external-account transfer.
7. Normal settlement and CLI recovery return the same verified amounts and
   metadata for equivalent deposit and withdrawal evidence.

Mock only Lighter public API responses. Use Anvil receipts and token balances
where an EVM transaction can be represented locally.

### Typer CLI black-box tests

Extend the current block-number-anchored Lagoon Lighter integration test:

1. `check-accounts` detects an unfinished or completed-but-unreconciled Lighter
   transfer without changing the state file.
2. `correct-accounts` verifies a completed transfer, updates reserve and
   exchange-account quantities once, and is idempotent on a second run.
3. `repair` performs the same verified completion when invoked first and does
   not create a counter-trade or duplicate reserve credit.
4. A withdrawal request without a claim transaction makes all three commands
   report the incomplete state; only the mutating commands are permitted to
   save when they actually reconcile a completed transfer.
5. The existing fixed-fork automatic deposit/withdrawal flow remains covered,
   with the Lighter sequencer and delayed withdrawal mocked and all Ethereum
   transactions executed on Anvil.
6. The existing `lighter-move-funds` fixed-fork test asserts the same
   `planned -> started -> broadcasted -> success` lifecycle, a single deposit
   reserve debit, and idempotent final accounting.

Anchor Anvil to the existing fixed block and reuse current fixtures. Do not mock
Safe execution, ERC-20 balances, transaction receipts, Lagoon queue balances,
or executor state persistence.

### Focused verification

Run focused tests one command at a time with the required environment and
worktree path override:

```shell
source .local-test.env && PYTHONPATH="$(pwd):$PYTHONPATH" poetry run pytest tests/exchange_account/test_exchange_cash_manager.py --timeout=300
source .local-test.env && PYTHONPATH="$(pwd):$PYTHONPATH" poetry run pytest tests/exchange_account/test_exchange_account_transfer.py --timeout=300
source .local-test.env && PYTHONPATH="$(pwd):$PYTHONPATH" poetry run pytest tests/lagoon/test_lagoon_lighter_cash_management.py --timeout=300
source .local-test.env && PYTHONPATH="$(pwd):$PYTHONPATH" poetry run pytest tests/lagoon/test_lighter_move_funds.py --timeout=300
source .local-test.env && PYTHONPATH="$(pwd):$PYTHONPATH" poetry run pytest tests/exchange_account/test_lighter_cli_accounts.py --timeout=300
```

The account-command tests use the same fixed Ethereum fork block but their own
existing fixture layout. Extend that harness with the new withdrawal and
`repair` cases; do not create a shared fixture hierarchy merely to deduplicate
the block setup. Do not run the entire suite unless focused tests expose a
wider regression.

## Documentation cleanup

Update `docs/lagoon-lighter-deployment.md` and
`tradeexecutor/exchange_account/README-Lighter.md`:

- explain the net redemption liquidity formula including pending deposits;
- state that the withdrawal timeout is a CLI infrastructure setting;
- document that startup aborts on an unfinished transfer;
- direct operators to read-only `check-accounts` first and then use
  `correct-accounts` or `repair` only for a transfer whose claim transaction
  completed;
- explain that a persisted request without a claim transaction remains a
  manual diagnostic case and is never resubmitted automatically; and
- remove claims that startup resumes pending withdrawals.

Update the original implementation plan only to mark superseded recovery and
configuration statements as obsolete or link to this cleanup plan. Do not
rewrite its historical implementation record as if the first draft had used
the final design.

## Acceptance criteria

- The strategy returns custody `TradeExecution` objects through the existing
  execution and router pipeline.
- Pending Lagoon deposits reduce the amount reclaimed from Lighter for pending
  redemptions.
- Strategy code performs no direct Web3 calls and owns no global Lighter API
  session or operator credentials.
- Withdrawal timeout and operator-record configuration enter only through the
  Typer CLI and are passed explicitly without globals or lower-level
  environment reads.
- `check-accounts` never mutates state.
- `correct-accounts` and `repair` complete only protocol-verified transfers and
  use the persisted custody amount and measured Safe credit, including fees.
- An incomplete Lighter request stays unclean and is never resumed,
  rebroadcast, refunded, or duplicated automatically.
- Generic repair cannot release an uncertain external deposit allocation.
- Automatic and manual deposits use one reserve-allocation lifecycle and debit
  reserves exactly once.
- Normal execution and CLI recovery use the same read-only Lighter verifier.
- Lagoon treasury code contains no Lighter-specific configuration name.
- The redundant automatic-transfer flag and duplicated routing-state
  dependencies are removed.
- Focused policy, state, fixed-fork, and Typer black-box tests pass.
