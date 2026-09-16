# Lighter move funds command

## Goal

Add a small interactive Typer command named `lighter-move-funds` for moving
USDC between an Ethereum Lagoon Safe and its Lighter account.

This is an operator-only diagnostics and error-recovery tool. It is not a
strategy execution path and must not be presented as the normal way to manage
strategy capital. The executor service must be stopped while the command is
running so it cannot write the same state file concurrently.

The command must:

1. Inspect the configured Lagoon vault, Safe, Lighter account and executor
   state.
2. Show Safe USDC and a useful Lighter balance and margin breakdown.
3. Ask whether to deposit or withdraw (`d`/`w`).
4. Ask for the amount in USDC and show a final confirmation.
5. Execute and verify the requested movement.
6. Record the completed custody movement in trade-executor accounting.
7. Print the balances again and finish with `All ok` only after read-back
   verification and state persistence succeed.

## Scope decisions

- Support only Ethereum Lagoon vaults with one configured Lighter exchange
  account.
- Reuse the deployment-created `LIGHTER_OPERATOR_RECORD_FILE`; never accept or
  print the delegated private key as a command-line value.
- Reuse the existing Safe deposit, secure-withdrawal, claim and API-token retry
  code from `lagoon-lighter-test-trade`.
- Keep the interaction intentionally simple. Do not add batch operation files,
  strategy hooks, a generic cross-exchange transfer engine or a second recovery
  journal.
- Do not open or close perpetual positions and do not automatically settle
  Lagoon investor flows. `lagoon-settle` remains a separate operator action.

## Accounting model

### Use a transfer trade, not paired balance corrections

Record a Safe/Lighter movement as one synthetic `TradeExecution` on
the existing `LIGHTER-ACCOUNT` exchange-account position:

- Safe to Lighter deposit: a positive buy of synthetic account value;
- Lighter to Safe withdrawal: a negative sell of synthetic account value;
- `TradeType.rebalance`, because this is an intentional capital allocation;
- a new `TradeFlag.external_account_transfer`, so reports and future generic
  exchange transfer support can identify it without parsing notes; and
- human-readable notes identifying the manual `lighter-move-funds` operation.

This is preferable to paired `BalanceUpdate` corrections. A transfer changes
custody but not total strategy NAV, while exchange-account balance updates are
currently used for observed PnL and reconciliation. Treating the movement as a
trade gives one durable audit record and moves the same amount out of one side
of the portfolio and into the other.

Add small accounting helpers to `tradeexecutor/exchange_account/state.py` to
create, advance and complete the transfer trade. They should:

1. Require an existing exchange-account position and one positive USDC transfer
   amount.
2. Create the buy or sell trade against that existing position with quantity
   equal to the signed transfer amount and planned and executed price `1.0`.
3. Manage the synthetic trade directly instead of calling
   `State.start_execution()` or `State.mark_trade_success()`, whose reserve and
   position-closing behaviour is for normal routed trades.
4. Decrease Safe reserves by the verified deposit amount or increase them by
   the verified withdrawal amount.
5. Update the position's internal share-price state using the existing trade
   update helper, so capital flow is not mistaken for investment return.
6. Store only public audit metadata in `TradeExecution.other_data`: direction,
   protocol, account index, requested and received amounts, before/after Safe
   and Lighter balances, and public transaction or withdrawal identifiers.
7. Reject any result that would create a negative reserve or exchange-account
   quantity.
8. Call `TradeExecution.mark_success()` directly after the helper has moved the
   trade through started and broadcasted, and keep the persistent
   exchange-account position open when a full withdrawal leaves it at zero.

For the current Lighter flow, the transfer is one-for-one in USDC. A deposit
must produce the same raw-unit Safe debit and confirmed Lighter credit. A
withdrawal uses the matching withdrawal-history claimable amount and requires
the Safe raw-unit credit to equal it. Any mismatch is a verification failure,
not an inferred fee. Do not derive withdrawal quantity from collateral
snapshots taken hours apart because realised PnL, funding and trading can move
collateral during `withdrawalDelay`.

If Lighter later exposes separate gross and net transfer values, extend the
helper then and record the explicit protocol fee. Do not infer a fee from an
incomplete Safe credit, and do not mark an unverified trade successful.

Do not add exchange adapters, protocol classes or other abstractions for future
Derive or GMX support. The helper only records the two verified sides of one
exchange-account transfer; this command is its sole caller for now.

### Reconciliation around the transfer

Before prompting, read and display both physical and tracked balances without
mutating state. If physical and tracked balances differ by more than the
existing accounting epsilon, stop and direct the operator to run
`correct-accounts` first. Do not silently accept pre-existing drift as part of
the transfer. After final confirmation, take the state backup and revalidate
the requested amount against fresh balances.

After the physical movement succeeds:

1. Complete the transfer trade using the confirmed deposit amount or matching
   withdrawal-history amount and the verified Safe debit or credit.
2. Revalue Lighter once more so any concurrent trading PnL is recorded as a
   separate exchange-account `BalanceUpdate`.
3. Re-read the Safe reserve and require it to match the expected transfer
   result. Do not hide a mismatch by creating a correction.
4. Save the state and verify that the tracked reserve and account value agree
   with the final observations.

Take a normal command-specific state backup before the live operation. Do not
persist a successful transfer trade until the chain/API operation and balance
read-back have succeeded. A deposit is not crash-atomic: if the process exits
after the physical deposit but before accounting is saved, the next run must
report the custody mismatch and direct the operator to `correct-accounts`.
Document this narrow limitation instead of adding deposit recovery machinery.

A secure withdrawal is asynchronous and needs the existing trade fields:

1. Create and save the planned withdrawal trade before submission, including
   the amount, request time and pre-transfer balances in public metadata.
2. On planned, started or broadcasted recovery, query withdrawal history before
   doing anything else. If submission is ambiguous and no matching row appears,
   fail closed instead of automatically resubmitting.
3. Refactor the existing request helper to return its public transaction
   identifier. After Lighter accepts the request, set `started_at`, persist the
   identifier, move the trade to broadcasted and save it again.
4. If the API definitively rejects the request, mark the saved trade failed so
   a fresh command run can retry. Keep ambiguous transport failures pending.
5. After a successful L1 claim, persist the public claim transaction identifier
   and verified Safe credit before final accounting. On rerun, inspect the
   history status, claim receipt or Safe credit before deciding whether another
   claim call is needed.
6. Call `TradeExecution.mark_success()` directly only after the claim receipt
   and balance read-back pass.

Do not add a second recovery journal. While a transfer trade is unfinished,
extend `State.check_if_clean()` narrowly so any non-terminal trade carrying
`external_account_transfer` is unclean, including planned and started trades.
Do not change the global meaning of `TradeExecution.is_unfinished()`. This
prevents `check-accounts`, `correct-accounts` and strategy execution from
interpreting the temporary custody mismatch as profit or loss.

### `check-accounts` and `correct-accounts`

The transfer trades must use normal position quantity semantics so the existing
account checks can reason about them without a command-specific ledger:

- a successful deposit trade increases `position.get_quantity()` by the
  confirmed deposit amount;
- a successful withdrawal trade decreases it by the matching withdrawal-history
  amount; and
- the opposite verified delta is applied to the Lagoon reserve position.

`check-accounts` must remain read-only and check both custody locations in the
same invocation:

1. Compare the tracked Lagoon reserve with physical Safe USDC through the
   normal on-chain balance check.
2. Compare `position.get_quantity()`, including all successful transfer trades,
   with current public Lighter equity through the exchange-account check.
3. Report clean immediately after either `lighter-move-funds` direction when
   there is no unrelated PnL or balance drift.
4. Report the Safe and Lighter mismatches separately after an interrupted or
   out-of-band transfer, without modifying the transfer trades or state.

The existing Lighter value check already uses `position.get_quantity()`, so it
should naturally include these trades. Keep that behaviour and add regression
coverage rather than adding flag-specific arithmetic.

`correct-accounts` needs one narrow fix for Lagoon Lighter strategies. It
currently synchronises Lighter equity, but skips generic on-chain correction
when every open position is an exchange account. That shortcut also skips the
Lagoon Safe reserve, even though the reserve is on-chain. Change the condition
so a Lagoon vault with a Lighter exchange account still runs reserve
correction, while exchange-only hot-wallet and unrelated exchange integrations
retain their current behaviour.

In one `correct-accounts` run:

1. Synchronise the Lighter position through `ExchangeAccountSyncModel`, adding
   only the residual valuation `BalanceUpdate` required to match public equity.
2. Run the normal Lagoon on-chain correction for Safe USDC and apply any reserve
   correction.
3. Perform the final account check against both the Safe reserve and Lighter
   public equity. Do not use the current final-check shortcut merely because
   all trading positions are exchange accounts, and pass the Lighter account
   value reader to the final check.
4. Leave all `external_account_transfer` trades unchanged. Corrections are
   appended as `BalanceUpdate` events and must not manufacture replacement
   transfer trades because an out-of-band flow has no trustworthy direction or
   transaction metadata.

The generic expected-asset map and on-chain correction pass must continue to
exclude the synthetic Lighter exchange-account asset. Only physical Safe USDC
is checked through Web3; Lighter equity is checked through the exchange-account
value reader. Add a regression assertion for this distinction when removing
the current all-exchange-account shortcut.

`correct-accounts --dry-run` must report the proposed Safe and Lighter
corrections against a copied state or `SimulateStore`, without persisting either
one. A clean state containing completed transfer trades must produce no
corrections in normal or dry-run mode. Both account commands must refuse to run
while an `external_account_transfer` trade is unfinished; they must not repair
an expected in-flight custody mismatch.

## Command behaviour

Create `tradeexecutor/cli/commands/lighter_move_funds.py` and register it in
`tradeexecutor/cli/main.py`.

Use the standard live-command configuration:

- `EXECUTOR_ID`, `STRATEGY_FILE` and `STATE_FILE`;
- `JSON_RPC_ETHEREUM` and `PRIVATE_KEY`;
- `ASSET_MANAGEMENT_MODE=lagoon`;
- `VAULT_ADDRESS` and `VAULT_ADAPTER_ADDRESS`;
- optional `LIGHTER_ACCOUNT_INDEX` public cross-check, matching the existing
  Lighter test-trade command; and
- `LIGHTER_OPERATOR_RECORD_FILE` for the owner-only delegated key record.

Use `@shared_options.with_json_rpc_options()`, `prepare_executor_id()`,
`setup_logging()`, the existing single-chain Lagoon bootstrap helpers and the
normal state backup/store APIs. Reject `SIMULATE=true`, non-mainnet Ethereum,
non-Lagoon asset management, an operator record which does not match the vault
and Safe, and any state which does not contain exactly one matching Lighter
exchange-account position.

### Balance display

Print the following before the prompts and again after completion:

- physical Safe USDC;
- tracked Safe reserve USDC;
- Lighter collateral;
- Lighter total equity;
- Lighter available balance;
- cross initial margin requirement;
- cross maintenance margin requirement;
- allocated margin summed from returned position records;
- gross open-position notional summed from `position_value`;
- unrealised PnL; and
- returned position-record count and tracked exchange-account value.

Use the public account response already read by the command. Market metadata
and price lookups are unnecessary for this first utility.

### Prompts and validation

Use Typer prompts in this order:

1. `Move direction [d/w]`;
2. `Amount in USDC`; and
3. a final confirmation which spells out `Safe -> Lighter` or
   `Lighter -> Safe`, the amount and the account index, defaulting to no.

Parse money as `Decimal`, require a positive amount with no more than six USDC
decimal places, and validate before confirmation:

- a deposit cannot exceed physical Safe USDC;
- a deposit must meet Lighter's mainnet minimum;
- a withdrawal cannot exceed Lighter `available_balance`;
- a withdrawal must meet Lighter's mainnet minimum; and
- all observed Safe, Lighter and resulting accounting balances must remain
  non-negative.

### Deposit success

For `d`:

1. Record the pre-transfer Safe balance and Lighter collateral.
2. Call `deposit_usdc_from_lagoon_safe_into_lighter()`.
3. Let the eth-defi helper confirm both Safe transactions and reject failed
   receipts.
4. Poll the public Lighter account until collateral has increased by at least
   the confirmed deposit amount or the existing sensible deposit timeout
   expires. Do not require an exact snapshot delta when other account activity
   may occur.
5. Verify the Safe raw-unit debit and transaction/event deposit amount match,
   then log the public transaction hash and observed balance changes.
6. Record and save the successful transfer trade. If the process exits after
   the physical deposit but before this save, leave the mismatch for
   `correct-accounts`; do not resubmit automatically.

### Withdrawal success

For `w`:

1. Record the pre-transfer Safe balance, Lighter collateral and current
   `withdrawalDelay`, and log the delay.
2. If a planned, started or broadcasted transfer already exists, use its saved
   amount, request time and baselines and query withdrawal history before any
   external action. Never blindly resubmit an ambiguous request.
3. Otherwise create and save a planned transfer trade, then submit the secure
   withdrawal exactly once with the existing authenticated helper.
4. As soon as submission is accepted, return and save its public identifier,
   then move the trade through started to broadcasted.
5. Use rotating Lighter auth tokens only for idempotent withdrawal-history
   polling.
6. Wait for the matching withdrawal to become claimable. A rerun resumes here
   from the saved public metadata.
7. Claim the exact amount reported by the matching history row to the Lagoon
   Safe, wait for the Ethereum receipt and require the Safe raw-unit credit to
   equal that amount. Persist the public claim result before final accounting.
8. On recovery, detect an already completed claim from public history, receipt
   or the verified Safe credit instead of claiming twice.
9. Log only public identifiers, complete the transfer accounting from the
   history amount, and save the successful trade.

Reuse the existing operator-record and withdrawal helpers from
`lagoon_lighter_test_trade.py`. Do not restructure the test-trade command only
to introduce another support module for this small utility.

## Failure and security behaviour

- Never log the delegated private key, bearer token, signed request or raw SDK
  exception body.
- Keep the owner-only permission check for the operator record.
- Convert SDK failures to short public errors containing the operation and
  exception type only.
- On receipt failure, timeout or balance mismatch, do not create a successful
  transfer trade and do not print `All ok`. Preserve the non-terminal
  withdrawal trade so a rerun can resume it safely.
- Always close the Lighter session and Web3 configuration in a `finally` path.
- State backup and documentation must make it explicit that the executor must
  be stopped before this command mutates its state file.

## Tests

### Accounting helper

Add one focused test for the small accounting helper. In the same test:

1. Open an exchange-account position and seed its share-price state.
2. Record a deposit and assert reserve decreases, account quantity increases,
   NAV stays unchanged and the trade has the transfer flag and public metadata.
3. Record an equal-amount withdrawal and assert the inverse reserve and account
   movements preserve NAV at execution price `1.0`.
4. Reject a withdrawal when the Safe raw-unit credit differs from the
   withdrawal-history claimable amount.
5. Withdraw the remaining quantity and assert the zero-valued persistent
   exchange-account position stays open for a future deposit.
6. Assert an unverified trade cannot succeed and negative resulting balances
   are rejected.
7. Assert `State.check_if_clean()` rejects flagged planned, started and
   broadcasted transfer trades without changing the global
   `TradeExecution.is_unfinished()` semantics.

### Typer black-box integration

Add a fixed-block Ethereum Anvil test modelled on
`tests/lagoon/test_lagoon_lighter_test_trade.py`. Use a real Lagoon deployment,
Safe, USDC transfers, state file and Typer command. Mock only Lighter's public
sequencer, authenticated withdrawal/history calls and proof/claim behaviour
which Anvil cannot provide.

Run the command through Typer input, once with `d` and once with `w`, and
assert:

- the deposit and claim affect the real forked Safe USDC balance;
- the public Lighter mock observes the matching balance changes;
- state contains one deposit and one withdrawal transfer trade;
- reserve and Lighter position quantities return to their starting values;
- total NAV never becomes negative and is unchanged apart from explicitly
  mocked PnL;
- transaction identifiers are retained, but the delegated key is absent from
  persisted public state.

In the same withdrawal flow, stop the first command after Lighter accepts the
request but before the history row becomes claimable. Assert a non-terminal
transfer trade is persisted, no success or `All ok` is reported, and both account
commands refuse to correct the expected in-flight mismatch. Rerun the command,
make the existing history row claimable, and assert it claims and completes the
same trade without a second withdrawal submission.

The command also recognises an exact Safe credit on a later rerun, allowing it
to complete accounting after a claim succeeded but final state persistence did
not. Keep this recovery narrow rather than adding a failure-injection framework.

Extend the same fixed-block flow, or the existing Lighter account CLI black-box
coverage where fixture reuse is clearer, to prove account command behaviour:

1. Run `check-accounts` after the recorded deposit and after the recorded
   withdrawal; both runs must exit successfully and leave state byte-for-byte
   unchanged.
2. Run `correct-accounts` on the clean transfer state; it must create no
   correction and must preserve the flagged trades.
3. Create a controlled out-of-band Safe/Lighter balance change without adding
   a trade, then verify `check-accounts` reports both affected custody rows.
4. Run `correct-accounts` once and verify it repairs both the block-anchored Safe
   reserve and mocked Lighter equity, preserves the original transfer trades,
   and makes a following `check-accounts` pass.
5. Exercise `correct-accounts --dry-run` against the same mismatch and verify it
   reports both proposed corrections without changing the state file
   byte-for-byte.
6. Assert the generic on-chain expected-asset map contains Safe USDC but not the
   synthetic `LIGHTER-ACCOUNT` asset.

Use the real Lagoon Safe and USDC on fixed-block Anvil. Mock only the Lighter
equity change which cannot be represented by Anvil.

Follow the repository pytest docstring, numbered-step comment and fixture type
conventions. Set `LOG_LEVEL=disabled` in the black-box environment so parallel
CLI tests cannot race over global logging handlers.

## Documentation and release notes

Update `tradeexecutor/exchange_account/README-Lighter.md` with:

- an operator warning that this is diagnostics/error-recovery tooling;
- a complete environment and invocation example;
- the displayed balance fields and withdrawal-delay behaviour;
- the requirement to stop the executor first;
- the successful transfer-trade accounting model; and
- how an interrupted withdrawal resumes from its unfinished transfer trade,
  the interrupted-deposit reconciliation limitation, and
  when to use `check-accounts`, `correct-accounts` or `lagoon-settle`.

Replace the existing statement that Lighter transfers have no durable executor
record with the narrower limitation: completed transfers performed through
`lighter-move-funds` receive this record. A deposit interrupted after its
physical movement but before state persistence is repaired as a balance
correction, as are strategy-driven and out-of-band transfers.

Document that `check-accounts` validates both sides of recorded transfer trades
and that `correct-accounts` repairs residual Safe/Lighter drift without
rewriting those trades. An accounting correction proves the final balances,
not the history or direction of an out-of-band transfer.

Add a dated `CHANGELOG.md` entry because this is a new operator feature.

## Acceptance criteria

The work is complete when:

1. `trade-executor lighter-move-funds --help` describes the command as a manual
   diagnostics and recovery tool.
2. A stopped Lagoon Lighter deployment can deposit or withdraw interactively.
3. Both paths show before/after Safe and Lighter balance and margin details.
4. The command records one explicitly flagged transfer trade, verifies the
   physical result before marking it successful, and safely resumes an accepted
   withdrawal without resubmitting it.
5. Reserve plus exchange-account NAV is preserved and is never negative.
6. `check-accounts` is clean after either completed transfer, detects drift on
   either custody side and refuses an unfinished transfer.
7. One `correct-accounts` invocation repairs both Lagoon Safe reserves and
   Lighter equity while preserving transfer trades; dry-run persists nothing.
8. The focused helper and fixed-block Typer black-box tests pass.
9. No private Lighter key or auth token appears in output or persisted public
   state.
