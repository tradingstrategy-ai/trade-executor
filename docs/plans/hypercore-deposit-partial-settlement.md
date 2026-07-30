# Fix plan: recovery-safe HyperCore vault deposits

## Goal

Make a HyperCore vault buy safe when HyperCore accepts only part of a
multi-action settlement. The executor must verify every balance transition
before starting the next one, stop the remaining trade batch on failure, and
must not make already-bridged USDC spendable in internal reserve accounting.

This plan is based on:

- `.claude/docs/vault-deposit-redeem.md`, especially the continuous-equity and
  no-double-spend accounting invariants;
- `.claude/docs/hypercore-vault.md`, especially the separate HyperCore state
  machine and the rule that an EVM receipt does not prove a HyperCore action
  settled; and
- the production evidence for HyperAI trade #1486 on 2026-07-30.

The ERC-7540/Lagoon asynchronous vault state machine is not reused. HyperCore
has a separate spot → perp → vault flow and separate recovery semantics.

## Incident and root cause

Trade #1486 attempted to deposit 48.884068 USDC into Citadel:

1. The HyperEVM approval and deposit receipts succeeded, moving the USDC from
   the Safe towards HyperCore.
2. The former phase-2 EVM transaction submitted
   `transferUsdClass(spot→perp)` and `vaultTransfer(perp→vault)` together.
3. The EVM transaction succeeded, but HyperCore applied only the first action.
   The perp account rose to 49.384068 USDC, which is the 48.884068 USDC
   deposit plus its pre-existing 0.5 USDC. Spot stayed at its 0.217882 USDC
   baseline and the vault equity never rose.
4. Vault verification correctly timed out, but generic failed-buy accounting
   then added 48.884069 USDC back to reserves even though the money was in the
   HyperCore perp account. This overstated spendable cash and could allow the
   same capital to be allocated twice.

The protocol boundary is the cause: HyperCore actions requested through
CoreWriter settle asynchronously, so EVM atomicity and receipt status do not
make a group of HyperCore actions atomic. The bundled settlement assumed an
atomicity guarantee that does not exist.

Recent merged pull requests are not the trigger:

- #1585, #1586, and #1587 changed HyperCore withdrawal and dust-cleanup paths,
  not the deposit action bundle.
- #1588 changed typed vault JSON handling.
- #1589, the deployed commit `85c8f9cd`, added GuardV0-aware Lagoon treasury
  settlement. It did not change `hypercore_routing.py`; its eth_defi submodule
  bump did not change the relevant CoreWriter deposit helper.

The incident exposed an existing deposit-path assumption. It was not a
regression introduced by the deployed GuardV0 change.

## Safety invariants

The implementation must preserve these invariants:

1. A successful EVM receipt proves only EVM inclusion. Each HyperCore balance
   transition must be observed through the Info API.
2. The perp → vault action must not be sent until the expected spot → perp
   increase is visible.
3. Once phase 1 has removed USDC from the Safe, a failed trade must not return
   that USDC to spendable reserves or available bridge capital.
4. Failure location must be conservative. If a poll cannot distinguish two
   locations, record both rather than guessing.
5. The position remains frozen and sequential execution stops after a failed
   settlement. Recovery is explicit; the executor must not retry a transfer
   blindly and risk depositing twice.
6. A failure before phase 1 consumes Safe capital keeps the existing generic
   failed-buy refund behaviour.
7. A process crash or indeterminate broadcast must fail closed. Generic
   startup or trade repair must not refund a HyperCore buy until the phase-1
   transaction and live Safe/HyperCore balances prove the capital never left
   the Safe.

## Implementation

### 1. Split the deposit settlement legs

Replace the bundled deposit phase with two separately broadcast and verified
CoreWriter transactions in
`tradeexecutor/ethereum/vault/hypercore_routing.py`:

1. Wait for the phase-1 EVM escrow to clear into spot.
2. Snapshot current vault equity, spot USDC and the current perp withdrawable
   balance.
3. Broadcast only `transferUsdClass(spot→perp)`.
4. Check its EVM receipt and use a deposit-specific verifier to observe the
   full spot decrease and corresponding perp increase. Compare human USDC
   values after the correct per-account conversion; six-decimal USDC values
   exactly represent the raw transfer amount, so no monetary tolerance is
   permitted. Do not reuse the vault performance-fee tolerance. Unrelated
   positive movement may exceed the requested amount, but either side must
   still have moved by at least the complete requested amount.
5. Only after that poll succeeds, broadcast
   `vaultTransfer(perp→vault)`.
6. Check its EVM receipt and use
   `wait_for_vault_deposit_confirmation()` to prove the vault equity rose.

Append both transactions to `trade.blockchain_transactions` with distinct
phase notes. Keep nonce synchronisation between the two broadcasts. A
broadcast exception must retain the partially constructed transaction for
diagnostics, as the existing `SettlementBroadcastError` contract does.

The Safe is not expected to hold open perp positions, but the verifier must
still tolerate unrelated positive balance movement and must log both sides of
the transfer. The existing vault-equity drift tolerance remains appropriate
only for the final vault confirmation. It must not be used to accept a missing
spot → perp transfer.

### 2. Classify partial settlement failures

Every failure after the phase-1 bridge has consumed Safe USDC calls
`_mark_stranded_usdc()` with the narrowest defensible location:

| Failure point | Recorded location | Next action sent? |
|---|---|---|
| Escrow clear times out | `hypercore_evm_escrow_or_spot` | No |
| Vault equity baseline cannot be read | `hypercore_spot` | No |
| Spot → perp receipt definitely reverted | `hypercore_spot` | No |
| Spot → perp broadcast result is indeterminate | `hypercore_spot_or_perp` | No vault transfer |
| Spot → perp balance poll times out | `hypercore_spot_or_perp` | No vault transfer |
| Perp → vault receipt definitely reverted | `hypercore_perp` | No further action |
| Perp → vault broadcast result is indeterminate | `hypercore_perp_or_vault` | No further action |
| Vault equity confirmation times out | `hypercore_perp_or_vault` | No further action |

The last row reproduces the failure class seen in trade #1486. Its diagnostic
snapshot showed the exact amount in perp, but a timeout alone does not prove
that an accepted vault action cannot settle later. Recovery therefore rechecks
both the perp balance and vault equity instead of blindly repeating
`vaultTransfer`.

Diagnostics should name the failed phase, Safe, vault, raw and human amount,
all three HyperCore balances, Safe EVM balance, and transaction hashes. The
operator message must be location-aware:

- spot: transfer spot → perp to finish the deposit, or bridge spot → EVM;
- perp: finish perp → vault, or transfer perp → spot before bridging to EVM;
- ambiguous: inspect with `check-hypercore-user.py` before taking either
  action.

### 3. Fail closed across crashes and repair

Persist a `hypercore_deposit_capital_at_risk` marker before broadcasting the
phase-1 deposit transaction. This is earlier than the normal stranded-location
classification because the process can die after a node accepts the
transaction but before receipt handling or the next state save.

- A definite phase-1 revert or a definite pre-broadcast failure clears the
  marker and keeps the ordinary failed-buy refund path.
- A successful receipt, missing receipt or indeterminate broadcast keeps the
  marker until live reconciliation locates the capital.
- Once a location is known, `_mark_stranded_usdc()` adds the amount and
  location metadata. Ambiguous results remain ambiguous.

`EthereumExecution.repair_unconfirmed_trades()` and the state-only
`repair_trades()` flow must refuse to turn an at-risk HyperCore deposit into a
generic refunded failed buy. They must require receipt lookup plus a live Safe,
escrow, spot, perp and vault reconciliation. State-only repair cannot perform
that reconciliation and must stop with operator instructions.

This is a safety checkpoint, not automatic settlement resumption. It ensures a
crash cannot bypass the accounting marker even if no exception handler ran.

### 4. Preserve accounting for stranded capital

`_mark_stranded_usdc()` stores both:

- `hypercore_stranded_usdc`, containing amount, location, Safe and recovery
  guidance; and
- `retain_reserve_allocation_on_failure = True`.

Update `State.mark_trade_failed()` so a failed buy with this explicit marker:

- is still marked failed and freezes its position;
- does not call `portfolio.adjust_reserves()` for a reserve-funded buy; and
- does not release `bridge_currency_allocated` for a bridge-funded buy.

This is deliberately conservative: NAV may temporarily under-report the
recoverable transit balance, but spendable cash cannot be overstated. Normal
pre-bridge failures, which do not have the marker, continue to unroll their
allocation.

The marker is specific to externally stranded capital. The earlier at-risk
marker is allowed to retain capital conservatively until a live check proves a
refund safe; neither condition is inferred from failure text.

### 5. Keep recovery manual and idempotent

Do not automatically re-run a failed phase from the normal live loop. The
failed trade, its transaction list, its phase-specific location and the live
balance inspection provide the recovery checkpoint. An operator first
confirms the actual location, then either completes the vault deposit or
returns the USDC to the Safe and reconciles state.

Automatic crash-resume across these HyperCore legs is a separate change. It
would require a persisted phase enum plus pre-action balance checks that prove
whether an action has already settled. The at-risk marker in this fix prevents
unsafe accounting repair but does not automatically repeat a transfer. Adding
an unverified automatic retry would recreate the double-spend/double-deposit
risk.

### 6. Make reconciliation marker-aware

Audit reserve sync, HyperCore account checks, `correct-accounts` and stranded
USDC cleanup consumers. While either recovery marker is present:

- Safe EVM reserve sync must observe the lower real Safe balance and must not
  manufacture a reserve correction for the missing amount;
- HyperCore account checks should report the known transit amount and location
  rather than treating it as unexplained drift or vault dust; and
- no cleanup or repair command may clear the marker or release allocation
  without recording the operator's confirmed recovery destination.

When recovery finishes, one explicit reconciliation operation clears the
marker and accounts for exactly one destination: increased vault equity or
USDC returned to the Safe. This avoids both permanent under-reporting and a
second credit.

## Mocked regression tests

HyperCore/CoreWriter settlement cannot be reproduced on a normal EVM testnet,
so add focused unit tests in
`tests/hyperliquid/test_hypercore_deposit_settlement.py` with `MagicMock` and
module-level patches. The mocks are necessary to deterministically reproduce
the protocol behaviour where an EVM receipt succeeds but only one HyperCore
action changes balance. Keep activation-cost-specific coverage in
`test_hypercore_activation_cost.py`.

### Deposit ordering and happy path

1. Build a live buy with phase-1 receipts and mock the escrow clear.
2. Mock the vault-equity/spot/perp baselines, spot → perp broadcast, dual
   balance poll, perp → vault broadcast and vault confirmation.
3. Assert strict call ordering: the vault broadcast happens only after the
   perp poll returns successfully.
4. Assert the raw amount passed to both legs, separate phase-2/phase-3
   transactions are persisted, and the trade succeeds with no failure report.

### Production incident regression

In the same module:

1. Use the trade #1486 amount, a 0.5 USDC perp baseline, the 0.217882 USDC spot
   baseline and a successful spot → perp receipt/poll returning 49.384068
   USDC.
2. Return a successful perp → vault EVM receipt, then make
   `wait_for_vault_deposit_confirmation()` raise
   `HypercoreDepositVerificationError`.
3. Assert the trade reports failure, records `hypercore_perp_or_vault`, sets
   `retain_reserve_allocation_on_failure`, and persists both settlement
   transactions.
4. Assert no additional transfer is broadcast after the failed confirmation.

### Intermediate failure guard

1. Make the spot → perp EVM receipt succeed but its balance poll raise
   `HypercoreWithdrawalVerificationError`.
2. Assert `_broadcast_deposit_perp_to_vault()` is never called.
3. Assert the location is `hypercore_spot_or_perp` and the allocation-retention
   marker is set.

### Broadcast ambiguity and crash repair

1. Raise a broadcast exception after returning a signed transaction for each
   settlement leg.
2. Assert spot → perp records `hypercore_spot_or_perp`, while perp → vault
   records `hypercore_perp_or_vault`.
3. Create a broadcast HyperCore buy with
   `hypercore_deposit_capital_at_risk` but no stranded-location metadata,
   reproducing a process death before failure classification.
4. Assert unconfirmed-trade repair cannot refund it without live
   reconciliation, and state-only `repair_trades()` refuses it.
5. Add the definite phase-1 revert guard and assert it clears the at-risk
   marker before the normal refund.

### State accounting tests

In `tests/test_state.py`:

1. Start a reserve-funded buy so its planned reserve is debited.
2. Mark it as recoverably stranded and fail it.
3. Assert the trade is failed but portfolio cash is not increased.
4. Repeat for a bridge-funded buy and assert available bridge capital is not
   released.
5. Add a negative guard without the marker and assert an ordinary failed buy
   still restores unused reserve capital.

### Reconciliation test

In `tests/hyperliquid/test_hypercore_accounting.py`:

1. Construct a failed deposit with the reserve debited and a known perp transit
   balance.
2. Run the relevant account-check/correction path with mocked Safe and
   HyperCore balances.
3. Assert no reserve correction, dust cleanup or bridge-capital release occurs
   while the recovery marker exists.
4. Simulate one explicit recovery destination and assert the marker is cleared
   only as the returned Safe cash or increased vault equity is accounted once.

All new pytest tests must have a docstring that lists their high-level steps
as `1.`, `2.`, `3.`, with matching numbered comments in the body. Fixtures
must be type hinted, and mock comments must explain why live HyperCore cannot
be used. Use `pytest.approx()` for USDC/balance comparisons where decimals are
not deliberately exact.

## Documentation updates

Update the following documentation in the same pull request:

- `.claude/docs/hypercore-vault.md`: show separate deposit phases, the perp
  balance checkpoint, the no-EVM-atomicity rule, persisted stranded-capital
  and at-risk metadata, the failure-location table, crash/repair guard and test
  pattern.
- `.claude/docs/vault-deposit-redeem.md`: retain the scope boundary, and
  cross-reference the HyperCore document when discussing committed capital
  and no-double-spend accounting. Do not imply HyperCore uses the ERC-7540
  pending-event model.
- `tradeexecutor/ethereum/vault/README-vault-position-manual.md`: document
  location-aware inspection and recovery. In particular, perp USDC cannot be
  sent directly to EVM; it must first move perp → spot.
- `CHANGELOG.md`: record the recovery-safe deposit settlement and accounting
  correction with the pull request date.

## Validation

Run focused tests from the worktree with the repository environment and parent
Poetry environment:

```shell
source .local-test.env && PYTHONPATH="$(pwd):$PYTHONPATH" poetry run pytest -n auto tests/hyperliquid/test_hypercore_deposit_settlement.py
source .local-test.env && PYTHONPATH="$(pwd):$PYTHONPATH" poetry run pytest tests/test_state.py -k failed
source .local-test.env && PYTHONPATH="$(pwd):$PYTHONPATH" poetry run pytest tests/hyperliquid/test_hypercore_accounting.py -k stranded
```

Then run `git diff --check`. Do not require a live mainnet transfer for CI.
Before production deployment, use `check-hypercore-user.py` to reconcile trade
#1486 separately; deploying the code does not recover its 48.884068 USDC.

## Acceptance criteria

- No EVM transaction contains both deposit settlement actions.
- The vault action cannot be broadcast before the perp increase is observed.
- The phase-2 proof rejects any partial spot or perp movement, including a
  shortfall that would otherwise be hidden by a monetary tolerance.
- Every post-bridge failure records a conservative recovery location.
- A crash or indeterminate phase-1 broadcast cannot reach generic refund
  without live reconciliation.
- A failed post-bridge buy cannot restore or release its stranded allocation.
- An ordinary pre-bridge failed buy still refunds unused allocation.
- Timeout and broadcast-unknown locations never instruct an operator to repeat
  an action without rechecking live balances.
- The mocked trade #1486 path fails safely with the USDC classified as
  `hypercore_perp_or_vault` until reconciliation confirms its final location.
- Account sync and correction cannot re-credit marked transit capital.
- Operator and architecture documentation match the implemented phases and
  recovery steps.
