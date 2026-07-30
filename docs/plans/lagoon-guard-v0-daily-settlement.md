# Lagoon GuardV0 settlement integration

## Objective

Integrate the GuardV0 Lagoon v0.5 settlement safety policy into the
trade-executor treasury sync.

Every successful Lagoon post-valuation treasury sync must post the freshly
calculated NAV. A non-zero deposit/redemption queue is settled automatically
only when its gross underlying flow is within the Guard-configured maximum and
the cooldown permits another automated settlement. An oversized queue is
left untouched for direct Safe-governance settlement and reported at `ERROR`
level.

The implementation must preserve these distinctions:

| Queue and Guard state | NAV transaction | Settlement transaction | Logging |
|---|---:|---:|---|
| No pending flow | Always | No | Normal diagnostics |
| Guard limit disabled or unsupported by an older module | Always | Existing automatic behaviour for pending flow | Normal diagnostics |
| Gross flow at or below the cap and cooldown available | Always | Yes | Normal settlement diagnostics |
| Gross flow at or below the cap but cooldown active | Always | No | Informational deferral with next eligible time |
| Gross flow above the cap | Always | No | `ERROR`: direct Safe settlement required |

The amount comparison follows GuardV0's inclusive rule: an automated
settlement is permitted when `grossSettlementAmount <= maxSettlementAmount`.

## Pre-change behaviour

`LagoonVaultSyncModel.sync_treasury()` currently uses
`check_nav_update_and_settle_needed()` as one combined gate. If the gate opens,
the method signs and broadcasts `updateNewTotalAssets()` and
`settleDeposit(uint256)` together, then analyses the settlement receipt and
creates a reserve `BalanceUpdate`.

This design cannot support the Guard policy because:

- the NAV and settlement transactions cannot be selected independently;
- the executor does not read `getLagoonSettlementSafetyConfig(address)`;
- it will submit a transaction which GuardV0 atomically reverts for an
  oversized queue or active cooldown;
- skipping settlement also skips the NAV update; and
- a failed module transaction does not provide an operator-friendly instruction
  to settle directly through Safe governance.

The web3-ethereum-defi dependency already contains the required GuardV0 and
TradingStrategyModuleV0 v0.5 ABIs. The complete safety getter returns:

1. whether the Lagoon vault is allowed;
2. whether the limit is enabled;
3. the measured underlying asset;
4. the pending-deposit Silo;
5. the maximum gross amount in raw underlying units;
6. the cooldown in seconds;
7. the previous non-zero settlement timestamp; and
8. the next eligible non-zero settlement timestamp.

GuardV0 measures the exact post-execution movement as:

```text
deposit assets = Silo balance before - Silo balance after
redeem assets  = vault balance after - vault balance before
gross flow     = deposit assets + redeem assets
```

Deposits and redemptions must never be netted. Guard validation checks the
amount before the cooldown, so an oversized queue remains a manual-settlement
condition even while a previous automated settlement is cooling down.

## Scope

Primary implementation:

- `tradeexecutor/ethereum/lagoon/vault.py`

Focused tests:

- `tests/lagoon/test_lagoon_guard_settlement.py` for the new deployment-heavy
  Guard integration and decision tests;
- `tests/lagoon/test_lagoon_deposit.py` for adjustments to existing assumptions
  about no-op post-valuation calls.

Documentation:

- add `.claude/docs/lagoon-treasury-settlement.md` as the operator and developer
  deep-dive for the strategy's own Lagoon treasury;
- add the new deep-dive to the reference-doc table in `AGENTS.md`;
- update `.claude/docs/vault-deposit-redeem.md` with a cross-link from its
  existing scope note, keeping external ERC-7540 investment flows clearly
  separate from the strategy's own investor settlement queue.

The deployment contract and Guard enforcement remain owned by
web3-ethereum-defi. Trade-executor consumes the existing v0.5 getter and custom
errors; it must not recreate or weaken their policy.

## Guard capability and state model

Read the returned safety tuple as raw integer values. The executor uses the
enabled flag and validates the configured asset and Silo before simulating a
non-empty queue; GuardV0 remains the source of truth for the exact gross amount.

Capability detection must be backward compatible:

- read the advertised module version through the version getter's typed ABI
  call, with the established pre-version-getter fallback for older modules;
- modules whose known version predates the settlement-safety getter retain
  unlimited legacy behaviour and must not be probed for the getter;
- the current settlement-safety implementation is enabled for the explicitly
  supported TradingStrategyModuleV0 version `v0.5`;
- a supported `v0.5` module with `limitEnabled == false` retains unlimited
  behaviour;
- an enabled policy must identify the same vault asset and Silo that
  `LagoonVault` reports;
- a malformed enabled policy, an unexpected zero address, or a supported
  version's getter failure is a configuration error and must fail closed; and
- do not add numeric, semantic or feature-probing version inference. When the
  smart contract is changed, bump its existing advertised module version and
  explicitly update the trade-executor version dispatch and tests for that new
  version.

The preflight result distinguishes:

- `no_pending_flow`;
- `automatic_settlement_available`;
- `cooldown_active`;
- `manual_settlement_required`; and
- `unlimited_legacy_settlement`.

It carries the pending raw queue sizes and, for GuardV0 rejections, the
contract-reported cap, gross amount or next eligible timestamp. Keeping this
decision explicit avoids spreading policy branches through transaction signing
and balance-update analysis.

## NAV and settlement transaction separation

Refactor the current combined transaction section into two independently
confirmed phases.

### NAV phase

For every `sync_treasury(..., post_valuation=True)` call that passes the
existing frozen-position and valuation-freshness safety checks:

1. reconcile the Safe reserve balance;
2. calculate the fresh strategy NAV;
3. build and broadcast `vault.post_new_valuation(valuation)`;
4. wait for and validate the successful NAV receipt before making the
   settlement decision; and
5. retain the NAV receipt block for treasury sync metadata when no settlement
   follows.

The production path and the Anvil path must both send only this transaction at
this stage. Do not pre-sign the settlement transaction because the decision is
made against the post-NAV on-chain state.

`min_nav_change_update` and `check_nav_update_and_settle_needed()` can no longer
gate NAV posting. Keep the constructor argument temporarily if removing it
would create unrelated caller churn, but document it as compatibility-only and
do not let it suppress a requested post-valuation update.

Calls with `post_valuation=False` remain read/sync-only and do not broadcast.
The frozen-position safety remains intentionally stronger than “always”: when
the executor cannot calculate a trustworthy NAV, it must still abort before
posting anything.

### Flow and Guard phase

After the NAV receipt is confirmed:

1. read pending deposits from the Silo in raw underlying-token units;
2. read pending redemption shares in raw share-token units;
3. read the complete Guard settlement safety tuple when supported;
4. for every non-empty guarded queue, simulate the exact wrapped module
   settlement using `eth_call` from the configured asset-manager hot wallet;
5. classify the returned Guard result, checking the amount-limit error before
   treating the queue as cooldown-only; and
6. broadcast settlement only when the simulation succeeds.

The simulation is the final authority because share conversion, Lagoon fee
accrual and raw-unit rounding can make a local redemption estimate differ from
the exact token movement. The diagnostic estimate must therefore never suppress
the simulation or authorise a broadcast. Simulating an apparently oversized
queue is necessary to obtain GuardV0's exact `actualAmount` and to preserve the
contract's amount-before-cooldown precedence. It also prevents wasting gas on a
transaction that GuardV0 will reject.

Decode at least:

- `LagoonSettlementLimitExceeded(uint256 actualAmount, uint256 maxAmount)`;
- `LagoonSettlementCooldownActive(uint256 currentTimestamp, uint256 nextSettlementTimestamp)`.

An amount-limit result takes precedence over cooldown. Other simulation errors,
including insufficient redemption liquidity or an unexpected Guard/configuration
failure, retain the existing fail-fast behaviour and must not be mislabelled as
a cooldown deferral.

### Settlement phase

When automatic settlement is available:

1. sign and broadcast the already simulated module call;
2. wait for its receipt using the existing robust receipt propagation helper;
3. run `analyse_vault_flow_in_settlement()`;
4. create the reserve `BalanceUpdate`; and
5. update treasury share count, pending redemptions and settlement metadata as
   today.

When settlement is skipped:

- return no `BalanceUpdate`;
- mark the treasury sync completed at the confirmed NAV block;
- leave pending deposits, shares and Safe/vault token balances unchanged;
- do not consume the Guard cooldown timestamp;
- do not analyse a synthetic or absent settlement receipt; and
- persist state normally through the existing runner/start-up/stats-refresh
  callers.

## Operator diagnostics

For an oversized queue, write one `logger.error()` record each time the
condition is observed. The message must be directly actionable and contain:

- a clear statement that automated settlement was skipped;
- a clear statement that direct Safe-governance settlement is required;
- chain id;
- Lagoon vault address;
- Safe address;
- TradingStrategyModuleV0 address;
- pending deposit assets;
- pending redemption shares;
- exact Guard-reported gross amount;
- maximum permitted gross amount;
- underlying token symbol and raw decimals-aware values; and
- confirmation that the NAV update succeeded and the queues remain pending.

Do not suggest `lagoon-settle` as the recovery command: it uses the same
asset-manager module and remains subject to the Guard cap. Direct Safe
governance deliberately bypasses the module policy.

For an active cooldown, use `INFO` or `WARNING`, include the UTC next-eligible
time, and state that the queue will be retried automatically. Cooldown alone
must not request manual intervention.

## Documentation changes

Create `.claude/docs/lagoon-treasury-settlement.md` alongside the existing
architecture deep-dives. It must document:

- that this flow concerns external investors entering or leaving the strategy's
  own Lagoon vault, not the executor depositing into a third-party ERC-7540
  vault;
- the two independent transactions, `updateNewTotalAssets()` and guarded
  `settleDeposit(uint256)`, and why NAV posting can succeed while settlement is
  deliberately skipped;
- the GuardV0 safety tuple, inclusive cap and 24-hour default cooldown;
- gross-flow accounting, including why simultaneous deposits and redemptions
  are added instead of netted;
- the executor decision table for no flow, automatic settlement, cooldown
  deferral and oversized manual settlement;
- the meaning and expected fields of the `ERROR` log;
- the operational recovery boundary: `lagoon-settle` still uses the guarded
  asset-manager module, while an oversized epoch requires a direct
  Safe-governance transaction;
- the invariant that skipped or rejected settlement leaves both queues,
  balances and Guard cooldown state unchanged; and
- focused troubleshooting checks for module version, configured asset/Silo,
  cap, next eligible timestamp, pending deposits and pending redemption shares.

Update the reference table in `AGENTS.md` so future work in
`tradeexecutor/ethereum/lagoon/vault.py` is routed to this document. Add a short
link from the scope note in `.claude/docs/vault-deposit-redeem.md`; do not merge
the two documents because they describe opposite sides of a vault relationship
and different executor code paths.

## Anvil Base-fork tests

Add two focused Base-mainnet-fork tests using the existing
`tests/lagoon/conftest.py` Anvil infrastructure. Deploy a fresh stock Lagoon
v0.5 vault with `max_settlement_amount=Decimal("10")` and the default 24-hour
cooldown. The fixture must use the same `LagoonVaultSyncModel` and hot-wallet
transaction path as production; do not mock Guard decisions or settlement
transactions.

Both tests must follow the repository pytest rules: function-style tests,
typed fixtures, numbered docstring steps repeated as body comments, no stdout
or permanent info-level logging, and `pytest.approx()` for money comparisons.

### Happy path: automatic settlement below the cap

Exercise a real non-zero automated settlement:

1. initialise executor state and reserve tracking;
2. queue a 9 USDC deposit against the 10 USDC cap;
3. run `sync_treasury(post_valuation=True)`;
4. assert the NAV transaction succeeded;
5. assert the settlement transaction succeeded and the Silo queue is empty;
6. assert the Safe received the deposited USDC;
7. assert exactly one reserve `BalanceUpdate` records the 9 USDC flow;
8. assert total supply/share-count metadata was updated;
9. assert GuardV0 recorded a non-zero `lastSettlementTimestamp` and a
   `nextSettlementTimestamp` 24 hours later; and
10. assert no manual-settlement error was logged.

Queue a second below-cap deposit during the first settlement's cooldown and run
another post-valuation sync. Verify it is NAV-only: no additional balance
update, no new cooldown timestamp, no settlement transaction, and an
informational automatic-retry message. Use receipt/nonce or event evidence,
rather than only comparing an unchanged NAV value, to prove the NAV transaction
occurred. The updated existing no-op test separately proves that an empty queue
also receives a NAV-only transaction.

### Bad path: NAV posted but gross flow requires Safe governance

Stress gross rather than net flow so the test would fail if deposits and
redemptions were incorrectly offset:

1. bootstrap the vault with a below-cap deposit and finalise the depositor's
   shares;
2. queue a 9 USDC deposit and a redemption worth more than 1 USDC, making the
   gross flow exceed the 10 USDC cap even though the net Safe movement is below
   it;
3. run `sync_treasury(post_valuation=True)` while the bootstrap settlement's
   cooldown is still active;
4. assert the new NAV was posted successfully;
5. assert amount-limit handling takes precedence over cooldown handling;
6. assert no settlement transaction was broadcast;
7. assert the deposit assets and redemption shares remain in their queues;
8. assert Safe, Silo and Lagoon vault underlying balances show no settlement
   movement;
9. assert no reserve `BalanceUpdate` was created for the rejected queue;
10. assert GuardV0's previous and next settlement timestamps did not change;
11. assert the asset-manager nonce advanced only for the NAV transaction; and
12. capture logs with `caplog` and assert an `ERROR` contains the manual Safe
   instruction, actual gross amount, cap, vault and Safe addresses.

The test must not broadcast an intentionally reverting settlement and merely
accept a failed receipt. Its purpose is to prove the executor recognises the
condition before broadcast while still completing the independent NAV phase.

Update existing Lagoon treasury tests whose assumption was that a no-op
post-valuation sync broadcasts nothing. They should continue to expect no
balance event, but now verify a NAV-only transaction and completed treasury
metadata.

## Verification

Prepare the environment first:

```shell
source .local-test.env
```

Run focused tests one at a time with the required extended timeout:

```shell
source .local-test.env && poetry run pytest tests/lagoon/test_lagoon_guard_settlement.py::test_lagoon_guard_automatically_settles_flow_below_settlement_limit
source .local-test.env && poetry run pytest tests/lagoon/test_lagoon_guard_settlement.py::test_lagoon_guard_posts_nav_but_defers_oversized_flow_to_safe
```

Run the existing Lagoon deposit tests affected by the transaction split. Do not
run the complete test suite for this change.

## Acceptance criteria

- A trustworthy NAV is posted on every requested post-valuation sync,
  independently of whether a queue can settle.
- No Guard-capped settlement is broadcast without a successful preflight
  simulation.
- A gross flow at or below the cap settles automatically once cooldown allows.
- A cooldown-only queue remains pending and is retried automatically.
- An oversized queue remains unchanged and produces an actionable
  manual-Safe `ERROR`.
- Deposits and redemptions are added as gross movement, never netted.
- Exact-cap settlement follows GuardV0's inclusive boundary.
- Legacy and uncapped Lagoon modules keep working.
- Skipped settlements produce no false balance updates or cooldown changes.
- The Base-fork happy and bad paths exercise real deployed GuardV0 enforcement,
  transaction signing, Lagoon queues and executor state accounting.
- The new Lagoon treasury Markdown deep-dive and existing external-vault flow
  documentation clearly distinguish the two settlement systems and describe
  manual Safe recovery.

## Out of scope

- Changing GuardV0, LagoonLib or their on-chain policy.
- Partial settlement of an oversized Lagoon epoch; stock Lagoon v0.5 settles a
  snapshotted queue in full.
- Automatically bypassing GuardV0 through Safe governance.
- Adding a new CLI command to construct or sign the direct Safe recovery
  transaction.
- Persisted alert de-duplication or acknowledgement state.
- Changing the frozen-position safety rule or posting a NAV when valuation is
  not trustworthy.
