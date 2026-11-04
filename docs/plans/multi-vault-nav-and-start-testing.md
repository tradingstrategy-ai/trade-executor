# Account for Lighter idle collateral and test strategy execution through `start`

## Goal and boundaries

Include the dedicated Lighter account's idle USDC in NAV before its first pool purchase, so outside subscriptions receive correctly priced shares. Validate Lighter, Atoma and Upshift through strategy-produced trades using the existing dual-fork simulation.

Baseline: executor `3a4f79428` and its pinned eth-defi revision. Reuse the existing state store, trade checkpoints, settlement resolver and simulation harness. Exclude actual NB05 conversion, historical investor-share repairs, production changes and funded transactions.

Approved on 4 November 2026. The text-only Claude CLI review used `claude-opus-5-5` and completed with a successful terminal result. Its corrections concern consistent snapshots, shared treasury enforcement, bootstrap acquisition cost, redemption liquidity and simulation clocks.

## Idle collateral accounting

### State and valuation

- Extend `Portfolio` with a default-empty mapping of typed Lighter idle-collateral observations. Use string account-identity keys and record custody chain, Safe, account index, Decimal USDC amount and naive UTC observation time. Older states load with an empty mapping.
- Discover activated accounts from public deployment metadata before any pool position exists. Validate the account's association with the Ethereum satellite Safe; public valuation must not require loading private API keys.
- Include idle collateral once in NAV, total equity and chain totals. Keep it distinct from reserve cash and immediately available bridge funding. Display the saved amount and timestamp in `show-valuation`, which remains an offline reader.
- Use observed collateral rather than aggregate account equity, whose relationship to pool holdings remains unverified. Reuse the executor's existing Lighter API session and retry policy.
- Require a fresh, consistent observation in the shared Lagoon treasury path before posting NAV or settling investor flows. Command-specific setup must not be the only protection. Cover scheduled treasury sync, `start`, `lagoon-settle` and `perform-test-trade`.
- Refuse a custom valuation callback combined with pool idle-collateral accounting until that combination explicitly supports the same accounting contract. Do not silently add the observation to an arbitrary callback result.

### Attribution and first purchase

Use one account snapshot for pool valuation and idle-collateral attribution. Reconcile saved operations before interpreting balances. Preserve the existing one-pending-operation restriction for each account.

| Phase | Accounting treatment |
|---|---|
| Activated, no pool trade | Observed collateral is idle NAV; remaining bootstrap capital is Safe or bridge cash. |
| Deposit submitted, not credited | Deposit value remains in the existing pending trade; do not subtract uncredited capital from collateral. |
| Deposit credited, mint pending | Attribute credited capital to the pending buy; retain only unrelated idle collateral as idle NAV. |
| Mint confirmed | Pool holdings replace consumed collateral; clear the consumed idle amount. |
| Burn confirmed | Existing exit valuation owns released proceeds; idle NAV excludes those proceeds. |
| Secure withdrawal pending | The saved exit owns its receivable, including any swept idle amount transferred into that exit. |
| Claim confirmed | Verified Safe/bridge cash replaces the receivable. |

- Treat the strategy's buy amount as the **total investment budget**. Consume existing idle collateral first and deposit only the shortfall. A 5-USDC target with 1 USDC idle therefore requires approximately 4 USDC of new funding, subject to actual fees and protocol sizing.
- Record the consumed idle amount on the existing trade. Extend allocation/completion bookkeeping to distinguish Safe-funded capital from idle-funded capital. The position's acquisition cost includes both sources; reserve and bridge balances decrease only by their actual contribution.
- Book consumption and pool acquisition together through existing state callbacks. A successful mint must neither manufacture a bootstrap profit nor temporarily count the same cash and shares twice.
- Refresh or invalidate observations whenever operations change collateral. Invalid observations must not appear as valid current totals in statistics or permit settlement; diagnostics must identify incomplete valuation.
- Refuse inconsistent ownership, ambiguous operation state, unexplained balance differences or failed reads. Preserve evidence and resume through existing reconciliation; never substitute zero for an unknown balance.
- Do not automatically write off idle dust. Full investor redemption must refuse while material idle collateral remains without a supported withdrawal path. For this scope, recover it through the documented pool purchase/exit lifecycle before final redemption.

Document the bootstrap reconciliation explicitly: the 20-USDC subscription becomes approximately 18.99 USDC in the hub Safe plus 1 USDC on Lighter, with the actual bridge fee explaining the difference. Verify that no existing bridge position already represents the same activation USDC.

## Simulated `start` coverage

Add `tests/lagoon/test_multi_vault_start_e2e.py`, reusing the existing simulation helpers and `strategies/test_only/multi_vault_arbitrum.py`.

- Reuse persistent Arbitrum/Ethereum forks, the fake Lighter sequencer and forged attestations for real CCTP messages. Retain real Safe, Guard, Lagoon, Atoma and Upshift transactions.
- Use Arbitrum block `520900000` and the existing Ethereum midnight block. The code-built universe and seeded fake pool supply fixed inputs.
- Invoke the registered `start` command in separate bounded one-cycle processes with durable state and explicit fork RPCs. Do not use production `--simulate`, whose persistence semantics differ.
- Keep test controls in the harness. Advance the executor's test clock and both relevant fork clocks consistently between cycles; pass the simulated UTC time through the existing CLI debug-clock mechanism, extending test plumbing only if necessary.
- Preserve the existing exit trigger: the mixed strategy closes an existing position when its redemption check permits it. Pending/test positions remain excluded. Use existing public reclaim commands for any final idle satellite cash.

Use two end-to-end scenarios:

1. **Complete lifecycle.** Deploy, initialise, verify idle bootstrap NAV, then settle an outside subscription before any pool purchase. Run `start` cycles to open all three ordinary positions, advance protocol time/operator settlement, complete exits, return satellite cash and redeem investors.
2. **Pending and restart.** Delay sequencer confirmation and Atoma settlement across process boundaries. Restart from the same state and venue data; confirm original operations resume without duplicate submission or premature funding. Inject an account-read failure before subscription settlement, verify refusal, restore the provider and complete successfully.

Add a focused interruption regression at the existing pre-submission/submission checkpoint boundary. Demonstrate recovery of a submitted operation whose result was not yet booked, without requiring a broad crash-testing framework.

## Validation and acceptance

- Accounting regressions cover legacy state loading, Decimal/datetime round trips, repeated refresh, bootstrap reconciliation, mixed idle/Safe funding, pending attribution, withdrawal dust and failed or inconsistent observations.
- Derive investor shares from verified pre-subscription NAV and on-chain supply, allowing actual fee and raw-unit rounding effects.
- Reconcile phase NAV independently from Safe balances, bridge transit, idle collateral, net pool holdings and pending claims.
- Assert no false bootstrap profit, unintended immediate rebalance sale, duplicate operation, duplicate claim or spending of pending proceeds.
- Confirm all three positions originate from `decide_trades()`, finish through normal execution/recovery, and leave no unexplained exposure before investor redemption.
- Run focused accounting tests, the new fork tests and affected existing Lighter/manual-CLI regressions. Use the repository environment, worktree `PYTHONPATH`, extended deadlines and `-n auto` where required. Do not format code.
- Update the deployment guide and readiness checklist with results and simulation limitations. Record missing credentials/archive access as validation gaps. Simulated success does not establish live sequencer or operator availability.

## Implementation and verification, 4 November 2026

Implemented the portfolio observation, total-budget pool funding and shared
treasury gate. During an unresolved pool operation the observation is marked
invalid: NAV reads refuse rather than showing a partial total. The daemon defers
decisions, treasury publication and statistics while its settlement sweep keeps
running. Offline
`show-valuation` still lists positions and identifies the incomplete observation.
The settlement sweep retries invalid observations even when no pending ticket
remains. Confirmed sequential trades and resolved sweep outcomes are checkpointed
before reporting or an external-account refresh can fail.

The no-fee fork rehearsal also exposed the unused 0.01 USDC activation bridge
allowance in the Ethereum Safe. The first valuation books its actual balance
through an existing accounting-correction trade on the bridge position, not as
another investor deposit. The simulation bootstrap reconciles to 18.99 hub cash
+ 0.01 satellite cash + 1 Lighter collateral = 20 USDC. Live runs use the actual
remaining balance and fee. Pool withdrawals return swept idle cash separately
from pool PnL, including through the public test-trade bridge-back path.

The normal `start` path exposed four additional wiring defects: multichain
`init` selected the first RPC instead of the strategy's primary chain; account
checks treated Lighter shares as ERC-20s; an empty asset mapping assigned a
position object instead of a USD difference; and the mixed test strategy passed
integer budgets to live pricing. These were corrected without changing NB05
selection or allocation policy.

The simulation wrapper supplies the fork-based UTC clock to command processes
and lets it advance with real monotonic elapsed time, so a recovered buy's lock
can expire normally. Production `--simulate` is not used. Simulation controls
refuse execution without local Anvil custody chains.

### Recorded checks

| Check | Result |
|---|---|
| `tests/lagoon/test_lighter_idle_collateral.py` | 7 passed; includes serialisation, fractional/mixed/idle-only funding, pending multi-pool batches, expired top-ups, missing account metadata, submitted-operation reuse, cash/PnL separation, torn/failed reads, empty-sweep recovery and checkpointing through transport or invariant failures |
| `tests/lagoon/test_lighter_pool_universe.py` | 4 passed |
| `test_multi_vault_start_lifecycle` | Passed in the final two-worker fork run (331.64 seconds for both scenarios): subscription before first purchase, full redemption queued before recovery trades, three ordinary round trips, independent custody/NAV checks, bridge-back and investor redemption |
| `test_multi_vault_start_restart_and_refusal` | Passed in the same final fork run: provider refusal without transactions, repeated startup and manual settlement during a pending mint without new statistics, original operation identity reused, delayed Atoma settlement and completed exits |
| `tests/lagoon/test_multi_vault_manual_cli_e2e.py` | Earlier implementation run passed in 535 seconds: the existing shell rehearsal completed all three round trips and investor redemption; investor received 130.503998 simulated Arbitrum USDC. Not repeated during clean-up; the strengthened strategy fork tests exercise the changed terminal redemption path. |
| Existing CCTP funding regressions | Partial idle funding, sub-raw-unit rounding and insufficient funding: 3 selected cases passed |
| Existing multichain balance routing regression | 1 selected case passed |
| Existing vault settlement regressions | 9 cases passed across pending, claimable, operator payout, already-mined claim recovery and reclaimed capital; checkpoint-count expectation includes the new completed-outcome save |
| `tests/cli/test_lagoon_redeem.py` | 6 passed, including completed-claim restart without duplicate settlement, terminal NAV for drained pool deployments, refusal of started/broadcasted trades without tickets and held shares with a stale zero price |
| Existing portfolio statistics regressions | 2 selected cases passed: legacy initial-cash handling and position share-price statistics |

The original implementation passed 20 selected cases. PR clean-up passed 26
focused cases: seven collateral, six redemption CLI, nine settlement, two
statistics and both fork scenarios. The final small terminal-state checks were
verified by the CLI regressions and focused review after the fork rerun.
`git diff --check` and local documentation-link checks also passed. The full
test suite was not run.

Run each module or selected node separately, sourcing `.local-test.env` and
putting the worktree and pinned dependency sources first on `PYTHONPATH` while
using the installed parent Poetry environment. The fork inputs are Arbitrum
520900000 and Ethereum 25598869, with the executor's pinned dependencies. The
strategy universe is code-built; no live market-data universe is substituted.

Claude CLI `claude-opus-5-5` completed a grounded read-only code review and a
focused follow-up. Its recovery findings were addressed with outcome checkpoints
and invalid-observation refresh before NAV reads; a regression reproduces a
post-completion API failure, reloads the checkpoint and recovers without another
trade. The previously missing Decimal import was also corrected.

A read-only real-provider check of Lighter account 766589 confirmed Safe owner
`0x8DbC6562306834b39b88Be77614617548210132b`, zero collateral and no pool shares,
with identical economic snapshot inputs across two reads. Aggregate equity
telemetry is deliberately excluded from the stability comparison. This closed
account check establishes parser/ownership compatibility, not live execution
or the relationship between aggregate account equity and pool holdings.

All EVM custody, Guard, vault and CCTP calls in the fork tests are real local
transactions. Lighter's sequencer and Circle attestations are simulated; fork
funding, clock advancement and Atoma operator settlement are explicit test
actions. No funded transaction, production state mutation or historical
investor-share repair was performed.

### PR clean-up, 4 November 2026

An idle-only purchase now skips signing Safe approval/deposit transactions,
preserving the transaction builder's nonce. A focused regression completes
the mint and checks that no EVM transaction is signed. The accounting module's
seven cases passed after the review corrections. The agent integration guide records
the collateral and checkpoint rules; the changelog and diagnostic wording have
also been updated. Changed narrative text was checked for UK English,
sentence-case headings and agreement with the implementation.

The completed Claude Opus 5.5 clean-up review identified fractional-budget
rounding, queued-redemption deadlock, daemon shutdown on pending operations,
missing account configuration, outcome checkpoints and sell/buy idle allocation
as correctness issues. Budgets now round to executable USDC units before funding
allocation; mints stay within credited collateral. Investor settlements defer
until hub cash is sufficient, allowing recovery trades to proceed. Pending
operations defer statistics and decisions without terminating `start`. Pool
universes require public account metadata, every changed settlement outcome is
checkpointed in `finally`, and a sell/buy batch does not allocate swept idle cash
to its buy. The new fork checks cover a full redemption queued before the first
purchase and a second restart while the same mint is still pending.

The follow-up review also caught expired top-ups being mistaken for unresolved
operations and later pool orders trying to allocate cash after a pending mint.
Expired trades now leave NAV available; later planned pool orders expire without
allocation when the account is busy. Manual settlement and stop-loss triggers
defer while portfolio valuation is incomplete. This pauses triggers on other
venues too, until the pending operation reconciles.

The queued-redemption fork exposed an obsolete per-controller request after a
completed claim. `lagoon-redeem` checks the global queue before another settlement,
including after a restart. Its pool-specific terminal path requires the state and
public deployment metadata, refreshes account ownership and collateral, rejects
remaining tracked exposure and reads the actual Ethereum Safe USDC balance. Only
a drained deployment uses hub Safe cash as terminal NAV; pool holdings and
collateral never use the legacy exchange-account dust exclusion.

Claude CLI `claude-opus-5-5` completed the grounded clean-up review and focused
follow-ups. Every review result was checked for successful completion. The final
blocking-only follow-up reported no blocking findings after explicit checks for
all started/broadcasted submissions, non-zero held quantities despite a zero
price, and persistence of refreshed observations before terminal transactions.
The final follow-up reviewed supplied excerpts; the earlier passes inspected the
repository directly. No reviewer ran tests or changed the worktree.

Remaining limits: fork success does not establish live sequencer/operator
availability; the real-provider compatibility check used a closed account.
Budgets still need to respect Lighter's minimum deposit and mint requirements.
Terminal redemption deliberately refuses residual holdings or satellite cash
instead of writing them off. The existing per-controller redemption getter can
remain ambiguous when another investor has a queued request; this can require
another invocation after that queue settles. No historical investor shares were
repaired and no production or funded transaction was performed.

### CI follow-up, 4 November 2026

The first PR run passed the slow suite and both strategy fork scenarios, but
the main suite found eight failures. `init` now passes the existing parsed
`UNIT_TESTING` setting into Web3 configuration, so multichain test nodes can
use chain ID 31337 while live commands retain chain validation. Minimal test
models now declare the optional pool account and empty collateral mapping;
generic-vault mocks explicitly report that they are not Lighter pools. The
HyperCore pre-broadcast regression also expects the new outcome checkpoint.
Nine focused cases covering these failures passed locally. A completed
Claude CLI Opus 5.5 review of the small correction diff found no blocking
findings.
