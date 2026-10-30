# Atoma executor integration plan

## Objective and scope

Prepared on 2026-10-29 following executor PR [#1694](https://github.com/tradingstrategy-ai/trade-executor/pull/1694),
as a subplan of [NB05 ERC-4626 live support](nb05-erc4626-live-support.md#6-atoma).

Map Atoma onto the existing vault integration used by Lagoon and Ostium:
`VaultDepositManager`, routed `TradeExecution`, serialised redemption tickets,
`vault_settlement_pending`, the shared settlement resolver and ordinary executor
state. Complete the remaining adapter wiring and demonstrated consumer fixes.
**Do not add integration infrastructure.** Use existing accounting, valuation,
CLI, custody and testing mechanisms throughout.

| Vault | Chain | Proxy |
| --- | --- | --- |
| Atoma Index | Arbitrum, 42161 | `0xcc56410e1a136af0eceb7241c6ae394f4d8b581c` |
| Atoma RWA | Arbitrum, 42161 | `0x1c788e14d8e5b446e3f71b5142e2edabcab36da1` |

Required entry points are strategy-generated trades, `perform-test-trade`,
`lagoon-deploy-vault`, normal settlement retry, account checks and Lagoon NAV.
Satellite execution and production governance remain in the parent plan. This
document records the implementation scope; no funded live run is included.

Implementation and focused validation were completed locally on 2026-10-30;
see the [Anvil validation report](../reports/atoma-executor-anvil-2026-10-30.md)
for the exercised behaviour, local receipt evidence, test results and remaining
publication/live-availability limits. Historical review notes below describe the
plan before implementation.

## Baseline and reference implementation

- Executor baseline is `1fd2880ae`, with eth-defi pinned to
  `de5538c4c8ab185001b3e6fadd14f0b10fdb6aef`. That revision lacks the specialised
  Atoma manager; its later commit date does not imply inclusion.
- Upstream [Atoma #1695](https://github.com/tradingstrategy-ai/web3-ethereum-defi/pull/1695)
  merged as `752decd1ed56a1fff224ffa35492d494a02bf942`. The inspected candidate
  `2939ee63ee44de354b827668f64b5f23ffd52d7f` includes it and the ERC-4626 sizing
  fixes from #1700. Review the full dependency difference before selecting the
  final pin, lockfile and matching contract artefacts.
- That candidate already provides synchronous deposits, epoch redemption
  tickets, request/claim analysis, implementation checks, aggregate ticket
  reconciliation, strict Anvil settlement and `whitelistAtoma()`. Consume these.
  [Upstream execution evidence](https://github.com/tradingstrategy-ai/web3-ethereum-defi/blob/2939ee63ee44de354b827668f64b5f23ffd52d7f/docs/README-atoma-execution.md)
  covers guarded forks for both vaults, not funded executor production use.

Read these local references before implementation:

- [Vault deposit and redeem flows](../../.claude/docs/vault-deposit-redeem.md):
  shared lifecycle, accounting and command behaviour.
- [Lagoon treasury settlement](../../.claude/docs/lagoon-treasury-settlement.md):
  the strategy's own investor treasury, NAV and Guard settlement limits.
- [State persistence](../state-persistence.md): existing records, stores and
  persistence callbacks.
- [Lagoon ERC-7540 integration tests](../../tests/erc_4626/test_vault_async_lagoon_erc_7540.py):
  the template for pending balance checks, revaluation, state reload, close
  reruns and already-completed claim recovery.
- [Shared settlement tests](../../tests/ethereum/test_vault_settlement_lifecycle.py)
  and [async backtests](../../tests/backtest/test_backtest_async_vault.py):
  consumer regressions, including the existing Atoma fee/delay case.

Keep the two uses of Lagoon distinct. Atoma is an **external vault position**
owned by the strategy Safe. The strategy's **Lagoon treasury** serves investors.
Atoma claims finish ordinary trades; they are not Lagoon investor-flow
`BalanceUpdate` events and must not enter the Lagoon settlement event scanner.

## Mapping to existing primitives

Paths below are relative to `tradeexecutor/`, except dependency paths.

| Atoma behaviour | Existing primitive | Integration work |
| --- | --- | --- |
| Immediate deposit | Manager synchronous-deposit capability and `ethereum/vault/vault_routing.py` | Select the upstream Atoma manager; account for the actual `Deposit` receipt. No pending-deposit lifecycle. |
| Share lock and admission | Manager preflight, shared vault admission and `VaultFlowUnavailable` | Use the adapter's unlocked-share, minimum-deposit and implementation checks. Keep protocol/account restrictions distinct from strategy policy. |
| `requestWithdrawal(shares)` | `RedemptionRequest`, manager parser, routing transaction roles | Parse the mined request into the upstream `AtomaRedemptionTicket`; fix post-broadcast reconstruction as described below. |
| Epoch and request evidence | Manager ticket serialisation in `TradeExecution.other_data` | Store the upstream ticket through `State.mark_vault_settlement_pending()`. Epochs, implementation and request parts already belong to this ticket. |
| Waiting for operator settlement | `TradeStatus.vault_settlement_pending`; `ExecutionModel.resolve_pending_vault_settlements()` | Poll through `ethereum/vault/settlement_retry.py`, already called at startup and each strategy cycle. No Atoma worker, scheduler or new trade status. |
| Claimable epoch and `claimWithdrawal(epoch)` | Manager status, `can_finish_redeem()`, `finish_redemption()` and claim analysis | Let the shared resolver build, sign and analyse the claim with the existing custody transaction builder. Temporary illiquidity leaves the trade pending. |
| Saved request/claim and restart | `BlockchainTransaction`, `StateStore`, execution-model persistence callbacks | Reuse transaction hashes, signed bytes and receipts; wire the existing save callback before claim broadcast. |
| Shares unavailable while exiting | `TradingPosition.get_available_trading_quantity()` and `has_pending_vault_settlement()` | Reuse unchanged. A repeated close must resume the pending trade. |
| Wallet balance differs from position quantity | `TradingPosition.get_held_assets()` through `strategy/asset.py` | Existing pending-sell quantities already exclude escrowed shares from expected wallet balances. Verify the same behaviour after the Atoma share burn; no reconciliation ledger. |
| Pending position equity | `VaultValuator`, `ValuationUpdate`, existing position and portfolio value | Keep the holding in the same position; value its settled entitlement using manager claim amounts. No receivable token, extra position or separate portfolio balance. |
| Proceeds cannot fund buys before claim | `PositionManager.is_async_vault_sell_pair()`, AlphaModel and YieldManager gates | Preserve the existing Atoma delayed-redemption classification and gates. Test the routed strategy path. |
| Public close/resume | `cli/close_position.py` and `cli/testtrade.py` resolver/force-settlement helpers | Extend the synchronous-only test-trade gate using the existing close rerun order; keep test-position ownership checks. |
| Safe custody and permissions | `LagoonTransactionBuilder`, existing Lagoon deployment configuration and upstream `whitelistAtoma()` | Select the specialised policy and matching artefacts. No new contracts or governance framework. |
| Investor NAV and withdrawal | `LagoonVaultSyncModel.calculate_valuation()` / `sync_treasury()` and `lagoon-settle` | Consume the corrected position value and normal reserve cash. Retain existing budgets, cooldowns, freshness checks and investor-flow accounting. |
| Test settlement | `AnvilForkPool`, snapshot fixtures and manager `force_settle(ticket)` | Swap the target vault/manager in the established Lagoon test pattern. Use the real Atoma operator and deployed ABI. |

## Remaining changes within these paths

### 1. Dependency selection and request parsing

Ship the dependency bump and compatible executor consumers in one change, after
validation. No feature flag is needed. If delivery must be split, use the existing
admission mechanism to refuse new requests until consumers are compatible.
Preserve recovery of existing holdings; their absence in deployed strategies
has not been established.

The concrete parsing gap is in `vault_routing.py`: after a successful request it
reconstructs a request with `create_redemption_request(check_enough_token=False)`.
The inspected Atoma adapter still performs unlocked-share and same-epoch
admission checks there, although the transaction already escrowed the shares.
Make the existing reconstruction flag usable for mined requests without
repeating balance/lock and same-epoch admission checks. Keep admission mandatory
for new requests and receipt owner/share/version checks mandatory when parsing.
Solve this upstream at the existing manager boundary; do not catch admission
exceptions in routing or introduce another request-construction abstraction.

Reuse `vault_raw_amount`, `vault_redeem_shares`, owner/direction metadata and
`vault_transaction_role` / request indices already recorded by routing. Reuse
`vault_redemption_preflight` for any required preflight evidence, rather than
adding parallel metadata. The upstream parser already reads the implementation
at the receipt block; retain its upgrade/reconciliation checks, including any
required comparison with preflight evidence. Raw quantities remain decimal
strings. Derive epochs from the mined event, including an epoch-boundary crossing.

### 2. Persistence and recovery

Use the existing shared resolver for pending tickets and the existing execution
repair path for failed/unparsed execution. A pending operator settlement is not
a failed trade and must not be sent through `repair` just to advance it.

The claim path currently appends its signed transaction and broadcasts directly.
Call the installed `sync_state_before_broadcast()` after appending it, as normal
execution already does, then use normal state saves after accounting/errors.
The resolver already recovers saved hashes/receipts and falls back to the
manager's completed-redemption lookup when generic `Withdraw` history finds
nothing. Its recovered `direct_payout` branch still calls `analyse_redemption()`;
verify Atoma through that path rather than adding another recovery route.

Keep the shared resolver's existing handling of confirmed reverted claims: drop
the failed claim attempt from its active transaction list and retry only after
checking the still-outstanding ticket. Do not redesign transaction history for
this integration. A successful but unparsed receipt stays unresolved. Indexer
lag, RPC errors and unknown implementations must never become zero payouts or
permission to repeat a request. No new scanner, journal or recovery store is needed.

Check recovery ordering at the actual callers. `StrategyRunner.tick()` currently
revalues and syncs the treasury before resolving pending vault trades, and
`make_test_trade()` also syncs the treasury before its trade flow. A claim can
already have paid on-chain while its saved trade is still pending. Ensure the
existing resolver can recover that receipt before pending-claim valuation or
NAV publication would misread it, or before a valuation refusal would prevent
recovery from ever running. Use the existing hooks and consistent receipt/state
blocks; do not add a recovery loop inside the valuator or reorder unrelated
protocol flows without a demonstrated need.

### 3. Valuation and final trade accounting

Use the existing position throughout this sequence:

| Stage | Existing state representation | Valuation and cash |
| --- | --- | --- |
| Deposit confirmed | Successful buy, normal vault position | Actual minted shares; reserve reduced by actual deposit |
| Withdrawal requested | One pending sell with serialised ticket | Position retains economic quantity; committed shares excluded from wallet expectations and further sells; no reserve credit |
| Epoch settled, claim outstanding | The same pending sell and ticket | Attributed net claim entitlement replaces the value of burned shares within the same position; no reserve credit |
| Claim confirmed | Existing trade success and position closure/remaining quantity | Credit actual net USDC once; remove redeemed quantity; use existing cost basis for realised PnL |

`VaultValuator` currently quotes the whole logical share quantity. Adapt this
existing valuation path to combine remaining wallet/unsettled shares with the
pending ticket's settled entitlement, using block-consistent manager
`fetch_claim_amounts()` evidence. Feed that total through the existing position
valuation and `ValuationUpdate`; retain the existing quantity until trade
completion. Do not also add pending-sell value to portfolio equity: those shares
already contribute through the position. Verify partial exits as well as full
closure so remaining shares and claim proceeds are each counted once.

The inspected `fetch_claim_amounts()` checks ticket structure/version but quotes
the entire `userEpochShares` balance without checking it equals the ticket's
shares. Tighten that existing manager quote path for an attributable outstanding
ticket, at the same block used for its amounts, rather than inventing a getter
or copying protocol checks into executor valuation. Reuse canonical ticket and
implementation checks. A zero on-chain request balance can mean an already-paid
claim and must enter receipt recovery; an outstanding request with a zero net
entitlement is the distinct total-loss case. Do not call `can_finish_redeem()`
as a valuation prerequisite: its liquidity check can refuse a valid entitlement.

The entitlement can be illiquid or subsequently haircut. Illiquidity defers
payment; it does not erase value. A confirmed zero-payout claim consumes shares
and realises a loss. Narrowly update `convert_vault_flow_analysis()` and existing
trade/state success and reporting checks to accept that evidenced redemption,
without relaxing positive-amount validation for swaps/deposits or dividing by
zero. A zero quote or missing event is not successful completion.

`WithdrawalClaimed.assets` is already net of the withdrawal fee. Retain the
manager's fee evidence for reporting without deducting it again. Existing NAV
calculation must see the resulting position value and cash without any
Atoma-specific treasury accounting.

### 4. Assert one redemption request per trade

Scope decision (30 October 2026): the supported operating model will not create
aggregate claims. Use one executor/state writer and one pending redemption per
custody account/vault; wait for completion before another exit. Independent
manual requests, concurrent writers and imported aggregate components are
outside this model. The adapter also refuses a new request in an occupied epoch.
See the [shared vault operating assumption](../../.claude/docs/vault-deposit-redeem.md#one-redemption-request-per-trade).

Use assertions to halt if competing pending exits, multiple Atoma request parts,
a reconciliation marker or an adapter-reported aggregate mismatch appears.
Preserve mined evidence and propagate the assertion out of settlement retry,
before further claims or NAV publication. Inspect custody/state before resuming;
never clear a marker or distribute a payment to make the assertion pass.

Aggregate reconciliation, ticket supersession and multi-trade payout allocation
are outside scope for now, not outstanding Atoma integration requirements. Do
not add scanners, allocation policies or recovery infrastructure for this case.
Ordinary claim recovery, indexer lag and temporary illiquidity continue through
the existing shared lifecycle. Generic `repair` does not resolve aggregates.

### 5. Commands, deployment and documentation

Reuse the existing `close_single_or_all_positions()` flow: resolve claimable
trades first, skip still-pending exits, then construct new sells. Apply this to
`perform-test-trade --close-only` before its blanket unfinished-trade refusal,
while preserving ownership checks and refusals for unrelated unfinished work.
Respect the current import direction: `close_position.py` already imports its
force-settlement helper from `testtrade.py`, so importing the whole close command
back into `testtrade.py` creates a cycle. Reuse the resolver and position methods
directly at the command orchestration boundary; move only a small existing helper
if sharing it is necessary. Preserve the resolve/skip/new-sell order without
introducing another command framework or pulling treasury/UI side effects into
test-trade execution.
`--buy-only` remains immediate; pending exits report an incomplete outcome under
the command's existing policy. The normal startup/cycle resolver completes them.

Use the existing Anvil force-settlement helper; the test harness can hold
settlement to observe pending/restart stages without adding a production switch.
Update command help and the existing manual ERC-4626 tutorial with the Atoma
deposit/request/resume sequence. Reuse ordinary `STATE_FILE` throughout.

Consume the specialised whitelist through existing deployment configuration.
Requests and claims have no receiver parameter: the Safe caller owns the claim;
deposit receivers must also be the Safe. Preserve real Guard refusals for direct
`redeem`/`withdraw`, unauthorised share transfers/approvals and operator/admin
calls. Deploy the matching artefacts under normal EIP-170 limits in the main
fixture and check selected module-version support; no separate size-validation
exercise or new contracts are needed. Existing broad grants are additive and
need the parent plan's authorised governance procedure before claiming that an
old deployment has the narrow policy.

### 6. Documentation and code discovery

Deliver these cross-references with the integration, so an agent starting from
repository instructions, documentation or a code search reaches the existing
async vault mechanisms. Keep
[vault-deposit-redeem.md](../../.claude/docs/vault-deposit-redeem.md) as the
canonical shared lifecycle guide. Extend its existing sections rather than
creating another architecture guide or copying the lifecycle into each adapter.

| Discovery point | Required update |
| --- | --- |
| [CLAUDE.md](../../CLAUDE.md), also exposed through the existing `AGENTS.md` symlink | Strengthen the existing vault reference row: read the shared guide before adding or changing async vault integration, settlement, recovery or accounting. Name Lagoon, Ostium and Atoma, and direct agents to reuse its code/test map. Edit the canonical file once. |
| [docs/README.md](../README.md) | Add a discoverable **Async vault integration** link to the shared guide, next to existing vault references. Link Lagoon treasury settlement separately and explain that it handles investor flows. |
| [Shared lifecycle guide](../../.claude/docs/vault-deposit-redeem.md) | Add Atoma's synchronous deposit / asynchronous redemption mapping and its settled-claim valuation difference. Update the existing code/test tables with concrete symbols and relative file links for routing, tickets, persistence callbacks, settlement retry, valuation, reconciliation and funding gates. Link the Atoma plan as implementation context, keeping delivered behaviour in the guide itself. |
| [State persistence](../state-persistence.md), [Lagoon treasury settlement](../../.claude/docs/lagoon-treasury-settlement.md) and [CLI guide](../../.claude/docs/cli-command.md) | Preserve the existing two-way treasury/lifecycle links. Add only missing useful references, and document CLI request/resume semantics with a link to the shared guide rather than another resolver explanation. No rewrite of unchanged persistence or treasury documentation is required. |
| [Routing](../../tradeexecutor/ethereum/vault/vault_routing.py), [settlement retry](../../tradeexecutor/ethereum/vault/settlement_retry.py) and [execution-model hook](../../tradeexecutor/strategy/execution_model.py) | Add concise module/method docstring references to the shared guide and related symbols. Make the hand-off explicit: manager request/parser → persisted pending ticket → `resolve_pending_vault_settlements()` → `check_and_resolve_vault_settlements()` → trade completion. Document the existing pre-broadcast persistence callback at the claim submission site. |
| [State](../../tradeexecutor/state/state.py), [position](../../tradeexecutor/state/position.py), [valuation](../../tradeexecutor/ethereum/vault/vault_valuation.py) and [position manager](../../tradeexecutor/strategy/pandas_trader/position_manager.py) | At `mark_vault_settlement_pending()`, `get_held_assets()`, `VaultValuator` and `is_async_vault_sell_pair()`, cross-reference the guide's accounting/funding sections and the related existing helpers. Explain why pending economic quantity differs from wallet shares and why pending proceeds cannot fund buys. Keep comments local to these mechanisms rather than scattering protocol-specific instructions. |
| [Lagoon lifecycle tests](../../tests/erc_4626/test_vault_async_lagoon_erc_7540.py), [shared resolver tests](../../tests/ethereum/test_vault_settlement_lifecycle.py) and the new Atoma lifecycle test | Link the worked examples from the guide; have the Atoma test module docstring point back to the guide and Lagoon reference test, identifying the protocol-specific differences and Anvil concessions. |

Use repository-relative Markdown links in docs and the existing Python docstring
cross-reference conventions for symbols, with an explicit repository path to the
shared guide. Ensure searches for `async vault`, `vault_settlement_pending` and
`Atoma` lead to the guide or a linked implementation entry point. Keep permanent
code references aimed at the shared guide, not solely this temporary plan.
Deliver the documentation with the consumer changes, describing verified support
and the aggregate-refusal limitation. No separate provisional documentation pass
is required.

Validate discovery as part of review: follow the paths from `AGENTS.md` and
`docs/README.md` to the guide, from the guide to implementation/tests, and from
the touched core docstrings back to the guide. Check relative links, heading
anchors and referenced symbols. Confirm that an agent can find the existing
resolver, state store, escrow reconciliation and funding gates without reading
this plan. Use ordinary documentation review; no new link-checking framework
or additional test infrastructure is needed.

## Anvil validation with minimal spoofing

Use Arbitrum block **519150162**, where both proxies use implementation
`0x9161be115867c347adfb4c88a80e759920de86dc`, with the reviewed
[ABI provenance](https://github.com/tradingstrategy-ai/web3-ethereum-defi/blob/2939ee63ee44de354b827668f64b5f23ffd52d7f/eth_defi/abi/atoma/README.md).
Block 487039644 is historical reader evidence, not this execution lifecycle.
At the lifecycle block the minimum deposit is **100 USDC** and share decimals
are 12; verify these through getters and use token conversion helpers.

Adapt the existing Lagoon async integration fixture and guarded custody setup,
parameterising the target over Index and RWA. Reuse `AnvilForkPool`, snapshot
restoration, `create_multi_provider_web3()` and robust receipt waits. Fund the
test investor by transferring existing fork USDC from an impersonated holder;
fund gas and impersonate the actual operator only for real settlement calls.
Require `is_anvil()` before any privileged mutation. Asset-manager transactions
must travel through the real Safe/module/Guard, without impersonating the Safe.
Do not overwrite storage/code, mint assets, inject protocol liquidity or fake
events in the lifecycle test.

### Cooldown handling

The inspected Atoma ABI has **no cooldown-to-zero setter**. Its
`setEpochDuration(uint64)` schedules a duration subject to protocol bounds; it
does not establish that zero is legal or unlock existing deposits immediately.
Use Anvil time travel instead: demonstrate the initial deposit lock, advance to
the actual lock end, then read back the advanced epoch and unlocked shares.
Later call the manager's strict `force_settle(ticket)` to advance to settlement
and invoke the real operator. Time travel eliminates wall-clock waiting without
changing protocol rules. No extra timing setter test or storage spoof is needed.

Lagoon treasury Guard cooldowns and window budgets are separate. Retain them;
advance the fork clock when needed. `lagoon-settle --force-valuation` only
bypasses the NAV tolerance, not these limits.

### Lifecycle acceptance

1. Deploy the existing Lagoon Safe/Guard/module setup with Atoma permissions,
   fund and subscribe an investor through the normal treasury flow, then run
   public `perform-test-trade --buy-only` for 100 USDC. Assert actual deposit
   calldata/receipt, custody share balance and immediate successful accounting.
2. Refuse an immediate locked exit without leaving a new failed/planned trade
   or changing reserves. Advance to the actual unlock and run `--close-only`
   once, holding automatic test settlement. Assert actual request shares/epoch,
   serialised ticket, unavailable shares and unchanged cash.
3. Run account checks and valuation while escrowed; persist and reload normal
   state. Repeat close and verify it resumes rather than requests again.
4. Settle through strict `force_settle(ticket)`. Assert the real settlement
   receipt/share burn, then `can_finish_redeem(ticket)` and actual idle cash.
   Do not rescue an illiquid happy-path fixture with funding. Revalue and run
   account checks after settlement, before claim: entitlement remains counted
   once and cannot fund another buy. Reload state again.
5. Resume through the normal resolver/public close. Assert real guarded claim
   calldata, `WithdrawalClaimed` and USDC transfer evidence, net payout, fee,
   consumed shares and closed position. Repeating resolution changes nothing.
6. Reconcile and post terminal NAV through existing Lagoon treasury settlement,
   using forced valuation when needed for small fee differences, then redeem
   the investor. Assert proceeds net of actual fees/rounding, no remaining
   Atoma claims and no open/frozen test position.

Public restart invocations must share one already-running Anvil RPC and a
temporary normal JSON `STATE_FILE`, with `SIMULATE=false`. `SimulateStore.sync()`
does not persist across invocations, so `--simulate` cannot establish this
recovery result. All endpoints and state files remain local test resources.

Reuse this fixture for the strategy-generated route, following the existing
Lagoon/Ostium direct-driver tests. During its pending stage run a strategy cycle
and check the existing AlphaModel funding gate. A no-buy assertion alone does
not prove strategy-generated execution: also route its actual deposit/sell
`TradeExecution` through the shared executor. Reuse setup and lifecycle assertions
instead of creating another fixture or test runner.

### Typer start black-box integration

Add a dummy strategy module under `strategies/test_only/`, following
`docs/strategy-parameters.md`, and an Anvil integration test that invokes the
registered Typer application through `init` and repeated `start --max-cycles 1`
runs. Reuse the existing async CLI test pattern. Keep one local fork RPC and
normal JSON state file across invocations (`SIMULATE=false`); simulation here
means executing on Anvil, so persistence remains real.

The dummy strategy produces an actual deposit, waits for its shares to unlock,
requests redemption once and avoids duplicate trades while pending. The test
harness advances fork time and impersonates the real settlement operator between
command invocations. Fund the test custody with a real fork USDC transfer and
gas; document those spoofs. Do not mock routing, manager calls, execution,
receipts or state persistence. The final `start` invocation must discover and
claim the settled ticket through the normal resolver. Reload the state after
every command and assert receipt/ticket identities, pending valuation and cash,
exactly one request/claim, the fee-net final proceeds and repeat safety. Exercise
both Atoma deployments using the existing fork/snapshot fixtures. Keep operator
spoofing outside the dummy strategy and production command code.

### Focused additional checks

| Check | Reuse and expected evidence |
| --- | --- |
| Claim save/restart | Interrupt after the signed claim is saved, let its real transaction complete without the final accounting save, then reload through the public resume/runner path. Recover before valuation/NAV, credit once and do not mistake the consumed request's zero balance for a loss. Reuse existing request/reverted-claim tests; extend them only if the parser change exposes a gap. |
| Post-settlement haircut and zero payout | Combine them in one fork scenario: settle and value, induce a reduced entitlement and revalue, then induce total loss and claim zero. Reuse real operator `pause()` and owner `resyncTotalAssets()` calls from upstream; assert receipt, realised loss and NAV. |
| Partial exit and illiquid entitlement | One focused valuation test with a labelled stub manager checks remaining-share plus claim arithmetic and an illiquid claim retaining value. Reuse upstream real-liquidity failure evidence and shared resolver deferral tests instead of adding another capital-withdrawal fork scenario. Explain that only valuation arithmetic is stubbed. |
| Aggregate refusal and missing history | Start with a recent prior valuation, then introduce the mismatch. Verify the trade stays pending, no claim is submitted, valuation fails without refreshing its timestamp, and the NAV-posting caller sends no transaction even though the old quote is within its freshness window. Reuse provider-boundary tests for missing indexed history. An external indexer cannot observe local Anvil blocks. |
| Epoch boundary, unsupported implementation and Guard refusals | Reuse adapter/Guard coverage; add executor assertions only where selection, persistence or refusal behaviour changes. Mined epoch wins; unknown implementation never silently succeeds. |

For induced loss, follow the real methods and call order in upstream
[`test_guard_atoma.py`](https://github.com/tradingstrategy-ai/web3-ethereum-defi/blob/2939ee63ee44de354b827668f64b5f23ffd52d7f/tests/guard/test_guard_atoma.py).
Its loss scenario claims while paused, including a zero payout; do not add an
unpause prerequisite. Privileged failure setup is separate from the primary
lifecycle and never expands the asset-manager permissions. Group related checks
into a small number of tests rather than reproducing every upstream adapter test.

## Delivery and completion

1. Land the smallest request reconstruction and attributable-claim quote fixes
   in the existing manager methods; select the coherent dependency/artefact revision.
2. Wire the existing routing, resolver persistence, valuation and trade success
   paths; preserve the adapter's aggregate-refusal checks. Retain old ticket
   loading and ordinary Lagoon/ERC-7540 behaviour.
3. Extend public close/resume and deployment selection, deliver the documentation
   and code cross-references in section 6 with the existing tutorial updates,
   then run the parameterised guarded lifecycle and focused consumer regressions.
4. Record versions, fork block, ABI, actual transaction hashes/amounts and Anvil
   concessions in an ordinary report. Update the two inventory rows when that
   assessment is available, distinguishing fork mechanics from live owner
   admission, liquidity and operator availability.

Run only relevant tests using the repository Poetry environment, unchanged
private test environment, worktree `PYTHONPATH`, required deadlines and pytest
parallelism rules. The unchanged `.local-test.env` was copied from the main
checkout for implementation tests, following the worktree instructions. Earlier
planning reviews did not execute tests; subsequent results are recorded in the
linked validation report.

Completion requires both guarded lifecycles, strategy funding protection,
restart/repeat safety, safe aggregate refusal, net-fee/zero-loss accounting and
consistent account checks and Lagoon NAV. Documentation completion also requires
the discovery paths in section 6 to work in both directions, with current support
status and no duplicated lifecycle specification. Run relevant existing Lagoon, generic
ERC-4626 and affected Hyper AI regressions for shared/dependency changes. Missing
credentials/history or skipped checks remain explicit validation gaps.

## Review record

An earlier version received two completed bounded plan-text reviews through
Claude CLI with `claude-opus-5-5`; the final review reported no blocking findings.
Those tools-disabled reviews did not independently inspect repository code.

The subsequent local `sanity-check-plan` reviews inspected repository docs and
call sites. This revision explicitly maps Atoma to existing Lagoon/shared
primitives, removes proposed prepared-operation and reconciliation infrastructure,
and retains only demonstrated adapter/consumer fixes. It also distinguishes
external-position accounting from the strategy's Lagoon investor treasury.

The subsequent simplification review used Claude CLI with explicit
`claude-opus-5-5`, the full current plan and selected repository documentation/code
excerpts, with tools disabled. The run completed with a validated successful
terminal result. It returned six findings; the primary agent checked the relevant
adapter and executor paths before applying changes:

- Replace multi-trade aggregate allocation with the adapter's existing refusal;
  record the reduced scope here and in the parent plan. Unlike the review's
  assumption, concurrent requests can race, so refusal remains required.
- Preserve existing claim recovery and reverted-attempt handling; the missing
  pre-broadcast callback is the concrete persistence fix.
- Ship the dependency and consumers together, reuse preflight metadata, and
  avoid a feature flag or speculative migration work. Deployment holdings remain
  unknown, so compatibility was not removed on an assumption of no users.
- Combine loss cases and use focused valuation arithmetic coverage instead of
  duplicating the upstream illiquidity fork setup.
- Reuse close/resume code and fixture setup. Keep the terminal investor withdrawal
  as the integration's end-to-end acceptance, and keep actual strategy-generated
  execution; a no-buy cycle alone would not prove it.
- Avoid redundant documentation edits, while retaining the explicitly requested
  two-way discovery checks as ordinary review, not a separate tool or workflow.

These post-review edits were checked locally, not submitted for a second Claude
pass. No implementation or lifecycle tests ran during this review.

The 2026-10-30 direct sanity check verified the remaining caller and valuation
boundaries. It made completed-claim recovery ordering explicit, identified the
missing attribution check in the upstream quote method, replaced the assumption
that valuation age alone blocks an invalid NAV, and removed a proposed CLI
delegation that would create a circular import. These are changes within existing
paths, with assertions folded into the planned focused checks. Documentation
cross-references and the aggregate-refusal scope remain required. The dependency
fixes and missing test environment remain implementation prerequisites; no tests
or external reviewer were run for this sanity check.

Implementation review subsequently completed with Claude Opus 5.5. The full
review and focused follow-up were grounded in the repository code; the latter
reported no blocking or important findings after the corrections. See the
validation report for the tested local dependency revision, final checks and
publication constraints.
