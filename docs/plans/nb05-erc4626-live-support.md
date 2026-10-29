# NB05 ERC-4626 live support plan

## Scope and baselines

Prepared on 2026-10-29. Plan the remaining adapter and executor work for seven
families; this file does not implement, deploy or certify them. GMX liquidity is
excluded. Strategy conversion, Lighter and Hypercore integration are outside this
plan.

The recorded NB05 inventory contains 347 identities. The requested families
account for **243** of them:

| Family | Recorded vaults | Existing execution route | Main remaining work |
| --- | ---: | --- | --- |
| Generic ERC-4626: Morpho and Yearn | 205 | Synchronous, with specialised readers/managers where selected | Complete guarded exits, capacity, residual shares and deployment evidence |
| Upshift | 27 | 10 standard; 17 multi-asset with synchronous deposits and queued exits | Directional classification, accepted-asset funding and dated claims |
| Accountable | 3 | Synchronous deposits; operator-settled request and claim | Correct deposit timing, admission and restart evidence |
| Ember | 3 | Synchronous deposits; queued operator direct payout | Early exit classification and payout recovery evidence |
| Plutus | 2 | Hedge: synchronous deposit and queued exit; Dolomite: synchronous | Distinguish deployments and preserve the correct claim selector |
| Atoma | 2 | Specialised manager merged upstream, beyond executor's dependency pin | Consume manager, recover requests, value receivables and accept evidenced zero payouts |
| Symbiotic | 1 | Reader and inherited generic manager; withdrawal lifecycle uncertified | Resolve deployed version, implement deallocation/queue route and partial claims |
| **Total in scope** | **243** | | |

The generic group comprises 180 Morpho, 21 Yearn and four generic-labelled Yearn
Compounders. Its implementation split is 46 Morpho V1, 134 Morpho V2, 15 Yearn
Compounders and 10 Yearn V3. These are adapter selections, not independent
certificates of live support. The other 104 inventory rows stay in the full
assessment with their existing status; they are not silently removed or certified
by this work.

The 10/17 Upshift split comes from recorded features, including
`upshift_multi_asset_like` on Tori Ecosystem Vault. Tori is provisionally in the
17; its missing manager-capability and asset evidence must still be resolved.
The expected 20 delayed exits below are regression expectations derived from
those features and Ember, not a second maintained admission list.

Use these immutable source baselines rather than the older notebook dependency:

- [trade-executor master `eb439746`](https://github.com/tradingstrategy-ai/trade-executor/tree/eb43974659488d70de8736ea4f8e2bf95b7a54a8), whose eth-defi pin is `d19a5a6b9fcd29914b8749fcb438cf068002850d`.
- [eth-defi master `9bd1452d`](https://github.com/tradingstrategy-ai/web3-ethereum-defi/tree/9bd1452daf734826374f048048738e5f85541781), including the merged [Atoma adapter `752decd1`](https://github.com/tradingstrategy-ai/web3-ethereum-defi/commit/752decd1ed56a1fff224ffa35492d494a02bf942).
- [The recorded 347-vault assessment](https://github.com/tradingstrategy-ai/trade-executor/blob/9679aa2930f054e40e4b3729cf5fba8e824034d9/docs/reports/nb05-live-adapter-assessment.md) and [per-identity inventory](https://github.com/tradingstrategy-ai/trade-executor/blob/9679aa2930f054e40e4b3729cf5fba8e824034d9/docs/reports/nb05-live-adapter-coverage.json) are published separately in PR #1693. Their recorded metadata and historical results remain snapshots, not fresh on-chain probes.

In the tables, `eth-defi` means `tradingstrategy-ai/web3-ethereum-defi` and
`trade-executor` means `tradingstrategy-ai/trade-executor`.

## Existing code to extend

Paths below are relative to their named repository at the baseline above. Test
files are existing starting points, not tests run as part of writing this plan.

| Repo | Code inspected | Existing coverage to extend |
| --- | --- | --- |
| eth-defi | `eth_defi/vault/deposit_redeem.py`, `eth_defi/erc_4626/deposit_redeem.py`, `eth_defi/erc_4626/flow.py` | `tests/erc_4626/vault_protocol/test_morpho_v2_redemption.py`, `test_yearn_deposit_redeem.py`, `test_yearn_compounder.py`, `test_yearn_morpho_compounder.py` |
| eth-defi | `eth_defi/erc_4626/vault_protocol/{upshift,accountable,ember,plutus,atoma}/deposit_redeem.py`; `symbiotic/vault.py` in the same parent directory | `tests/erc_4626/vault_protocol/test_upshift.py`, `test_accountable_settlement.py`, `test_ember_settlement.py`, `test_ember_operator_liquidity.py`, `test_plutus.py`, `test_atoma_deposit_redeem.py`, `test_atoma_integration.py`, `test_symbiotic.py` |
| eth-defi | Family-specific whitelist/deployment machinery and matching Guard artefacts | `tests/guard/test_guard_simple_vault_{accountable,ember,plutus}.py`, `test_guard_atoma.py`, `test_guard_atoma_policy.py` |
| trade-executor | `tradeexecutor/ethereum/vault/{vault_routing,vault_live_pricing,settlement_retry}.py`; `tradeexecutor/state/{identifier,trade,state,portfolio}.py` | `tests/ethereum/test_vault_settlement_lifecycle.py`, `tests/erc_4626/test_vault_trading_e2e.py` |
| trade-executor | `tradeexecutor/strategy/alpha_model.py`, `tradeexecutor/strategy/pandas_trader/yield_manager.py`, `tradeexecutor/cli/commands/{perform_test_trade,lagoon_deploy_vault}.py`, `tradeexecutor/ethereum/lagoon/universe_config.py` | `tests/cli/test_vault_test_trade.py`, existing vault planning and deployment fixtures |

## Shared prerequisites

Existing live routing already asks each deposit manager whether deposits and
redemptions are synchronous. Ticket serialisation, pending settlement state,
escrow-aware balance reconciliation and operator direct-payout recovery also
exist. Extend these paths rather than introducing a parallel execution system.

| Repo | Change description | Before | After |
| --- | --- | --- | --- |
| trade-executor | Publish directional lifecycle information before allocation | Early pair flags differ from manager behaviour; routing sets `vault_async_flow` only after trade planning | Universe construction resolves deposit and redemption lifecycles from the selected adapter/deployment; `decide_trades()`, AlphaModel, YieldManager, CCTP planning and backtesting consume the same information |
| trade-executor | Verify scoped early classifications using the shared resolver | Ember and multi-asset Upshift exits are missing from delayed-redemption classification; ERC-7540 labels can delay synchronous Accountable/Hedge deposits; Dolomite can be treated as delayed | Assert the resolver returns the expected 20 delayed exits from Ember and multi-asset Upshift; keep standard Upshift and Dolomite synchronous, and Accountable/Hedge deposits synchronous; retain regression coverage for existing families |
| eth-defi | Separate preparation from mined-request parsing | Executor can reconstruct a request by calling the admission-checking constructor after execution | Define a manager-owned receipt parser/reconstruction API accepting serialisable preparation evidence; retain existing ticket formats and avoid fresh admission checks during receipt parsing |
| trade-executor | Persist preparation alongside existing transaction records | `TradeExecution.blockchain_transactions` already holds signed transactions/hashes, but preparation evidence is absent and public `perform-test-trade` calls omit the persistence callback | Add serialisable preparation fields to the existing operation record; checkpoint before submission and after receipt/ticket parsing; wire the callback through every public path and reuse existing `repair`/`repair_trades` recovery rather than creating another transaction journal |
| trade-executor | Recover uncertain and legacy operations | A crash can leave only broadcast evidence, or an older ticket without the new preparation fields | Reconcile existing hashes and receipts before any retry; use a versioned state representation and manager-specific recovery for legacy tickets; an ambiguous operation remains unresolved and cannot create a replacement request |
| trade-executor | Make receipt application idempotent | Recovery spans ticket state, trade state, position balances and cash | Apply each confirmed economic effect once, keyed by chain, transaction hash, log index and protocol request/component identity; one operator transaction can pay multiple requests; commit accounting and ticket transitions together, including Atoma aggregate supersession and Symbiotic partial claims |
| trade-executor | Separate cash, escrow and claims in allocation/NAV | Early planning can assume delayed sell proceeds fund buys; pending redemption valuation is primarily share-based | Pending proceeds never fund new trades; value either remaining shares or their replacement claim receivable exactly once, and credit reserve cash only on actual payout; reconcile each bucket in Lagoon NAV |
| eth-defi; trade-executor | Separate implementation, evidence and availability | Protocol labels, capability snapshots and successful deposits can imply too much | Record directional implementation support separately from exact-deployment lifecycle evidence and current owner-specific admission/liquidity; allow exits needed for existing holdings while blocking new allocations without the required evidence |
| trade-executor | Make assessment reproducible after changes | Current assessment gap labels include static, feature-based findings | Refresh exact adapter selection, dependency/proxy versions and lifecycle evidence for all 243 scoped identities; keep all 347 records and derive summary counts from records; unknown metadata remains an evidence gap |

Preparation evidence includes chain/vault/owner, selected implementation, direction,
raw assets or shares, funding token, call roles/indices and transaction identity.
Use existing transaction role markers and request indices. Do not serialise live
Python request objects or copy an ABI into strategy state. Preserve existing
owner-specific limit checks, and revalidate implementation and eligibility before
submission; a mined transaction must be parsed using its historical context.

For backtests, reuse existing asynchronous timing hooks with explicit, configurable
assumptions for Upshift epochs, Ember FIFO/operator delays and Symbiotic queue
fills, including partial fills. These are simulation assumptions, not live payment
schedules. The synchronous directions must remain immediate.

## 1. Generic ERC-4626: Yearn and Morpho

The [generic ERC-4626 test-trade subplan](nb05-generic-erc4626-test-trade-subplan.md)
details the first implementation milestone through public `perform-test-trade`.

Existing support: `ERC4626DepositManager` already builds deposit/redeem calls,
checks limits and analyses flows. Wrapper-aware readers and Morpho V2 exact-call
redemption simulation already exist. Factory precedence matters: some Morpho
records carry Yearn features, so matching a protocol label must not replace the
actual selected class. Yearn capacity is owner-specific; a global zero capacity
does not establish that the selected Safe cannot exit.

| Repo | Change description | Before | After |
| --- | --- | --- | --- |
| eth-defi | Establish complete exit evidence per implementation | Morpho/Compounder classes lack complete certification; V2 already has typed exact-call failure handling, including `TransferReverted()` | Pin representative deployments and blocks, preserve existing decoded failure evidence, and prove deposit through final payout before extending certification; a liquidity failure remains a temporary unavailability or unresolved implementation gap |
| eth-defi | Preserve capacity checks and expose executed-share evidence | V1 capacity/rounding can constrain a full exit; V2 already simulates exact calls because getter results can be insufficient | Keep manager-specific checks; report actual shares redeemed and remaining capacity/residuals without pretending an illiquid vault can be forced to pay |
| trade-executor | Account for partial synchronous exits and residual shares | Planned full closes can exceed executable capacity or leave shares after epsilon rounding | Only spend actual receipts and remove actual redeemed shares; retain the residual position, block duplicate commitments and plan a later separately evidenced redemption; any dust write-off requires an explicit existing policy |
| trade-executor | Admit the exact reader/manager combination | Feature overlap and sparse capability metadata can misclassify standard wrappers | Resolve all 205 recorded identities to the actual class, owner-specific preflight and evidence status; preserve fee-aware pricing and wrapper payout analysis |
| eth-defi; trade-executor | Validate existing Guard permissions and CLI behaviour | Standard ERC-4626 policies and executor routes exist, but do not prove every selected deployment | Reuse bounded ERC-4626 approvals/selectors; verify owner and receiver are the Safe, unsupported calls fail, and `perform-test-trade`/strategy-generated trades complete both directions |

Extend the existing Morpho V2 redemption, Yearn/Compounder, guarded ERC-4626 and
executor vault lifecycle tests. Include a capacity-limited failure and a residual
exit; do not mark all 205 deployments supported from one family-level test.

## 2. Upshift

Existing support: standard Upshift uses synchronous ERC-4626. The multi-asset
manager already discovers accepted assets, estimates deposits per asset, builds
the chosen asset's approval/deposit, and implements request → operator settlement
→ dated claim. Its optional instant exit is a separate path. Executor pricing
already selects `estimate_deposit_for_asset`, but assumes the reserve amount is
denominated in that asset; routing also adds a denomination-token approval.

| Repo | Change description | Before | After |
| --- | --- | --- | --- |
| trade-executor | Distinguish the two variants throughout planning | A shared protocol feature obscures 10 standard and 17 queued-exit deployments | Standard variants stay synchronous both ways; multi-asset variants deposit synchronously and reserve exit proceeds until the dated claim actually pays |
| eth-defi; trade-executor | Make funding token and approval ownership explicit | Manager-provided chosen-token approval coexists with executor denomination-token approval; per-run override does not provide conversion accounting | Define one approval owner per request; use the selected token's decimals and raw units consistently across quote, approval and deposit; avoid duplicate/unrelated approvals |
| trade-executor | Fund non-USDC inputs using supported swap routes | Three known USDC-accounted deployments do not accept USDC: Clear RWA, Sentora USD Earn and Liquity V2 Airdrop Farming | Add per-deployment accepted-token and validated swap-route configuration using existing supported DEX routing; represent conversions as separate `TradeExecution` instances, price in reserve terms, bound slippage and enable required deployment permissions; retain an explicit gap until a route is validated |
| eth-defi; trade-executor | Resolve missing accepted-asset evidence | Tori Ecosystem Vault `1:0xcd69123b3fbbfc666e1f6a501da27b564c00de54` lacks recorded capability/asset data | Query the actual deployment at a pinned block, select its manager and accepted assets, and record funding/exit evidence without assuming it accepts USDC |
| eth-defi | Make dated request identity restart-safe | Request parsing verifies an event but reads the current withdrawal epoch | Recover the request's historical epoch from receipt-time evidence/block-pinned reads; serialise and restore the same dated ticket across epoch changes |
| trade-executor | Track and value intermediate tokens and order dependent trades | Reserve denomination and selected deposit/payout assets can differ | Use existing spot positions for intermediate tokens with a validated executable price source; settle the entry swap before sizing the deposit from actual received tokens, and settle the exit payout before sizing its conversion; residual/interrupted conversion holdings remain explicit positions in reconciliation/NAV, never assumed USDC |
| eth-defi; trade-executor | Validate settlement and optional instant exits | Async manager/claim code exists; some upstream comments still describe absent redemption support | Update stale documentation; test guarded request and claim, unavailable epochs and restart; enable instant redemption only with explicit capability/preflight evidence, never as an implicit fallback |

Acceptance: cover both variants, one accepted-USDC path, each distinct conversion
route and receipt recovery across an epoch change. Operator settlement remains
external in production. Label any mock settlement; it does not prove live payment.
If no supported priced spot pair can represent a required intermediate asset,
record that deployment as an implementation gap until the representation and
funding route exist; do not enable it using a denomination override alone.

## 3. Accountable

Existing support: `AccountableDepositManager` implements synchronous deposits,
ERC-7540 requests/tickets/claims and admission checks, including minimum deposit,
permission level, KYC and whitelist restrictions. It has mock settlement support,
but does not advertise a real Anvil operator-settlement driver. All three NB05
deployments are on Monad.

Before enabling them, verify configured Monad RPC and fork support, Safe/Lagoon/
Guard artefact availability and a supported reserve funding/return route. Missing
chain infrastructure is an explicit deployment gap rather than a manager failure.

| Repo | Change description | Before | After |
| --- | --- | --- | --- |
| trade-executor | Preserve synchronous deposit semantics | The shared ERC-7540 classification can introduce an artificial deposit queue | A deposit receives and accounts for actual shares immediately; only withdrawals wait for operator settlement |
| eth-defi; trade-executor | Validate the actual Safe's admission | Permissionless historical snapshots do not prove current owner eligibility | Check minimums, permission level, owner eligibility and capacity for each selected deployment; classify KYC/whitelist refusal as unavailable without bypassing admission |
| trade-executor | Exercise the existing ticket lifecycle through strategy trades | Manager serialisation exists; complete executor/deployment evidence is incomplete | Route actual `decide_trades()` output through request, persist/restart, operator fulfilment and claim; no pending proceeds fund a rebalance |
| eth-defi; trade-executor | Validate narrow deployment permissions | Accountable Guard machinery and settlement tests exist | Consume existing policies for depositor operations, test the Safe's receiver/owner and rejection cases, and never grant operator `fulfillRedeemRequest` authority to the asset manager |

Acceptance: test one allowed owner and one rejected owner, immediate deposit
accounting and delayed exit recovery. Use existing settlement/Guard fixtures for
mechanics; require separate genuine operator-settlement evidence per deployment.

## 4. Ember

Existing support: `EmberDepositManager` implements synchronous deposits,
share approval plus withdrawal request, FIFO tracking and operator-paid proceeds.
There is **no depositor claim transaction**. Manager transaction-hash recovery
and executor direct-payout recovery already exist. The fork settlement driver can
inject liquidity, which must be distinguished from strict settlement.

| Repo | Change description | Before | After |
| --- | --- | --- | --- |
| trade-executor | Classify queued exits before allocation | Live routing knows the manager is asynchronous, but early allocation flags do not | All three Ember identities reserve requested shares and exclude future payouts from spendable cash from the first planning step |
| trade-executor | Preserve and harden direct-payout recovery | A generic completed-request path can look for `Withdraw` before the manager's protocol-specific payout evidence | Prefer manager-owned recovery for this lifecycle; consume `RequestProcessed`/payout evidence, handle indexer lag as unresolved, and never construct a depositor claim |
| eth-defi; trade-executor | Verify actual payments and repeated reconciliation | Ticket and direct-payout helpers exist; a terminal request status alone does not establish credited cash | Match request identity, recipient, actual net payout and receipt; repeated resolver runs credit cash once, including a crash after payment and before state persistence |
| eth-defi; trade-executor | Separate strict and injected-liquidity evidence | Existing Anvil settlement can provision missing assets | Run strict liquidity checks first; record injected amounts and simulation labels separately; unavailable live liquidity leaves the request pending and does not certify redemption availability |
| eth-defi; trade-executor | Validate guarded requests and owner limits | Existing manager minimums and Ember Guard tests cover parts of the route | Reuse these controls and test deposit/request through the selected Safe; the asset manager receives no operator-processing permissions |

Acceptance: FIFO delay, below-minimum refusal, operator payout during executor
downtime, duplicate recovery and no fabricated claim. Extend the existing executor
operator direct-payout recovery test rather than replacing its mechanism.

## 5. Plutus

Existing support: the Hedge deployment selects `PlutusAsyncDepositManager`:
synchronous deposit → `requestRedeem(shares, owner, owner)` → operator fulfilment
→ `redeem(requestId, receiver)`. The latter is a two-argument claim, distinct from
three-argument ERC-4626 redemption. Dolomite uses the standard synchronous route.

| Repo | Change description | Before | After |
| --- | --- | --- | --- |
| trade-executor | Resolve capabilities by actual deployment | Broad Plutus/async flags can delay Dolomite exits or Hedge deposits | Record and use Hedge's synchronous-deposit/async-redemption lifecycle and Dolomite's synchronous lifecycle independently |
| trade-executor | Reuse Hedge request and recovery APIs | Specialised manager, ticket serialisation and claim builder already exist | Exercise them through routed strategy trades and restarted `perform-test-trade`; retain the request ID and wait for fulfilment without duplicate requests |
| eth-defi; trade-executor | Validate overloaded selectors and Safe custody | Existing Plutus Guard policy supports the special route, but generic routing must select it consistently | Allow only the required depositor calls with the Safe as owner/receiver; demonstrate the two-argument claim succeeds and inappropriate three-argument redemption is not substituted |
| eth-defi; trade-executor | Record deployment-specific operator evidence | Anvil operator discovery/settlement helpers and mock fixtures exist | Test genuine available operator settlement separately from mock fixtures, and keep a pending request unresolved when fulfilment is unavailable |

Acceptance: complete one lifecycle for each of the two variants; Hedge deposit
must not create an artificial pending deposit. Preserve existing guarded Plutus
tests and add restart/strategy-level coverage.

## 6. Atoma

Existing support changed upstream: Atoma's specialised adapter is now merged.
It provides synchronous deposits, `requestWithdrawal(shares)`, epoch tickets,
`claimWithdrawal(epoch)`, aggregate-ticket reconciliation, zero-payout/fee-aware
analysis, implementation checks and narrow Guard permissions. Updating the
executor dependency alone does not fix its consumer incompatibilities.

| Repo | Change description | Before | After |
| --- | --- | --- | --- |
| trade-executor | Consume the merged adapter with its consumer fixes | Pinned `d19a5a6b` predates the specialised Atoma manager/Guard policy; factory selection changes automatically with the dependency | Update to a reviewed revision containing `752decd1`, regenerate the lockfile and use matching artefacts in the same release as receipt parsing and safe accounting; otherwise explicitly disable new Atoma requests/allocations until compatible recovery and accounting land |
| eth-defi; trade-executor | Parse a mined request without repeating admission | Routing calls `create_redemption_request()` after broadcast; escrowed balance or the same-epoch duplicate check can now reject | Retain preparation evidence and use receipt parsing/reconstruction without rerunning request admission; keep admission checks mandatory for genuinely new submissions |
| trade-executor | Accept evidenced zero-proceeds redemptions end to end | Flow conversion and subsequent success/accounting assertions require positive reserve/price | Represent a confirmed total-loss redemption with zero realised proceeds, correct burned shares, realised loss and position closure; update PnL/reporting paths; preserve positive-amount checks for unrelated swaps and invalid/unconfirmed vault flows |
| trade-executor | Value escrow, settled receivables and cash separately | Pending exits generally retain share-based valuation; Atoma burns shares at settlement before paying | Replace settled burned shares with the protocol-valued claim receivable, include subsequent haircut and fee evidence, then replace the receivable with actual cash at claim; no double counting or fee deduction twice |
| trade-executor | Integrate upstream aggregate reconciliation atomically | Components can resolve to one aggregate epoch claim | Reconcile and acknowledge the upstream aggregate ticket, supersede its component requests in one state commit, and allocate the single actual payout/loss to components without duplicate credits or stranded pending trades |
| trade-executor | Recover claims without assuming generic events or liquidity | Generic `Withdraw` scanning does not match Atoma; settled epochs may lack idle cash | Use the manager's completed-claim evidence first; tolerate indexer lag, preserve temporarily unpayable receivables, and reconcile already-completed claims before any new broadcast; implementation changes require revalidation rather than blind retries |
| eth-defi; trade-executor | Validate narrow permissions for new and existing Guards | Upstream deployment helpers add narrow Atoma permissions; older Guards may retain broad permissions | Consume the specialised whitelist automatically on new deployments; inventory and explicitly revoke/migrate prior broad grants on existing deployments through authorised governance; test forbidden calls and deployed artefact compatibility |

Acceptance: cover both Arbitrum identities, same-epoch aggregation, a normal
fee-net claim, zero payout, a haircut after settlement, insufficient idle cash,
indexer lag, completed-claim recovery and proxy implementation change. Use a
post-upgrade fork block (the [upstream integration fixture](https://github.com/tradingstrategy-ai/web3-ethereum-defi/blob/9bd1452daf734826374f048048738e5f85541781/tests/erc_4626/vault_protocol/test_atoma_integration.py) uses 519150162);
older reader-era blocks do not test the new lifecycle. Reuse upstream strict
settlement tests; injected assets or mock settlement are separate evidence.

## 7. Symbiotic

Existing support: `SymbioticVault` reads the selected KPK USDC LiquidLane vault
(`1:0x8bcd746976885b5832bad07b4921e3f2dd1d3703`) and inherits generic execution.
Its withdrawal/deallocation lifecycle is explicitly uncertified.

Primary interface research at [Symbiotic core `25a065a3`](https://github.com/symbioticfi/core/tree/25a065a3dd71150a588162bd2ad4833bbd4e2423)
shows [Vault V2](https://github.com/symbioticfi/core/blob/25a065a3dd71150a588162bd2ad4833bbd4e2423/src/interfaces/vault/IVaultV2.sol)
with deallocation-aware `withdrawable()`/`redeemable()` simulation and a linked
[withdrawal queue](https://github.com/symbioticfi/core/blob/25a065a3dd71150a588162bd2ad4833bbd4e2423/src/contracts/vault/WithdrawalQueue.sol).
The queue takes shares with `requestRedeem(shares, receiver)`, mints an ERC-721
request ID, can be filled incrementally and pays `claim(tokenId, receiver)` in
parts. It requires the caller to own the NFT; a fully claimed NFT is not burned
by this implementation. **This source is a candidate implementation, not proof
that the selected deployment uses that version.**

| Repo | Change description | Before | After |
| --- | --- | --- | --- |
| eth-defi | Resolve actual vault, queue and proxy implementations first | Reader selection does not establish the deployed withdrawal mechanism | Pin the selected deployment's bytecode/proxy version, source/ABI and queue relationship at an actual fork block; implement its verified lifecycle; reject unknown versions instead of assuming latest core matches |
| eth-defi | Add a specialised direction-aware manager | Inherited ERC-4626 calls do not certify deallocation or queue behaviour | Keep validated synchronous deposits; simulate exact immediate-exit capacity with `eth_call` where supported, and use the verified queue route for illiquid/pending-queue exits; never broadcast non-view getter-like simulations merely to read capacity |
| eth-defi | Implement persistent queue ownership and receipt analysis | No certified ticket/status/claim/recovery path for the request NFT | Approve only the verified queue, parse the minted request ID, persist vault/queue/owner/version/request evidence, poll claimable shares/assets, claim to the Safe and recover all prior claim events; determine completion from request accounting rather than NFT disappearance |
| trade-executor | Support incremental settlement on one pending request | Current vault routing assumes a full redemption; a claim is generally treated as terminal | Persist per-claim executed shares, actual assets, cumulative totals and receipt identity; credit each partial payout once, keep the remainder pending and unavailable for a second request, and complete the original `TradeExecution` only after its full obligation resolves; final success records cumulative execution without crediting prior payouts again |
| trade-executor | Value remaining obligations and available cash correctly | Escrow-share accounting alone cannot describe a partially paid queue request | Remove paid shares/receivables as cash arrives, value only the unresolved remainder, and reconcile wallet shares, queue escrow/NFT rights and cash without double counting; handle zero-asset rounding without falsely completing an unpaid remainder |
| eth-defi; trade-executor | Add a narrow queue Guard policy and deployment discovery | Generic vault permissions do not authorise a separate request/claim target | Validate `queue.vault()` against the approved vault; bind share approvals, NFT request ownership and payout receiver to the Safe; reject NFT `approve`, `setApprovalForAll`, `transferFrom`, `safeTransferFrom` and arbitrary targets; verify the Safe fallback handler supports receiving NFTs if the deployed queue uses safe minting; integrate verified queue collection in `lagoon-deploy-vault` |
| eth-defi | Verify queue filling and deallocation authority | Latest core `fill()` is public, while actual deployed/operator mechanics remain unverified | Establish who fills the deployed queue and what underlying liquidity permits; use external filling initially and grant no privileged deallocation/strategy/operator calls to the asset manager; add depositor-triggered filling only if independently required and reviewed |
| eth-defi; trade-executor | Prove full exit and Guard size compatibility | No complete guarded lifecycle evidence; baseline deployed bytecode is 22,173 bytes for GuardV0 and 24,259 for TradingStrategyModuleV0 (317 bytes below EIP-170 for the latter) | Measure generated artefacts when adding selectors; refactor existing validation or split a reviewed policy component if required; pass guarded deposit, queue, partial claim and final payout before enabling new allocations |

Acceptance: pin the deployed lifecycle before implementation. Test an unavailable
instant exit, queue request, two partial payouts with restart between them,
duplicate receipt replay, wrong NFT owner/receiver and eventual full exit. The
unfilled remainder is never cash. Fork impersonation can test privileged
deallocation mechanics but cannot establish live operator willingness or timing.

## Required coverage across executor surfaces

| Surface | Required result |
| --- | --- |
| `perform-test-trade` | Exercise each distinct manager and direction; pass state persistence through public calls; support separate invocations for deposit/request/claim or direct payout; report pending, unavailable, unsupported and completed outcomes distinctly; resume the original operation after a crash |
| `decide_trades()` and `TradeExecution` | Drive actual returned trades through routing and settlement; preserve immediate deposits, exclude delayed proceeds from funding, block duplicate share commitments, use actual fee-net receipts, support residual positions/zero loss payouts/partial claims, and keep equity continuous apart from evidenced PnL and fees |
| `lagoon-deploy-vault` | Resolve exact selected managers and approved funding/payout assets; reuse existing family-specific policies, consume merged Atoma permissions, add verified Symbiotic queue policy, and whitelist only validated conversion routes; verify owner/receiver restrictions, compatible artefacts and bytecode size |
| Restart and accounting | Interrupt after signing, submission, confirmation, request parsing, operator settlement and payout; recover without replacements or repeated credits; preserve existing ticket/state compatibility and test Lagoon NAV before/after each economic event |

Do not expose protocol-operator settlement methods to the asset manager to make a
test pass. Existing deployment governance is required to alter old Guard grants;
a successful new deployment does not establish that existing permissions narrowed.

## Release compatibility and operational recovery

- **Dependency compatibility:** review the entire eth-defi change range from
  `d19a5a6b` to the chosen target, including APIs, ticket formats, ABI/bytecode and
  client/dependency changes. Run focused regressions for existing Lagoon/ERC-7540
  managers and other active strategy routes, including Hyper AI; adding seven
  families must not strand unrelated pending operations. Keep existing tickets
  readable and test fixture states from both sides of the upgrade.
- **Reference owners:** use a deployed fork Safe/Guard as the reference depositor
  for mechanics tests and record its address. No production NB05 Safe has been
  established by this plan. Before admission, repeat owner-specific permissions,
  balances, limits and funding checks for the actual production Safe on each chain.
- **State inventory and migration:** inventory pending tickets and broadcast trades
  in every deployment receiving the release. Prefer backward-compatible optional
  fields and lazy reconstruction; add a versioned migration only where required.
  Its preview must reconcile receipts and list exact proposed changes, ambiguous
  records and recovery evidence. Run production maintenance under the existing
  [Docker Compose runbook](../docker.md): stop the writer, verify the authoritative
  state/mounts and deployed image, back up before writes, apply the reviewed
  migration, validate balances/tickets/NAV, then restart and check state endpoints.
- **Ambiguous operations:** extend the existing `repair` command's read-only
  diagnostic output to show the request, signed transaction/hash, nonce and receipt
  evidence, and missing indexer/implementation data. Recovery attaches verified
  historical evidence to the existing trade; it never releases commitments or
  rebroadcasts merely because an indexer has not returned a result. Any governed
  manual correction needs an audited receipt/account reconciliation.
- **Rollback:** document which executor releases can read the new state. After a
  new partial claim or other on-chain effect, use a compatible reader/recovery
  release; restoring a pre-effect backup is not a safe rollback. Preserve backups
  and migrate forward with reconciled chain evidence when older readers cannot
  represent the current operation.
- **Existing Guard/module upgrades:** Symbiotic validation requires new code,
  not just new whitelist entries. Inventory deployed module versions, required
  conversion selectors and permission grants. Also snapshot settlement-policy
  version, configured limits, active window/usage and legacy cooldown deadlines.
  A replacement must not silently reset the consumed budget or cooldown, and a
  v0.5-to-v0.6 change requires an explicit governance-approved policy/limit mapping.
  Transfer policy state only through a supported, tested mechanism. Stop all
  settlement writers at the start of maintenance, including keepers and manual
  settlement procedures. If policy-state transfer is unavailable, disable the old
  module through Safe governance at the start of the waiting period and keep the
  replacement disabled until the old window/cooldown constraints expire. This
  fallback deliberately incurs downtime; the old module cannot settle during it.
  Immediately before activation, refresh the source policy snapshot from chain
  and simulate the batch against it; any unexpected intervening settlement forces
  a new assessment/wait rather than use of the original snapshot.
  For a supported state-transfer migration, use one atomic Safe batch that
  disables the old module before enabling the configured replacement. For the
  fallback, verify the old module remains disabled before enabling the replacement.
  Remove obsolete broad permissions/allowances as part of activation. There must
  be no interval with both modules active, and all writers remain stopped until
  activation and validation finish.
  Validate policy continuity, pending tickets, owner/receiver restrictions and
  budget enforcement before restarting. Test a partly consumed window and an
  active legacy cooldown, including a v0.5-to-v0.6 transition and an unexpected
  settlement after the initial snapshot; replacement must neither reset nor widen
  effective limits. Until this migration is executed,
  existing deployments cannot use routes requiring new code; new deployments use
  the matched artefacts.
- **Loss posting and monitoring:** test an evidenced haircut/total loss through
  Lagoon NAV posting and settlement, including empty queues and pending investor
  flows. Respect the actual module version: v0.6 enforces gross-flow window budgets;
  v0.5 has legacy caps/cooldowns. A NAV loss is not itself gross investor flow.
  Preserve safety limits, keep deferred queues explicit and use the existing
  governed recovery runbook when required. Alert on aged pending requests,
  ambiguous transactions, unpayable receivables, indexer lag and residual tokens
  awaiting conversion, with evidence and a concrete recovery action.

## Ordered delivery and evidence

1. **Freeze and reconcile inputs.** Start from the executor master above; record
   exact eth-defi/client pins, selected managers and proxy versions for all 243
   scoped identities. Verify Symbiotic's deployed lifecycle first. Preserve the
   347-row inventory and historical evidence with their original provenance.
2. **Land upstream contracts and dependency support.** Add prepared-request parsing
   APIs, historical Upshift request identity and the verified Symbiotic manager /
   queue policy. Reuse existing managers and merged Atoma work. Prepare the
   dependency/artefact upgrade and complete its full-range compatibility review;
   release it atomically with step 3's required consumers, or keep newly selected
   incompatible routes explicitly disabled pending those consumers.
3. **Fix common executor state and planning.** Unify directional capabilities,
   durable request checkpoints and idempotent recovery; implement receivable valuation,
   partial settlement, residual shares and narrowly scoped zero-payout accounting.
   Prove migration/restart behaviour before exposing new live paths; ship Atoma's
   pin and compatible request parsing/accounting together, including recovery for
   already-pending tickets, before enabling new Atoma requests.
4. **Complete funding and deployment configuration.** Wire selected assets and
   validated Upshift conversions, eliminate duplicate approvals, consume narrow
   family permissions and add Symbiotic queue discovery. Check contract size and
   existing-Guard/module migration requirements and their governance procedure.
5. **Integrate and verify one lifecycle per distinct implementation.** Start with
   generic ERC-4626, standard Upshift and Plutus Dolomite, then Accountable, Ember,
   Hedge and multi-asset Upshift, then Atoma and Symbiotic. Exercise the three
   executor surfaces above with actual strategy-produced trades. Deliver the
   [manual Arbitrum Lagoon tutorial](nb05-generic-erc4626-test-trade-subplan.md#manual-integration-tutorial-arbitrum-lagoon-vault)
   in the format of the merged Lighter subaccount shell tutorial: deploy Lagoon
   on Arbitrum, subscribe, open and close positions using `perform-test-trade`,
   reconcile and redeem the investor. Start with the generic subplan's matrix;
   extend it for every newly supported protocol here, keeping Hedge/Dolomite and
   standard/multi-asset Upshift separate. For delayed exits, persist the request,
   wait for actual operator settlement and resume claim/payment reconciliation
   across invocations; request submission alone never completes a close. Keep
   GMX liquidity excluded. An unavailable route remains a visible incomplete
   matrix row, including when its chain requires satellite funding from Arbitrum.
6. **Collect exact-deployment evidence.** Use pinned EVM forks with actual contract
   calls, receipts and Safe/Guard execution. Record chain/block, owner, dependency
   and implementation versions, funding token, admission/limits, requested and
   paid amounts, fees, residuals and all simulation concessions. Mock settlement,
   impersonation and injected liquidity demonstrate only the labelled mechanics.
   Live admission/liquidity/operator availability needs separate observed evidence.
7. **Refresh the coverage report.** Every scoped identity ends with directional
   implementation/evidence status and current availability: supported with
   evidence, temporarily unavailable, or an explicit implementation/evidence gap.
   Unknown or unavailable deployments remain listed and are not admitted on the
   strength of a family label. Publish counts derived from all 347 records, with
   the 104 identities outside this implementation scope still visible.

Keep tests focused on distinct implementations and failure modes, reusing existing
upstream and executor fixtures. Group related scenarios within happy-path and
failure-path tests where practical rather than adding one test per listed check.
No new funded transaction, deployment or operator
permission change is authorised by this planning document.

## Review record

Reviewed on 2026-10-29 with Claude CLI model `claude-opus-5-5`, using a bounded
plan-text review after the source exploration above. Findings were incorporated
for atomic Atoma integration, dependency compatibility, production state recovery,
intermediate token accounting and existing Guard/module migration. Follow-up
reviews tightened settlement-budget/cooldown preservation and prevented the old
module from settling during migration downtime. The final completed review
reported **no blocking findings**.

The later manual integration requirement in delivery step 5 links to the generic
subplan's Arbitrum tutorial. Its separate Opus 5.5 amendment reviews and resulting
recovery clarifications are recorded in that subplan; this does not constitute a
funded validation of the parent protocol matrix.

Document checks reconciled all 243 scoped identities against the 347-row inventory
and verified the seven required four-column change tables, local report/runbook
links and whitespace. No lifecycle tests or on-chain operations were run while
preparing this plan.
