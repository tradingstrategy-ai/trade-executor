# Vault test corrective fixes

## Objective

Correct the classifications and protocol gaps exposed by the 129-vault
production-data simulation on 2026-07-31, then rerun the same matrix with a
representative deposit amount.

The work is complete when:

- HYPE++ and texasHedge are never classified as requiring KYC or account
  whitelisting, while their funding, trading, redemption and lock-up delays
  remain visible;
- canonical Morpho V1 vaults are classified as permissionless, while Morpho V2
  derives its policy from its configured deposit gates;
- canonical Euler Earn vaults and ungated Euler EVK vaults are classified as
  permissionless, while EVK deposit hooks are classified by their actual
  behaviour;
- the standard vault test uses 1,001 USDC and the matrix does not override it
  with 1 USDC;
- supported redemption lifecycles can be exercised on Anvil even when live
  redemption liquidity is unavailable, with the intervention disclosed in a
  distinct success result;
- the successful Arche USD redemption receipt is analysed correctly;
- a vault which does not accept USDC is classified as
  `incompatible_deposit_asset`; and
- a fresh 129-vault report records the command, amount, dependency commits and
  result changes needed for a direct comparison with the 2026-07-31 report.

This requires coordinated changes in eth-defi and trade-executor. Protocol
semantics, transaction construction, forced settlement and receipt analysis
belong in eth-defi. CLI defaults, orchestration, result mapping and report
provenance belong in trade-executor.

## Current findings

| Issue | Vault | Current result | Root cause | Required result |
|---|---|---|---|---|
| Public D2 vault treated as KYC-gated | HYPE++ (`42161-0x75288264fdfea8ce68e6d852696ab1ce2f3e5004`) | `whitelisting-needed` | D2's historical `whitelisted` eligibility check is passed through the shared KYC whitelist check | Successful lifecycle when the funding phase is open, otherwise a typed amount or window result; never `whitelisting-needed` |
| Public D2 vault with stale delay metadata | texasHedge (`999-0x208f63a7f60c319597c05fa5ec67fde41839bad6`) | Production page omits permission metadata and reports “Funding phase closed (opens in -485h)” | D2 permission is not exported, an expired epoch end is retained as `deposit_next_open`, and failed phase reads can be swallowed as open | `permissionless`, with independently derived current deposit/redemption status, future-only next-opening timestamps and current epoch lock-up |
| Permissionless Morpho vault exported as permissioned | 9S Mount Kosciuszko USR (`1-0x00b6f2c15e4439749f192d10c70f65354848cf4b`) and other canonical Morpho vaults | Public page says “permissioned vault” | Both Morpho V1 and V2 adapters unconditionally return `True` without inspecting the contracts; Morpho's unrelated `not_whitelisted` risk flag makes the output more misleading | V1 `permissionless`; V2 `permissionless` when deposit gates are unset, otherwise derived from its gates |
| Permissionless Euler vaults exported as permissioned | Hyperithm Euler USDC (`1-0x3cd3718f8f047aa32f775e2cb4245a164e1c99fb`), Hyperithm mHYPER (`1-0x8aff4fe319c30475d27ec623d7d44bd5ecfe9616`), Sentora RLUSD (`1-0x9bd52f2805c6af014132874124686e7b248c2cbb`) and Sentora PYUSD (`1-0xab2726daf820aa9270d14db9b18c8d187cbf2f30`) | Public pages say “permissioned vault”; all four lifecycle simulations succeeded | Both Euler EVK and Euler Earn adapters unconditionally return `True` without inspecting EVK hooks or the canonical Euler Earn deposit path | Euler Earn `permissionless`; EVK `permissionless` when the deposit operation is unhooked, otherwise derived from the hook |
| Deposit amount below protocol minimum | Accountable and Ember minimum checks | `below_minimum` | The matrix command supplied `--amount 1.0`, overriding the CLI default | Test with 1,001 USDC and pass the protocol minimum |
| Redemption path unavailable | YieldNest RWA MAX (`1-0x01ba69727e2860b37bc1a2bd56999c1afb4c15d8`) | `redemption_unavailable` | Maturity-aware redemption is not implemented | Implement the lifecycle and, when only live liquidity or an open redemption phase is missing, complete an explicitly disclosed forced simulation |
| Redemption path unavailable | Apyx USDC (`1-0x069662d2588fcac24b5c209456db965d151556f0`) | `redemption_unavailable` | Morpho V2 redemption asset transfer fails and is not handled by the adapter | Fix and analyse the lifecycle; use the same disclosed forced simulation only for a liquidity/closed-redemption constraint |
| Successful receipt rejected | Arche USD (`1-0x33ffc177a7278ff84aab314a036bc7b799b7cc15`) | `receipt_analysis_failed` | The ERC-4626 analyser assumes an event amount has a direction/sign which standard unsigned event fields do not carry | `success (simulated)` with correct asset and share amounts |
| Unsupported selected asset misclassified | Sentora USD Earn (`1-0x74ad2f789ed583dbd141bbdafc673fe1f033718b`) | `deposit_closed` | Upshift's missing explicit asset selection falls through to a generic closure result | `incompatible_deposit_asset` for USDC; success only when an accepted asset is explicitly selected and funded |

`DEFAULT_VAULT_TEST_TRADE_AMOUNT` is already 1,001 USDC and already has line
comments explaining the Accountable Hyperithm minimum and the one-dollar
buffer. Preserve those comments. The regression is in the matrix invocation,
which explicitly used `--amount 1.0`.

## Known-correct whitelist control

TESS USDT sUSDs Loop Vault
(`1-0x9fec8a63a6c6ef9eadddfbd79daba5918965794e`) is a genuine permissioned
IPOR Fusion vault. Its current production status must remain `whitelisted`
while the Morpho, Euler and D2 false positives are corrected.

The check is confirmed independently against IPOR documentation, IPOR's
published contract source, OpenZeppelin AccessManager semantics and the live
deployment. IPOR defines role `800` as `WHITELIST_ROLE` and
`type(uint64).max` as `PUBLIC_ROLE`. Its vault initializer assigns the deposit
selector to one of those two roles according to the vault's `isPublic` setting,
and `convertToPublicVault()` changes that exact selector to `PUBLIC_ROLE`.
`PlasmaVault.deposit()` is protected by OpenZeppelin's `restricted` modifier.
OpenZeppelin specifies that every address has `PUBLIC_ROLE`, while any other
selector role requires the caller to hold that role.

At Ethereum block `25,651,723`:

- eth-defi autodetects the deployment as `IPORVault`;
- its deposit selector is `0x6e553f65`;
- `getAccessManagerAddress()` and the vault's inherited `authority()` both
  resolve to `0xD3783CA8113e6EcA8716067302f49311aACbf8D7`;
- `isTargetClosed(vault)` is false;
- the IPOR AccessManager assigns that selector role `800`, rather than
  `PUBLIC_ROLE` (`18,446,744,073,709,551,615`);
- `hasRole(800, caller)` and `canCall(caller, vault, selector)` both return
  `(False, 0)` for the executor Safe
  `0xa2b04c6a053AB2EFBC699f5DD0F0957742A41629` and for an arbitrary address;
- direct zero-value deposit simulations from both callers revert before amount
  validation with selector `0x068ca9d8`,
  `AccessManagedUnauthorized(address)`; and
- the earlier observation found `maxDeposit()` non-zero for both denied
  addresses.

The public Bitcoin Dollar USDC IPOR vault
(`1-0xf8f226da66244f89e70c5b5d1a5c5b0d505eb1d8`) provides the positive
control at the same live state. Its deposit selector has `PUBLIC_ROLE`,
`canCall()` returns `(True, 0)` for the same arbitrary outsider and eth-defi
reports `is_whitelisted_deposit() is False`. Its zero-value deposit reaches
amount validation and reverts with `NoAssetsToDeposit()` (`0x182b250f`), not
`AccessManagedUnauthorized(address)`.

This proves that the page's combination of `deposit_status=open` and
`deposit_permission=whitelisted` is internally consistent: global capacity is
open, but outsiders cannot call the deposit selector. `maxDeposit()` must never
be used as a substitute for the AccessManager policy.

Add the exact deployment as a fixed-block negative-control regression:

1. assert the vault's `authority()` matches `getAccessManagerAddress()` and the
   target is not globally closed;
2. assert its selector role equals IPOR's `WHITELIST_ROLE` (`800`) rather than
   merely checking that it differs from `PUBLIC_ROLE`;
3. assert the executor Safe does not hold role `800`, is not admitted and has
   no scheduling delay;
4. assert a direct deposit simulation from the Safe reverts with
   `AccessManagedUnauthorized(address)`;
5. assert eth-defi reports `is_whitelisted_deposit() is True`;
6. assert trade-executor returns `whitelisting-needed` before transaction
   construction when the correct USDT deposit asset is selected;
7. assert production JSON and the public page retain `whitelisted` after
   regeneration; and
8. assert the Morpho, Euler and D2 changes do not alter IPOR's
   selector-specific AccessManager logic. Pair this with the public Bitcoin
   Dollar USDC control, which must retain `permissionless`.

For IPOR, use the selector role for the global policy and `canCall()` for the
specific account. Keep target closure, deposit capacity and account admission
as separate dimensions. A non-zero `canCall()` delay means the caller holds the
role but must use scheduled execution; `(False, 0)` means it is not admitted
when the target is open.

## Cross-cutting evidence and result model

Do not replace blanket `whitelisted` assumptions with blanket
`permissionless` assumptions. Before changing any exported permission:

1. establish the deployment generation from an authoritative factory event,
   registry, resolved proxy implementation or source-matched runtime bytecode;
2. record the chain, observation block, implementation or factory evidence,
   relevant gate/hook/role values and account-specific predicate result;
3. treat an unrecognised wrapper, implementation, gate or hook as `unknown`;
   and
4. require the regenerated JSON diff to carry this evidence for every row
   changing from `whitelisted` to `permissionless`.

Factory membership and runtime-bytecode matching are alternative forms of
provenance, not mandatory simultaneous checks. Proxy deployments must resolve
the implementation before matching. Maintain recognised gate and hook
addresses or code hashes as versioned eth-defi protocol metadata with source
links and fixed-block tests. Publish total `whitelisted`, `permissionless` and
`unknown` counts before and after regeneration.

Permission, availability and capacity are separate machine dimensions:

- an unrecognised permission mechanism produces
  `deposit_permission_unknown`; the runner fails closed before broadcasting,
  while a research-only `eth_call` may be recorded as evidence without
  upgrading the exported permission;
- an RPC revert or undecodable lifecycle view produces
  `vault_state_read_failed`, not an open or closed status;
- insufficient capacity produces `deposit_capacity_exceeded` only when the
  adapter declares the capacity source authoritative. A protocol-defined
  sentinel or deliberately non-authoritative `maxDeposit()` must be resolved
  through an adapter-specific capacity view or a deposit simulation at the
  requested amount; an unresolved read is `vault_state_read_failed`, not a
  capacity claim;
- inability to source the requested asset for the Anvil wallet produces
  `test_wallet_funding_failed`, which is an experiment/infrastructure result
  rather than a vault restriction;
- retain `below_minimum` for compatibility, but attach
  `minimum_leg=deposit|redemption`, the requested amount and the required
  amount; and
- replace generic `redemption_unavailable` with
  `redemption_not_implemented`, `redemption_closed`,
  `redemption_liquidity_unavailable` or
  `redemption_not_yet_matured`.

The scanner may publish `deposit_permission=unknown` independently of a
research simulation which happened to succeed for one caller at one block.
One successful call is not proof that a vault is generally permissionless.
Apply the same separation to redemption: inspect any withdrawal/redeem
gate, hook or role independently and export `redemption_permission` where the
protocol supports distinct policies.

## Work item 1: correct D2 admission and timing semantics

Implement this first in eth-defi's D2 adapter.

1. Prove the admission semantics for the exact HYPE++ and texasHedge
   deployments from D2 documentation, published or verified contract source
   and live calls at pinned blocks. Record the implementation identity, the
   exact predicate currently named `onlyWhitelisted`, its inputs and results
   for the executor Safe and arbitrary control addresses.
2. If that evidence confirms the predicate is balance-, funding-phase- or
   schedule-based rather than an identity allow-list, make the exact
   deployments `permissionless`. Do not make every future D2 deployment
   permissionless by protocol name alone; an unrecognised D2 generation is
   `unknown`.
3. Ensure `D2Vault.is_whitelisted_deposit()` reports false only for the
   source-proven public generation. Override the shared whitelist preflight for
   that generation so `VaultDepositManager.check_deposit_whitelist()` cannot
   raise `WhitelistingRequired` from a non-identity eligibility failure.
4. Keep the actual D2 eligibility checks. Return structured failures for:
   insufficient eligible balance, a deposit below the required amount, or a
   closed funding phase. Include the requested value, required value and next
   opening time where the contract exposes them.
5. Apply the same distinction to the GuardV0 validation request path. Guard
   allow-list validation and D2 funding eligibility must not be conflated.
6. Keep trade-executor's `WhitelistingRequired` mapping unchanged for genuinely
   permissioned vaults. It should simply stop receiving that exception from D2.

Keep permission and timing as independent dimensions:

1. Derive the current D2 phase from `fundingStart`, `epochStart`, `epochEnd`,
   custody state and the chain block timestamp. Do not use the host wall clock
   when evaluating a fixed-block fork.
2. Export `deposit_status`, `redemption_status`, `deposit_next_open`,
   `redemption_next_open` and estimated lock-up independently of
   `deposit_permission`.
3. A next-opening timestamp must be strictly later than the observation block
   timestamp. If the recorded epoch has ended and no later funding window is
   committed on-chain, publish no next-opening timestamp rather than treating
   the expired epoch end as the next opening.
4. A closed-reason formatter must never emit a negative “opens in” duration.
   Omit the countdown when the next opening is unknown.
5. Do not convert an RPC revert or undecodable phase view into `None` when
   `None` means open. Propagate a typed unknown/read-failed state so the scanner
   and test runner cannot falsely report deposits or redemptions as open.
6. Recompute the estimated lock-up from the current epoch. Do not retain the
   previous epoch's duration after a new epoch has been scheduled.

As of HyperEVM block `41,917,827`, texasHedge's current epoch has:

- funding start `2026-07-31 08:00:00`;
- trading start `2026-08-04 02:00:00`;
- epoch end `2026-10-02 08:00:00`; and
- an estimated trading lock-up of 59 days and 6 hours.

At that observation block the vault is in its funding window, so the correct
combination is `deposit_permission=permissionless`,
`deposit_status=open` and `deposit_next_open=null`. These values are
time-dependent; the regression must pin the block rather than hard-code them
as permanent vault attributes.

Capture the raw `fundingStart`, `epochStart`, `epochEnd` and custody-state call
results in the fixed-block fixture or report appendix so these timestamps can
be reproduced without relying on the prose values above.

Add pinned regressions for both exact D2 addresses.

For HYPE++ on Arbitrum:

- assert that its permission metadata is public;
- fund the test Safe with 1,001 USDC;
- assert that preflight does not raise `WhitelistingRequired`;
- exercise the full lifecycle when the forked funding phase permits it; and
- test a deliberately unmet D2 eligibility condition and assert a typed amount
  or window result rather than `whitelisting-needed`.

For texasHedge on HyperEVM:

- assert autodetection as `D2Vault` and `permissionless`;
- cover one block in a funding window and one block in a trading/closed window;
- assert the exact deposit and redemption status independently at each block;
- assert a future next-opening time or `None`, never an expired timestamp;
- assert the lock-up from that block's current epoch;
- reproduce the old epoch-end-in-the-past case and ensure the report contains
  no negative countdown; and
- prove a delayed lifecycle is typed as a window or lock-up result rather than
  `whitelisting-needed`.

Regenerate the production vault JSON and verify that the texasHedge page shows
both facts at once: “permissionless” admission plus the current funding,
trading/redemption and lock-up timing. The page must not hide delays merely
because the vault is public.

Acceptance criterion: neither HYPE++ nor texasHedge may produce
`whitelisting-needed`. An actually closed funding phase remains an honest
`deposit_closed` result, an active lock-up remains visible, and no D2 countdown
is negative.

## Work item 2: correct Morpho deposit permissions

This is a confirmed eth-defi and production-data bug. The exact
9S Mount Kosciuszko USR address autodetects as `MorphoV1Vault`, but the current
adapter returns `True` from `is_whitelisted_deposit()` solely because of a
blanket operating assumption. Its exported caveat simultaneously says that no
permissioned hook checks were performed.

Morpho's API warning named `not_whitelisted` is a listing and curation-risk
signal. It does not mean that an investor must be allow-listed before
depositing. Keep this warning in `morpho_vault_flags` and
`morpho_market_flags`, but never use it to populate `deposit_permission`.

### Morpho V1

Canonical Morpho V1, or MetaMorpho, has no depositor gate in its public
`deposit()` and `mint()` paths.

1. Change `MorphoV1Vault.is_whitelisted_deposit()` to return `False` for the
   canonical source-proven implementation. Establish canonical status from a
   Morpho factory/registry deployment record or resolved implementation
   bytecode, not from adapter autodetection alone.
2. Remove the “No permissioned hook checks were performed” caveat from this
   source-proven V1 classification.
3. Do not infer permissioning from owner, curator, guardian, allocator, market
   cap, supply queue, a permissioned underlying token or Morpho API warning.
   These are independent operational or asset facts.
4. If autodetection later includes a non-canonical wrapper or proxy generation,
   classify that generation separately. Do not inherit V1's permissionless
   answer until its deposit path is source-verified.

### Morpho V2

Morpho V2 is permissionless by default, but it has configurable gates. Deposit
admission depends on both the caller sending assets and the receiver obtaining
shares.

1. Add the stable gate views needed by the adapter:
   `sendAssetsGate()`, `receiveSharesGate()`, `canSendAssets(address)` and
   `canReceiveShares(address)`. Confirm each view against the deployed V2 ABI
   and source before adding it; an absent or reverting view yields `unknown`,
   never an assumed zero address.
2. Return `permissionless` when both deposit-relevant gate addresses are zero.
3. When either gate is non-zero, classify the vault as `whitelisted` only when
   the gate's semantics demonstrably implement account admission. Return
   `unknown` for an unrecognised gate rather than making a KYC claim from its
   mere presence.
4. Implement `is_account_whitelisted(address)` for recognised gated
   deployments by requiring the account to pass both relevant predicates. The
   executor's caller and share receiver are currently the same Safe; if that
   changes, expose a two-address admission preflight instead of testing only
   one role.
5. Keep deposit availability, liquidity-adapter state and V2's deliberately
   zero `maxDeposit()` result separate from deposit permission.
6. Inspect the deployed `withdraw()` and `redeem()` paths for send/receive
   gates or other account predicates. Export and preflight redemption
   permission independently instead of assuming deposit admission implies
   redemption admission.

Token-level transfer restrictions also remain separate. A permissionless
Morpho vault holding a KYC-restricted asset is not a permissioned vault; the
selected asset should carry its own incompatibility or account-eligibility
result.

### Scanner and production output

1. Regenerate the production vault JSON after the adapter correction rather
   than patching individual rows.
2. Update both top-level `deposit_permission` and the compatibility
   `whitelist.status`/deposit-manager copy from the same scanner result.
3. Inventory all exported Morpho V1 and V2 rows before and after regeneration.
   Every changed row must list its detected generation, old status, new status
   and gate addresses where applicable. Include its factory/implementation
   evidence and fail the regeneration review if any
   `whitelisted` → `permissionless` change lacks it.
4. Verify that the 9S Mount Kosciuszko USR page no longer displays “permissioned
   vault”. It may continue to display `not_whitelisted` and `short_timelock` as
   Morpho risk warnings.
5. Retain trade-executor's stale-JSON mismatch result. A stale
   `whitelisted` JSON row against a permissionless fork should be
   `whitelisted-incorrectly` until regenerated data is deployed, not
   `whitelisting-needed`.

Add regressions for:

- the exact 9S Mount Kosciuszko USR address at a pinned Ethereum block,
  asserting `MorphoV1Vault`, `permissionless`, open deposits and independence
  from its `not_whitelisted` risk warning;
- another canonical V1 vault with no Morpho API warning;
- a V2 vault with both deposit gates unset;
- an ungated V2 vault whose deliberate sentinel-zero `maxDeposit()` does not
  become `deposit_capacity_exceeded` and whose requested deposit is checked by
  the adapter's authoritative path;
- a V2 vault with a recognised account gate, checking both an admitted and a
  refused address;
- a V2 vault with an unrecognised non-zero gate, which must remain `unknown`;
- scanner and lifetime-metrics propagation into production JSON; and
- trade-executor's old-JSON mismatch followed by normal operation with the
  regenerated permissionless row.

Acceptance criterion: canonical V1 rows are permissionless, default ungated V2
rows are permissionless, genuine V2 gates remain visible, and no Morpho API
`not_whitelisted` warning can produce `whitelisting-needed` or a permissioned
vault banner.

## Work item 3: correct Euler deposit permissions

This is also a confirmed eth-defi and production-data bug. Both
`EulerVault.is_whitelisted_deposit()` and
`EulerEarnVault.is_whitelisted_deposit()` currently return `True` under the
same unverified operating assumption as Morpho.

The four Euler rows in the 2026-07-31 lifecycle matrix all accepted the newly
created executor Safe and completed their simulated lifecycle. Direct reads of
the three EVK rows show `hookConfig() == (address(0), 0)`, meaning no operation
hook is installed. Their public pages nevertheless display the
permissioned-vault warning.

### Euler Earn

Canonical Euler Earn is a MetaMorpho-derived ERC-4626 vault. Its public
`deposit()` and `mint()` implementation has no depositor admission gate.

1. Change `EulerEarnVault.is_whitelisted_deposit()` to return `False` for
   canonical factory deployments whose implementation is source-verified.
   Record the factory event/registry or resolved implementation evidence; do
   not treat successful autodetection alone as canonical provenance.
2. Remove the “No permissioned hook checks were performed” caveat from the
   source-proven permissionless classification.
3. Keep owner, curator, guardian, allocator, strategy caps, supply queue,
   liquidity and the permission policy of an underlying strategy separate from
   the Euler Earn depositor policy.
4. Treat an unrecognised wrapper, factory or implementation generation as
   `unknown` until its public deposit path is verified.

### Euler EVK

Euler EVK supports arbitrary operation hooks. Hooks can implement KYC, but can
also implement pausing, utilisation caps, minimum position sizes, synthetic
asset controls or other non-identity policies. A hook's presence alone is
therefore not evidence of depositor whitelisting.

1. Add the EVK `hookConfig()` view and operation constants to the adapter ABI.
   `OP_DEPOSIT` is the authoritative bit for the test trade's `deposit()` path;
   keep `OP_MINT` available for callers which use `mint()`.
2. Return `permissionless` when `OP_DEPOSIT` is not set, including the common
   `(address(0), 0)` configuration.
3. When `OP_DEPOSIT` is set and the hook target is zero, classify the vault
   policy as permissionless but the deposit operation as disabled. This is
   `deposit_closed`, not `whitelisting-needed`.
4. When `OP_DEPOSIT` points to a non-zero hook, identify the hook by deployed
   code or a source-verified interface:
   - a recognised access-control/KYC hook is `whitelisted`;
   - a recognised pause, amount, utilisation or synthetic-asset hook remains
     permissionless and reports its own typed availability condition; and
   - an unrecognised hook is `unknown`, never automatically whitelisted.
5. Implement `is_account_whitelisted(address)` only for recognised
   access-control hooks. Query their real account predicate or simulate the
   documented hook call for the same caller and receiver that the executor
   will use.
6. Keep `maxDeposit()`, supply cap, cash, borrowing configuration, EVC account
   health and token-level transfer restrictions separate from KYC status.
7. Inspect `OP_WITHDRAW`, `OP_REDEEM` and any EVC-mediated account checks used
   by the actual redemption path. Classify and preflight redemption permission
   separately from `OP_DEPOSIT`.

The scanner should persist enough evidence for review:

- canonical factory or implementation provenance;
- EVK hook target and hooked-operation bitfield;
- recognised hook kind, if any;
- whether `OP_DEPOSIT`, `OP_WITHDRAW` and `OP_REDEEM` are active;
- the source of a `whitelisted` classification; and
- an explicit note for `unknown` hooks.

### Production output and regressions

Regenerate production vault JSON after the adapter fix and inventory all Euler
Earn and EVK status changes. Do not patch the four known pages by hand.

Add fixed-block regressions for:

- Hyperithm Euler USDC, asserting canonical `EulerEarnVault`,
  `permissionless` and no depositor gate;
- Hyperithm mHYPER, Sentora RLUSD and Sentora PYUSD, asserting
  `EulerVault`, `(address(0), 0)` hook configuration and `permissionless`;
- an EVK vault with `OP_DEPOSIT` disabled by a zero hook target, asserting
  `deposit_closed` rather than `whitelisting-needed`;
- a recognised EVK access-control hook with admitted and refused accounts;
- a recognised non-KYC hook affecting deposits without changing
  `deposit_permission`;
- an unrecognised non-zero hook, asserting `unknown`;
- scanner and lifetime-metrics propagation into production JSON; and
- each known page losing its permissioned-vault banner after regenerated data
  is deployed.

Acceptance criterion: canonical Euler Earn and EVK vaults without a deposit
access-control hook are permissionless. Only source-recognised account
admission hooks may produce `whitelisted` or `whitelisting-needed`.

## Work item 4: make 1,001 USDC the effective matrix amount

The production default is already correct in
`tradeexecutor/cli/commands/vault_test_trade.py`:

- keep `DEFAULT_VAULT_TEST_TRADE_AMOUNT = 1_001.0`;
- keep adjacent line comments stating that this value exists for the
  Accountable Hyperithm vault's 1,000 USDC minimum and provides a 1 USDC
  buffer; and
- do not replace the protocol-specific explanation with a generic “large test
  amount” comment.

Then fix every invocation used for the 129-vault experiment:

1. Remove `--amount 1.0` so the tested command exercises the default, or pass
   `--amount 1001.0` explicitly when reproducibility requires the amount in a
   script.
2. Record `deposit_amount=1001.0` in report provenance and display it in the
   Markdown report header.
3. Audit alternate vault-test entry points. A shared helper may retain a
   generic 1 USDC default, but `vault-test-trade` must always pass its resolved
   CLI value explicitly.
4. Before broadcasting, preflight both the deposit minimum and the shares that
   will later be redeemed. Preserve typed `below_minimum` only when 1,001 USDC
   genuinely does not pass the relevant minimum, and record
   `minimum_leg=deposit|redemption`.
5. Do not silently increase the amount above 1,001 USDC. If another vault
   requires more, record its required amount and use a documented per-vault
   override in a focused test.
6. Preflight the vault's capacity at 1,001 USDC separately using the capacity
   source which its adapter declares authoritative. A confirmed lower
   `maxDeposit`, supply cap or utilisation cap is
   `deposit_capacity_exceeded`, not `below_minimum` or `deposit_closed`.
   Never treat Morpho V2's deliberate sentinel-zero `maxDeposit()` as
   authoritative; use its adapter-specific view or requested-amount deposit
   simulation instead.
7. Fund the Anvil Safe with the exact selected asset and raw amount before
   protocol preflight. If the test harness cannot source it, emit
   `test_wallet_funding_failed` and do not attribute the result to the vault.

Add focused tests which assert:

- the CLI's omitted `--amount` value resolves to 1,001 USDC;
- the resolved amount reaches the simulation runner unchanged;
- report provenance records 1,001 USDC; and
- the exact Accountable vault and Ember Apollo ACRED do not produce
  `below_minimum` at that amount.

Acceptance criterion: the rerun command contains no 1 USDC override, and any
remaining `below_minimum` row includes a protocol minimum greater than the
actual 1,001 USDC deposit or its resulting redeemable shares.

## Work item 5: implement redemption and disclosed liquidity injection

Treat the YieldNest and Apyx rows as eth-defi implementation work, not accepted
`redemption_unavailable` endpoints.

### Adapter implementation

1. Implement YieldNest RWA MAX's maturity-aware redemption request, readiness,
   claim and receipt-analysis lifecycle.
2. Reproduce Apyx's Morpho Vault V2 transfer failure, decode it, and implement
   the correct request or immediate redemption route.
3. Separate three states in the capability contract:
   - the adapter does not implement the protocol, producing
     `redemption_not_implemented`;
   - the adapter implements redemption but the current live state has closed
     redemptions, insufficient liquidity or unmet maturity, producing
     `redemption_closed`, `redemption_liquidity_unavailable` or
     `redemption_not_yet_matured`; and
   - redemption is currently executable.
4. Do not advertise `can_redeem=False` merely because current liquidity is
   unavailable. Capability describes implemented code; availability describes
   live protocol state.

### Anvil-only forced simulation

Extend the eth-defi forced-settlement interface for adapters whose transaction
lifecycle is implemented but whose forked state cannot currently redeem:

1. Attempt the unmodified redemption first and capture its typed unavailability
   reason.
2. Permit intervention only in auto-simulated Anvil mode.
3. Add only the denomination-token liquidity needed to make the redemption
   executable. If the protocol requires an authorised settlement role to open
   or settle a redemption phase, impersonate that role only through a
   protocol-specific eth-defi hook and only when that role could legitimately
   perform the action at the observation block with no unmet protocol time
   condition.
4. Execute the real adapter request, settlement and claim functions after the
   intervention. Do not fabricate a receipt or bypass receipt analysis.
5. Return structured evidence including the original reason, injected token,
   raw injected amount, privileged action if any, transaction hashes and
   analysed assets returned.
6. Never use this mechanism to bypass KYC, an account allow-list, an
   incompatible deposit asset, an amount minimum, or an irreversible time
   lock. Real execution mode must never inject liquidity or impersonate an
   actor.

Add one canonical machine result in trade-executor:
`success_simulated_with_intervention`, with a required structured
`interventions` list. Supported intervention kinds are:

- `liquidity_injected`, including token, target and raw amount;
- `authorised_phase_action`, including role, actor, function and transaction;
  and
- any future intervention only after its semantics and report rendering are
  explicitly added.

Render the result from its interventions, for example
`success (simulated, liquidity injected)` or
`success (simulated, authorised phase action)`. If both occurred, disclose
both. Keep these rows separate from ordinary `success (simulated)` in JSON,
Markdown totals and PR headline numbers. Do not label a privileged phase action
as liquidity injection.

Do not advance time past YieldNest maturity merely to manufacture success. If
the fork block is pre-maturity, implement and verify the request/readiness path
and finish with the legitimate terminal result
`redemption_not_yet_matured`. A post-maturity fixed-block regression must then
exercise claim completion. A failed forced attempt retains its specific typed
failure and all intervention evidence.

Add pinned mainnet-fork tests for YieldNest RWA MAX and Apyx USDC. Each test
must first prove the natural redemption state and then:

- for a liquidity or immediately actionable phase constraint, prove the
  disclosed intervention completes the real lifecycle;
- for pre-maturity YieldNest, prove the request is correctly typed as
  `redemption_not_yet_matured` and use a separate post-maturity block to prove
  completion; and
- record the injection target and exclude intervention rows from price, return
  or other economic aggregation because injected assets may change NAV.

Add trade-executor unit coverage for result mapping, JSON serialisation,
Markdown aggregation, real-mode rejection and a forced simulation that still
fails.

Acceptance criterion: both exact adapters implement their real redemption
lifecycles and no row remains generic `redemption_unavailable`. An unmet
maturity is an acceptable, explicit terminal result. Any completed intervention
is disclosed and never counted as ordinary success.

## Work item 6: fix ERC-4626 receipt analysis

Fix this in eth-defi's ERC-4626 analyser rather than adding a Yearn or Arche
special case to trade-executor.

1. Reproduce the exact Arche USD redemption receipt on a pinned Ethereum fork.
2. Remove the assumption that ERC-4626 event amount fields encode trade
   direction with a negative sign. Standard event integers are unsigned.
3. Derive direction from the operation and event roles, then normalise the
   asset and share amounts into the established `TradeSuccess` sign contract.
4. Filter events by the concrete vault, asset, owner and receiver so unrelated
   transfers in a nested receipt cannot be selected. Let an adapter declare
   protocol escrow or claim-contract counterparties for async flows rather than
   assuming the user is always the direct event counterparty.
5. Replace contradictory assertions with a typed
   `VaultReceiptAnalysisError` containing the transaction hash and relevant
   decoded event values when the receipt is genuinely inconsistent.
6. Where a vault emits no standard event, use pre-recorded owner/vault asset
   and share balance deltas as a documented fallback. Never infer a successful
   amount from receipt status alone.

Add unit coverage for standard ERC-4626 deposits and redemptions with unsigned
event amounts, an adapter-declared async counterparty and balance-delta
fallback, plus the exact Arche USD fork regression. Assert the returned asset
amount, share amount and price, and confirm that a mined status-1 redeem does
not become `receipt_analysis_failed`.

Acceptance criterion: Arche USD produces `success (simulated)` without any
trade-executor protocol-specific receipt logic.

## Work item 7: classify and support deposit assets correctly

Fix the exception at the eth-defi Upshift boundary and thread asset selection
through trade-executor.

### eth-defi

1. Determine and expose the exact accepted deposit assets for Sentora USD Earn.
2. When the selected asset is USDC and USDC is not accepted, raise
   `IncompatibleDepositAsset` from capability, estimation and request-building
   paths. Include the selected asset and all accepted assets.
3. Do not translate missing asset selection into a generic closed-deposit
   reason.
4. When an accepted asset is supplied, build and analyse the real Upshift
   multi-asset deposit and redemption lifecycle.
5. Audit every existing `deposit_closed` row from the 129-vault report for the
   same unsupported-selected-asset failure. Fix the classification generically
   rather than only for Sentora USD Earn.

### trade-executor

1. Thread `--deposit-asset` into pair configuration, pricing model creation,
   router creation, wallet funding and the executable vault manager before
   preflight.
2. Remove the incomplete behaviour which patches only already-materialised
   routers. Lazy routers must receive the selected asset when they are created.
3. Keep the existing `IncompatibleDepositAsset` normalisation, including
   selected and accepted asset metadata.
4. With no override, retain USDC as the native test asset and report
   `incompatible_deposit_asset` when it is unsupported.
5. Do not use liquidity injection or automatic asset substitution to turn the
   default USDC mismatch into success. A supported-asset lifecycle is a
   separate, explicitly selected test.
6. Record amounts in the selected asset's native units. Report a USD value only
   when the normal pricing model has a valid quote for that asset, and include
   the price source; do not assume a non-USDC asset is worth one dollar.

Add an exact-vault fork test with two cases:

- the default USDC attempt ends as `incompatible_deposit_asset` before
  transaction construction and lists the accepted assets; and
- an explicitly selected accepted asset is funded, priced, routed, deposited
  and redeemed successfully.

Also add a trade-executor regression proving the result remains correctly typed
when the router is created lazily.

Acceptance criterion: Sentora USD Earn is never reported as
`deposit_closed` solely because USDC is unsupported.

## Implementation order

1. Define the cross-cutting permission evidence, typed failure and
   `success_simulated_with_intervention` schemas. Update JSON serialisation,
   Markdown aggregation, production-page rendering and any other in-repository
   consumers in the same landing.
2. Land eth-defi's D2 permission/eligibility correction.
3. Land eth-defi's Morpho V1/V2 permission correction and regenerate the
   production vault JSON.
4. Land eth-defi's Euler Earn/EVK permission correction and include it in the
   same production vault JSON regeneration.
5. Land the 1,001 USDC CLI and report-provenance regressions in trade-executor.
6. Land the generic eth-defi forced-settlement disclosure contract.
7. Implement YieldNest and Apyx redemption support against that contract.
8. Land the ERC-4626 receipt analyser correction.
9. Land eth-defi's Upshift asset contract, then thread explicit asset selection
   through trade-executor.
10. Update the trade-executor eth-defi submodule to the final tested commit.
11. Run focused exact-vault tests before running the complete matrix.

Keep protocol-specific eth-defi fixes in separate commits so a regression in
one adapter can be reviewed or reverted independently. The trade-executor
result-schema change and its reporting changes should land together.

## Verification and rerun

Run tests from the trade-executor worktree with its path first on
`PYTHONPATH`, as required for worktree imports. Run eth-defi tests from the
submodule checkout using that repository's own instructions.

The focused acceptance matrix is:

| Vault | Default 1,001 USDC result | Additional focused result |
|---|---|---|
| HYPE++ | Full success when open, otherwise typed funding-window/eligibility result | Never `whitelisting-needed` |
| texasHedge | Full success when its HyperEVM funding window is open, otherwise typed window result | `permissionless`; current lock-up and future-only next opening remain visible |
| 9S Mount Kosciuszko USR | Not part of the USDC lifecycle matrix | `permissionless`; Morpho risk warnings remain independent |
| Morpho V2 ungated vault | Full success when otherwise available | `permissionless` with zero deposit gates |
| Morpho V2 gated vault | `whitelisting-needed` only for a recognised gate and refused Safe | Gate addresses and account predicate evidence |
| Hyperithm Euler USDC | `success (simulated)` | `permissionless` Euler Earn metadata |
| Hyperithm mHYPER | `success (simulated)` | `permissionless`; EVK hook target zero and operations bitfield zero |
| Sentora RLUSD | `success (simulated)` | `permissionless`; EVK hook target zero and operations bitfield zero |
| Sentora PYUSD | `success (simulated)` | `permissionless`; EVK hook target zero and operations bitfield zero |
| Euler EVK gated vault | Depends on recognised hook behaviour | Only a recognised account-admission hook may produce `whitelisting-needed` |
| TESS USDT sUSDs Loop Vault | `whitelisting-needed` when explicitly tested with USDT | Known-correct IPOR AccessManager control; retain `whitelisted` |
| Accountable Hyperithm | Not `below_minimum` | Report confirms 1,001 USDC |
| Ember Apollo ACRED | Not `below_minimum` | Redeem-share minimum preflight is recorded |
| YieldNest RWA MAX | `success_simulated_with_intervention` for an actionable liquidity/phase constraint, or `redemption_not_yet_matured` before maturity | Post-maturity completion; ordinary success when natural redemption is available |
| Apyx USDC | `success_simulated_with_intervention` when an allowed intervention is required | Ordinary success when natural redemption is available |
| Arche USD | `success (simulated)` | Analysed redeem receipt and amounts |
| Sentora USD Earn | `incompatible_deposit_asset` | Success with an explicitly selected accepted asset |

After focused tests pass:

1. Freeze the same ordered list of 129 vault IDs. “Same” refers to this
   population, not to stale metadata.
2. Choose and record one fork block per chain and use the same blocks and
   provider configuration for every controlled run. If an original report block
   cannot be recovered, state that the old report is observational and run the
   following three-way control at newly pinned blocks.
3. Run control A using the PR 1588 baseline trade-executor commit, its pinned
   eth-defi commit, baseline production JSON and `--amount 1.0`.
4. Run control B using exactly the same baseline code, JSON and blocks but
   `--amount 1001.0`. The A→B delta isolates the amount change.
5. Run candidate C using the corrected code, final eth-defi commit,
   regenerated production JSON, the same blocks and `--amount 1001.0`. The
   B→C delta measures the complete code-plus-metadata correction without an
   amount or chain-state confound.
6. Publish the production JSON diff. Every permission change must include the
   evidence required by the cross-cutting model. Run the focused stale-JSON
   mismatch tests separately; do not use stale JSON for candidate C.
7. Start every matrix from a fresh state/report so cached terminal results are
   not reused.
8. Record trade-executor and eth-defi commit hashes, JSON commit/hash and
   generation timestamp, chain blocks, command line, resolved amount, start/end
   time, wall-clock runtime and concurrency.
9. Publish headline counts which separate ordinary simulated success,
   intervention success by intervention kind, permission/status unknown,
   protocol constraints and experiment/infrastructure failures.
10. Publish row-by-row A→B and B→C deltas, plus a clearly labelled comparison
    with `cross-chain-vault-test-2026-07-31-results.md`.

After the controlled comparison, optionally run candidate C against fresh live
blocks as the production-state snapshot. Report that separately because D2
phases, liquidity, caps and gate configuration may have changed; do not
attribute its delta from the old report solely to code.

## Definition of done

- Production metadata and runtime behaviour agree that HYPE++ and texasHedge
  are public while retaining their independent D2 lifecycle delays, backed by
  deployment-specific contract and account-predicate evidence rather than a
  protocol-wide assumption.
- D2 phase reads cannot silently turn an RPC/read failure into an open status,
  and D2 next-opening countdowns are never negative.
- Canonical Morpho V1 and ungated V2 vaults are exported as permissionless;
  recognised V2 gates are evaluated per account and unrecognised gates remain
  unknown. Every changed production row carries canonical-deployment and gate
  evidence.
- Morpho `not_whitelisted` remains a risk/listing warning and never becomes a
  depositor whitelist result.
- Canonical Euler Earn and EVK vaults with no deposit access-control hook are
  exported as permissionless, with factory/implementation and hook evidence.
- Euler EVK operation hooks are classified by verified behaviour; disabled,
  non-KYC and unknown hooks do not become depositor whitelist results.
- TESS USDT sUSDs Loop Vault remains `whitelisted`, proving genuine IPOR
  AccessManager restrictions survive the permission-classification fixes.
- The effective default experiment amount is 1,001 USDC and is covered by a
  regression test.
- No `below_minimum` result is caused by the old explicit 1 USDC command.
- Capacity, minimum and test-wallet funding failures are separate and carry
  their requested/required values or infrastructure reason.
- YieldNest and Apyx have implemented redemption lifecycles. Allowed
  interventions complete honestly and are disclosed; unmet YieldNest maturity
  remains `redemption_not_yet_matured` until a post-maturity regression proves
  completion.
- Every intervention kind is visibly distinct in machine data and human
  reports, and intervention rows are excluded from economic aggregates.
- Arche USD's successful receipt is analysed successfully.
- Unsupported USDC is always `incompatible_deposit_asset`, with accepted assets
  listed.
- Focused tests and the complete 129-vault rerun contain no unexplained
  `receipt_analysis_failed`; Arche USD no longer produces it.
- The controlled A→B comparison isolates the amount change and B→C holds the
  amount and fork blocks fixed while measuring the corrected code and
  regenerated metadata.
- The new report explains every remaining non-success row as a genuine live
  protocol restriction, an explicitly unsupported asset, or a concrete code
  defect with an owner.
