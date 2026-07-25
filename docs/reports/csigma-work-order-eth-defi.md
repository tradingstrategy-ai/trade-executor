# eth-defi work order — cSigma cSuperior redemption preflight

Scope: one vault, one defect class. This is a **standalone** work order; it does
not change the round-3 matrix contract, it corrects the cSigma part of it.

Target vault: **cSuperior Quality Private Credit USDC**
`1-0x438982ea288763370946625fd76c2508ee1fb229`
(ERC-1967 proxy → verified implementation `0xa5b7555775a33ca79818702f63b34b14dc9aec4d`, `ContractName=CsigmaV2Pool`)

---

## 1. The issue

In the trade-executor simulation matrix, cSuperior currently ends in
`transaction_reverted` — a **forbidden** result:

```
deposit  -> OK
redeem   -> tx status=False
            revert_reason = "execution reverted: Withdrawal pending"
            (custom error WithdrawalPending(), selector 0xb34f5c6c)
result: transaction_reverted
```

`csigma/deposit_redeem.py` already *has* a redemption preflight intended to stop
exactly this (it raises `VaultFlowUnavailable(decoded_error="WithdrawalPending",
preflight_result="redemption_capacity_limited")`). **It does not fire.** The raw
revert escapes to the caller, violating invariant I1 ("never an on-chain revert
that reaches the caller").

### Why it does not fire — measured evidence

The preflight compares the request against `maxRedeem(owner)`. On this pool that
is not a valid authority. Probed on Ethereum mainnet:

| Probe (address) | share balance | `maxRedeem()` |
|---|---:|---:|
| `0x00…01` | 0 | **4,795,294** |
| `0x00…02` | 0 | **4,795,294** |
| `0xd8dA…6045` (vitalik) | 0 | **4,795,294** |

`maxRedeem()` returns the **same value for every address, including addresses
holding zero shares**. It is therefore *not* owner-scoped and does not satisfy
ERC-4626 semantics (`maxRedeem(owner)` must be bounded by the owner's balance).
It appears to report gross idle cash at the pool level.

Consequence in the matrix run: our Safe requested **626,911** raw shares,
`maxRedeem` reported **4,795,294**, so `fetch_redemption_preflight()` returned
`available=True` — and the on-chain `redeem()` reverted `WithdrawalPending()`
anyway. The preflight passed on a number that does not describe what can
actually be redeemed.

Share/denomination decimals are both 6; share `totalSupply` = 266,506,399,862.

---

## 2. Ground truth — how cSigma pools actually work

From cSigma's own design documentation
([Designing the Withdraw Request Flow: Queues, Reserves & Lender Repayments](https://www.csigma.finance/articles/designing-the-withdraw-request-flow-queues-reserves-lender-repayments)):

1. **Withdrawals are partial-fill, explicitly not all-or-nothing.**
   *"whatever the reserve can cover is returned immediately, and the remainder
   enters the queue."* This design deliberately **replaced** an older
   all-or-nothing model in which insufficient liquidity blocked withdrawals.
2. **The reserve is queue-adjusted.** It is *"the portion of the pool's total
   funds that hasn't been deployed"* — and critically *"cash on hand **minus what
   earlier queued lenders are already owed**."*
3. **The queue is strict FIFO**, and *"a pending position keeps earning yield
   right up until it's fulfilled."*
4. The reserve is replenished by a held-back percentage of new deposits (a
   manager must trigger distribution) and by returning deployed capital.

The published design document does **not** specify `maxRedeem` semantics,
`WithdrawalPending` trigger conditions, or ERC-4626 conformance — those are
implementation details that must be read off the verified contract, not assumed.

### What the verified ABI confirms

`eth_defi/abi/csigma/CsigmaV2Pool.json` (87 functions) contains:

- custom errors: `AccessDenied`, `AssetsFrozen`, `InvalidDepositAmount`,
  `InvalidPoolSize`, `InvalidStatusUpdate`, `PoolIsNotActive`,
  **`WithdrawalPending`**
- withdrawal/reserve surface: `withdrawalManager`, `setWithdrawalManager`,
  `nonReservePercentage`, `updateNonReservePercentage`,
  `sendReserveToFundManager`, `emergencyWithdraw`, `maxWithdraw`,
  `previewWithdraw`, `withdraw`
- **no** `requestWithdraw` / pending-position getter / `claim` / request id

So eth-defi's existing conclusion that there is **no on-chain per-lender
request/claim surface is correct** — the FIFO queue is serviced off-chain by the
`withdrawalManager`. ERC-7540-style ticket modelling is genuinely not achievable
here, and must not be attempted.

---

## 3. Root causes to fix

**C1 — The preflight uses a non-authoritative number.**
`maxRedeem()` is pool-wide gross idle cash, not `min(owner shares,
queue-adjusted reserve)`. Any preflight built on it will keep passing requests
that revert.

**C2 — The verified ABI is committed but never bound.**
`eth_defi/abi/csigma/CsigmaV2Pool.json` exists and its README states it is *"used
to model cSuperior redemption as an asynchronous queued request rather than a
synchronous ERC-4626 redeem."* But `csigma/vault.py` contains **no reference to
it**, and the runtime vault instance binds a generic ERC-4626 ABI whose error
list is empty (`[]`). The `WithdrawalPending` selector therefore cannot decode
from the bound ABI.

**C3 — Documented intent contradicts the implementation.**
The ABI README says async-queue modelling; the `csigma/deposit_redeem.py` module
docstring says the manager *"stays synchronous"* and is *"**not** an ERC-7540
async flow"*. Both are shipped. Given §2, the docstring is the achievable one and
the README is wrong. Resolve the contradiction rather than leaving both.

---

## 4. The goal (measurable)

> `1-0x438982ea288763370946625fd76c2508ee1fb229` must **never** produce
> `transaction_reverted` in the trade-executor matrix. Every redemption that
> cannot be filled **in full, immediately** must be refused as a typed
> pre-broadcast `VaultFlowUnavailable`, carrying `decoded_error="WithdrawalPending"`,
> `error_selector=0xb34f5c6c`, `preflight_result="redemption_capacity_limited"`,
> `direction="redeem"`, `phase="preflight"`, and the requested/available raw
> share amounts.

Accepted results for this vault: `redemption_capacity_limited` (expected in the
current pool state), or `success (simulated)` if the full amount genuinely is
immediately redeemable, or `redemption_paused` if `AssetsFrozen` /
`PoolIsNotActive` is the authoritative state.

**Forbidden:** `transaction_reverted`, `broadcast_failed`, `execution_failed`,
`receipt_analysis_failed`.

### Consumer contract you can rely on

trade-executor **assumes redemptions fill in full** and will not be changed in
this round. It does not model partial payouts or queued remainders. So:

- Do **not** build an API that returns "partially fillable, remainder queued" and
  expect the consumer to split it. It will not.
- A redemption that can only be partially filled is, for our purposes, a
  **refusal** — return the typed refusal above with `available_raw_amount` set to
  the true immediately-redeemable amount.

---

## 5. Required work

1. **Find the authoritative immediate-redeemability figure.** It is not
   `maxRedeem()`. Determine, from the verified `CsigmaV2Pool` implementation,
   the queue-adjusted reserve actually applied by `redeem()` before it reverts
   `WithdrawalPending()` — i.e. cash on hand minus amounts owed to earlier queued
   lenders (§2.2). Candidate surfaces to read: `withdrawalManager` (and the
   manager contract's own accounting), `nonReservePercentage`, `maxWithdraw`,
   plus the owner's share balance as an upper bound. Confirm the derivation by
   reproducing the revert boundary on a fork: a request at or below the figure
   must succeed, and one above it must be the case the preflight refuses.

2. **Rewrite `fetch_redemption_preflight()` on that figure**, bounded by the
   owner's actual share balance. Remove or clearly quarantine the `maxRedeem()`
   path, and document in the method why `maxRedeem()` is not usable on this pool
   (it is not owner-scoped — §1 evidence).

3. **Bind the verified ABI** (C2) so `WithdrawalPending` and the other custom
   errors decode from the contract the adapter actually talks to. Ensure the
   bound ABI for `0x438982ea…` is `CsigmaV2Pool.json`, not the generic ERC-4626
   ABI. Add an assertion or test that the bound ABI exposes the
   `WithdrawalPending` error.

4. **Resolve the intent contradiction** (C3): correct
   `eth_defi/abi/csigma/README.md` so it no longer claims async-queued-request
   modelling, and state plainly that the pool has no on-chain claim surface and
   is modelled as a synchronous, reserve-limited redemption with a typed
   capacity refusal.

5. **If the authoritative figure cannot be read on-chain**, do not guess and do
   not leave the revert. Raise a typed refusal before broadcast using the best
   available bound (e.g. treat immediate capacity as zero) with a concrete reason
   in the message. A conservative typed refusal is acceptable; a raw revert is
   not.

---

## 6. Acceptance criteria

- [ ] A focused fork test at a **pinned block** against the exact address
      `0x438982ea288763370946625fd76c2508ee1fb229` (I4 — no same-ABI substitute)
      that deposits, then attempts a redemption, and asserts a typed
      `VaultFlowUnavailable` is raised **with no redeem transaction broadcast**.
- [ ] That test asserts the structured fields: `decoded_error="WithdrawalPending"`,
      `error_selector=0xb34f5c6c`, `preflight_result="redemption_capacity_limited"`,
      `direction="redeem"`, `phase="preflight"`, `requested_raw_amount`,
      `available_raw_amount`.
- [ ] A test asserting the bound ABI for this address exposes the
      `WithdrawalPending` custom error (guards C2 from regressing).
- [ ] A boundary test: a request at/below the authoritative figure completes the
      redemption; a request above it is refused (proves the figure is real, not
      a constant).
- [ ] **cSigma USD `1-0xd5d097f278a735d0a3c609deee71234cac14b47e` still returns
      `redemption_capacity_limited`** — existing regression control, must not
      change.
- [ ] Matrix row for cSuperior pasted in the PR showing a non-forbidden result.

---

## 7. Explicit non-goals

- **Do not model partial fills.** The consumer assumes full fills this round
  (§4). Partial payout + queued remainder is tracked as future work on the
  trade-executor side and must not be a prerequisite here.
- **Do not model this as ERC-7540 async.** §2 shows there is no on-chain
  request/ticket/claim surface; it is not achievable.
- **Do not advertise `supports_anvil_settlement=True`** for this pool. The queue
  is serviced off-chain by the `withdrawalManager`; there is nothing to force on
  a fork (invariants I2/I3).
