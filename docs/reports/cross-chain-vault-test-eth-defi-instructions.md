# Eth-defi vault simulation — acceptance-driven work order

This is the **binding work order** for the next eth-defi vault-simulation round.
It replaces the prose "fixes needed" list, which was interpreted loosely last
round. Read section 0 first: it explains exactly how the previous round
(PR #1374) fell short, and section 1–3 turn the goal into a contract that can be
checked mechanically.

## 0. Why the last round did not close the gaps

PR #1374 landed real improvements (Lagoon Safe provisioning, YieldNest and
cSigma-USD typing, Upshift multi-asset manager). But four items were "fixed" in
name only, and the reason each slipped through is the same class of problem:
**the goal was described as a list of per-vault suggestions, not as a
machine-checkable acceptance contract.**

| Miss | What #1374 did | Why it slipped |
|---|---|---|
| **Ember async settlement** (4 vaults) | Implemented `force_settle()` and advertised `supports_anvil_settlement=True` | On the fork the driver runs but the ticket stays `pending`. The capability was advertised without a mandatory in-code proof that the exact ticket becomes claimable, and eth-defi's own tests used a happy-path deployment, so the real vaults were never exercised. |
| **Accountable Monad** (1) | Added the `InsufficientAmount` typed error against a *different* same-ABI deployment | The instruction gave the exact `vault_id` but did not make "this exact address must reach a typed result" binding. A substitute deployment satisfied the prose; the reported vault still reverts raw `0x5945ea56`. |
| **cSuperior "Withdrawal pending"** (1) | Deferred (documented: off-chain FIFO, no on-chain claim) | The deferral was legitimate, but it still lets `redeem()` revert on-chain. The instruction prescribed one solution ("model as async") instead of the *fallback contract*: if you keep it synchronous you must **preflight-detect** the queued state and raise a typed error **before broadcast**. |
| **Ember redemption minimum** (1) | Typed the deposit `InsufficientAmount`, missed the redemption minimum | `ember/deposit_redeem.py` still does `raise ValueError(f"Ember redemption shares … below minimum …")`. No acceptance test asserted this exact path emits a typed error, so it was silently missed. |

Cross-cutting root causes, in order of impact:

1. **No measurable goal.** "Fix these 21" invites 21 local judgement calls.
2. **Capability was advertised without proof.** `supports_anvil_settlement=True`
   was treated as a label, not a self-checked promise.
3. **No shared acceptance harness keyed to the exact vault ids.** eth-defi
   verified against its own unit tests (convenient deployments), never against
   the executor matrix on the exact reported vaults. The feedback loop was
   one-way.
4. **"or keep it false / with a reason" softened the target** and let deferrals
   ship that still emit raw reverts.
5. **Exact deployment identity was not binding**, so a same-ABI substitute
   counted as a fix.

Sections 1–3 remove every one of these outs.

## 1. The goal, as one measurable contract

There is exactly one source of truth: the executor simulation matrix. From the
trade-executor checkout (eth-defi submodule pointed at your branch):

```shell
source .local-test.env
ASSET_MANAGEMENT_MODE=lagoon \
PYTHONPATH="$PWD:$PWD/deps/web3-ethereum-defi" \
poetry run trade-executor vault-test-trade \
  --id acceptance --state-file /tmp/acceptance/state.json \
  --auto-simulated --settle-async-on-anvil --amount 1.0 \
  --vault-id "<comma-separated ids from section 4>" \
  --report-json /tmp/acceptance/report.json
```

**Done means:** for every `vault_id` in the section-4 table, the matrix produces
the **required result string** in that table, and the run contains **zero** rows
whose result is `transaction_reverted`, `broadcast_failed`, `execution_failed`,
or `receipt_analysis_failed`. Paste the per-vault result table from
`report.json` into the PR. A PR without that pasted table is not ready for
review.

## 2. Binding invariants (apply to every adapter, no exceptions)

- **I1 — Typed refusals only.** Any predictable state discovered before or
  during a flow (closed, paused, whitelisted, below minimum, over capacity,
  windowed, cooldown, queued, wrong asset) must surface as
  `VaultFlowUnavailable` (or `WhitelistingRequired`) or
  `UnsupportedVaultSimulation`, with the structured fields populated
  (`decoded_error`, `requested_raw_amount`, `available_raw_amount`,
  `minimum_raw_amount`, `next_open`, `access_delay`, `direction`, `phase`,
  `protocol`, `vault_address`). **Never** a bare `assert`, `ValueError`,
  `ABIFunctionNotFound`, or an on-chain revert that reaches the caller.

- **I2 — Capability is a proof, not an intent.**
  `get_deposit_manager_capability().supports_anvil_settlement` may be `True`
  **only if** `force_settle(ticket)` is guaranteed to end with
  `status_after == claimable`. `force_settle` **must assert that itself** before
  returning, and raise `UnsupportedVaultSimulation` with the concrete reason
  otherwise. A driver that *can* leave the ticket pending **must** advertise
  `supports_anvil_settlement=False`. Advertising `True` on a pending-leaving
  driver is a defect, not a partial success.

- **I3 — Deferral is typed, never raw.** If a settlement or flow genuinely
  cannot be reproduced on a fork (role-gated operator, off-chain queue, no
  on-chain claim surface), that is acceptable — **only** as a typed
  `UnsupportedVaultSimulation` / `VaultFlowUnavailable` raised **before
  broadcast**, carrying the concrete reason. A deferral that still lets a
  transaction revert on-chain does not pass.

- **I4 — Exact deployment identity.** Each fix is verified against the exact
  `vault_id` (chain-address) in section 4, using that address in the focused
  manager test. A same-ABI substitute deployment does **not** satisfy
  acceptance. If the exact vault's current on-chain state makes a full lifecycle
  impossible, the acceptance is the correct **typed** result for that exact
  `vault_id`, asserted at that vault's pinned fork block.

- **I5 — Synthetic liquidity is disclosed.** Any fork settlement that injects
  synthetic liquidity must set `VaultForcedSettlementResult
  .synthetic_assets_injected_raw > 0` (kept from #1374).

- **I6 — Every known custom-error selector decodes.** At minimum:
  `0xa73449b9` EndOfEpoch (gains), `0xb8b8b59c` ExceededMaxRedeem (yieldnest),
  `0xb34f5c6c` WithdrawalPending (csigma), `0x5945ea56` InsufficientAmount
  (accountable), plus Plutus `UseRequestRedeem` / `WithdrawalsArePaused`. A
  reverting selector that reaches the caller undecoded is an I1 violation.

## 3. The only accepted matrix outcomes

A listed vault must end in one of these, and nothing else:

| Result | Meaning | Extra requirement |
|---|---|---|
| `success (simulated)` | Full deposit + redemption lifecycle completed | If via forced settlement: `status_after == claimable` and synthetic injection disclosed (I2, I5) |
| `deposit_closed` | Deposits currently closed on-chain | Typed, from a preflight |
| `whitelisting-needed` | Owner not on deposit whitelist | `WhitelistingRequired` |
| `below_minimum` | Amount below protocol minimum | `decoded_error="InsufficientAmount"` + `minimum_raw_amount` |
| `redemption_capacity_limited` | Immediate redemption capacity exceeded | `available_raw_amount` = immediate capacity |
| `redemption_window_closed` | Epoch/window shut | `decoded_error="EndOfEpoch"` + `next_open` when known |
| `redemption_paused` | Withdrawals paused | `decoded_error="WithdrawalsArePaused"` |
| `redemption_unavailable` | No redemption path in current state | Typed, with reason |
| `simulation_unsupported_async` | Async lifecycle cannot be forced on a fork | **Only** with `supports_anvil_settlement=False` + concrete typed reason (I2, I3) |

**Never accepted for a listed vault:** `transaction_reverted`,
`broadcast_failed`, `execution_failed`, `receipt_analysis_failed`. Any of these
means a raw or re-untyped failure still leaks.

## 4. Target table (exact ids → required result)

Signatures marked *(confirmed 2026-07-25, eth-defi 38fa4f945)* were observed
after PR #1374 merged; *(pending matrix)* rows are being re-measured and will be
finalised from the full run — but the required result already applies.

| # | vault_id | Vault | Current signature | Required result |
|--:|---|---|---|---|
| 1 | `1-0x9be9294722f8aad37b11a9792be2c782182cafa2` | Ember Earn | force_settle runs, ticket stays `pending` *(confirmed)* | `success (simulated)` **or** `simulation_unsupported_async` with `supports_anvil_settlement=False` (I2) |
| 2 | `1-0x0b9342c15143e8f54a83f887c280a922f4c48771` | Ember Polymarket | as above *(confirmed)* | same as #1 |
| 3 | `1-0xf3190a3ecc109f88e7947b849b281918c798a0c4` | Ember Third Eye | as above *(confirmed)* | same as #1 |
| 4 | `1-0x373152feef81cc59502da2c8de877b3d5ae2e342` | Ember UDL | as above *(confirmed)* | same as #1 |
| 5 | `1-0x2b13311fd553e74b421d4ccc96e348f71e179dcf` | Ember Apollo ACRED | raw `ValueError` "shares … below minimum" *(confirmed)* | `below_minimum` (typed, `minimum_raw_amount`, `direction=redeem`) |
| 6 | `1-0x438982ea288763370946625fd76c2508ee1fb229` | cSuperior | on-chain revert `Withdrawal pending` *(confirmed)* | typed preflight before broadcast → `redemption_capacity_limited` (or `redemption_paused`), **no revert** |
| 7 | `143-0x7cd231120a60f500887444a9baf5e1bd753a5e59` | Accountable Hyperithm (Monad) | raw revert `0x5945ea56` on this exact address *(confirmed)* | typed for **this** address (`below_minimum` / `whitelisting-needed` / typed unsupported) |
| 8 | `42161-0xd3443ee1e91af28e5fb858fbd0d72a63ba8046e0` | Gains gTrade (Arbitrum) | revert `custom error 0xa73449b9` *(pending matrix)* | `redemption_window_closed` (EndOfEpoch + `next_open`) **or** `success (simulated)` |
| 9 | `8453-0xad20523a7dc37babc1cc74897e4977232b3d02e5` | Gains gTrade (Base) | async, driver added in #1374 *(pending matrix)* | `success (simulated)` (I2) **or** `simulation_unsupported_async` false-capability |
| 10 | `8453-0x2bff679b1a9fbcc202316c1402172747ba2fbf56` | Lagoon For Yield v2 (Base) | `settleDeposit()` `transfer amount exceeds allowance` *(pending matrix)* | `success (simulated)` **or** typed unsupported — **no broadcast_failed** |
| 11 | `8453-0x63b04d3ce2c14f6d308657ab73ac92fc1a0b1075` | Lagoon RB Capital (Base) | as #10 *(pending matrix)* | same as #10 |
| 12 | `8453-0xbe7db44f4ce20dac83b578b94fd35087f66e9754` | Lagoon TruMarket (Base) | as #10 *(pending matrix)* | same as #10 |
| 13 | `1-0xa00f63e85b3d242568a9edecb48f5e2cf879b07b` | Lagoon Moon Digital | `pending -> pending` before #1374 *(pending matrix)* | `success (simulated)` **or** typed unsupported |
| 14 | `42161-0x1723cb57af58efb35a013870c90fcc3d60174a4e` | Lagoon Angmar (Arbitrum) | `pending -> pending` before #1374 *(pending matrix)* | same as #13 |
| 15 | `42161-0x58bfc95a864e18e8f3041d2fcd3418f48393fe6a` | Plutus Hedge (Arbitrum) | async manager added, `force_settle` deferred *(pending matrix)* | `simulation_unsupported_async` (typed, false capability) **or** `success (simulated)` |

Already accepted after #1374, listed so they are not regressed:
`1-0xd17049…` Lagoon Syntropia → `success (simulated)`;
`1-0xd5d097…` cSigma USD → `redemption_capacity_limited`;
`1-0x01ba69…` YieldNest RWA MAX → `redemption_capacity_limited`.

`1-0x74ad2f…` Upshift Sentora is now an **executor-side** result
(`incompatible_deposit_asset`: the vault accepts RLUSD/PYUSD/USDT, not USDC) —
no further eth-defi work unless we want USDC accepted.

## 5. Protocol-specific notes (only beyond the invariants)

- **Ember (#1–4).** Either make the operator-impersonation driver actually
  advance the queue so `status_after == claimable` on the pinned fork block
  (then I2's assert passes and the vault reaches `success`), or set
  `supports_anvil_settlement=False` with the concrete reason the operator path
  cannot be reproduced. Do not advertise `True` and leave the ticket pending.
- **Ember minimum (#5).** Type the redemption-minimum branch in
  `ember/deposit_redeem.py` exactly as the deposit `InsufficientAmount` branch
  is typed. This is a two-line change plus a test that asserts the type.
- **cSuperior (#6).** Keeping it synchronous is fine, but add a preflight that
  detects the queued/withdrawal-pending state (read the queue or `maxRedeem`)
  and raises `VaultFlowUnavailable(decoded_error="WithdrawalPending")` **before**
  building the redeem transaction, so no revert is broadcast.
- **Accountable (#7).** Fix and test the **exact** Monad address. If it is
  unusable in current state, that is fine — but then that exact address must
  yield a typed refusal at a pinned block, and the test must use that address.
- **Lagoon Base (#10–12).** The `settleDeposit()` allowance revert must be made
  impossible from the caller's view: provision the approval on the fork so the
  lifecycle completes, or raise a typed unsupported result. A `broadcast_failed`
  row is an automatic fail.

## 6. Mandatory self-verification before requesting review

1. Run the section-1 matrix over the section-4 ids. Paste the per-vault result
   table into the PR.
2. For every `success (simulated)` reached by forced settlement, show
   `synthetic_assets_injected_raw` and that `status_after == claimable`.
3. For every typed refusal, show `decoded_error` and the populated structured
   fields.
4. Add one focused fork test **per exact `vault_id`** that asserts the required
   terminal result at a pinned block (I4). Same-ABI substitutes do not count.

## 7. Definition of done

- [ ] Matrix result table for all section-4 ids pasted in the PR.
- [ ] Zero `transaction_reverted` / `broadcast_failed` / `execution_failed` /
      `receipt_analysis_failed` rows among the listed ids.
- [ ] No `supports_anvil_settlement=True` on any driver that can leave a ticket
      pending (I2 assert present in `force_settle`).
- [ ] Every deferral is a typed pre-broadcast exception with a concrete reason
      (I3), never an on-chain revert.
- [ ] One fork test per exact `vault_id`, pinned block, asserting the required
      result (I4).
- [ ] Accepted-now vaults (Syntropia, cSigma USD, YieldNest) still reach their
      accepted result (no regression).
