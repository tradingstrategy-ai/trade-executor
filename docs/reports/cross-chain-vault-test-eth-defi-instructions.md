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
  --auto-simulated --settle-async-on-anvil \
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

All signatures are from the **authoritative full 129-vault matrix run on
2026-07-25 against eth-defi `38fa4f945` (#1374 merged)**, trade-executor
`71607299`. #1374 genuinely closed a lot (52 success, up from 43; Lagoon 6→1
gaps; Accountable Monad and Gains-Arbitrum now typed). Nine eth-defi items
remain, and **six of them are the same defect: `supports_anvil_settlement=True`
on a driver that leaves the ticket `pending` (invariant I2).**

### 4a. MUST-FIX (eth-defi) — still failing at #1374

| # | vault_id | Vault | Current result @#1374 | Required result |
|--:|---|---|---|---|
| 1 | `1-0x9be9294722f8aad37b11a9792be2c782182cafa2` | Ember Earn | `simulation_unsupported_async` — force_settle runs, ticket stays `pending`, but capability advertises `True` | `success (simulated)` (driver actually settles, I2 assert passes) **or** set `supports_anvil_settlement=False` + typed reason |
| 2 | `1-0x0b9342c15143e8f54a83f887c280a922f4c48771` | Ember Polymarket | same capability-lie | same as #1 |
| 3 | `1-0xf3190a3ecc109f88e7947b849b281918c798a0c4` | Ember Third Eye | same capability-lie | same as #1 |
| 4 | `1-0x373152feef81cc59502da2c8de877b3d5ae2e342` | Ember UDL | same capability-lie | same as #1 |
| 5 | `8453-0xad20523a7dc37babc1cc74897e4977232b3d02e5` | Gains gTrade (Base) | `simulation_unsupported_async` — satellite force_settle leaves `pending`, capability `True` | same as #1 |
| 6 | `8453-0x4efc07dca8697792119484af33549f33ab11bf3c` | Lagoon MoneyFi FlowForge (Base) | `simulation_unsupported_async` — satellite force_settle leaves `pending` | same as #1 |
| 7 | `1-0x2b13311fd553e74b421d4ccc96e348f71e179dcf` | Ember Apollo ACRED | `execution_failed` — raw `ValueError` "shares 904 below minimum 9170000" | `below_minimum` (typed `VaultFlowUnavailable`, `minimum_raw_amount`, `direction=redeem`) |
| 8 | `1-0x438982ea288763370946625fd76c2508ee1fb229` | cSuperior | `transaction_reverted` — on-chain `execution reverted: Withdrawal pending` | typed **preflight before broadcast** → `redemption_capacity_limited` / `redemption_paused`, **no revert** |
| 9 | `1-0x093272c07700d3ca5301c3bf9b3a392624179e2f` | Morpho Hyperithm USDC Degen | `transaction_reverted` — undecoded `custom error 0xace2a47e` on redeem | decode `0xace2a47e`, raise the matching typed result **before broadcast** |

### 4b. CONFIRM (typed, verify it is the intended terminal state)

| vault_id | Vault | Current result @#1374 | Action |
|---|---|---|---|
| `42161-0x58bfc95a864e18e8f3041d2fcd3418f48393fe6a` | Plutus Hedge (Arbitrum) | `redemption_unavailable` (typed) | Confirm this is correct for current state, or implement the request/claim flow and reach `success` |

### 4c. ACCEPTED after #1374 — do NOT regress

- `143-0x7cd231120a60f500887444a9baf5e1bd753a5e59` Accountable Hyperithm (Monad)
  → `below_minimum` ✓ (was raw `0x5945ea56` — now fixed on the exact address)
- `42161-0xd3443ee1e91af28e5fb858fbd0d72a63ba8046e0` Gains gTrade (Arbitrum)
  → `redemption_window_closed` ✓ (EndOfEpoch typed)
- Lagoon `1-0xd17049…` Syntropia, `1-0xa00f63…` Moon Digital,
  `42161-0x1723cb…` Angmar, `8453-0x2bff67…` For Yield v2,
  `8453-0x63b04d…` RB Capital, `8453-0xbe7db4…` TruMarket → `success`/closed ✓
  (5 of the original 6 Lagoon gaps closed by the Safe-provisioning change)
- `1-0xd5d097…` cSigma USD → `redemption_capacity_limited` ✓
- `1-0x01ba69…` YieldNest RWA MAX → `redemption_capacity_limited` ✓

### 4d. NOT eth-defi (do not touch here — tracked on the executor side)

- `8453-0xd6701905c59ee618dc36dc747506bce0a4ac760a` IPOR Autopilot (Base) —
  `transaction_reverted` satellite close; executor must surface the reason and
  reconcile.
- `43114-0x124d00b1ce4453ffc5a5f65ce83af13a7709bac7` 40acres Pharaoh (Avalanche)
  — `transaction_reverted` satellite share reconciliation (executor).
- `1-0x3cd3718f8f047aa32f775e2cb4245a164e1c99fb` Euler Hyperithm —
  `infrastructure_failed` (transient Anvil disconnect); rerun, not an adapter
  defect.
- `1-0x74ad2f789ed583dbd141bbdafc673fe1f033718b` Upshift Sentora — now
  `incompatible_deposit_asset` (accepts RLUSD/PYUSD/USDT, not USDC); executor
  concern, no eth-defi work unless USDC support is wanted.

## 5. Protocol-specific notes (only beyond the invariants)

- **The capability-lie group (#1–6): Ember Earn/Polymarket/Third Eye/UDL, Gains
  Base, Lagoon MoneyFi FlowForge.** This is the dominant remaining defect. For
  each, `force_settle()` runs but the ERC-7540/queue ticket is still `pending`
  afterwards, while `get_deposit_manager_capability().supports_anvil_settlement`
  returns `True`. Per I2 that is a defect. Do **one** of:
  1. make the driver actually advance the queue so `status_after == claimable`
     at the pinned fork block (impersonate the real operator/keeper, advance
     time/epoch, top up and disclose via `synthetic_assets_injected_raw`), so
     the mandatory I2 assert in `force_settle` passes and the vault reaches
     `success (simulated)`; **or**
  2. set `supports_anvil_settlement=False` and raise `UnsupportedVaultSimulation`
     with the concrete reason the operator path cannot be reproduced.

  Add the I2 self-assert to `force_settle` so this class of defect cannot
  regress silently again.
- **Ember redemption minimum (#7).** Type the redemption-minimum branch in
  `ember/deposit_redeem.py` (currently `raise ValueError(f"Ember redemption
  shares … below minimum …")`) exactly as the deposit `InsufficientAmount`
  branch is typed. Two-line change plus a test asserting the raised type and
  `minimum_raw_amount`.
- **cSuperior (#8).** Keeping it synchronous is fine (off-chain FIFO), but add a
  preflight that detects the queued/withdrawal-pending state (read the queue or
  `maxRedeem`) and raises
  `VaultFlowUnavailable(decoded_error="WithdrawalPending")` **before** building
  the redeem transaction, so no revert is broadcast.
- **Morpho Hyperithm USDC Degen (#9).** Redemption reverts with an undecoded
  `custom error 0xace2a47e`. Add the deployed ABI error, decode the selector,
  and raise the matching typed result (capacity / cooldown / window) before
  broadcast. An undecoded revert reaching the caller is an I1 violation.
- **Do not regress the §4c wins.** In particular the Lagoon Safe-provisioning
  path (Syntropia, Moon Digital, Angmar, For Yield v2, RB Capital, TruMarket)
  and the Accountable/Gains-Arbitrum typing must still hold in your run.

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
