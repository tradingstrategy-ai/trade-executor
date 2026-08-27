# HyperCore same-cycle two-stage full close

## Status

Implemented on 2026-08-27.

This supersedes the next-cycle-only handling in
`docs/plans/hypercore-full-close-residual-cleanup.md` for ordinary strategy full
closes. The existing later-cycle retry remains the fallback when one additional
protected withdrawal cannot safely reduce the residual to 5 USD or less.

## Goal

When a HyperCore strategy target is zero:

1. Preserve full-close intent from the alpha signal through settlement.
2. Withdraw the amount that can safely fund the current cycle.
3. Read the remaining vault equity after the first withdrawal is verified.
4. If one more protected withdrawal can leave at most 5 USD, execute it before
   the shared downstream sweep.
5. Credit only USDC observed back in the HyperEVM Safe and close the ledger
   position only from a verified final residual.

This is a best-effort live target, not an unconditional guarantee. Lockups,
limited `max_withdrawable`, NAV movement, unavailable API data, or an ambiguous
HyperCore action can prevent a same-cycle close.

## Architecture decision

Use one `TradeExecution`, at most two `vaultTransfer(vault -> perp)`
transactions, and one combined `perp -> spot -> HyperEVM` sweep.

A second trade or cleaning job is unnecessary for the eligible happy path. The
residual is not known until the first transaction settles, while HyperCore
routing already performs additional verified settlement transactions during
sequential execution. The later-cycle cleaner remains useful only as fallback.

No change is needed in `strategies/hyper-ai.py`. Generic alpha-model and
HyperCore routing code own these semantics.

## Implementation

### Preserve full-close intent

`TradeFlag.close` is the durable close signal across the stack.

For a cap-bound zero-target HyperCore signal:

- The alpha signal retains the conservative redemption amount for thresholds
  and same-cycle cash planning.
- `PositionManager.close_position()` creates one trade for the full available
  position quantity with `TradeFlag.close`, `TradeFlag.reduce`, and
  `trade.closing == True`.
- The conservative first-stage value is stored as
  `hypercore_first_stage_conservative_reserve_usd`.
- Cash sufficiency checks use that conservative value rather than treating the
  trade's full marked value as immediately spendable proceeds.

An exact zero weight is a full close even when
`close_position_weight_epsilon == 0`.

### Execute the protected residual leg

Routing limits the first physical withdrawal to the smaller of:

- fresh live equity minus the normal withdrawal safety margin; and
- `hypercore_first_stage_conservative_reserve_usd`, when present.

After the first perp increase is verified, an explicit close obtains fresh,
uncached vault equity and fresh `max_withdrawable`. If the residual is already
5 USD or less, no second transaction is needed.

Otherwise routing applies the same safety rule to the second request:

```text
margin(amount) = max(0.5% × amount, 1.50 USD)
```

It broadcasts only when the protected, capacity-capped request can leave at
most 5 USD after raw USDC rounding. A residual above 1,000 USD cannot meet this
condition with the current percentage margin and therefore waits for another
cycle.

If the first withdrawal needed its existing silent-no-op retry, that retry has
already consumed the cycle's second vault transaction. The intentional
residual leg is then skipped.

### Persist and verify the second transaction

Before broadcasting the optional transaction, routing:

1. appends the signed transaction to the original trade;
2. records `hypercore_second_stage_status = "broadcast_pending"` and its hash;
3. invokes the routing state checkpoint supplied by live execution.

The transaction must then have both:

- a verified increase relative to the post-first-stage perp balance, allowing
  for the vault's performance fee; and
- a fresh vault-equity decrease relative to the equity immediately before the
  second leg.

Successful first and second withdrawals are aggregated before the existing
phase-2 and phase-3 settlement code runs once.

## Failure handling

The second transaction is an optimisation after a verified first withdrawal:

| Outcome | Action |
| --- | --- |
| Preparation fails before broadcast | Skip the second leg and settle the verified first leg. |
| Receipt has `status = 0` | Treat it as a definite revert and settle the verified first leg. |
| Broadcast or confirmation outcome is unknown | Mark the combined amount stranded, fail the trade, and do not sweep shared perp USDC. |
| Receipt has `status = 1` but balance movement is not verified | Treat it as ambiguous, fail the trade, and do not sweep shared perp USDC. |
| Final vault equity is unavailable | Fail the explicit close instead of inferring a residual or writing it off. |

Ambiguous outcomes use the existing `hypercore_stranded_usdc` repair guard.
The normal repair command must not create a synthetic counter-trade or unfreeze
the position. An operator resolves the persisted hash and live vault, perp,
spot, and Safe balances with `correct-accounts` before adjusting accounting.
Automatic rebroadcast of an unresolved HyperCore action is outside this
implementation.

## Accounting

Cash and position quantity have different sources of truth:

- Reserve credit is the combined USDC actually observed in the HyperEVM Safe.
- Planned close value uses the latest valued share price because HyperCore's
  execution pricing is deliberately fixed at 1:1 USDC; executed price still
  comes from observed settlement proceeds.
- A partial close composes the independently verified equity-reduction
  fraction of each completed vault leg. This prevents NAV movement between
  legs from appearing as sold shares.
- A verified final residual of 5 USD or less books the full planned quantity
  and records the residual as an accepted write-off.
- A verified residual above 5 USD books only the proven reduction and leaves
  the position open with retry metadata.
- Executed quantity must never exceed the full-position planned quantity.

## Diagnostics

The trade records:

- conservative and requested first-stage amounts;
- fresh first-stage residual;
- second-stage status, requested amount, transaction hash, error, and final
  residual;
- total requested gross amount and total observed perp increase; and
- accepted residual or stranded-capital metadata.

## Verification

Focused tests cover:

1. exact-zero alpha signal to full-close trade propagation;
2. conservative cash and first-stage routing caps;
3. accepted and capacity-limited residual planning;
4. successful two-stage aggregation into one downstream sweep;
5. ambiguous second-stage settlement failing before phase 2;
6. unavailable final equity failing an explicit close;
7. partial-quantity composition across NAV movement;
8. simulated/mainnet-fork full closes without live API residuals; and
9. Hyper AI live-loop close, partial reduction, and fee-slippage regressions.

The earlier Claude Fable plan review agreed with the one-trade, two-vault-leg
architecture. Its safety findings are reflected in the pre-broadcast
checkpoint, ambiguity guard, final-equity requirement, performance-fee-aware
verification, and per-leg quantity accounting.
