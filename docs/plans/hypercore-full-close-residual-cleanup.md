# HyperCore full-close residual handling

## Status

Superseded on 2026-08-27 by
`docs/plans/hypercore-same-cycle-two-stage-full-close.md` for ordinary strategy
full closes. The next-cycle retry described here remains the fallback when a
second protected withdrawal cannot safely reach 5 USD.

## Decision

Use the normal alpha cycle to retry incomplete HyperCore closes. An open
zero-target position is already durable retry state, so a separate scheduler
would duplicate transaction persistence, nonce serialisation, and unfinished
trade safeguards.

## Behaviour

- A full-close intent is an existing position with a reducing signal and a
  target below `close_position_weight_epsilon`.
- Full closes bypass only the portfolio and individual rebalance softbands.
  Problematic pairs, frozen positions, lockups, pending settlement, redemption
  limits, quantity dust, and routing preflight checks still apply.
- If the whole rebalance is below the portfolio threshold, only full-close
  intents are emitted; unrelated small adjustments remain skipped.
- With `cap_buys_to_sync_cash` enabled, a softband-bypassed close contributes
  no assumed cash to same-cycle buys. Only settled USDC is available on a later
  cycle. Strategies without that optional cash guard retain their existing
  marked-value financing behaviour.
- HyperCore's accepted residual is exactly 5 USD. This is separate from the
  2 USD transaction-dust threshold and is always converted to vault shares
  before a quantity comparison.

## Residual lifecycle

| Observation after a closing withdrawal | State action |
| --- | --- |
| Verified residual `<= 5 USD` | Close the ledger position, credit only received USDC, and record accepted-residual metadata. |
| Verified residual `> 5 USD` | Attempt one eligible protected residual withdrawal before the shared downstream sweep; if it cannot safely reach 5 USD, book only proven redemption and retry on the next normal alpha cycle. |
| Residual unavailable | An explicit full close fails closed for live reconciliation; do not infer that the physical residual is within the accepted boundary. |

The 5 USD boundary applies only to a verified close residual in state
settlement. Ordinary close planning and fresh positions retain the narrower
2 USD transaction-dust threshold. `correct-accounts` ignores any untracked
HyperCore vault balance at or below 5 USD, because it cannot safely distinguish
an external micro-deposit from an account-scoped retained residual.

Reattaching residual equity to a later new deposit remains deliberate operator
recovery work; this change does not invent a synthetic reserve credit or balance
update for it.

## Operations

No recurring cleanup job or external cron process is configured. The existing
manual `correct-accounts` and HyperCore dust-repair commands remain recovery
tools and must be run with the live executor stopped.

Escalate after three successful close attempts still verify more than 5 USD
remaining. Lockups, unavailable redemption capacity, pending settlement, and
unroutable pairs are deferrals, not write-off conditions.

## Verification

- Full-close softband bypass and non-bypass partial reduction tests.
- Same-cycle cash-cap test proving bypassed proceeds do not fund buys.
- USD-to-share dust conversion tests at non-unit share prices.
- Verified 5 USD residual versus fresh small-deposit tests.
- Correct-accounts regression test preventing residual recreation.
