# HyperCore full-close residual handling

## Status

Implemented on 2026-08-24.

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
| Verified residual `> 5 USD` | Book only proven redemption, keep the position open, and retry on the next normal alpha cycle. |
| Residual unavailable | Preserve the normal successful-close result. The post-withdrawal equity read is diagnostic and must not turn a verified EVM withdrawal into a phantom open position. |

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
