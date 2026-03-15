# P8: Activation cost persistence and single-deduction

## Severity: MEDIUM

## Problem

Two issues with activation cost handling:

1. **Persistence**: When a Safe is not yet activated on HyperCore,
   `setup_trades()` performs activation and stores the cost (2 USDC) in
   `self._activation_cost_raw`. If the executor crashes between setup and
   settlement, the cost is lost. On restart, `_settle_deposit()` would try
   to deposit the full amount (including the 2 USDC already consumed).

2. **Double-deduction**: `_settle_deposit()` previously read
   `self._activation_cost_raw` for every trade. If two buys were in the
   same cycle, both would have the activation cost deducted — the second
   buy would settle as 48 USDC instead of 50.

## Fix

`setup_trades()` stores the activation cost in `trade.other_data` only
for the first buy trade of the cycle:

- `trade.other_data["hypercore_activation_cost_raw"]` is set only on the
  first buy (via `activation_cost_applied` flag).
- `_settle_deposit()` reads the cost exclusively from `trade.other_data`,
  not from `self._activation_cost_raw`. A second buy has no key in
  `other_data`, so activation_cost = 0 — correct.
- `self._activation_cost_raw` is reset to 0 at the start of each
  `setup_trades()` call to prevent stale leakage between cycles.

## Files modified

- `tradeexecutor/ethereum/vault/hypercore_routing.py` — `setup_trades()` resets and stores per-trade; `_settle_deposit()` reads from `trade.other_data` only

## Test coverage

- `tests/hyperliquid/test_hypercore_activation_cost.py` — 4 tests covering setup deduction, cycle reset, sell exclusion, and settlement per-trade cost
