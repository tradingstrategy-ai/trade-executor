# P16: Phase-1 perp wait crashes on vault leader performance fee

## Severity: CRITICAL

## Problem

On 2026-06-13 HyperAI crashed during a sequential rebalance on trade #1022, IKAGI-USDC:

```
ExecutionHaltableIssue: Sequential trade execution stopped after trade failure.
Failed phase: phase1_perp_wait
Perp withdrawable USDC did not reach 45.28023341 ... Current balance: 42.64074,
tolerance: 0.45232559 ... The vaultTransfer action may have failed silently on HyperCore.
```

The `vaultTransfer(vault->perp)` phase-1 action **succeeded**. It was not a silent no-op:

- vault equity dropped 837.85 → 794.98 USDC (a real ~42.86 USDC redemption)
- perp withdrawable rose to 42.64 USDC

But `_wait_for_perp_withdrawable_balance()` expected the *gross* planned reserve
(45.23 USDC) to arrive in perp, with only a 1% relative/slippage tolerance
(0.45 USDC). HyperCore had deducted the **vault leader performance fee** from the
redeemed profit (IKAGI: PnL 544 USDC, leader fraction 0.36), so the net perp
arrival was ~5-6% below the gross request — far more than 1%.

A successful but fee-reduced withdrawal was treated as a failure, `report_failure()`
halted the sequential executor, and the main loop died with `LiveSchedulingTaskFailed`.

### Why earlier fixes did not catch it

- The partial-sell accounting fix (`7e673dc6`, PR for the IKAGI gross-vs-net case)
  only widened the phase-1 tolerance by the **trade slippage tolerance** (1%),
  which is far smaller than the performance fee (~5-6%).
- The `_is_withdrawal_already_reflected_in_vault_equity()` fallback used the
  fixed 1% `HYPERCORE_RELATIVE_BALANCE_TOLERANCE`, so it also rejected the
  genuinely-settled withdrawal (equity decrease 42.86 < gross request − 1%).
- `_get_phase1_noop_retry_raw()` returns `None` when fresh equity (794.98) far
  exceeds the request (45.23) — correct for a partial sell of a large position,
  but it means no salvage path remained.
- The regression test for fee-shaped shortfalls passed an artificial 20%
  `relative_tolerance`, masking the fact that live trades carry only ~1% slippage.

## Fix design

The HyperCore leader performance fee may be deducted before withdrawal (already
reflected in vault equity) or on withdrawal, so in the worst case the net perp
arrival equals `gross_request − performance_fee_rate * gross_request`. Use that
worst-case fee as the **maximum acceptable phase-1 shortfall**.

1. **`HYPERCORE_DEFAULT_PERFORMANCE_FEE = Decimal("0.10")`** — assumed when the
   vault does not report a fee.
2. **`_fetch_vault_performance_fee(vault_address)`** — reads
   `VaultInfo.commission_rate` (the `vaultDetails` `leaderCommission` field) via
   `HyperliquidVault.fetch_info()`; falls back to the 10% default when the field
   is `None`/zero or the API read fails.
3. In `_settle_withdrawal()`, compute the worst-case fee with
   `estimate_max_withdrawal_commission(gross_request, performance_fee_rate)` and
   pass it as `performance_fee_tolerance` to:
   - `_wait_for_perp_withdrawable_balance()` (both the first attempt and the
     phase-1 retry), and
   - `_is_withdrawal_already_reflected_in_vault_equity()`.
   Both helpers widen their accepted tolerance to the performance fee when it is
   larger than the relative tolerance.
4. **Logging**: log the resolved performance fee rate and worst-case tolerance
   at settlement, and log an explicit "performance fee deduction observed"
   message when the accepted perp arrival falls short of the gross request by
   more than ordinary drift. The diagnostics dump also prints the vault
   commission rate.

## Files to modify

- `tradeexecutor/ethereum/vault/hypercore_routing.py` — constant, helper,
  widened tolerances, settlement wiring, logging, diagnostics line.
- `tests/hyperliquid/test_hypercore_dual_chain.py` — regression tests:
  - `test_phase1_perp_wait_accepts_performance_fee_shortfall`
  - `test_withdrawal_already_reflected_accepts_performance_fee_reduced_equity_decrease`
  - `test_fetch_vault_performance_fee_uses_vault_data_or_default`
- `tests/hypercore_writer/test_hyper_ai_live_loop.py` — monkeypatch signature
  updated for the new `performance_fee_tolerance` argument.
