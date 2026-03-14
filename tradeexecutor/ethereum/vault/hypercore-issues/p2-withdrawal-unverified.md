# P2: Withdrawal not verified on EVM

## Severity: CRITICAL

## Problem

In `hypercore_routing.py:_settle_withdrawal()`, the code assumes `executed_reserve = planned_reserve` with no verification that USDC actually arrived in the EVM Safe. The withdrawal multicall's three CoreWriter actions can fail silently on HyperCore:

- `vaultTransfer(withdraw)` fails due to lock-up
- `transferUsdClass` fails
- `spotSend` fails because bridge is dry

## Fix design

### Layer 1: Helper methods on HypercoreVaultRouting

1. **Exception**: `HypercoreWithdrawalVerificationError` in `hypercore_routing.py`
2. **`_fetch_safe_evm_usdc_balance()`**: Reads Safe's EVM USDC balance using ERC-20 `balanceOf`
3. **`_wait_for_usdc_arrival()`**: Polls Safe's USDC balance until expected increase observed
   - timeout=30s, poll_interval=2s, initial 2s delay
   - Raises `HypercoreWithdrawalVerificationError` on timeout

### Layer 2: Modified settlement

In `_settle_withdrawal()`:

1. Capture baseline USDC balance at start (before processing receipt)
2. After EVM receipt verified, poll for USDC arrival via `_wait_for_usdc_arrival()`
3. Use actual verified balance increase as `executed_reserve`
4. On timeout: `report_failure()` -- trade marked FAILED
5. In simulate mode: skip balance verification (mock CoreWriter doesn't bridge)

### Partial amounts

- `increase >= expected_increase_raw` threshold means any amount >= expected passes
- Partial arrival (less than expected) causes timeout → trade marked FAILED
- This is correct: partial withdrawal means a CoreWriter action failed

### Tests

New file: `tests/hyperliquid/test_hypercore_withdrawal_verification.py`

1. `test_withdrawal_verification_success` -- balance increases after polling
2. `test_withdrawal_verification_timeout_marks_trade_failed` -- balance never increases
3. `test_withdrawal_verification_polls_until_arrival` -- takes multiple polls
4. `test_withdrawal_uses_actual_balance_increase` -- uses real amount not planned
5. `test_withdrawal_evm_receipt_failure_skips_verification` -- receipt fails → no polling

## Files to modify

- `tradeexecutor/ethereum/vault/hypercore_routing.py` -- exception + helpers + modified settlement
- `tests/hyperliquid/test_hypercore_withdrawal_verification.py` -- new test file
