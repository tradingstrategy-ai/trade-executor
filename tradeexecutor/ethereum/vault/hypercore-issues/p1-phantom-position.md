# P1: Phase 2 silent failure creates phantom vault position

## Severity: CRITICAL

## Problem

In `hypercore_routing.py:_settle_deposit()`, after phase 2 succeeds on HyperEVM, the code queries `fetch_user_vault_equity()`. CoreWriter actions are NOT atomic -- they can succeed on EVM but fail silently on HyperCore.

If the vault deposit was silently rejected on HyperCore:
1. `fetch_user_vault_equity()` returns `None`
2. The fallback sets `executed_amount = actual_deposit_human`
3. Trade is marked SUCCESS with phantom equity
4. USDC is stranded in HyperCore spot/perp

## Fix design

### Layer 1 (eth_defi): New polling function

Add to `eth_defi/hyperliquid/api.py`:

1. **Exception**: `HypercoreDepositVerificationError` -- raised when deposit cannot be verified
2. **Function**: `wait_for_vault_deposit_confirmation()` -- polls `fetch_user_vault_equity()` until equity appears/increases

Poll loop (modelled on `wait_for_evm_escrow_clear`):
- timeout=60s, poll_interval=2s, initial 2s delay
- bypass_cache=True on every poll
- Handles two cases:
  - **New position**: waits for any equity > 0
  - **Existing position**: waits for equity to increase by expected_deposit - tolerance

### Layer 2 (tradeexecutor): Modified settlement

In `hypercore_routing.py:_settle_deposit()`:

1. Before phase 2: snapshot existing vault equity
2. Replace single-shot equity query with `wait_for_vault_deposit_confirmation()`
3. On `HypercoreDepositVerificationError`: `report_failure()` -- trade marked FAILED

### Tests

New file: `tests/hyperliquid/test_hypercore_deposit_verification.py`

1. `test_vault_deposit_verification_succeeds_new_position` -- mock returns None then equity
2. `test_vault_deposit_verification_succeeds_existing_position` -- equity increases
3. `test_vault_deposit_verification_timeout_raises_error` -- always returns None
4. `test_settle_deposit_marks_failed_on_silent_rejection` -- integration test
5. `test_settle_deposit_succeeds_with_verified_equity` -- happy path

## Files to modify

- `deps/web3-ethereum-defi/eth_defi/hyperliquid/api.py` -- exception + poll function
- `tradeexecutor/ethereum/vault/hypercore_routing.py` -- modify `_settle_deposit()`
- `tests/hyperliquid/test_hypercore_deposit_verification.py` -- new test file
