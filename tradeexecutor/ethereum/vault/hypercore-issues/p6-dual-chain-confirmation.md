# P6: Confirm both HyperCore and HyperEVM updated after trades

## Severity: HIGH

## Problem

After any Hypercore vault trade (deposit or withdrawal), only one chain's
state was verified. A deposit only checked the EVM receipt; a withdrawal only
checked the EVM receipt. Because CoreWriter actions can fail silently on
HyperCore, the other chain's state could be inconsistent.

For deposits: the EVM tx succeeding only means the action was *queued*; the
vault equity on HyperCore might never appear.

For withdrawals: the EVM tx succeeding only means the multicall ran; but
USDC might not arrive back in the Safe's EVM balance if the HyperCore-to-EVM
bridge is delayed or one of the CoreWriter actions failed silently.

## Fix

### Deposits (P1 provides most of the dual-chain verification)

The P1 fix already verifies both chains:
- EVM receipt is checked (phase 1 and phase 2)
- HyperCore vault equity is polled via `wait_for_vault_deposit_confirmation()`

### Withdrawals (P2 + P6)

The P2 fix polls the Safe's EVM USDC balance to verify USDC arrived.
The P6 addition queries `fetch_user_vault_equity()` after the USDC arrives
to confirm vault equity decreased on HyperCore. This is a non-fatal check
(logged as warning on failure) since the EVM verification is the primary
confirmation.

## Files modified

- `tradeexecutor/ethereum/vault/hypercore_routing.py` - `_settle_withdrawal()`: added HyperCore equity check after EVM USDC verification
