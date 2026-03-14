# P15: Make escrow wait more robust

## Severity: MEDIUM

## Problem

The `wait_for_evm_escrow_clear()` function only checks that the
`evmEscrows` field is empty. This is necessary but not sufficient:
it's possible (in edge cases) for the escrow to disappear without
USDC actually arriving in the user's HyperCore spot balance — e.g.
if the bridge action silently fails.

Relying solely on escrow emptiness means phase 2 could fire when
the user has no USDC in spot, causing silent failures.

## Fix

Enhanced `wait_for_evm_escrow_clear()` with an optional `expected_usdc`
parameter. When provided:

1. Captures baseline USDC spot balance before the wait loop.
2. After escrows clear, checks that spot USDC increased by at least
   the expected amount (with 1% tolerance for fees/rounding).
3. Logs a warning if the increase is insufficient (possible silent
   bridge failure).

The check is non-fatal (warning, not error) because:
- The escrow clearing is still the primary signal
- Phase 2 deposit verification (P1) provides the ultimate safety net
- A strict failure here would block recovery of partially-bridged USDC

The caller in `_settle_deposit()` now passes `expected_usdc` to the
escrow wait, enabling dual verification (escrow clear + spot balance).

## Files modified

- `deps/web3-ethereum-defi/eth_defi/hyperliquid/evm_escrow.py` — `wait_for_evm_escrow_clear()` enhanced with `expected_usdc` parameter, added `_get_usdc_spot_balance()` helper
- `tradeexecutor/ethereum/vault/hypercore_routing.py` — `_settle_deposit()` passes `expected_usdc` to escrow wait

## Test coverage

- `tests/hyperliquid/test_hypercore_escrow_robust.py` — 5 tests covering `_get_usdc_spot_balance`, escrow wait with/without expected_usdc, and shortfall warning
