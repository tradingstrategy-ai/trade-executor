# P9: HyperCore-to-EVM bridge can run dry

## Severity: LOW

## Status: COMMENT ONLY

## Problem

The `spotSend` action bridges USDC from HyperCore back to HyperEVM. If
more USDC exists on HyperCore than in the bridge contract on HyperEVM,
`spotSend` fails silently -- the EVM transaction succeeds but no USDC
is transferred.

## Decision

We cannot affect the bridge liquidity. The P2 withdrawal verification
(`_wait_for_usdc_arrival`) will detect this as a timeout and mark the
trade as failed. The error message references P9 to help operators
diagnose the root cause.

Referenced in `hypercore_routing.py:367-370` (withdrawal verification
error message).
