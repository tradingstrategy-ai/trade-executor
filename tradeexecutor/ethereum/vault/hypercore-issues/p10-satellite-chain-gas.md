# P10: Satellite chain HYPE gas not checked by preflight

## Severity: LOW

## Status: COMMENT ONLY

## Problem

The primary-chain gas preflight check (`EthereumExecution.preflight_check`)
verifies the deployer has enough native token for gas on the primary chain.
In multichain setups, this does NOT cover satellite chains like HyperEVM.
If the deployer runs out of HYPE on HyperEVM, multicall transactions fail
with out-of-gas errors.

## Decision

Not fixed with code. The failure is caught by the normal trade failure
path (transaction reverts), but the error message does not specifically
diagnose insufficient HYPE. A dedicated satellite chain gas check could
be added later if this becomes a recurring issue.

Comment added at `hypercore_routing.py:535-540`.
