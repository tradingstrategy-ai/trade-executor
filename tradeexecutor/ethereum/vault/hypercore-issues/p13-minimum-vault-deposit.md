# P13: Minimum vault deposit silently rejected

## Severity: LOW

## Status: COMMENT ONLY (non-issue)

## Problem

HyperCore silently rejects vault deposits below 5 USDC (5,000,000 raw).
If such a deposit were attempted, the EVM transaction would succeed but
the vault equity would never increase.

## Decision

Already handled at the encoding level. `encode_vault_deposit()` in
`eth_defi/hyperliquid/core_writer.py` has an assertion that rejects
amounts below the minimum at transaction build time, before any on-chain
interaction.

Comment added at `hypercore_routing.py:442-446`.
