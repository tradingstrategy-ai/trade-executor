# P11: Hyperliquid API failure crashes tick cycle

## Severity: LOW

## Status: COMMENT ONLY (intentional behaviour)

## Problem

If the Hyperliquid info API is down during valuation, the
`HypercoreVaultValuator` raises an exception that crashes the entire
tick cycle.

## Decision

This is intentional. If the API is down, we cannot value the vault
position and must halt rather than use stale or incorrect data. Using
a cached or fallback price would risk making trading decisions based on
wrong valuations.

Comment added at `hypercore_valuation.py:132-133`.
