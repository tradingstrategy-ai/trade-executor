# P12: Precompile reads return stale data within same block

## Severity: LOW

## Status: COMMENT ONLY (non-issue in practice)

## Problem

HyperEVM precompiles (e.g. `coreUserExists` at `0x...0810`) return stale
data when called in the same block as a CoreWriter action. If
`is_account_activated()` were called in the same transaction as
`depositFor`, it might return `False` even though the account was just
created.

## Decision

Not a practical issue because the trade executor always waits for the
activation transaction receipt before calling `is_account_activated()`.
Since they are in separate blocks, the precompile returns fresh data.

Docstring note added to `is_account_activated()` in `evm_escrow.py:151-156`.
