# P14: Big blocks mode causes slow confirmations

## Severity: HIGH

## Problem

HyperEVM has a dual-block architecture. When "big blocks" mode is enabled
for an address, transactions go to the large block mempool with ~1 minute
confirmation times instead of ~1 second. If big blocks were left enabled
from a previous deployment or manual testing, all strategy transactions
would be silently slow, causing timeouts in escrow waits and settlement.

This is not detectable from transaction behaviour alone -- the tx succeeds
but takes much longer than expected.

## Fix

Added a check in `HypercoreVaultRouting.__init__()` that queries the
deployer's big blocks status via `fetch_using_big_blocks()`. If enabled,
raises `AssertionError` immediately at startup rather than failing
mysteriously during trading.

The check is wrapped in a try/except to gracefully handle environments
where the RPC method is unavailable (Anvil forks, non-HyperEVM chains).
In simulate mode the check is skipped entirely.

## Files modified

- `tradeexecutor/ethereum/vault/hypercore_routing.py` -- `__init__()` big blocks assertion
