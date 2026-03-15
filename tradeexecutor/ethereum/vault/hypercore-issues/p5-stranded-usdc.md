# P5: USDC stranded in HyperCore spot after phase 1 success

## Severity: HIGH

## Problem

When phase 1 (bridge USDC to HyperCore spot) succeeds but phase 2 fails,
USDC is stranded in HyperCore spot. The trade is marked as failed, but
there's no record of where the USDC is or how to recover it.

## Fix

Store stranded USDC info in `trade.other_data["hypercore_stranded_usdc"]`
when phase 2 fails after phase 1 succeeded. This includes:
- Amount in raw USDC
- Location (HyperCore spot)
- Safe address
- Human-readable recovery instructions

The P1 fix (deposit verification) also marks the trade as failed when
the vault deposit is silently rejected, which catches the case where
phase 2 EVM tx succeeds but HyperCore silently rejects it. Both failure
paths now record stranded USDC info.

## Files modified

- `tradeexecutor/ethereum/vault/hypercore_routing.py` - `_settle_deposit()`
