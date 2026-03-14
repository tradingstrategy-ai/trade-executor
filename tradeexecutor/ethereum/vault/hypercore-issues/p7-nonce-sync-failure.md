# P7: RPC failure during nonce sync between deposit phases

## Severity: LOW

## Status: COMMENT ONLY

## Problem

Between phase 1 (bridge USDC to HyperCore spot) and phase 2
(transferUsdClass + vaultTransfer), the deployer nonce is synced via
`self.deployer.sync_nonce(web3)`. If the RPC connection fails at this
exact moment, USDC is stranded in HyperCore spot with no phase 2 to
move it into the vault.

## Decision

Not fixed with code. The window for this failure is very small (a single
RPC call between two phases), and the recovery path is manual: use
`check-hypercore-user.py` to verify spot balance, then either retry the
vault deposit or bridge USDC back to EVM via `spotSend`.

Comment added at `hypercore_routing.py:812-814`.
