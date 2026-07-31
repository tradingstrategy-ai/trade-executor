# Cross-chain vault test matrix — 2026-07-31 corrective rerun

This is the corrected run of the ordered 129-vault population from PR #1588.
The full run used trade-executor `99dfd2df`, eth-defi `576f5aeb9` and a
1,001 USDC amount. Its machine-readable output is
`cross-chain-vault-test-2026-07-31-corrective.report.json`.

The full run exposed one final simulation-order issue for HYPE++: public USDC
balance eligibility was checked before funding its Safe. Trade-executor
`f072f5a0` fixes that order. The exact-vault amendment in
`cross-chain-vault-test-2026-07-31-hype-amendment.report.json` changes only
HYPE++ from `below_minimum` to `success-deposit-closed`; the effective counts
below include that amendment.

The permission-default amendment then re-ran the 23 vaults previously reported
as `deposit_permission_unknown`, using trade-executor `51d8395f`, eth-defi
`9a50a0dc` and the same 1,001 USDC amount. Its machine-readable output is
`cross-chain-vault-test-2026-07-31-permission-defaults.report.json`. All 23 now
reach a concrete lifecycle outcome, so the effective counts below also include
this amendment.

## Result comparison

| Result | Corrective rerun | PR #1588 rerun | Change |
|---|---:|---:|---:|
| Ordinary simulated success | 88 | 55 | +33 |
| Successful deposit with a closed protocol phase | 18 | 0 | +18 |
| Simulated success with disclosed liquidity intervention | 2 | 0 | +2 |
| Genuine account whitelisting needed | 14 | 14 | 0 |
| Deposit permission unknown | 0 | 0 | 0 |
| Incompatible deposit asset | 1 | 1 | 0 |
| Redemption liquidity unavailable | 1 | 0 | +1 |
| Redemption not yet matured | 1 | 0 | +1 |
| Async simulation unsupported | 1 | 5 | -4 |
| Below minimum | 1 | 2 | -1 |
| Redemption capacity limited | 1 | 0 | +1 |
| Raw transaction/execution/receipt failures | 1 | 2 | -1 |

The intermediate `deposit_permission_unknown` result was deliberate fail-closed
reporting. The follow-up establishes public defaults for adapters whose
contracts expose no account-admission mechanism, while preserving explicit
contract-backed gates.

## Headline protocol checks

- HYPE++ is permissionless and currently has a delayed funding phase. Its final
  result is `success-deposit-closed`, opening in 56.3 days at the pinned block.
- Canonical Morpho and Euler metadata is no longer reported as whitelisted.
  All four Euler rows succeeded; Morpho produced nine ordinary successes and
  two disclosed intervention successes (Apyx and Saturn).
- The six USDC-denominated IPOR Fusion rows remain genuine
  `whitelisting-needed` results. The separate TESS USDT/sUSDS exact-vault check
  also remains gated by IPOR's selector role.
- Accountable exposes `permissionLevel()` with `None=0`, `KYC=1` and
  `Whitelist=2`. `allowed(address)` is the account predicate for Whitelist mode;
  KYC uses a signed per-call payload. Hyperithm was deployed with level 0 and is
  therefore permissionless. Its current `success-deposit-closed` result is an
  admission-capacity limit, not an identity gate.
- Sentora USD Earn is correctly `incompatible_deposit_asset`: it accepts RLUSD,
  not the matrix's native-USDC deposit asset.
- Arche USD no longer fails receipt analysis. Its redemption mined but returned
  zero USDC, now reported as `redemption_liquidity_unavailable`.
- YieldNest RWA MAX is correctly `redemption_not_yet_matured` until
  2026-10-15.

## Remaining work after permission-default amendment

| Adapter/protocol | Affected vaults | Current result | Required fix or action |
|---|---|---|---|
| 40acres | Aerodrome USDC | `transaction_reverted` | Diagnose its redemption: the vault reverts with `ERC20: transfer amount exceeds balance`. This is independent of deposit permission policy. |
| 40acres | Pharaoh USDC | `redemption_capacity_limited` | Add safe simulated redemption liquidity or retain this typed capacity outcome when the vault cannot immediately return underlying. |
| Ember | Apollo ACRED | `below_minimum` | Use a redemption quantity above Ember's share minimum, or retain the typed minimum outcome when a complete round trip cannot satisfy it. |
| IPOR Fusion | Six USDC vaults | `whitelisting-needed` | Add the simulated Safe to each IPOR AccessManager selector role, or use an authorised test identity. No code classification fix is needed. |
| Lagoon Finance | AltaETF; Block4Block; Der USDC; Der base USDC; Muchacho USDC; Noon STS USDC; Strada USDC; pyUSDC | `whitelisting-needed` | Add the simulated Safe to each vault's investor allow-list. Expanding only the outer Lagoon Guard whitelist is insufficient. |
| Upshift | Sentora USD Earn | `incompatible_deposit_asset` | Run a dedicated lifecycle with RLUSD, one of the vault's accepted assets. Keep native USDC classified as incompatible. |
| Yearn | Arche USD | `redemption_liquidity_unavailable` | Add safe Yearn withdrawal-queue/liquidity preparation for simulation, or wait for immediate underlying liquidity. Preserve the typed zero-output outcome in production. |
| YieldNest | YieldNest RWA MAX | `redemption_not_yet_matured` | Re-run at or after 2026-10-15 and prove the post-maturity redemption path. |
| Lagoon Finance | For Yield v2 | `simulation_unsupported_async` | Implement and test the exact satellite operator/asset-manager settlement path so forced Anvil settlement reaches `claimable`. |

The original full report contains zero `transaction_reverted`,
`broadcast_failed`, `execution_failed` or `receipt_analysis_failed` rows. The
permission-default amendment exposes one independent 40acres redemption revert
that was previously hidden behind the fail-closed permission classification.
