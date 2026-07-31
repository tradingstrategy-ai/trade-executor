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

## Result comparison

| Result | Corrective rerun | PR #1588 rerun | Change |
|---|---:|---:|---:|
| Ordinary simulated success | 82 | 55 | +27 |
| Successful deposit with a closed protocol phase | 4 | 0 | +4 |
| Simulated success with disclosed liquidity intervention | 2 | 0 | +2 |
| Genuine account whitelisting needed | 14 | 14 | 0 |
| Deposit permission unknown | 23 | 0 | +23 |
| Incompatible deposit asset | 1 | 1 | 0 |
| Redemption liquidity unavailable | 1 | 0 | +1 |
| Redemption not yet matured | 1 | 0 | +1 |
| Async simulation unsupported | 1 | 5 | -4 |
| Below minimum | 0 | 2 | -2 |
| Raw transaction/execution/receipt failures | 0 | 2 | -2 |

The new `deposit_permission_unknown` result is deliberate fail-closed
reporting. PR #1588 attempted these vaults without a contract-backed permission
classification, so it did not expose this category separately.

## Headline protocol checks

- HYPE++ is permissionless and currently has a delayed funding phase. Its final
  result is `success-deposit-closed`, opening in 56.3 days at the pinned block.
- Canonical Morpho and Euler metadata is no longer reported as whitelisted.
  All four Euler rows succeeded; Morpho produced nine ordinary successes and
  two disclosed intervention successes (Apyx and Saturn).
- The six USDC-denominated IPOR Fusion rows remain genuine
  `whitelisting-needed` results. The separate TESS USDT/sUSDS exact-vault check
  also remains gated by IPOR's selector role.
- Sentora USD Earn is correctly `incompatible_deposit_asset`: it accepts RLUSD,
  not the matrix's native-USDC deposit asset.
- Arche USD no longer fails receipt analysis. Its redemption mined but returned
  zero USDC, now reported as `redemption_liquidity_unavailable`.
- YieldNest RWA MAX is correctly `redemption_not_yet_matured` until
  2026-10-15.

## Remaining work by adapter

| Adapter/protocol | Affected vaults | Current result | Required fix or action |
|---|---|---|---|
| 40acres | Aerodrome USDC; Blackhole USDC; Pharaoh USDC | `deposit_permission_unknown` | Implement a contract-backed account permission accessor for the 40acres adapter; retain fail-closed behaviour until its gate semantics are proven. |
| Accountable | Hyperithm Delta Neutral Vault | `deposit_permission_unknown` | Implement Accountable account-policy discovery. The experiment already uses 1,001 USDC, so this is no longer the old 1 USDC minimum issue. |
| Ember | Apollo ACRED; Earn; Polymarket; Third Eye; UDL | `deposit_permission_unknown` | Implement Ember account-policy discovery before attempting its queue and minimum checks. |
| Plutus | Dolomite vault; Hedge Token | `deposit_permission_unknown` | Implement contract-backed Plutus permission discovery. |
| Yearn Compounder | Moonwell WETH/cbBTC borrowers; Morpho Hyperithm; Morpho V2 Sentora PYUSD; two USD3 Pendle PT Maxi vaults; USDC-to-SKY; sUSDS Pendle PT Maxi; yPT-aUSDC | `deposit_permission_unknown` | Add a permission accessor for `YearnCompounderVault`, using its actual deposit policy rather than inheriting the V3 vault assumption. |
| Legacy Yearn | StrategyGearboxLenderUSDC | `deposit_permission_unknown` | Identify the exact implementation and policy. Its V3 `deposit_limit_module()` selector reverts, so the adapter now fails closed instead of leaking an execution error. |
| cSigma | cSigma USD; cSuperior Quality Private Credit USDC | `deposit_permission_unknown` | Implement contract-backed cSigma permission discovery. |
| IPOR Fusion | Six USDC vaults | `whitelisting-needed` | Add the simulated Safe to each IPOR AccessManager selector role, or use an authorised test identity. No code classification fix is needed. |
| Lagoon Finance | AltaETF; Block4Block; Der USDC; Der base USDC; Muchacho USDC; Noon STS USDC; Strada USDC; pyUSDC | `whitelisting-needed` | Add the simulated Safe to each vault's investor allow-list. Expanding only the outer Lagoon Guard whitelist is insufficient. |
| Upshift | Sentora USD Earn | `incompatible_deposit_asset` | Run a dedicated lifecycle with RLUSD, one of the vault's accepted assets. Keep native USDC classified as incompatible. |
| Yearn | Arche USD | `redemption_liquidity_unavailable` | Add safe Yearn withdrawal-queue/liquidity preparation for simulation, or wait for immediate underlying liquidity. Preserve the typed zero-output outcome in production. |
| YieldNest | YieldNest RWA MAX | `redemption_not_yet_matured` | Re-run at or after 2026-10-15 and prove the post-maturity redemption path. |
| Lagoon Finance | For Yield v2 | `simulation_unsupported_async` | Implement and test the exact satellite operator/asset-manager settlement path so forced Anvil settlement reaches `claimable`. |

The full report contains zero `transaction_reverted`, `broadcast_failed`,
`execution_failed` or `receipt_analysis_failed` rows. The HYPE++ amendment also
contains no raw failure.
