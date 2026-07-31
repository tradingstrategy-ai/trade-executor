# Cross-chain vault test matrix — 2026-07-31

This is a full rerun of the ordered 129-vault matrix from PR #1588, using
production vault JSON data and trade-executor `d3e2a268` with eth-defi master
`3636729f`. The report JSON is
`cross-chain-vault-test-2026-07-31.report.json`. The command recovered from 13
transient Anvil/RPC generations; those are infrastructure retries, not vault
outcomes.

## Result comparison

| Result | 2026-07-31 | 2026-07-25 rerun | Change |
|---|---:|---:|---:|
| Success (simulated) | 91 | 55 | +36 |
| Deposit closed | 17 | 45 | -28 |
| Whitelisting needed | 15 | 14 | +1 |
| Typed flow unavailable (minimum, capacity or redemption unavailable) | 5 | 11 | -6 |
| Receipt-analysis failure | 1 | 2 raw failures | -1 |

The status movement is partly live vault-state drift, especially Yearn vaults
whose deposit windows are now open. The code-attributable improvements include
the Ember queue flows, Plutus Hedge, MoneyFi FlowForge, Pharaoh's capacity
preflight, and the former Hyperithm/Morpho raw-redemption paths.

## Whitelist inventory

All 15 `whitelisting-needed` results are genuine per-account admission
denials for the simulated Safe. There is no incorrect whitelist status in this
matrix. The group is six IPOR Fusion vaults, eight Ethereum Lagoon vaults
(AltaETF, Block4Block, Der USDC, Der base USDC, Muchacho, Noon STS, Strada and
pyUSDC), and D2 Finance HYPE++ on Arbitrum. A lifecycle test requires adding
the simulated Safe to each vault's own allow-list (or a protocol-authorised
test identity); it must not be fixed by expanding the Lagoon Guard whitelist.

## Remaining non-executable trades

| Vault | Result | Required fix |
|---|---|---|
| Arche USD (`1-0x33ffc177a7278ff84aab314a036bc7b799b7cc15`) | `receipt_analysis_failed` | Fix eth-defi `analyse_4626_flow_transaction()` to handle this Yearn redeem event's amount direction. The on-chain redeem succeeded; the current positive/negative assertion is an analyser defect. Add an exact-vault fork regression test. |
| Ember Apollo ACRED (`1-0x2b13311fd553e74b421d4ccc96e348f71e179dcf`) | `below_minimum` | Keep the typed result. A lifecycle test needs an amount above the 9,170,000-share redemption minimum, or a per-vault test amount. |
| Accountable Hyperithm (`143-0x7cd231120a60f500887444a9baf5e1bd753a5e59`) | `below_minimum` | Keep the typed result. The 1 USDC test amount is below the 1,000 USDC minimum; use a larger test amount to exercise the lifecycle. |
| Pharaoh USDC (`43114-0x124d00b1ce4453ffc5a5f65ce83af13a7709bac7`) | `redemption_capacity_limited` | Correct typed preflight: the vault has no immediate underlying liquidity. No broadcast should be attempted until liquidity exists. |
| YieldNest RWA MAX (`1-0x01ba69727e2860b37bc1a2bd56999c1afb4c15d8`) | `redemption_unavailable` | Implement the maturity-aware redemption flow, or retain this explicit adapter limitation. |
| Apyx USDC (`1-0x069662d2588fcac24b5c209456db965d151556f0`) | `redemption_unavailable` | Investigate and decode the Morpho V2 redemption asset-transfer failure; convert any predictable capacity/cooldown state into a more specific typed preflight. |
| Sentora USD Earn (`1-0x74ad2f789ed583dbd141bbdafc673fe1f033718b`) | `deposit_closed` | The reason is actually that Upshift needs an explicitly selected accepted deposit asset. Restore `incompatible_deposit_asset` rather than labelling this as a closed deposit window, then select an accepted asset for a lifecycle test. |

The only remaining untyped failure is Arche USD's receipt analysis. The other
six rows are typed, actionable protocol state or an explicit adapter
limitation.
