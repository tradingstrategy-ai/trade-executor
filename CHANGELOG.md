    **Note**: A full changelog is not available as long as `trade-executor` package is in active beta developmnt.

## 0.2


- Breaking API changes

- Fix `correct-accounts` to create and sync exchange account positions in one pass (2026-02-23)

- Fix `correct-history` to update trading period start date and share price return baseline after pruning (2026-02-21)

- Add `correct-history` CLI command to prune early time-series data from state files (2026-02-20)

- Add `deposits_disabled` strategy tag to disable deposit box on the website (2026-02-15)

- Add `--trade-type` option to `show-positions` command to control trade display (2026-01-31)

- Add `--sync-interest` option to `lagoon-settle` command to fix credit supply position valuation timestamp issues (2026-01-31)

- Make token cache configurable and persistent across all commands via `CACHE_PATH` environment variable (2026-02-02)

- Add exchange account position support for external perp DEXes like Derive (2026-02-04)

- Add CCXT exchange account position support for tracking CEX account balances like Aster futures (2026-02-05)

- Add internal share price profit calculation method inspired by ERC-4626 vault mechanics (2026-02-06)

- Add `print_vault_rebalance_status()` console utility for displaying vault allocations and cash balance (2026-02-08)

- Add premium harvest strategy module for Derive options manual trading, make DERIVE_OWNER_PRIVATE_KEY optional when DERIVE_WALLET_ADDRESS is provided (2026-02-12)

- Wire exchange account support into CLI `start` command for Derive strategies with `TradeRouting.default` (2026-02-12)

- Add `print_vault_deposit_status()` console utility for displaying Lagoon vault deposit and redemption queue status (2026-02-09)

- Add vault universe loading with full metadata from JSON blob in trading-strategy module (2026-02-08)

- Upgrade to Python 3.12 for Docker images and CI workflows (2026-02-08)
