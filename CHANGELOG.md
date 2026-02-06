    **Note**: A full changelog is not available as long as `trade-executor` package is in active beta developmnt.

## 0.2


- Breaking API changes

- Add `--trade-type` option to `show-positions` command to control trade display (2026-01-31)

- Add `--sync-interest` option to `lagoon-settle` command to fix credit supply position valuation timestamp issues (2026-01-31)

- Make token cache configurable and persistent across all commands via `CACHE_PATH` environment variable (2026-02-02)

- Add exchange account position support for external perp DEXes like Derive (2026-02-04)

- Add CCXT exchange account position support for tracking CEX account balances like Aster futures (2026-02-05)

- Add internal share price profit calculation method inspired by ERC-4626 vault mechanics (2026-02-06)

- Add `print_vault_rebalance_status()` console utility for displaying vault allocations and cash balance (2026-02-08)

- Add vault universe loading with full metadata from JSON blob in trading-strategy module (2026-02-08)

- Upgrade to Python 3.12 for Docker images and CI workflows (2026-02-08)
