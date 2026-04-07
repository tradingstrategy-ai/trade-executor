    **Note**: A full changelog is not available as long as `trade-executor` package is in active beta developmnt.

## 0.2


- Breaking API changes

- Add `StrategyTag.closed_source` to hide strategy source code from the `/source` webhook endpoint (2026-04-07)

- Record underlying asset price (e.g. vault share price) in `PositionStatistics` time-series so price history is available from state without re-reading the price feed (2026-04-06)

- Auto-close dusty Hypercore vault positions with repair trades instead of attempting failed withdrawals (2026-04-03)

- Expose Derive public addresses (wallet, owner EOA, session key) in strategy metadata for frontend display (2026-04-03)

- Enable internal share price PnL tracking for exchange account positions (GMX, Derive, CCXT) so the frontend displays unrealised profit instead of 0% (2026-03-31)

- Add `check-state.py` script for remote state sanity checks with Lagoon vault sync diagnostics and on-chain pending deposit/redemption queries (2026-03-30)

- Populate live Hyperliquid vault lockup remaining hours in open position data for trade-ui (2026-03-26)

- Add forward-only auto-generated CCTP bridge pairs for multichain universes and fork-simulated xchain master-vault coverage (2026-03-26)

- Add rolling live cycle trigger mode based on previous cycle end timestamps, with CLI integration coverage for 1s scheduler offsets (2026-03-19)

- Add Hyper AI redeem-accounting helpers for staged Lagoon redemptions and deployable-cash tracking in live loop tests (2026-03-18)

- Add `prepare-report` CLI command to inject iframe CSS/JS into externally generated HTML files for use as backtest reports (2026-03-17)

- Add live vault tradeability checks for ERC-4626 and Hypercore pricing, including Hyperliquid leader-fraction deposit gating and user-specific redemption limits (2026-03-16)

- Use `fetch_gmx_total_equity()` for GMX valuation with per-block reads and block number tracking in ValuationUpdate/BalanceUpdate state events (2026-03-15)

- Fix Lagoon vault NAV double-counting when exchange account positions (GMX) transfer USDC from Safe outside the trade engine, causing share price to drop on deposit (2026-03-15)

- Speed up grid search result serialisation ~5-13x by clearing unused indicators before save, switching from joblib+gzip to pickle+zstd, and disabling GC during serialisation (2026-03-15)

- Add explicit vault sort priority for Hypercore deposits and withdrawals, ensuring withdrawals execute before deposits (2026-03-14)

- Fix exchange account treasury sync crash where `last_updated_at` was never set, causing assertion failure on first trade cycle (2026-03-13)

- Switch notebook CLI runner from `ipython` to `jupyter execute` for multiprocess support in optimiser and grid search (2026-03-13)

- Add Markdown deployment report generation and `/file?type=deployment-report` API endpoint for Lagoon vault deployments (2026-03-13)

- Optimise backtest performance: skip expensive state serialisation validation and cache function hash results, reducing backtest time by ~44% (2026-03-11)

- Add Monad chain RPC support, Hypercore→HyperEVM chain mapping for vault deployment, and master-chain-multichain deployment test (2026-03-11)

- Fix indicator cache key for multipair universes to include a pair composition hash, preventing stale cache reuse when pairs change (2026-03-10)

- Add chain and address columns to multipair analysis and vault position tables (2026-03-10)

- Add equity curve by chain chart showing USD allocation per blockchain over time (2026-03-10)

- Add `distribute-gas-funds` CLI command to bridge native gas tokens to hot wallets across all strategy universe chains via LI.FI (2026-03-10)

- Add multichain gas balance check to validate hot wallet has native tokens on all universe chains at startup (2026-03-10)

- Support multi-chain simulated Lagoon vault deployment with multiple Anvil forks per CLI invocation (2026-03-10)

- Add cross-chain CCTP bridge + Hypercore vault lifecycle support with simulate mode for Anvil fork testing (2026-03-10)

- Add Hypercore native vault position support for Lagoon vaults on HyperEVM with API-based valuation, routing, and CLI pipeline (2026-03-09)

- Extract deployment pre-flight report into a reusable function for both single-chain and multichain Lagoon vault deployments (2026-03-06)

- Auto-detect GMX from strategy universe in single-chain `lagoon-deploy-vault` so the guard whitelists GMX routers (2026-03-06)

- Add `approve_gmx_trading()` console helper to initialise GMX trading approvals for Lagoon vaults (2026-03-06)

- Add `lagoon-redeem` CLI command for redeeming all vault shares from a Lagoon vault (2026-03-06)

- Add `lagoon-first-deposit` CLI command for making the initial deposit into a Lagoon vault (2026-03-06)

- Add GMX perpetuals exchange account support for Lagoon vaults with auto-discovery, on-chain position valuation, and market whitelisting (2026-03-03)

- Add multichain CCTP bridge position support with pricing, valuation, routing and `correct-accounts` auto-creation (2026-02-26)

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
