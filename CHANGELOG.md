    **Note**: A full changelog is not available as long as `trade-executor` package is in active beta developmnt.

## 0.2


- Breaking API changes

- Fix multichain Lagoon cross-chain (CCTP) `trade-ui` test trades failing on the satellite-chain buy. The sequential/MEV-blocker broadcast path had two defects: (1) its buy-side treasury trip-wire compared the buy value against the home-chain reserve balance, which is legitimately zero after the reserve was bridged to the satellite chain via CCTP, so it tripped and then crashed with `UnboundLocalError: cannot access local variable 'failed_tx'` (masking the real error); and (2) it selected the broadcast provider once from the home-chain web3 and submitted every transaction through it, so a satellite-chain (e.g. Base) tx would be sent to the home-chain (e.g. Arbitrum) provider. The trip-wire now skips cross-chain (satellite) trades and always initialises `failed_tx` (a genuine same-chain underfunded buy still raises a clean `ExecutionHaltableIssue`), and the sequential path now resolves the provider per `tx.chain_id` just like the multi-node path (2026-06-11)

- Fix multichain Lagoon cross-chain (CCTP) trades crashing with "No satellite vault configured for chain X" in every CLI command except `start`, `perform-test-trade` and `lagoon-reclaim-satellites`: satellite vault modules were only populated when `create_execution_and_sync_model()` received the deployment-artifact path, which most commands (notably `trade-ui`, the path that stranded a production test trade) never passed. A shared `resolve_deployment_file()` helper now derives the `{id}.deployment.json` path and all live-trading commands pass it (2026-06-11)

- Fix master test failures when translating Hypercore-native vault pairs (e.g. HLP): the vault metrics data server now emits `share_token_decimals=null` for these vaults because they have no on-chain ERC-20 share token, so `base_token_decimals` arrived as `None` and `translate_trading_pair()` asserted. We now default only the base/share token of a Hypercore-native vault to 18 decimals — never the quote/denomination token (defaulting 6-decimal USDC to 18 scales raw amounts by 10\*\*12 and reverts CCTP transfers) and never a non-Hypercore vault, whose missing decimals still surface as an error (2026-06-11)

- Fix Docker release image (v1392) crashing at runtime with "Chain data folder ... not found": the build-time submodule optimisation switched to a non-recursive checkout but missed the nested `trading-strategy/tradingstrategy/chains` submodule (ethereum-lists/chains), whose `_data/chains/*.json` is read by `Chain.get_slug()`. The release workflow now explicitly checks out the chains submodule alongside `lagoon-v0` (still skipping the unused chains `website` submodule and eth_defi's other contract submodules) (2026-06-11)

- Add `lagoon-reclaim-satellites` CLI command that consolidates capital scattered across multichain Lagoon satellite Safes back into the master vault: it displays a table of all on-chain Safe USDC balances, auto-detects unreceived CCTP burns by scanning `DepositForBurn` events from our Safes and checking Circle's Iris API plus the destination chain's `usedNonces()`, lists the planned reclaim actions and asks for y/n confirmation, completes any in-transit CCTP transfers (including burns missing from the state file, auto-detected or given via `--complete-burn-tx <chain>:<tx_hash>`), bridges each satellite Safe's USDC back to the master Safe via CCTP, and verifies every destination-chain mint confirmed. Supports `--dry-run` to stop after the action list without moving funds (2026-06-10)

- Fix CCTP `receiveMessage` reverting out-of-gas on the destination chain: the mint was signed with a hardcoded 200,000 gas limit (less than the 300,000 used for the lighter `depositForBurn`), so the attestation-verify-plus-mint ran out of gas and stranded the burn `cctp_in_transit`. Gas is now estimated live with a 2x buffer and a generous fixed fallback, in both the live settlement and startup-retry paths (2026-06-10)

- Speed up Docker release image builds: use a registry-backed BuildKit cache (a dedicated `:buildcache` ghcr tag) instead of the GitHub Actions cache that was being LRU-evicted between releases, add a Poetry cache mount so PyPI wheels survive `eth_defi`/`trading-strategy` dependency bumps, and check out only the `lagoon-v0` contracts submodule instead of all 23 of web3-ethereum-defi's nested submodules (2026-06-10)

- Auto-discover multichain Lagoon satellite vault modules from the `lagoon-deploy-vault` deployment artifact placed next to the state file, removing the manual `SATELLITE_MODULES` env hand-off that stranded bridged USDC on satellite chains. `perform-test-trade` and `start` now fail fast with an actionable error when a cross-chain destination has no satellite module configured, before any irreversible CCTP bridge (2026-06-10)

- Fix CCTP bridge burning 10^12x too much USDC ("ERC20: transfer amount exceeds balance") in vault-only universes: native USDC is pinned to 6 decimals when generating synthetic bridge pairs, and the burn amount is converted from authoritative on-chain token decimals (`TokenDetails.convert_to_raw()`) instead of trusting pair metadata. This supersedes the 2026-06-03 fix, which trusted `PandasPairUniverse.get_token()` decimals that are themselves wrong (18) when the upstream vault dataset omits decimals (2026-06-09)

- Show estimated settlement time for pending async vault deposits in the trade-ui Lockup column — Ostium V1.5 shows a countdown (`settles 18.0h`), operator-driven ERC-7540 vaults (Lagoon) show `pending` (2026-06-09)

- Fix Hypercore vault deposit reverts when Safe EVM USDC balance is less than planned deposit due to cumulative withdrawal drift (2026-06-09)

- Add vault share balance and total supply to `check-wallet` command output; fix `lagoon-redeem` to claim unclaimed redemptions before starting and poll `maxRedeem` to avoid stale-read failures (2026-06-09)

- Fix phantom Hypercore vault positions from untracked withdrawals: detect and close positions where the Hyperliquid API reports zero equity but state has positive deposited USDC (2026-06-08)

- Fix CCTP bridge USDC decimals: resolve token decimals from pair universe instead of trusting `reserve_asset.decimals` which could be wrong (18 instead of 6) in ERC-4626 vault pair universes (2026-06-03)

- Add cross-chain CCTP bridge support to `perform-test-trade` and `trade-ui` for satellite-chain vault pairs (2026-06-03)

- Add `diagnose_hyperliquid_vault_redemption_failure()` to capture full HyperCore state on vault settlement failures, replacing uninformative `revert_reason=None` errors (2026-06-01)

- Update cross-chain master vault strategy from 01-initial.ipynb notebook with Ethereum/Base/Arbitrum CCTP vault universe (2026-05-29)

- Add `claim-hypercore-vault-dust` CLI command to recover untracked Hypercore vault dust into Lagoon reserves with operator safety guards (2026-04-16)

- Add `--remove-share-price-outliers` flag to `correct-history` CLI command to detect and remove spurious share price data points using rolling median comparison (2026-04-08)

- Add `--all-test-trades` flag to `close-position` CLI command to close all positions flagged as test trades (2026-04-07)

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
