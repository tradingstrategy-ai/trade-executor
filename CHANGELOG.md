    **Note**: A full changelog is not available as long as `trade-executor` package is in active beta developmnt.

## 0.2


- Add historical vault deposit/redemption availability to backtests so the alpha model skips impossible rebalances. The backtest pricing model now answers `can_deposit()` / `check_redemption()` from a per-(vault, timestamp) state frame (`deposits_open` / `redemption_open` / hard caps) threaded from the vault price loader through the trading universe. Unknown / missing / out-of-tolerance state is treated as allowed, so existing backtests are unchanged (verified by the exact-trade-count vault rebalance regression test) (2026-06-25)

- Make the cross-chain CCTP bridge planner cash-aware so a master-vault backtest no longer crashes with a silent `OutOfSimulatedBalance` mid-run. `inject_cctp_bridge_trades()` now sizes both bridge directions against the per-chain liquidity ledger (`available_bridge_capital`): a bridge-out is netted against capital already idle on the satellite chain and bounded by the fundable primary reserve (raising a clear `NotEnoughMoney` with full diagnostics instead of underflowing the simulated wallet), and a bridge-back is capped to the available bridge capital so it cannot move satellite USDC that an in-flight async deposit still needs to settle. Adds `tests/ethereum/test_cctp_bridge_cash_aware.py` (idle-capital netting, partial shortfall, and underfunded cases) (2026-06-25)

- Fix async vault deposit/redeem settlement in the backtester debiting/crediting the home reserve currency instead of the satellite-chain token for CCTP-bridged satellite vaults. `_settle_async_vault_trade()` now moves the pair's quote token (the vault denomination, e.g. USDC on the satellite chain), which equals the reserve currency for home-chain vaults, so the bridged satellite USDC is actually consumed rather than stranded (2026-06-25)

- Exclude in-flight capital from the cross-chain deployable target. `calculate_portfolio_target_value()` now subtracts in-transit (CCTP burned-not-received) and pending async vault deposit value, so a strategy does not plan cross-chain deployments against capital that is mid-flight and cannot be allocated until it settles (2026-06-25)

- Fix vault lockup expiry estimation raising `OverflowError: date value out of range` and crashing trade execution when vault metadata reports a nonsensically large `lockup_days` (e.g. a lockup expressed in seconds, ~3.9M "days"). `maybe_set_vault_lockup_expiry()` now clamps the display-only estimated expiry to `datetime.datetime.max` (2026-06-25)

- Fix `correct-accounts` spamming the closed-positions list with hundreds of empty (zero-quantity) Hypercore vault positions. Hypercore vault withdrawals cannot fully exit — the protocol refuses exact full withdrawals when NAV moves between planning and execution, leaving a small on-chain residual (the ~1.5 USDC withdrawal safety margin). The original position is already marked closed at exit because the close epsilon tolerates the residual, but `create_missing_vault_positions()` ran on every correct-accounts cycle and re-materialised that same on-chain dust as a brand-new closed position each run. Sub-dust vault equity with no open position is now written off and skipped, matching how leftover spot dust is treated for other trading pairs (2026-06-24)

- Add trade-ui redemption status and estimated vault lockup display for metadata-backed vaults. Vault metadata now carries `lockup_days` into trading pairs, successful vault deposits persist estimated per-position lockup expiry when no live reader exists, and the pair selection table shows both deposits and redemptions for Lagoon, Euler, Morpho, D2, PlutusDAO, Ostium, Gains and other metadata-backed vaults (2026-06-23)

- Fix `trade-ui` / `perform-test-trade` crashing with `AssertionError: Test sell failed` when closing an asynchronous home-chain vault position (Ostium V1.5 or ERC-7540 Lagoon). Closing such a position issues a `requestWithdraw` / `requestRedeem` that confirms on-chain but leaves the trade in `vault_settlement_pending` until the vault operator settles the queue off-chain; the close path fell straight through to an `is_success()` assertion and crashed even though nothing reverted (the revert reason was `None`). The open path already tolerated this inline while the close path did not — the two diverged. Both paths now route through a shared `_resolve_home_chain_async_settlement()` helper (the single-chain sibling of `_resolve_satellite_async_settlement()`), which force-settles on Anvil or stops cleanly on a real chain so the daemon / a re-run claims it. The gap went unnoticed because the only test exercising `perform-test-trade` against an async vault was skipped (its skip reason — needing an "unlocked Ostium keeper" — was a misdiagnosis; `tryNewSettlement()` is permissionless, but the forced-settlement transaction was paid from the executor's hot wallet, whose local signer the Anvil node cannot use). That test is re-enabled and now force-settles the deposit and the redeem from a node-unlocked dev account to complete the full open+close cycle, plus new unit tests — including one that drives the real `make_test_trade()` close path and reproduces the exact `Test sell failed` crash against the pre-fix code — lock the regression (2026-06-19)

- Fix `trade-ui` radio mode selection requiring an invisible extra Enter/Space after moving the highlight with arrow keys. The dialog now commits the highlighted trade mode immediately when navigating with Up/Down/Left/Right, focuses the trade-mode selector first, and shows keyboard-reachable Cancel / OK buttons in the modal. UI integration tests now cover the real Textual keyboard path for selecting `Sell all` and tabbing to the buttons (2026-06-19)

- Fix `trade-ui` leaving the amount input active when selecting `Sell all (close full position)` for an existing vault position. Textual's `RadioSet.Changed` event exposes the selected radio button as `event.index`, not `event.pressed_index`, so the mode-change handler failed before disabling the ignored amount field. Textual is updated to 8.2.7 and the dialog has a regression test for the close-all mode change (2026-06-19)

- Fix `trade-ui` / `perform-test-trade` crashing with `AssertionError: Satellite open failed: None` when a cross-chain (CCTP-bridged) test trade opened a position in an asynchronous satellite vault. An ERC-7540 (Lagoon) or Ostium V1.5 deposit on a satellite chain lands in `vault_settlement_pending` after its `requestDeposit` confirms, but `_make_cross_chain_test_trade()` asserted `is_success()` immediately — and because nothing reverted the revert reason was `None`. The single-chain test-trade path already tolerated this; the cross-chain path did not. Both satellite open and close now route through the shared `_resolve_satellite_async_settlement()` helper, which force-settles on the *destination* chain on Anvil (and forwards `web3config` for chain-aware claiming) or stops cleanly on a real chain so the daemon / a re-run claims it. The gap went unnoticed because every cross-chain fork test used a synchronous satellite vault (IPOR Fusion, Mo Earn Max), while the async ERC-7540 tests never drove the cross-chain test-trade flow (2026-06-19)

- Fix `trade-ui` / live trading crashing with `web3.exceptions.ABIEventNotFound` when settling an async ERC-7540 deposit into a non-legacy (Lagoon v0.5+) vault. `parse_deposit_transaction()` referenced a misspelled `DepositRequested` event; the ERC-7540 standard (and the Lagoon v0.5 ABI) names it `DepositRequest`. The bug went unnoticed because every Lagoon ERC-7540 test fixture pointed at the legacy 722 Capital vault, which takes a different (Referral-topic) parsing branch — the non-legacy branch was never exercised. `test_lagoon_erc_7540` now parses a `requestDeposit` receipt on the freshly deployed non-legacy vault as a regression (2026-06-18)

- Breaking API changes

- Fix HyperCore vault withdrawals crashing the live executor on `phase1_perp_wait` when the vault leader performance fee shrinks the redemption. A successful `vaultTransfer(vault→perp)` whose net perp arrival fell short of the gross request by more than the ~1% slippage tolerance (a fee of ~5-6% on a profitable vault like IKAGI) was mistaken for a silent no-op, halting the whole sequential rebalance. Withdrawal phase-1 verification now widens its accepted shortfall to the worst-case leader performance fee, resolved per vault (the fee differs: ~10% for leader vaults, 0% for protocol/HLP vaults) from the trading pair metadata (`other_data["vault_performance_fee"]`, populated from `VaultMetadata`), then a live `leaderCommission` read, and only a 10% default as a last resort; the same tolerance feeds the vault-equity reflection fallback, and the deduction is logged and shown in the failure diagnostics. `create_hypercore_vault_pair()` gained a `performance_fee` argument (2026-06-18)

- Fix the `/logs` webhook endpoint crashing with `AttributeError: 'NoneType' object has no attribute 'tb_frame'` when the in-memory ring buffer contained a log record whose `exc_info` carried no traceback (e.g. an exception instance that was never raised, or `exc_info=True` outside an active `except`). `ExportedRecord.export()` now only builds traceback data when a traceback is present, and `RingBufferHandler.export()` skips an individual malformed record instead of taking down the whole endpoint (2026-06-18)

- Add two-stage ERC-7540 async deposit/redeem support to backtesting and complete it for live trading. Backtests now simulate the request → settle → claim flow for async vaults with a configurable settlement delay (global `vault_settlement_delay` plus per-vault `vault_settlement_delay_overrides` on `run_backtest_inline()` and the other backtest entry points); pending trades resolve through the new polymorphic `ExecutionModel.resolve_pending_vault_settlements()` hook the runner calls each tick. **Note a behaviour change for existing backtests**: any vault whose metadata carries async features (`erc_7540_like`, `lagoon_like`, `ostium_like`) is now automatically simulated with the two-stage flow instead of instant settlement — the global delay defaults to one day (`DEFAULT_VAULT_SETTLEMENT_DELAY`), and Ostium-style vaults settle the next day at the epoch hour (`OSTIUM_BACKTEST_SETTLEMENT_HOUR`, 18:00 UTC) unless a per-vault override is given. `check-accounts`/`correct-accounts` no longer report a false mismatch for a position whose redeem is pending (`requestRedeem()` escrows the shares out of the owner wallet, so the expected on-chain balance now subtracts them). The new tests also uncovered and fixed state-persistence bugs: `translate_vault_to_trading_pair()` produced pairs the state file cannot serialise (160-bit `internal_id`, `set`-typed vault features) and async vault raw amounts overflowed the state file's JavaScript safe-integer limit — raw amounts are now persisted as strings (2026-06-12)

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
