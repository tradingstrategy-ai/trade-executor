# Arbitrum multi-vault live execution plan

## Objective and scope

Deploy one Lagoon vault on Arbitrum, with an Ethereum satellite Safe, and execute these external vault positions through normal `start` cycles and `perform-test-trade`:

| Position | Dataset chain | Custody | Execution |
| --- | --- | --- | --- |
| Lighter public pool pmalt, `lighter-pool-281474976552918` (fallback Steady Wealth Builder, `lighter-pool-281474976552443`) | Lighter, 9998 | Ethereum satellite Safe, chain 1 | Guarded Ethereum deposits/claims and Lighter API signatures on signing chain 304 |
| Atoma Index | Arbitrum, 42161 | Arbitrum home Safe | Existing synchronous deposit and queued redemption manager |
| Upshift Core USDC | Ethereum, 1 | The same Ethereum satellite Safe | Existing Upshift manager, funded through CCTP |

Deliver `scripts/lagoon/manual-multi-vault-cli.sh`. It must deploy, subscribe, open and close one position per protocol, reconcile accounts and NAV, return satellite capital, and redeem the investor. `SIMULATE=true` must exercise the same registered executor commands against persistent Anvil forks, a fake Lighter sequencer and forged CCTP attestations. Live mode uses mainnet contracts and real money.

Use the existing `State`, `TradeExecution`, `BlockchainTransaction`, `StateStore` and persistence callbacks. Store pool progress on its trade. Keep the API private key exclusively in the existing owner-only operator record. The shell derives progress from executor state and verified balances.

GMX liquidity, Symbiotic, LLP, sub-account perpetual trading, aggregate redemptions, NB05 strategy conversion, automatic asset conversion and production Guard migration are outside scope. Retain the existing one-writer and one-pending-redemption restrictions.

This plan records proposed work. Planning was limited to local documentation, targeted code inspection and read-only Git access. No tests, application commands, deployments or funded lifecycle were run.

Provenance: drafted on 3 November 2026 by Codex CLI (`gpt-6.1-sol`, read-only sandbox) from a brief containing the verified gap map in [the NB05 readiness plan](nb05-live-readiness.md) and [PR #1693](https://github.com/tradingstrategy-ai/trade-executor/pull/1693). A sample of its code citations was re-checked against master `6f4563b07` before saving. Three design points were simplified afterwards and are marked "Decision" below: deployment key handling reuses the existing key-backup convention instead of an upstream callback change, bootstrap collateral needs no new state field, and the simulated sequencer is file-backed rather than served over HTTP.

## Baseline and reference implementation

Use the supplied, verified baseline:

- Executor master: `6f4563b07`.
- eth-defi submodule: `02b6caf9942404ac6f411a0865592b5f8ae4bf6b`, including Lighter Core pools #1668, Atoma #1703 and Upshift #1705.
- Executor Atoma support: `c415b5744`, #1697.
- Executor Upshift support: `ee7b1fd32`.
- Guarded ERC-4626 test trades and pre-broadcast saves: #1694.
- Lighter master/sub-account cash routing: #1683, #1689 and #1691.

The eth-defi submodule is empty in this worktree. Dependency references below refer to files inspected at the pinned revision using the parent checkout. Read `deps/web3-ethereum-defi/docs/README-lighter-core-execution.md` first during implementation.

Consume these upstream symbols rather than implementing their protocol behaviour again:

| Upstream path | Symbols and purpose |
| --- | --- |
| `eth_defi/lighter/execution.py:50` | `LighterAccountIdentity`: separates account, API deployment, dataset, custody and signing identities. |
| `eth_defi/lighter/execution.py:74` | `LighterPoolState`, including `share_value`; `LighterPoolHolding` at line 113, including fee-aware `estimated_proceeds`. |
| `eth_defi/lighter/execution.py:139` | `LighterOperationStage`; `LighterOperation.as_dict()` / `from_dict()` at lines 179 and 189. |
| `eth_defi/lighter/execution.py:231` | `fetch_execution_account()`, `fetch_pool_state()` at line 258 and `fetch_pool_holdings()` at line 294. |
| `eth_defi/lighter/execution.py:323` | `prepare_pool_amount()`: fresh-state sizing, operator-share capacity, holdings and liquidity checks. |
| `eth_defi/lighter/execution.py:421` | `LighterPoolExecutor.prepare()`, `submit()` and `reconcile()`: sign, persist, submit once and reconcile the original hash. |
| `eth_defi/lighter/execution.py:589` | `fetch_withdrawals()`, `fetch_matching_withdrawal()` at line 612 and `wait_for_claimable_withdrawal()` at line 655. |
| `eth_defi/lighter/execution.py:691` | `reconcile_bootstrap()`: conservation before pool exposure, not aggregate trading NAV. |
| `eth_defi/lighter/execution.py:711` | `prepare_pool_deposit()`, `prepare_pool_redemption()` at line 735, `wait_for_available_collateral()` at line 756, `prepare_secure_withdrawal()` at line 779 and `wait_for_pool_shares()` at line 800. |
| `eth_defi/lighter/activation.py:72` | `activate_lighter_safe()`: accounted Ethereum deposit and Safe-owned key registration. |
| `eth_defi/erc_4626/vault_protocol/lagoon/lighter.py:50` | `LagoonLighterActivation`, `preflight_lighter_multichain()` and `fund_and_activate_lighter()`. |
| `eth_defi/erc_4626/vault_protocol/lagoon/deployment.py:447` | `LagoonConfig.lighter_deployment`; `deploy_multichain_lagoon_vault(lighter_activation=...)` at line 3588. |
| `eth_defi/lighter/sdk.py:79` | Existing signer construction and auth-token support. Preserve signing chain 304 when adapting it for explicit pool nonces. |
| `eth_defi/cctp/testing.py` | `craft_cctp_message()`, `forge_attestation()` and `replace_attester_on_fork()`, already imported by `tests/mainnet_fork/test_xchain_master_vault_multichain.py:23`. |

The upstream funded example, `scripts/lagoon/lagoon-lighter-pool-roundtrip.py`, is reference code for mint/burn observations and withdrawal identification. Its withdrawal-history baseline and exclusive-writer checks are at lines 150–195. No funded pool round trip has been run upstream; fixture success must not be presented as that evidence.

Use these executor references:

- `docs/state-persistence.md`: authoritative state and callback requirements.
- `.claude/docs/vault-deposit-redeem.md`: shared Atoma/Upshift lifecycle.
- `.claude/docs/hypercore-vault.md`: sequential multi-phase execution and retained capital.
- `.claude/docs/cli-command.md`: public commands and state saves.
- `docs/lagoon-lighter-deployment.md`: existing operator-record boundary.
- `docs/plans/nb05-live-readiness.md`: current backlog and Lighter expansion scope.
- `docs/strategy-parameters.md`: engine 0.5 strategy layout.
- `docs/vault-closure.md` and `docs/backtest-universe-preservation.md`: admission and historical coverage.

## Mapping to existing primitives

| Lighter pool behaviour | Existing primitive | Integration work |
| --- | --- | --- |
| Synthetic public-pool identity | `TradingPairIdentifier`, `TradingPairKind.vault`, dataset translation | Add `is_lighter_pool()` and one canonical pair constructor; retain chain 9998. |
| Ethereum custody behind an Arbitrum reserve | CCTP bridge positions, satellite modules and `LagoonTransactionBuilder` | Map custody to chain 1 for funding, signing and reconciliation without changing dataset identity. |
| Safe USDC deposit and claim | `LighterRouting._prepare_deposit()` / `_prepare_claim()` | Share the small custody-call construction, supplying the Ethereum token, master account and satellite Safe explicitly. |
| Sequential multi-phase trade | `HypercoreVaultRouting` shape and `RoutingModel.needs_sequential_trade_execution()` | Add the required `LighterPoolRouting`; reuse execution and save hooks, not Hypercore’s protocol verifiers. |
| Signed API operation | Upstream `LighterPoolExecutor` and `LighterOperation` | Persist its serialised operation on the trade before submission; reconcile that operation after restart. |
| Expected sequencer or withdrawal delay | Existing `vault_settlement_pending` status and execution-model settlement hook | Dispatch pool trades to their router before generic ERC-4626 ticket reconstruction. No new status or controller. |
| Deposit capital in transit | Existing allocated capital and pending-deposit value | Retain allocation until shares are confirmed; substitute observed transit value where principal changes. |
| Burned shares awaiting USDC | Existing unavailable sell quantity and pending-position valuation | Replace the burned portion’s share value with attributable collateral/withdrawal entitlement. |
| Delayed exit proceeds | `has_delayed_vault_redemption()` and `PositionManager.is_async_vault_sell_pair()` | Classify pools as delayed exits so AlphaModel, yield funding and CCTP cannot spend pending proceeds. |
| Pool valuation | Ordinary vault position, `ValuationUpdate`, generic valuation routing | Price actual shares from current pool equity/supply and holdings; keep account equity separate. |
| Account checks | Multichain balance reader and existing accounting checks | Read API shares, account for phase evidence and exclude synthetic assets from ERC-20 probes. |
| Lagoon investor NAV | `LagoonVaultSyncModel.calculate_valuation()` and portfolio NAV | Include home assets, satellite cash, pool value and transit once. |
| Separate public buy/close invocations | `make_test_trade()` and `_make_cross_chain_test_trade()` | Extend existing safety checks and resume ordering to satellite vaults and pool phases. |
| Simulated external settlement | `FakeLighterCore`, Anvil forks and CCTP testing helpers | Extend the fake and expose it across shell command processes; retain real guarded EVM transactions. |
| Manual progress | `erc4626_manual.position_phase()` and normal JSON state | Add a small Typer helper for chain-aware phase/audit decisions; no progress files or markers. |

## Required changes

### 1. Pair identity, custody mapping and routing selection

**Files:** `tradeexecutor/state/identifier.py`, `tradeexecutor/strategy/dex_data_translation.py`, `tradeexecutor/ethereum/lighter/lighter_pool.py` proposed, `tradeexecutor/strategy/generic/default_protocols.py`, `tradeexecutor/ethereum/ethereum_protocol_adapters.py`, `tradeexecutor/strategy/generic/generic_router.py`, `tradeexecutor/ethereum/web3config.py`, `tradeexecutor/cli/bootstrap.py`, `tradeexecutor/ethereum/cctp/{bridge_universe,planner}.py`, `tradeexecutor/state/{state,portfolio}.py`.

Add `TradingPairIdentifier.is_lighter_pool()` beside `is_hyperliquid_vault()`. The current native-vault predicate recognises only chain 9999 (`tradeexecutor/state/identifier.py:1420`). Require vault kind, chain 9998 and a valid `lighter-pool-<index>` identity; reject malformed indices before custody operations.

Add `create_lighter_pool_pair()` alongside the small Lighter pool utilities. Use `create_hypercore_vault_pair()` as the constructor reference (`tradeexecutor/ethereum/vault/hypercore_vault.py:120`), without copying its USDC-as-quantity convention.

Preserve these separate identities:

- Dataset chain: 9998.
- Custody chain: 1.
- Signing chain: 304.
- API deployment: canonical Ethereum Lighter.
- Owner: Ethereum satellite Safe.
- Account: activated Safe-owned master account from deployment/state and the operator record.

Keep the dataset’s existing representational share-decimal convention initially; translation already defaults synthetic share decimals to 18 (`tradeexecutor/strategy/dex_data_translation.py:101`). Define the conversion explicitly: position quantity is raw shares divided by the representational scale; its price is upstream USDC-per-raw-share multiplied by that scale. Never apply USDC decimals to share sizing. Persist exact raw shares and use them for full burns and terminal checks. Prevent ordinary quantity epsilon from erasing a remaining raw share; `TradingPosition.get_quantity()` currently snaps quantities at `QUANTITY_EPSILON` (`tradeexecutor/state/position.py:717`).

Extend both `default_match_router()` and `default_supported_routers()` with router id `lighter_pool`. Today all non-Hypercore vaults select `"vault"` (`tradeexecutor/strategy/generic/default_protocols.py:44`), and supported-router discovery separately classifies exchange metadata (`:166`). Discover pool routing from pair semantics and avoid also constructing a generic ERC-4626 adapter for a pool-only exchange.

Add `create_lighter_pool_adapter()` and its `EthereumPairConfigurator.create_config()` branch. Resolve Ethereum Web3, satellite custody, pool pricing/valuation and existing `LighterKeyManagement` directly. Generic vault construction is unsuitable (`tradeexecutor/ethereum/ethereum_protocol_adapters.py:429`). Do not relax `_auto_discover_lighter()`’s Ethereum-home account-valuation checks to make pools pass; those checks belong to exchange-account support (`:1000`, `:1022`, `:1031`).

Introduce one small custody-chain normalisation function beside identifier semantics, with the two explicit mappings 9999→999 and 9998→1. Reuse it at existing boundaries; do not add a chain registry. Extend:

| Boundary | Existing gap and required change |
| --- | --- |
| Deployment | `normalise_deployment_chain_id()` maps only Hypercore (`tradeexecutor/ethereum/lagoon/universe_config.py:53`); delegate to the common mapping. |
| Transaction preparation | `GenericRouting.setup_trades()` maps only 9999→999 (`tradeexecutor/strategy/generic/generic_router.py:187`); resolve chain 1 and the Ethereum satellite builder for pools. |
| Settlement | `GenericRouting.settle_trade()` uses the dataset chain (`tradeexecutor/strategy/generic/generic_router.py:294`); normalise custody before selecting Web3. |
| RPC configuration | Extend `get_rpc_env_var_name()` and `Web3Config.get_connection()` (`tradeexecutor/ethereum/web3config.py:76`, `:456`). Require `JSON_RPC_ETHEREUM`, never an RPC for 9998 or 304. |
| Bootstrap | Extend `check_universe_chains_have_rpc()` (`tradeexecutor/cli/bootstrap.py:1383`). In `check_universe_contracts_resolve()`, skip synthetic pool contract probes but still require the Ethereum satellite, native USDC and real Lighter contracts (`:1548`). |
| CCTP generation | `generate_primary_to_satellite_cctp_bridge_universe()` filters dataset chain ids against native-USDC chains (`tradeexecutor/ethereum/cctp/bridge_universe.py:75`). Normalise custody first and deduplicate Ethereum. |
| CCTP planning | `inject_cctp_bridge_trades()` groups by raw pair chain and skips only Hypercore (`tradeexecutor/ethereum/cctp/planner.py:334`, `:356`). Group pool funding under chain 1; do not skip its CCTP funding. |
| Funding allocation | `State.start_execution()` finds bridge capital by raw pair chain (`tradeexecutor/state/state.py:810`). Extend allocation, success and failed-allocation return paths consistently. |
| Bridge ledger | `Portfolio.move_capital_from_bridge_to_spot_trade()` and `return_capital_to_bridge()` use raw pair chain (`tradeexecutor/state/portfolio.py:1236`, `:1276`). Both pool and Upshift must use the same Ethereum bridge position. |

Extend `TradingPairIdentifier.has_delayed_vault_redemption()` explicitly for pools before its generic feature fallback (`tradeexecutor/state/identifier.py:997`). Reuse the resulting funding gates through `PositionManager.is_async_vault_sell_pair()` (`tradeexecutor/strategy/pandas_trader/position_manager.py:744`).

Implement pool admission through the existing pricing `check_deposit()` / `check_redemption()` interfaces, which `GenericPricing` already dispatches (`tradeexecutor/strategy/generic/generic_pricing_model.py:201`). Report owner restrictions separately from global closure using `get_vault_closure()` (`tradeexecutor/strategy/vault_closure.py:145`). Recheck live capacity immediately before a buy; do not introduce a pool eligibility registry or current-readiness filter into historical universes.

The existing backtest route works because pools currently select `"vault"`. Preserve that behaviour by adding `"lighter_pool"` to the ignored vault-router set when changing live routing (`tradeexecutor/backtest/backtest_generic_router.py:75`). Retain existing historical admission, pricing and configured settlement-delay assumptions.

### 2. Multichain deployment, operator records and bootstrap accounting

**Files:** `tradeexecutor/ethereum/lagoon/universe_config.py`, `tradeexecutor/cli/commands/lagoon_deploy_vault.py`, `tradeexecutor/exchange_account/{lighter_operator,sub_account}.py` and `tradeexecutor/cli/commands/check_wallet.py`. No upstream change and no `init`/state-model change.

Extend `_collect_erc4626_vault_addresses()` to exclude Lighter pools before checksum conversion (`tradeexecutor/ethereum/lagoon/universe_config.py:130`). Extend `_collect_chain_token_addresses()` to exclude synthetic share and denomination addresses (`:150`). Explicitly whitelist native Ethereum USDC through the Lighter configuration.

Extend `_apply_protocol_configs()` to set `LagoonConfig.lighter_deployment = LighterDeployment.create_ethereum()` on Ethereum when pools are present. Its current protocol settings contain CCTP, Uniswap and GMX, but no Lighter setting (`tradeexecutor/ethereum/lagoon/universe_config.py:164`).

Extend `_deploy_multichain()`:

1. Remove its blanket refusal of non-Ethereum source activation (`tradeexecutor/cli/commands/lagoon_deploy_vault.py:2048`).
2. Preserve the existing single-chain/Ethereum-home activation branch.
3. For an Arbitrum hub, pass `LagoonLighterActivation(hub_chain="arbitrum", ethereum_chain="ethereum", ...)` to `deploy_multichain_lagoon_vault()`. The current call omits it (`:2186`).
4. Leave per-chain `generate_lighter_api_key=False` for coordinated activation. Upstream explicitly refuses per-chain activation and sub-account creation in this path (`deps/web3-ethereum-defi/eth_defi/erc_4626/vault_protocol/lagoon/lighter.py:101`).
5. Keep Arbitrum’s underlying as Arbitrum native USDC. The existing source-only branch overwrites the source underlying with Ethereum USDC (`tradeexecutor/cli/commands/lagoon_deploy_vault.py:2098`).
6. Select `result.deployments["ethereum"].lighter_account_setup` when writing the private record; current code selects the source deployment (`:2192`).

Use upstream’s 20-USDC accounted bootstrap envelope, not the single-chain 1-USDC preflight. It is defined at `deps/web3-ethereum-defi/eth_defi/erc_4626/vault_protocol/lagoon/lighter.py:46`; its fresh-deployment, explicit-token, gas, fee and CCTP checks are in `preflight_lighter_multichain()` at line 65. Retain the default bridge fee ceiling unless the operator deliberately selects another permitted ceiling. Do not silently raise it or change finality.

Fix deployment discovery as part of this change:

- `_build_multichain_artifact_payload()` currently suppresses Lighter metadata on satellites (`tradeexecutor/cli/commands/lagoon_deploy_vault.py:1059`). Export public satellite account metadata.
- `read_lighter_account_resolution()` currently requires the account-bearing deployment’s vault address to match the home vault (`tradeexecutor/exchange_account/sub_account.py:113`). Validate the home deployment first, then resolve its Ethereum satellite account by the canonical deployment map.
- `load_lighter_operator_record()` already selects `deployments.ethereum`, but requires `vault_address` (`tradeexecutor/exchange_account/lighter_operator.py:84`). Record the Arbitrum hub vault association and Ethereum Safe/module explicitly; do not treat a satellite Safe wrapper as an Ethereum Lagoon vault.
- Preserve mode 0600, ownership checks and secret redaction. Verify the record’s account, Safe, module, hub association and API-key slot before signing.

#### Deployment progress without a second ledger

The pinned coordinator unconditionally creates a protected report and writes deployment progress to it (`deps/web3-ethereum-defi/eth_defi/erc_4626/vault_protocol/lagoon/deployment.py:3673`). Activation emits the private key through its progress callback before registration (`deps/web3-ethereum-defi/eth_defi/lighter/activation.py:127`).

**Decision:** treat that report exactly as the existing single-chain path treats `--lighter-key-backup-dir` (`docs/lagoon-lighter-deployment.md`): a mode-0600 pre-registration key backup inside the private run directory, deleted once the private operator record has been written. Point `LagoonLighterActivation.report_path` at that location from `_deploy_multichain()`; do not change upstream, add a progress callback, add a field to the `Deployment` state model or extend `init()` beyond its pristine-file contract (`tradeexecutor/cli/commands/init.py:173`). Deployment precedes executor state, so this is deployment evidence, not a second executor ledger. Executor state starts at `init`, as today.

Partial deployment remains a supported halt requiring evidence-led recovery from that report and the chain, not automatic redeployment, exactly as `scripts/lagoon/manual-erc4626-cli.sh` step 1 already instructs. An incomplete deployment must never be mistaken for permission to deploy again. If a later funded run shows the report is insufficient for recovery, propose the upstream callback then, as a separate change.

#### Account for the bootstrap before another NAV settlement

Upstream subscribes 20 USDC, bridges activation capital and leaves accounted collateral on Lighter (`deps/web3-ethereum-defi/eth_defi/erc_4626/vault_protocol/lagoon/lighter.py:197`). Recording only the remaining Safe balances would omit that collateral.

**Decision:** keep this to existing mechanisms and one explicit residual.

- `init` imports the 20 USDC bootstrap subscription through the existing Lagoon treasury event scan from the pre-deployment block, as the single-chain Lighter path already books its 1 USDC activation (`manual-lighter-subaccounts-cli.sh` step 3). The hub reserve then reads 19 USDC minus the actual bridge fee.
- The activation collateral already sits in the deployment artefact (`LighterAccountSetup.collateral`, written by `_write_lighter_private_record()`). Until the first pool buy, it is a documented bootstrap residual: `check-wallet` reports it, `check-accounts` expects it, and NAV excludes it. Do not add a state field, an exchange-account position or pending-trade accounting for it.
- The first pool buy consumes it: size the L1 deposit as the trade amount minus the uninvested collateral read live from the account, and record the full amount as the trade's executed reserve. After that the residual is zero and NAV conserves.
- Refuse NAV publication while a pool trade is mid-phase, through the existing pending-settlement accounting; no extra attribution rule is needed.

Validate the bootstrap import before `lagoon-first-deposit`, which otherwise updates reserves directly from the home Safe (`tradeexecutor/cli/commands/lagoon_first_deposit.py:146`).

Verify installed permissions after deployment: Atoma’s real request/claim selectors on Arbitrum; Upshift denomination/LP-token permissions on Ethereum; Lighter native-USDC custody calls and Safe recipient; CCTP domains and recipients; asset-manager permissions; and final Safe owner sets/thresholds. Dynamic pool selection is software policy, not an Ethereum Guard restriction.

For simulated deployment, use the same coordinator against both forks with fake activation observations and forged bootstrap attestations. Do not enable production `--simulate`, bypass Guard calls, or claim that a private fork activates an account on the live Lighter sequencer.

### 3. Pool execution phases, persistence and restart

**Files:** `tradeexecutor/ethereum/lighter/lighter_pool_routing.py` proposed, `tradeexecutor/ethereum/lighter/lighter_routing.py`, `tradeexecutor/ethereum/execution.py`, `tradeexecutor/strategy/execution_model.py`, `tradeexecutor/ethereum/vault/settlement_retry.py`, `tradeexecutor/state/{state,trade,position}.py`.

Add `LighterPoolRouting` and a small routing-state class. Follow `HypercoreVaultRouting.setup_trades()` / `settle_trade()` structurally (`tradeexecutor/ethereum/vault/hypercore_routing.py:2383`, `:4495`). Do not subclass its protocol machinery.

Require sequential execution and one active pool operation per master account/API key. `EthereumExecution.execute_trades()` already selects sequential execution (`tradeexecutor/ethereum/execution.py:855`).

Reuse the Lighter custody-call preparation by extracting only the shared code from `_prepare_deposit()` and `_prepare_claim()` into small module-level helpers, or making their inputs explicit. The current deposit reads an exchange-account id (`tradeexecutor/ethereum/lighter/lighter_routing.py:358`), the reserve-token reader uses the trade’s home reserve (`:623`), and `_get_lagoon_vault()` returns `routing_state.vault` (`:633`). Pool calls must instead use Ethereum native USDC, the activated master account and the satellite builder’s Safe.

Do not reuse `_prepare_withdrawal()`’s request submission lifecycle. Pool withdrawals must use persisted upstream `LighterOperation` records and exact withdrawal-history ids.

#### Phase state

Store a single `lighter_pool` mapping in `trade.other_data`, containing only execution evidence needed to resume:

| Field | Purpose |
| --- | --- |
| `phase` | Last durably established phase below. |
| Pool/account/custody identity | Detect state/configuration mismatch before signing. |
| Deposit, collateral and Safe balance baselines | Attribute changes without sweeping another operation’s money. |
| Exact requested and observed raw shares | Verify mint/burn and retain full-exit precision. |
| `operations` keyed by `mint`, `burn`, `withdraw` | Preserve each upstream `LighterOperation.as_dict()` and its terminal receipt. |
| Withdrawal-history baseline ids and selected id | Identify and resume the exact withdrawal. |
| Actual collateral, claim amount, fees and observation time | Value transit and complete accounting from observed amounts. |
| Relevant EVM transaction indices | Locate existing `BlockchainTransaction` records without duplicating their signed bytes. |

Keep raw monetary/share amounts as decimal strings where required by state serialisation. Upstream `LighterOperation.as_dict()` currently retains integer amounts (`deps/web3-ethereum-defi/eth_defi/lighter/execution.py:179`), whereas executor state rejects integers above the JavaScript safe range (`tradeexecutor/state/validator.py:67`). Extend upstream serialisation/loading narrowly, or encode/decode those fields at the existing state boundary; do not persist an incompatible operation or create another operation class.

Bind the upstream `persist(operation)` callback to update this mapping and call the existing pre-broadcast state callback. `EthereumExecution` already binds that callback before routing (`tradeexecutor/ethereum/execution.py:841`).

#### Buy

CCTP arrival is a separate bridge trade. Pool execution begins only after Ethereum bridge capital is confirmed and allocated.

| Phase | Action and completion evidence |
| --- | --- |
| `deposit_prepared` | Record baselines; append signed Safe approval/deposit transactions; save before broadcast. Include attributable bootstrap collateral when sizing the additional deposit. |
| `collateral_wait` | Verify the L1 receipt and wait for the attributable collateral increase through upstream readers. A successful Ethereum receipt alone does not prove Lighter credit. |
| `mint_prepared` | Fetch fresh pool state and holdings; call `prepare_pool_deposit()` and persist its signed operation before submission. |
| `mint_submitted` | Submit once, reconcile the signed hash, then verify exact expected holdings with `wait_for_pool_shares()`. API acceptance is not completion. |
| `complete` | Mark success with actual shares and actual committed USDC; retain any attributable uninvested collateral as transit until recovered or consumed. |

Use upstream capacity and monetary sizing, including operator-share capacity and available collateral (`deps/web3-ethereum-defi/eth_defi/lighter/execution.py:323`). Refuse LLP in this integration. Refresh admission after collateral arrives because capacity may have changed since strategy sizing.

#### Sell

| Phase | Action and completion evidence |
| --- | --- |
| `burn_prepared` | Read exact holdings and fresh liquidity; call `prepare_pool_redemption()`. Full close burns all attributable raw shares. Save before submission. |
| `burn_submitted` | Submit once and reconcile the original burn hash. |
| `collateral_wait` | Confirm the reduced share holding and attributable available collateral. Retain the burned portion as a receivable, not spendable cash. |
| `withdraw_prepared` | Save a complete withdrawal-history baseline; refuse unrelated pending/claimable withdrawals; call `prepare_secure_withdrawal()` for attributable available USDC. |
| `withdraw_submitted` | Reconcile the original operation; identify exactly one new withdrawal-history record under the exclusive-writer assumption and save its id. |
| `claimable_wait` | Poll `wait_for_claimable_withdrawal()` using that exact id and refreshable authentication. Use its actual claimable amount. |
| `claim_prepared` | Append the signed Safe claim to `BlockchainTransaction`; save before broadcast. |
| `complete` | Verify the actual Ethereum Safe receipt/balance increase, account once, and return proceeds to the Ethereum bridge position. |

Do not match withdrawals by amount or approximate timestamp. The upstream funded example uses a saved all-pages history baseline because the history schema lacks the submitted hash (`deps/web3-ethereum-defi/scripts/lagoon/lagoon-lighter-pool-roundtrip.py:183`). Missing or multiple new ids must halt attribution; never choose one heuristically.

Deduct operator fees only from positive investor profit, using upstream holding estimates. Record actual net proceeds on success and avoid another fee debit. Mint/burn signatures do not enforce a USDC slippage limit; expose estimates honestly and halt on unexplained discrepancies rather than claiming atomic price protection.

#### Pending and failure behaviour

Reuse `vault_settlement_pending` for expected pool delays. Extend `State.mark_vault_settlement_pending()` so pool metadata records custody chain 1 while preserving pair chain 9998; it currently stamps the raw pair chain (`tradeexecutor/state/state.py:1259`). Reuse its allocation assertion and existing unavailable-share/pending-capital semantics.

Extend `ExecutionModel.resolve_pending_vault_settlements()` and the existing sweep to dispatch pool trades to `LighterPoolRouting` before generic ticket reconstruction (`tradeexecutor/strategy/execution_model.py:220`; `tradeexecutor/ethereum/vault/settlement_retry.py:466`). Construct the same pool adapter dependencies used by execution; do not add a persistent router registry or a worker.

Cover started/broadcast pool operations as well as formally pending ones: a crash can precede the pending-status save. The sweep currently collects only `vault_settlement_pending` trades (`tradeexecutor/ethereum/vault/settlement_retry.py:293`). Reconcile saved EVM receipts and API operations before any replacement preparation.

After restart:

1. Load the saved operation with `LighterOperation.from_dict()`.
2. Reconcile non-terminal operations before preparing another.
3. For already confirmed operations, verify the next phase’s holdings or collateral evidence; upstream `reconcile()` rejects terminal operations (`deps/web3-ethereum-defi/eth_defi/lighter/execution.py:536`).
4. Recover a mined claim before querying a consumed withdrawal as though it had no value.
5. Apply final reserve/bridge accounting once.

Expected settlement delays return pending after bounded reads. Unknown submission outcomes block further account writes. Identity mismatches, ambiguous history, competing operations and unexplained asset changes propagate as execution-halting invariant failures.

Retain potentially moved capital on definitive downstream failure using `retain_reserve_allocation_on_failure`, already honoured by `State.mark_trade_failed()` (`tradeexecutor/state/state.py:1170`). Never refund an opening allocation merely because minting or receipt analysis failed after the L1 deposit.

Extend `_execute_trades_sequentially()` only where needed for API-only pool phases: its existing no-EVM branch covers internal account transfers exclusively (`tradeexecutor/ethereum/execution.py:780`). An empty transaction list must still dispatch pool settlement rather than imply success.

Pool exits cannot fund same-cycle buys. Bridge-back occurs on a later cycle or later explicit command invocation after the Safe claim is confirmed, even if simulation can advance every pool phase immediately.

### 4. Pricing, valuation, account checks and Lagoon NAV

**Files:** `tradeexecutor/ethereum/lighter/lighter_pool_pricing.py` and `lighter_pool_valuation.py` proposed, `tradeexecutor/ethereum/multichain_balance.py`, `tradeexecutor/strategy/{asset,account_correction}.py`, `tradeexecutor/cli/commands/{check_accounts,correct_accounts,check_wallet}.py`, `tradeexecutor/state/{trade,position,portfolio}.py`, `tradeexecutor/ethereum/lagoon/vault.py`.

Add a pool pricing model and ordinary position valuator through `create_lighter_pool_adapter()`. Use:

- `fetch_pool_state().share_value` for current USDC per raw share.
- `fetch_pool_holdings()` for actual investor shares, principal and estimated operator fee.
- Structured admission checks for pool status, capacity, available collateral, holdings and redemption liquidity.
- Existing `TradePricing` and `ValuationUpdate` records, with naive UTC observation times.

Do not route pools through `VaultPricing.get_vault()`, which constructs an ERC-4626 instance (`tradeexecutor/ethereum/vault/vault_live_pricing.py:64`).

Preserve share-based profit accounting. Do not extend `is_using_internal_share_price_profit()` to pools; its existing exchange-account/Hypercore branch serves different quantity semantics (`tradeexecutor/state/position.py:532`).

For partial exits, value remaining shares plus the pending burn’s attributable entitlement. After burn confirmation, replace that portion’s share valuation with observed collateral or withdrawal proceeds. Do not continue valuing burned shares and add the receivable.

For pending buys, use the existing pending-deposit value path, extending it to phase-aware observed value when necessary (`tradeexecutor/state/position.py:851`). Remove the corresponding value as shares become accounted holdings. A missing required API field must fail valuation without refreshing a stale quote.

Extend account inspection:

| Existing function | Gap and action |
| --- | --- |
| `_OFFCHAIN_CHAIN_IDS`, `_fetch_pair_aware()` | Only Hypercore is excluded/routed off-chain (`tradeexecutor/ethereum/multichain_balance.py:38`, `:158`). Add pool holding reads and never issue ERC-20 calls for synthetic assets. |
| `get_onchain_assets()` | Only Hypercore vaults return no ERC-20 asset (`tradeexecutor/strategy/asset.py:155`). Exclude pool shares from the on-chain map. |
| `TradingPosition.get_held_assets()` | The synthetic-vault exemption is Hypercore-only (`tradeexecutor/state/position.py:2771`). Preserve economic pool holdings through the API check path. |
| `check_accounts()` | It appends a dedicated Hypercore check but no pool check (`tradeexecutor/strategy/account_correction.py:1320`). Add compact share/phase checks using the shared pool reader. |
| `preflight_state_for_account_correction()` | Special unresolved-capital handling is Hypercore/exchange-transfer specific (`tradeexecutor/strategy/account_correction.py:1623`). Recognise unresolved pool capital and refuse corrections that would erase it. |
| `_normalise_wallet_check_chain_id()` | Maps only Hypercore (`tradeexecutor/cli/commands/check_wallet.py:42`). Use custody mapping, then report Ethereum Safe USDC, pool shares, collateral and pending withdrawal separately. |

`check-accounts` and `correct-accounts --dry-run` must inspect pending pools without minting, burning, claiming or silently correcting transit away. Explain a pending or ambiguous operation with its trade/hash/history id. Real corrections continue through existing correction mechanisms after execution evidence is reconciled.

Use the default Lagoon portfolio NAV path (`tradeexecutor/ethereum/lagoon/vault.py:633`). Do not extend `create_lighter_vault_valuation_func()`, whose contract is Safe cash plus total account equity (`tradeexecutor/exchange_account/lighter.py:633`).

Prove this decomposition at each transition:

> NAV = home cash and positions + unallocated Ethereum Safe USDC + pool shares and attributable pool transit + unsettled CCTP value, with each amount represented once.

Ethereum bridge positions already subtract capital allocated to satellite investments (`tradeexecutor/state/position.py:965`). Verify pool allocation, pending value and released claim proceeds against that mechanism.

Check CCTP transit in the actual NAV method: `calculate_total_equity()` adds it (`tradeexecutor/state/portfolio.py:789`), but `get_net_asset_value()` currently does not (`:881`). Extend the latter if the phase test demonstrates omission, while removing each receivable once destination holdings are counted.

Keep account equity observational only for this strategy. Bootstrap and uninvested pool collateral must have explicit state attribution; do not infer “cash excluding pools” by subtracting holdings from unverified account equity.

Do not copy Hypercore’s post-trade equity revaluation hook into `StrategyRunner` (`tradeexecutor/strategy/runner.py:735`). Pool reconciliation compares exact shares and phase evidence, not a changing equity-as-quantity balance. Leave Hypercore lockup, residual write-off and allocation branches unchanged unless a focused shared regression identifies a real dependency.

### 5. Public test trades, satellite settlement and startup ordering

**Files:** `tradeexecutor/cli/commands/{start,perform_test_trade}.py`, `tradeexecutor/cli/{testtrade,vault_test_trade}.py`, `tradeexecutor/cli/loop.py`, `tradeexecutor/strategy/execution_model.py`, `tradeexecutor/ethereum/vault/settlement_retry.py`.

Load pool credentials through existing `create_lighter_key_management()` (`tradeexecutor/cli/bootstrap.py:205`).

`start._create_lighter_key_management()` currently returns no configuration unless `Parameters.lighter_cash_management` is enabled (`tradeexecutor/cli/commands/start.py:90`). Allow an explicitly supplied operator record to configure pool execution without enabling account cash policy. Preserve the existing missing-record refusal for cash-management strategies.

Do not use “routing config exists” as the cash-management predicate after this change: `start` currently does so for Lagoon redemption behaviour (`tradeexecutor/cli/commands/start.py:652`). Keep actual strategy cash-policy selection separate from signer availability. Refuse a pool universe without its master signer before moving funds; the public test command currently makes this check only for exchange-account pairs (`tradeexecutor/cli/commands/perform_test_trade.py:308`).

Extend existing public test safety checks:

- Remove the home-chain-only restriction in `make_test_trade()` (`tradeexecutor/cli/testtrade.py:1313`).
- Retain exact pair selection, test ownership, custody balance checks and refusal of unrelated unresolved work.
- Extend `get_test_position()` to recognise supported pending pool buys/exits and satellite resume stages; it currently permits only pending sell tickets (`tradeexecutor/cli/vault_test_trade.py:45`).
- Add pool-specific holding/capacity checks beside the existing small helpers. `is_synchronous_deposit_vault_test()` deliberately accepts only `VaultPricing` managers (`tradeexecutor/cli/vault_test_trade.py:28`); do not pretend pools are synchronous ERC-4626 vaults.
- Resolve existing target execution before treasury sync, then skip a still-pending trade, then create a new request only when none exists.
- Save after each execution leg and on error through the existing callback (`tradeexecutor/cli/commands/perform_test_trade.py:260`).

Normalise satellite destination selection to custody chain 1. `make_test_trade()` currently searches for a bridge to raw pair chain 9998 (`tradeexecutor/cli/testtrade.py:1426`), and `_make_cross_chain_test_trade()` uses that raw chain for destination Web3 (`:443`).

Reuse `_record_satellite_redemption_balance()` and `_get_satellite_bridge_back_amount()` to attribute bridge-back to actual proceeds (`tradeexecutor/cli/testtrade.py:201`, `:239`). Pool and Upshift tests must not drain each other’s Ethereum cash. Preserve the saved pending bridge-back identity and do not create another burn.

#### Upshift coverage already present

The cross-chain path already calls `_resolve_satellite_async_settlement()` after satellite opens/closes (`tradeexecutor/cli/testtrade.py:445`, `:513`). That helper checks `vault_settlement_pending`, not a protocol allow-list, and forwards destination Web3 plus `web3config` (`:935`, `:943`). It therefore provides the required mechanism for queued Upshift exits.

The generic resolver also selects Web3 from saved `vault_chain_id` and builds a satellite transaction builder (`tradeexecutor/ethereum/vault/settlement_retry.py:454`, `:59`). Reuse this unchanged where possible. Add a real satellite regression because the existing Upshift fixture deploys Ethereum as home (`tests/erc_4626/test_upshift_cli.py:46`).

Core USDC’s zero-lag exit alone cannot prove queued satellite recovery. Reuse Sentora’s existing delayed fixture for that focused regression (`strategies/test_only/upshift.py:53`; `tests/erc_4626/test_upshift_cli.py:119`), without adding it to the three-position manual universe.

Extend `_resolve_satellite_async_settlement()` only to dispatch a pending Lighter pool through its own phase resolver. Do not send synthetic pools to `_force_vault_settlement_and_resolve()`’s ERC-4626 manager lookup.

#### Preserve delivered startup ordering

The normal loop already binds the state callback before startup settlement (`tradeexecutor/cli/loop.py:1393`), resolves vault settlements before startup treasury sync (`:1560`), and resolves live tick settlements before valuation/treasury (`tradeexecutor/strategy/runner.py:833`). Do not reimplement or reorder these delivered mechanisms.

Make pool dispatch reachable through that same hook, including scheduled refresh and manual NAV callers that can encounter a consumed claim. `close-position`, `close-all` and `trade-ui` already sweep through `resolve_pending_vault_settlements()` before acting (see "Commands cannot wait for settlement" in `.claude/docs/vault-deposit-redeem.md`); verify that a pending pool exit reaches the pool resolver from each of them and add no command-specific pool branch. In `make_test_trade()`, broaden the early recovery beyond its current selected home-chain close-only case (`tradeexecutor/cli/testtrade.py:1321`) before the treasury call at line 1345.

Keep CCTP startup semantics: an unresolved bridge still halts startup (`tradeexecutor/cli/loop.py:1545`). The manual script must receive/reconcile it before starting a strategy cycle.

### 6. Mixed strategy module

**File:** `strategies/test_only/multi_vault_arbitrum.py` proposed.

Use engine 0.5 and exactly one module-level engine variable:

```python
trading_strategy_engine_version = "0.5"
```

Follow `docs/strategy-parameters.md`. Put normal allocation/cycle settings in `Parameters`; keep deployment outputs, account ids, simulation switches and test amounts out. Load no environment variables in the strategy.

`create_trading_universe()` must declare:

| Target | Identity and metadata |
| --- | --- |
| Home reserve | Native Arbitrum USDC. |
| Atoma Index | `0xcc56410e1a136af0eceb7241c6ae394f4d8b581c`, `atoma_like`, correct 12-decimal shares. |
| Upshift Core USDC | `0xE9B725010A9E419412ed67d0fA5f3A5f40159D32`, Ethereum native USDC denomination, `upshift_like`. |
| Lighter pool | pmalt, `lighter-pool-281474976552918`, through `create_lighter_pool_pair()`, chain 9998. Fallback: Steady Wealth Builder, `lighter-pool-281474976552443`. Both are USDC public pools tagged in `eth_defi/lighter/tags.py:2069` and `:2093`. The script's preflight must confirm the chosen pool is open before any money moves: `fetch_pool_state().status == 0`, operator-share capacity and positive share value through `prepare_pool_amount()` (`eth_defi/lighter/execution.py:366`–`369`), read live from the API, not from the dataset snapshot's `deposit_permission`. A closed or full pool stops the run with the reason; it must not silently select the other pool. Refuse LLP. |
| CCTP | One Arbitrum→Ethereum forward bridge pair; its sells implement Ethereum→Arbitrum return. |

Reuse the hand-built `VaultMetadata` pattern from `strategies/test_only/atoma_async.py:65` and `strategies/test_only/upshift.py:65`. Preserve metadata through translation; `translate_trading_pair()` already copies features, protocol and fee information (`tradeexecutor/strategy/dex_data_translation.py:223`).

Use `create_pair_universe_from_code()` with explicit unique pair/exchange ids and real per-pair chain ids, then the existing CCTP generator. Avoid the reverse-universe helper for this mixed universe unless its metadata handling is extended: it currently replaces `other_data` with only token metadata (`tradeexecutor/strategy/reverse_universe.py:215`).

Implement `decide_trades()` as a small, state-driven lifecycle strategy:

1. Ignore bridge pairs as investment targets.
2. Open each selected protocol once, within actual funding and admission.
3. Skip positions with pending pool phases or ordinary vault settlements.
4. Close an opened position when the existing redemption check permits it.
5. Never reopen a successfully closed target.
6. Let normal CCTP planning fund satellite buys and sweep confirmed idle proceeds on later cycles.
7. If public test trades already own a target, leave those test positions to the manual command rather than creating competing strategy trades.

Use normal `PositionManager` operations. Do not copy the single-chain fixtures’ “no trades anywhere means open every vault” condition into the mixed strategy: bridge trades and bootstrap imports make that condition unsuitable (`strategies/test_only/atoma_async.py:103`; `strategies/test_only/upshift.py:101`).

Test strategy-generated execution separately from the manual test-position run, using fresh normal state and the same module.

### 7. Dual-fork simulation and fake sequencer

**Files:** `tradeexecutor/testing/lighter_core.py`, `tradeexecutor/testing/lighter_simulation.py`, `tradeexecutor/testing/multi_vault_simulation.py` proposed, and the small manual Typer helper.

Extend `FakeLighterCore`; do not create a second fake protocol implementation. Its current account model contains collateral/margin but no pool holdings (`tradeexecutor/testing/lighter_core.py:50`), and its withdrawal fake ignores request ids (`:241`).

Add:

- Current public-pool state, integer supply and operator shares.
- Per-account pool holdings, entry principal and timestamps.
- Mint/burn effects on holdings, collateral, supply and operator fees.
- Deterministic operation hash/nonce/status records, including accepted, pending, rejected and unknown outcomes.
- Withdrawal-history pagination and stable exact ids.
- Secure withdrawal debit, delayed claimability and one-time claimable credit.
- Receipt-based observation of real L1 deposits, key registration and claims.
- Controls for delayed execution and ambiguous/missing history in tests.

Exercise upstream sizing, serialisation and reconciliation rather than replacing `LighterPoolExecutor` with a fake success result. Retain `get_patches()` for existing in-process tests (`tradeexecutor/testing/lighter_core.py:334`).

#### Fork lifetime and command processes

The existing simulation launches only Ethereum and installs process-local patches (`tradeexecutor/testing/lighter_simulation.py:67`, `:102`). Those patches do not survive separate Bash-launched CLI processes.

Add a test-only dual-fork environment that:

1. Launches Arbitrum and Ethereum with their real chain ids.
2. Uses `create_multi_provider_web3()` and the existing fork helpers.
3. Funds the deterministic test deployer with gas and transfers existing fork USDC from holders.
4. Replaces CCTP attesters on both forks.
5. Holds one `FakeLighterCore` for the lifetime of the simulated run.
6. Exposes local RPC URLs and `MULTI_VAULT_SIM_DIR` as process environment in `env.sh`, never as execution checkpoints.

**Decision:** no HTTP façade. The manual Typer helper provides a narrow `cli -- <executor arguments>` entry that installs the fake-core patches in-process and invokes the registered executor Typer application, following `tradeexecutor.testing.lighter_tutorial.run_cli()` (`:303`). Patch the `eth_defi.lighter.execution` seams the pool router consumes (`fetch_pool_state`, `fetch_pool_holdings`, `fetch_next_nonce`, `fetch_operation_status`, `LighterPoolExecutor.submit`, `fetch_withdrawals`, `wait_for_claimable_withdrawal`) in addition to the existing `get_patches()` set, so upstream sizing, serialisation and reconciliation code still runs. Preserve the canonical logical API/signing identity; redirect nothing over the network.

Because each shell step is a separate process, the fake sequencer's venue state (accounts, pool supply, holdings, operations, withdrawal history) is saved to one file at helper exit and reloaded at helper start, the way the detached Anvil processes keep chain state between steps. Keep it, together with the Anvil PIDs, ports and fork logs, in a named simulation directory `MULTI_VAULT_SIM_DIR`, defaulting to `/tmp/multi-vault-sim-<run id>/`, separate from the run directory under `~/.tradingstrategy`: the run directory holds only what a live run would also hold, so a simulated run directory never contains fake venue state and the executor's own state stays in `STATE_FILE`. `simulate-start` creates the directory and records its path in `env.sh`; a reboot clears it and therefore requires a fresh simulated run. If this mechanism proves insufficient, propose the façade separately.

Live mode invokes `poetry run trade-executor`; simulated mode invokes the helper's thin CLI entry with the same command arguments. Both execute the same registered callbacks and normal JSON state store. Add no simulated execution path to the strategy or router.

Keep the Anvil forks alive across shell resumptions: `simulate-start` launches a detached host process that owns both forks through eth-defi's `fork_network_anvil()` (whose local proxy keeps upstream RPC credentials off the Anvil command line) until `simulate-stop`. Fork URLs go to `simulation.json`; stopping renames it, so a later resume is refused. Losing a fork requires a fresh simulated run, not fabrication of replacement chain state from trade logs.

#### CCTP receives

Use `eth_defi.cctp.testing`, following `tests/mainnet_fork/test_xchain_master_vault_multichain.py:129`. Derive message fields from the actual saved burn and its receipt: direction, domains, recipient, amount, fee, nonce and message identity. Never mint the requested amount without checking the burn.

Add a helper subcommand such as `receive-cctp`:

- Require `is_anvil()` on both connections.
- Select pending burns from executor state.
- Craft/forge the matching message and attestation.
- Sign the real destination `receiveMessage` transaction.
- Save its `BlockchainTransaction` on the existing bridge trade before broadcast.
- Verify the receipt and let the existing CCTP retry/accounting path complete the trade.
- On repetition, reconcile the saved receive transaction rather than mint again.

Run it between shell steps after a command leaves a bridge in transit, before resuming the originating command. For the deployment coordinator’s internal bootstrap, supply the equivalent forged attestation through its test-only provider seam while it performs its normal receive call.

Avoid existing simulation shortcuts:

- `CctpBridgeRouting.skip_attestation` marks success after burn alone (`tradeexecutor/ethereum/cctp/routing.py:324`).
- `_materialise_bridge_on_anvil()` directly supplies destination tokens (`tradeexecutor/cli/testtrade.py:126`).

The new rehearsal must disable the former and avoid the latter. Extend the materialisation call’s guard so a real received message is never credited twice. Preserve legacy tests until their callers are deliberately migrated.

While touching receive submission, bind the existing pre-broadcast save and use robust receipt waits: the normal route currently appends and broadcasts without a checkpoint and uses the direct Web3 waiter (`tradeexecutor/ethereum/cctp/routing.py:370`). Apply the same correction to the corresponding retry submission (`tradeexecutor/ethereum/cctp/retry.py:204`).

#### Fork blocks

The pinned dependency defines:

- `ETHEREUM_MIDNIGHT_BLOCK = 25_598_869`.
- `ARBITRUM_MIDNIGHT_BLOCK = 487_039_644`.

These are at `deps/web3-ethereum-defi/eth_defi/testing/fork_blocks.py:56`.

The existing Atoma lifecycle deliberately uses Arbitrum block `519_150_162` (`tests/erc_4626/test_atoma_async_cli.py:47`). Its reviewed implementation cannot be assumed at the older shared midnight block.

Use the existing shared midnight forks for custody/CCTP mapping coverage. Use the reviewed Atoma-compatible fixed block for the full mixed rehearsal through a dedicated `fork_network_anvil()` launch, as `tests/erc_4626/test_atoma_async_cli.py` already does; the pooled `AnvilForkPool` fixtures are keyed to the midnight blocks. Share that block constant with the Atoma fixture rather than redefining it, unless inspection establishes compatible Atoma execution at the shared block. Share that test-support block selection with the existing Atoma fixture; do not silently change global fork constants or use a moving tip.

Live activation cannot run on a private fork’s sequencer. The fake supplies account discovery, collateral credit, registered-key observation and pool sequencing while actual Ethereum deposit/change-key/claim transactions still execute. This proves executor wiring and Guard mechanics, not real Lighter settlement.

### 8. Manual shell lifecycle and resume rules

**Files:** `scripts/lagoon/manual-multi-vault-cli.sh` and `tradeexecutor/testing/multi_vault_manual.py` proposed.

Follow `scripts/lagoon/manual-erc4626-cli.sh:43` and `scripts/lagoon/manual-lighter-subaccounts-cli.sh:137`:

- Bash with `set -Eeuo pipefail`, `umask 077` and a private run directory `MULTI_VAULT_RUN_DIR`, defaulting to `$HOME/.tradingstrategy/examples/multi-vault-cli-<UTC timestamp>` exactly as `manual-erc4626-cli.sh:63` does. State, deployment artefact, vault record, the mode-0600 activation key backup, strategy snapshot, `env.sh` and `run.log` all live there; set `MULTI_VAULT_RUN_DIR` to resume an earlier run.
- Saved `env.sh`, strategy snapshot, normal state/deployment files and logs.
- `run` wrapper with elapsed time, heartbeat, exit propagation and interrupt handling (`scripts/lagoon/manual-erc4626-cli.sh:131`).
- Explicit deployment/subscription flags; reruns continue existing work.
- Only executor CLI commands and the small testing helper perform lifecycle actions.
- No `.done`, `progress.json`, deposit markers or second transaction journal.

Save public configuration and references to required credentials in `env.sh`, never private keys or upstream credential-bearing RPC URLs. Preserve the source strategy snapshot and refuse an unexplained strategy/permission mismatch on resume (`scripts/lagoon/manual-erc4626-cli.sh:95`).

Use outer `SIMULATE=true` to select the rehearsal. Inside each registered executor command set `SIMULATE=false`, retain fork RPC URLs, and use a normal JSON state store. `SimulateStore.sync()` does not persist (`tradeexecutor/state/store.py:151`), so production `--simulate` cannot prove cross-command recovery.

The helper should remain comparable in scope to `erc4626_manual.py`. Extend/share its small universe and phase functions where practical; do not copy the larger tutorial harness. Its current phase selector is address-only (`tradeexecutor/testing/erc4626_manual.py:44`), and its final audit assumes ERC-20 shares on Arbitrum (`:100`).

Provide chain-aware `identity`, `preflight`, `pairs`, `phase`, `audit` and `finished` actions, plus the necessary simulation/receive actions. Phase output must distinguish:

- New opening.
- Bridge receive/reconciliation required.
- Existing pool operation or vault ticket awaiting completion.
- Ready to close.
- Confirmed close with bridge-back remaining.
- Completed.
- Unresolved failure requiring inspection.

The helper must not turn an unrecognised state into a fresh opening.

| Step | Commands and evidence |
| --- | --- |
| 0. Prepare or reconnect | Check tools, fixed targets, signer, chain ids, hub USDC and gas on both chains against the minimum funding table below, and existing run identity. Confirm live that the selected Lighter pool is open for deposits (`status == 0`, capacity, share value), that Atoma `maxDeposit(owner) >= MIN_DEPOSIT` and that Upshift Core USDC accepts deposits, through the existing `check_deposit()` route. In simulated mode start/reconnect the dual-fork environment. |
| 1. Deploy | `lagoon-deploy-vault` with the mixed strategy, Arbitrum hub and coordinated Lighter activation. Verify both Safes/modules, account registration, owner record and bootstrap conservation. |
| 2. Initialise and subscribe | `init` creates the state and imports the 20 USDC bootstrap subscription from the pre-deployment block. Run `check-wallet`, which reports the activation collateral residual; optionally `lagoon-first-deposit` for additional capital. Verify total subscription and Lagoon shares. |
| 3. Lighter buy | `perform-test-trade --pair '(lighter, lighter-pool-281474976552918)' --amount 5 --buy-only --no-test-short`. Resume the bridge and original pool trade until exact shares are confirmed. |
| 4. Lighter close | `perform-test-trade --pair … --close-only --no-test-short`. Resume burn, secure withdrawal and exact-id claim. On a later invocation return confirmed Ethereum cash through the saved bridge path. |
| 5. Atoma buy/close | Use `(arbitrum, 0xcc56410e1a136af0eceb7241c6ae394f4d8b581c)` with separate buy-only and close-only commands. Live mode exits with resume instructions during the lock/epoch wait. Simulation advances time and uses the existing manager’s strict settlement helper. |
| 6. Upshift buy/close | Use `(ethereum, 0xE9B725010A9E419412ed67d0fA5f3A5f40159D32)`. Reuse bridge funding, existing manager execution and saved bridge-back attribution. |
| 7. Reconcile | Run helper `audit`, `check-accounts`, `correct-accounts --dry-run`, `check-wallet` and `show-valuation`. A discrepancy stops the run; no automatic balancing correction or top-up. |
| 8. Return idle satellite capital | Complete saved bridge-backs and use `lagoon-reclaim-satellites` for verified remaining idle capital. Its existing command retries state-recorded CCTP transfers (`tradeexecutor/cli/commands/reclaim_satellites.py:694`). |
| 9. Settle and redeem | Once positions, collateral and bridge transit are clear, run `lagoon-settle --force-valuation`, `show-valuation`, `lagoon-redeem`, then helper `finished`. Honour Guard budgets/cooldowns. |

In simulated mode, run `receive-cctp` between relevant invocations and advance only the protocol timing needed by existing Atoma/Upshift settlement helpers. Never impersonate an operator or mutate fork state in live mode.

Answer the deployment confirmation explicitly with `y`; give other commands no stdin unless their inspected implementation has a documented prompt. Do not blindly answer confirmation to `correct-accounts`; the script uses its preview only. Unexpected prompts must fail rather than hang.

Resume decisions come from the same `STATE_FILE`, saved transaction identities, protocol holdings and investor request/share state:

- Never repeat deployment because its final text report is missing.
- Never repeat a subscription solely because a shell step did not finish.
- Never create another mint, burn, secure withdrawal or CCTP burn while its original execution is unresolved.
- Never open Upshift while a Lighter operation owns the same satellite funds.
- A queued exit is incomplete, not failed; preserve it and print the exact resume command.
- A successful burn is insufficient for pool closure; require Safe claim and subsequent bridge reconciliation.
- Final completion requires zero protocol holdings, no pending claims/transit and zero investor Lagoon shares.

#### Minimum funding

The deployer holds everything on Arbitrum except Ethereum gas; the coordinator bridges the activation capital itself. All USDC returns to the deployer at the end apart from fees and sub-1-USDC Lighter dust. Amounts are the smallest that satisfy each documented protocol rule; the script refuses to start below them.

| Item | Minimum | Source of the rule |
| --- | ---: | --- |
| Bootstrap subscription on the hub, consumed by `lagoon-deploy-vault` | 20 USDC | `LIGHTER_MULTICHAIN_BOOTSTRAP_SUBSCRIPTION` (`eth_defi/erc_4626/vault_protocol/lagoon/lighter.py:46`); includes the 1 USDC activation deposit and the 0.01 USDC bridge-fee ceiling. The remaining ~18.99 USDC stays in the vault as reserve. |
| Atoma Index test deposit | 100 USDC | `MIN_DEPOSIT()` at the reviewed deployment (`eth_defi/erc_4626/vault_protocol/atoma/deposit_redeem.py:219`); reread live in step 0. |
| Lighter pool test deposit | 5 USDC | Lighter's 1 USDC L1 deposit and secure-withdrawal minimum (`LIGHTER_MIN_MAINNET_USDC`, `eth_defi/lighter/api.py:32`) plus margin so the exit is above the minimum after fees; 1 USDC is the activation collateral already on Lighter, so the L1 deposit is 4 USDC. Pools have no share minimum beyond one raw share. |
| Upshift Core USDC test deposit | 5 USDC | No documented protocol minimum; 5 USDC keeps the CCTP round trip and any withdrawal rounding visible. Confirm live in step 0. |
| CCTP fees and cash headroom | 1 USDC | Fast Transfer fee on two Arbitrum→Ethereum burns and two returns at current rates, plus rounding; the planner reads the live quote. |
| **Additional subscription (`SUBSCRIPTION_USDC`)** | **111 USDC** | 100 + 5 + 5 + 1. |
| **Total native Arbitrum USDC on the deployer** | **131 USDC** | 20 bootstrap + 111 subscription. |
| Ethereum gas | 0.02 ETH | The sub-account tutorial's full Ethereum deployment with verification plus ~30 transactions cost 0.0066–0.0074 ETH at 0.04–0.1 gwei (`manual-lighter-subaccounts-cli.sh:35`, `:54`); the satellite needs fewer contracts but adds approve/deposit/claim/CCTP-receive transactions. Budget for a gas-price spike. |
| Arbitrum gas | 0.005 ETH | Hub Lagoon/Safe/Guard deployment, Atoma deposit/request/claim, CCTP burns and settlements at Arbitrum prices. |

The defaults are `AMOUNT_ATOMA=100`, `AMOUNT_LIGHTER=5`, `AMOUNT_UPSHIFT=5` and `SUBSCRIPTION_USDC=111`; an operator can raise any of them, and the script recomputes the required balances. Record actual costs in the script header after the first live run, as the earlier tutorials do.

Expected end-to-end duration is unknown. Existing cash-withdrawal timings do not establish pool withdrawal timing, and Atoma operator settlement is externally controlled. Print elapsed time, current phase and observed eligibility instead of promising a fixed duration.

### 9. Focused validation

Use a small happy-path and refusal/recovery case per changed boundary. Combine related assertions and reuse existing fixture setup.

| Area | Files | Focused checks and environment |
| --- | --- | --- |
| Identity/routing/funding | `tests/test_generic_router_lighter_pool_satellite.py` proposed; extend `tests/ethereum/test_lagoon_universe_config.py` | Unit happy path for mixed routing, 9998→1, shared bridge allocation and metadata round-trip; bad path for missing Ethereum RPC/satellite, malformed pool or wrong token. |
| Deployment | Extend existing Lagoon Lighter deployment tests; add mixed case to `tests/ethereum/test_lagoon_universe_config.py` | Dual shared midnight forks through `AnvilForkPool`: Arbitrum hub activation, Ethereum private/public records and Guard permissions. Interrupt before key registration/final records; prove key and progress survive only in authorised stores. |
| Pool phases | `tests/lighter/test_lighter_pool_routing.py` proposed | Fake-core unit happy path across saved/reloaded operations and fees; bad path combining uncertain submission, exact-id ambiguity and retained allocation. Use real upstream sizing/operation code. |
| Pool value/accounts | `tests/lighter/test_lighter_pool_valuation.py` proposed | Shares, partial burn, collateral, claimable withdrawal and completed claim value once; missing/contradictory API evidence prevents NAV/correction. Include large raw-integer serialisation and one-raw-share residual. |
| Public satellite commands | `tests/lagoon/test_multi_vault_cli.py` proposed | Shared dual forks: Lighter public buy/close with an actual saved bridge receive and Guard claim; refusal of ordinary holdings/wrong signer before broadcast. Assert later-invocation bridge-back. |
| Queued satellite Upshift | Extend `tests/erc_4626/test_upshift_cli.py` or reuse its fixture assertions in the mixed CLI file | Ethereum midnight fork plus Arbitrum hub: Sentora request, reload, operator settlement and startup claim on Ethereum. Preserve one request/hash and credit once. Core USDC remains the manual target. |
| Normal `start` | `tests/lagoon/test_multi_vault_cli.py` | Fresh state, repeated `start --max-cycles 1`; actual strategy opens/closes all three protocols, waits safely, recovers claims and sweeps later. Use the reviewed Atoma-compatible Arbitrum block and Ethereum midnight block. |
| Actual Bash wrapper | `tests/lagoon/test_multi_vault_manual_cli_e2e.py` proposed | Execute `manual-multi-vault-cli.sh` itself in simulated mode, not an equivalent Python sequence. Interrupt/restart a command while the same forks/fake remain alive; complete investor redemption. A second case refuses live endpoints in a simulation helper. |

For CCTP, assert real burn/receive calldata, message identity, Safe recipient, destination receipt and no repeated mint. For Lighter, assert real Safe deposit/claim receipts plus separately labelled fake sequencer evidence.

Reuse Atoma loss/aggregate-refusal and Upshift manager/Guard evidence; add only consumer checks affected by satellite execution. Do not reproduce unchanged upstream adapter suites.

The full mixed test requires the Atoma-compatible block described in section 7. Do not claim it runs at `ARBITRUM_MIDNIGHT_BLOCK` without implementation/ABI evidence. Missing archive access, SDK support, credentials, current pool selection or protocol liquidity remain explicit validation gaps.

Follow repository pytest rules: no test classes, typed fixtures, numbered docstrings with matching numbered body comments, explained mocks, `pytest.approx()` for money and no permanent informational logging. Share helpers under `tradeexecutor.testing`, not imports from the tests tree.

During implementation, run focused files one at a time with the unchanged private environment, Poetry, worktree `PYTHONPATH` and a five-minute command deadline. Use `-n auto` for commands containing more than five tests or multiple slow cases. No validation command is authorised by this read-only planning task.

### 10. Documentation and code discovery

Update existing documentation with each consumer change:

| Path | Update |
| --- | --- |
| `docs/lagoon-lighter-deployment.md` | Add Arbitrum-hub activation, satellite operator-record association, pool commands, bootstrap conservation and the new manual script. Scope its existing Ethereum-home restrictions to that older path (`:58`). |
| `.claude/docs/vault-deposit-redeem.md` | Add one cross-reference identifying the dedicated Lighter pool lifecycle and shared settlement entry point; keep Atoma/Upshift manager guidance canonical. |
| `.claude/docs/hypercore-vault.md` | Cross-reference the new router as another user of sequential execution/state callbacks; explicitly retain separate protocol phases. |
| `.claude/docs/cli-command.md` | Update the home-chain-only close-only wording (`:507`), pool resume semantics and the distinction between outer shell simulation and executor `--simulate`. |
| `tradeexecutor/exchange_account/README-Lighter.md` | Link pool execution and explain that account equity remains separate from pool valuation. |
| `docs/plans/nb05-live-readiness.md` | Link this plan from its Lighter expansion milestone; do not mark NB05 conversion or funded readiness complete. |
| `docs/README.md` | Link the manual script from the existing Lagoon/Lighter documentation entry. |
| `CLAUDE.md` | Add one `.claude/docs` reference-table line pointing to the pool execution guidance before changing pool routing, settlement or NAV. Edit the canonical file once. |

Add short docstring links at `LighterPoolRouting`, its pricing/valuation models and the shared settlement dispatch. Do not write another top-level lifecycle document or duplicate the shared persistence guide.

Record eventual commands, revisions, blocks, transaction evidence and limitations in the existing documentation/reporting convention or PR evidence. This planning task creates no report file.

## Implementation status

Implemented on 3 November 2026 against executor master `6f4563b07` and eth-defi `02b6caf`. The funded run of the same day needed three eth-defi changes: Lighter `entry_timestamp` in milliseconds, the withdrawal history end cursor, and the reviewed Atoma fee-split implementation ([#1722](https://github.com/tradingstrategy-ai/web3-ethereum-defi/pull/1722)). Its results are in the [deployment guide](../lagoon-lighter-deployment.md#arbitrum-hub-with-lighter-public-pools).

| Area | Status |
| --- | --- |
| Pair identity and routing | `is_lighter_pool()`, `get_lighter_pool_index()`, `get_funding_chain_id()`, `normalise_custody_chain_id()`, `create_lighter_pool_pair()` and the `lighter_pool` router. Bridge capital, generic routing, RPC lookup, startup checks, `check-wallet` and deployment translation use the custody mapping. Backtests treat `lighter_pool` exactly as they treated `vault`. |
| Execution | `LighterPoolRouting` runs the buy and sell phases inside `setup_trades()`, saves each phase and each upstream `LighterOperation` (as JSON strings) before submission, and leaves a timed-out wait `vault_settlement_pending` for the shared sweep. |
| Pricing and valuation | `LighterPoolPricing` and `LighterPoolValuator` read live pool state; deposits into a closed, full or LLP pool are refused. Position quantity is raw shares; the asset keeps the dataset's 18 representational decimals. |
| Deployment | `lagoon-deploy-vault --generate-lighter-api-key` with a non-Ethereum hub runs the upstream coordinated activation and writes the operator record for the Ethereum satellite. The key-bearing activation report is deleted after the record is written. |
| Test trades | Satellite `--close-only` is accepted; a pending satellite exit or deposit is resumed rather than requested again; Anvil materialisation is skipped when a real `receiveMessage` delivered the funds. |
| Simulation | `FakeLighterCore` serves pool reads, `sendTx`, nonces, receipts and withdrawal history to the unmodified upstream executor, observes real Safe deposits and is saved between processes. `multi_vault_simulation.py` keeps Arbitrum and Ethereum forks in a detached host process and forges attestations for real CCTP burns, which are then relayed by real `receiveMessage` transactions. |
| Script | `scripts/lagoon/manual-multi-vault-cli.sh` with `tradeexecutor.testing.multi_vault_manual`. |

Deviations from the sections above, found during implementation:

- The first subscription after deployment is not `lagoon-first-deposit`, which refuses a vault that already holds the 20 USDC bootstrap. The script uses the ordinary investor request, `lagoon-settle` and claim, as the Lighter sub-account tutorial does.
- `check-accounts` and `correct-accounts` accept one chain only (`Web3Config.choose_single_chain()`), as for every multichain vault. The script reconciles with the state audit, `check-wallet` and the NAV; extending the account checks to satellites is follow-up work.
- The activation USDC is minted into the pool on the first buy, because the account is dedicated and a buy invests all available collateral. Until then it is outside NAV.
- Not implemented: the exit-classification change in `has_delayed_vault_redemption()` (pools already classify as delayed through their protocol slug) and a `start`-driven test of the strategy's `decide_trades()`; the strategy declares its primary chain and the CCTP planner groups pool funding under Ethereum, but the script exercises the public test-trade commands only.
- After the independent review on 3 November 2026: the trade is marked pending at its first checkpoint, a second pool trade is refused while one is unresolved, invariant failures raise `VaultRequestInvariantViolation`, pending exits are valued at released collateral and held shares net of the estimated operator fee, satellite `--close-only` applies the test-ownership checks, bridge reopening and the tutorial's phase selection follow the funding chain, a fresh sequencer rejection also halts, and burned shares are valued at the pool price until their released collateral is observed.
- Cleanup done: both "safe to delete now" items; from "merge during this plan", the custody mapping, the satellite metadata suppression, the private-record writer and the satellite close-only gate. Not done: moving `LighterRouting` call construction into shared helpers (the pool router builds its own approve, deposit and claim calls), the stale plan corrections and the archive items.

## Validation and acceptance

Completion requires all of the following:

1. A deployed Arbitrum Lagoon vault and one Ethereum satellite with verified Guard permissions, master activation and protected operator record.
2. Both public test commands and actual strategy-generated trades execute Lighter pools, Atoma and satellite Upshift through normal routing.
3. Restart resumes original API operations, EVM claims and bridge receives without duplicate submission or credit.
4. Pending pool/queued-vault proceeds cannot fund buys or bridge-back before confirmation.
5. Account inspection understands pool shares and transit without ERC-20 probes or destructive correction.
6. Bootstrap, fee, share, collateral, pending withdrawal and CCTP transitions preserve attributable NAV once.
7. The actual Bash script completes against persistent dual forks and the fake, including investor redemption.
8. Documentation distinguishes unit, fork, simulated-provider and funded evidence.
9. A separately authorised funded run establishes real Lighter mint/burn/history semantics and complete investor reconciliation.

Before funded use, retain upstream’s unresolved checks: share sizing, entry timestamp units, supported transaction statuses, fee observations and whether total account equity includes pool shares. The pool valuation design remains separate from account equity regardless of an unverified inclusion assumption.

## Delivery and completion

1. **Land identity and custody mapping.**  
   Complete when route discovery, pair translation, RPC/bootstrap checks, CCTP generation/planning and bridge allocation all resolve pools to Ethereum while retaining chain 9998. Preserve existing backtest routes and Hypercore behaviour.

2. **Complete deployment and bootstrap state integration.**  
   Deliver the Arbitrum-hub activation, the key-backup report placement, satellite metadata/private-record fixes and the bootstrap import together. Complete when an interrupted activation leaves its key only in the mode-0600 backup and operator record, `init` books the bootstrap subscription, and `check-wallet` reports the activation collateral residual.

3. **Implement pool phases and ordinary valuation.**  
   Complete when the fake-core happy/bad paths prove persisted operations, exact history identity, retained capital, net fees, remaining shares and idempotent claims. No API operation may be replaced merely because a process restarted.

4. **Extend public commands and satellite recovery.**  
   Complete when buy-only/close-only support both pool phases and satellite Upshift, with ownership checks before broadcast, destination-chain claims before valuation and later-invocation bridge-back.

5. **Deliver the mixed strategy and guarded dual-fork coverage.**  
   Complete when actual `start` cycles trade all three protocols, share one Ethereum satellite, survive reloads and respect pending funding. Document the Atoma fork-block exception.

6. **Deliver and run the actual manual shell rehearsal.**  
   Complete when the Bash wrapper resumes from state across command processes, uses forged real CCTP receives, reconciles NAV/accounts and redeems the investor. Finish the documentation updates and only the directly justified cleanup below.

7. **Perform the funded live run last.**  
   Prepare the concrete pool, vault, account, amounts, fee ceilings and signer configuration for user approval. Run only after that approval. Completion requires real receipt/API evidence, no unresolved exposure and investor redemption. This plan claims no funded result.

Do not push changes, publish a PR or perform production maintenance automatically.

## Open questions

| User decision | Recommended default |
| --- | --- |
| Which public Lighter pool should be used? | Decided: pmalt `lighter-pool-281474976552918`, with Steady Wealth Builder `lighter-pool-281474976552443` as the fallback chosen by the operator, never by the script. Step 0 checks the pool is open live. |
| What live capital and fee ceilings are authorised? | The minimum funding table: 131 native Arbitrum USDC on the deployer (20 bootstrap + 111 subscription), 0.02 ETH on Ethereum, 0.005 ETH on Arbitrum, upstream's 0.01 USDC activation bridge ceiling. Approve the exact live values before spending. |
| Who owns and manages the test Safes? | Use a dedicated test signer as owner/primary manager following the existing manual scripts. Use a separately reviewed owner arrangement for production. |
| Is the upstream key-bearing activation report acceptable as the pre-registration key backup? | Yes: it is the multichain equivalent of the existing `--lighter-key-backup-dir` convention. Keep it mode 0600 inside the private run directory and delete it after the operator record is written. Revisit only if a funded run shows recovery needs more. |
| May the simulated sequencer keep its venue state in a file between shell steps? | Decided: yes, under `/tmp/multi-vault-sim-<run id>/` with the fork metadata, never in the `~/.tradingstrategy` run directory. It is test-only venue state, the counterpart of Anvil's chain state, and never read by the executor. Recorded because `docs/state-persistence.md` requires explicit permission for any file that could be mistaken for executor state. |
| How is the 1 USDC activation collateral carried before the first pool buy? | As a documented bootstrap residual read from the deployment artefact and the live account, excluded from NAV and consumed by the first pool deposit. No state field. |
| Which Arbitrum block should the full rehearsal use? | Use reviewed Atoma block `519_150_162`; retain shared midnight blocks for unchanged custody/CCTP coverage. Do not advance global constants merely to make this test pass. |
| How should live waits be operated? | Exit with saved pending state and a resume command. Use repeated invocations or the normal daemon; do not keep the shell waiting indefinitely for an external operator. |
| What should happen to an unattributable withdrawal or material residual? | Halt and retain value/evidence. Do not choose a history row heuristically, write off shares automatically or add capital to force completion. |

## Cleanup and simplification opportunities

These findings identify redundancy and conflicting instructions. Code authorship cannot be inferred from them; “AI-generated” provenance is unverified.

### Safe to delete now

| Evidence | Finding | Suggested action |
| --- | --- | --- |
| `tradeexecutor/state/position.py:874` and `:896` | `TradingPosition.get_current_price()` is defined twice; the later identical implementation shadows the first. | **Delete** the first definition when touching position valuation. |
| `tradeexecutor/strategy/trading_strategy_universe.py:2228` | Both branches of the nested Hypercore test assign `ExchangeType.erc_4626_vault`. | **Merge** into the single vault assignment. This removes no protocol behaviour. |

No inspected large command, tutorial, report or plan is established as safe for immediate wholesale deletion.

### Merge during this plan

| Evidence | Finding and reason | Suggested action |
| --- | --- | --- |
| `tradeexecutor/cli/commands/lagoon_deploy_vault.py:2095` | The source-only activation comment and assignments contradict the required satellite activation. | **Delete/replace** that branch for coordinated activation while retaining the existing single-chain path. |
| `tradeexecutor/cli/commands/lagoon_deploy_vault.py:1059` | Satellite account metadata is deliberately suppressed, preventing normal discovery. | **Delete** the suppression and use the existing public-field filter. |
| `tradeexecutor/cli/commands/lagoon_deploy_vault.py:433`, `:2196` | Private-record writing is concentrated after deployment; coordinated activation needs the same record before key registration. | **Extend** the existing writer. Do not add another backup/report writer. |
| `tradeexecutor/ethereum/lagoon/universe_config.py:53`; `tradeexecutor/ethereum/web3config.py:76`; `tradeexecutor/strategy/generic/generic_router.py:187`; `tradeexecutor/cli/commands/check_wallet.py:42` | Repeated Hypercore-only custody mappings would otherwise become repeated Lighter branches. | **Merge** the two explicit mappings into one small identifier-level function used at these boundaries. |
| `tradeexecutor/ethereum/lighter/lighter_routing.py:347`, `:580` | Pool routing needs the same guarded custody calls, but their methods assume exchange-account trade data and home custody. | **Move** only call construction into small explicit-input helpers. Keep pool and account lifecycles separate. |
| `tradeexecutor/cli/testtrade.py:1313`, `:483` | The public front gate refuses satellite close-only although the cross-chain implementation already has a close-only branch. | **Merge** safety checks and resume ordering into the existing route; remove the contradictory gate. |
| `tradeexecutor/testing/erc4626_manual.py:44`, `:92` | Phase/audit logic is useful but fixed to addresses and Arbitrum ERC-20 shares. | **Extend/share** the small functions for chain-aware targets; do not copy the larger tutorial harness. |
| `docs/plans/lighter-accounting-commands-implementation.md:14` versus `tradeexecutor/cli/commands/perform_test_trade.py:260` | The plan says public test trades do not checkpoint before submission; current code installs the callback. | **Archive or correct** the obsolete decision and link to current persistence guidance. |
| `docs/plans/lagoon-lighter-cli-manual-test.md:203`, `:233` | Historical design prescribes marker files and says test trades save only at completion. Those instructions are unsuitable for the new script and conflict with current callbacks. | **Mark historical**, retain dated run evidence, and point current usage to the scripts/state guide. |

### Leave, with bounded follow-up

| Evidence | Assessment | Suggested action |
| --- | --- | --- |
| `tradeexecutor/cli/vault_trade/runner.py:1688` | The standalone `VaultTestBatchRunner._execute_real()` delegates to the shared test-trade function at line 1701. It is a catalogue/batch diagnostic layer, not a wholly separate execution stack. | **Leave** the batch runner, state reporting and simulation infrastructure. Extend shared execution in `cli/testtrade.py`; do not add a third runner. |
| `tradeexecutor/cli/vault_trade/state.py:700`; `tradeexecutor/cli/vault_trade/simulation.py:50` | Diagnostic positions, attempt results and isolated fork infrastructure serve standalone catalogue testing. Their removal is not justified by this manual lifecycle. | **Leave**; do not reuse diagnostic book-close or attempt recovery as live pool accounting. |
| `tradeexecutor/cli/commands/lagoon_lighter_test_trade.py:357`, `:381`; `tradeexecutor/cli/testtrade.py:1065` | The dedicated command opens/closes an ETH perpetual; generic Lighter test support moves account cash. The new pool route replaces neither capability. | **Leave** both commands. Do not add pool support to the perpetual diagnostic. |
| `tradeexecutor/cli/commands/lagoon_lighter_test_trade.py:119`, `:206`, `:442` versus `tradeexecutor/exchange_account/lighter_operator.py:43`, `:65`, `:162` | Operator-record types/loaders and withdrawal wrappers remain duplicated despite the earlier consolidation proposal in `docs/plans/lighter-multi-subaccount-implementation.md:398`. | **Merge** only the credential loader/type when changing satellite record association, preserving command error behaviour. Defer broader withdrawal migration because matching/recovery semantics differ. |
| `tradeexecutor/cli/commands/lagoon_lighter_test_trade.py:263` | The dedicated perpetual diagnostic has its own recovery journal. It remains actively used and is not the pool lifecycle. | **Leave** existing journal recovery intact. Do not copy or expand it. Any removal needs a separate migration for outstanding diagnostic runs. |
| `tradeexecutor/testing/lighter_tutorial.py:137`, `:184`; `tradeexecutor/testing/lighter_manual.py:231`, `:516` | Existing tutorial progress files and manual marker-based actions overlap operationally, but callers still use them. | **Leave** existing runs compatible. Build the new helper from the smaller ERC-4626 pattern; do not inherit these persistence mechanisms. |
| `scripts/lagoon/manual-trade-executor-lighter.py`; `manual-trade-executor-lighter-subaccounts.py:1`; `manual-trade-executor-lighter-transfers.py:17`; `manual-lighter-subaccounts-cli.sh:113` | The scripts cover different paths: perpetual trading, external-trader cash policy, executor-scripted transfers and shell CLI interactions. Wholesale redundancy is unproven. | **Leave**. Add a short selection link in existing documentation so operators choose the appropriate tutorial. |
| `tradeexecutor/testing/lighter_scripted_transfers.py:60`, `:147`; `tradeexecutor/testing/lighter_share_price_timelines.py:335`, `:443` | One supplies trade schedules/assertions; the other replays flow-neutral share-price accounting. They are not duplicate sequencers. | **Leave** and reuse applicable assertions only. Do not extend either into pool execution. |
| `tradeexecutor/ethereum/vault/hypercore_routing.py:826`, `:1318`, `:1417`, `:1497`, `:1765` | Multiple verifiers implement distinct Hypercore balance transitions; copying them would duplicate protocol-specific recovery code. | **Leave** Hypercore intact. Use upstream pool waits and the existing robust receipt helper. |
| `tradeexecutor/ethereum/vault/hypercore_routing.py:2659`, `:2703`, `:3854` | The residual planner always returns no second-stage request, leaving the second-stage executor unreachable through this caller. Persisted metadata may still matter to recovery. | **Delete** the unreachable executor/caller branch only in a separate bounded cleanup after checking recovery consumers. Do not copy it into pool routing. |
| `tradeexecutor/ethereum/ethereum_protocol_adapters.py:963`, `:1000`, `:1056` | `_auto_discover_*` methods repeat discovery structure but establish different account, custody and NAV contracts. A generic discovery framework would conceal those differences. | **Leave** the existing methods. Add the direct pool adapter branch without another account-NAV auto-discovery system. |
| `tradeexecutor/cli/commands/lagoon_deploy_vault.py:1153`, `:2285` | The inline Safe ABI is still consumed by Guard migration instructions. Its size alone does not establish dead code. | **Leave** unless a separate ABI-source consolidation is justified. |

### Plans and reports to archive or retain

| Path and evidence | Suggested action |
| --- | --- |
| `docs/plans/nb05-erc4626-live-support.md:3` | **Archive** as superseded research; retain a redirect to `nb05-live-readiness.md` and preserve useful linked protocol research. |
| `docs/plans/lighter-accounting-commands-compatibility.md:10` | **Archive** the baseline analysis once incoming links are updated; it already records resolved gaps and points to implementation. |
| `docs/plans/lighter-accounting-commands-implementation.md:14` | **Archive/correct** its contradicted no-checkpoint guidance before treating it as an active implementation reference. |
| `docs/plans/lagoon-lighter-manual-test.md:25` | **Archive** the pre-implementation design after preserving deviations and linking current tutorial usage. Its amount-based withdrawal matching at line 40 must not guide pool execution. |
| `docs/plans/lagoon-lighter-api-key-deployment.md:29` | **Archive candidate** after current deployment guidance contains its durable decisions; do not delete blindly because it contains implementation/review context. |
| `docs/plans/nb05-generic-erc4626-test-trade-subplan.md:137` | **Retain as delivered history**, clearly separated from remaining work; its lifecycle and funded references remain useful. |
| `docs/plans/nb05-atoma-integration-subplan.md:24` | **Retain as delivered history**, with the existing validation-report link. Do not re-plan its completed manager integration. |
| `docs/plans/lighter-cash-management-simplification.md:17` | **Leave as historical scope**, labelled for exchange-account startup behaviour; do not apply its “never resume” instruction to the accepted pool lifecycle. |
| `docs/reports/erc4626-manual-live-2026-10-29.md:1` | **Retain** funded command and transaction evidence. It is not superseded by a new plan. |
| `docs/reports/atoma-executor-anvil-2026-10-30.md:1` | **Retain** fork evidence and concessions. |
| `docs/reports/nb05-eligibility-corrected-backtest-2026-11-02.md` and its audit Parquet | **Retain** financial/eligibility evidence referenced by `docs/plans/nb05-live-readiness.md:15`. |

Archiving must preserve or update incoming links. Do not delete reports because their associated implementation plan is complete, and do not turn this integration into a repository-wide documentation reorganisation.