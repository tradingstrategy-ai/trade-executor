# Vault-test PR #1602 follow-ups

## Objective

Finish the trade-executor work exposed by the 2026-08-03 warm-cache rerun of
the 129-vault `vault-test-trade --auto-simulated` matrix. The run completed
without infrastructure failures after every chunk retained Ethereum as the
simulated Lagoon primary chain. The remaining work is to make reports
self-contained and to remove the one executor-owned satellite redemption
accounting revert.

Protocol-specific admission and liquidity decisions remain eth-defi work. The
executor must consume typed eth-defi results and must not add vault-address or
protocol-name exceptions.

## Evidence

The rerun used the following pinned fork blocks, chosen at the preceding UTC
midnight. Every result recorded `lagoon_deployment.primary_chain_id = 1`.

| Chain | Chain id | Fork block |
|---|---:|---:|
| Ethereum | 1 | 25670641 |
| Monad | 143 | 92626146 |
| Arbitrum | 42161 | 490490945 |
| Avalanche C-chain | 43114 | 91878709 |
| Base | 8453 | 49462926 |

The final report was generated with trade-executor commit
`432227eba017d7a53d7fd198fcd3ca8475d99eba` and eth-defi commit
`34351fca05b7662a9c670dfa34a7b4c3d2217c25`.

| Result | Count |
|---|---:|
| `success (simulated)` | 88 |
| `success-deposit-closed` | 18 |
| `success_simulated_with_intervention` | 3 |
| `whitelisting-needed` | 14 |
| `transaction_reverted` | 3 |
| `infrastructure_failed` | 0 |

## Transaction reverts

| Vault name | Root cause | Suggested fix |
|---|---|---|
| TAU InfiniFi Pointsmax (Ethereum, IPOR Fusion) | Deposit succeeds, but the full redemption reverts with `WithdrawManagerInvalidSharesToRelease(uint256,uint256,uint256)` (`0x3c71a1e7`). The generic IPOR manager only performs an exact redemption-liquidity simulation for the characterised Base Autopilot vault, so this deployment reaches broadcast without a capacity preflight. | **Deferred to eth-defi.** Characterise this PlasmaVault's withdrawal-manager limits, simulate its exact `redeem()` path for the caller and return `VaultFlowUnavailable(preflight_result="redemption_capacity_limited")` before signing. Add an Ethereum fork regression for this vault. |
| Chimi USDC Vault Turbo (Base, Yearn) | The CCTP bridge and USDC approval succeed, but the direct vault `deposit()` returns status 0 with no decoded revert payload. The Yearn/generic ERC-4626 path has no vault-specific admission preflight or error decoder for this deployment. | **Deferred to eth-defi.** Add a Yearn manager preflight using the exact sender and deposit amount, then expose either a typed closure/admission result or the decoded revert reason. Add a Base fork regression. Trade-executor should report the typed result without a Yearn-specific branch. |
| Aerodrome USDC (Base, 40acres) | The satellite deposit succeeds, but the executor constructs a full close from the accounting position quantity. The satellite custody account holds fewer vault shares, and `redeem()` reverts with `ERC20: transfer amount exceeds balance`. | **Trade-executor fix.** Read the satellite vault-share balance for the custody account immediately before constructing the close. Cap the redemption quantity to that observed balance, record planned versus observed shares in attempt metadata, and classify a zero balance as an explicit unavailable outcome. Add a Base fork regression covering lossy share accounting. |

Do not mark a status-0 receipt as success. The target outcomes are a successful
40acres full lifecycle and typed, pre-broadcast availability results for the
two eth-defi adapter gaps.

## Work item 1: reconcile satellite shares before close

Update `_make_cross_chain_test_trade()` in `tradeexecutor/cli/testtrade.py`.

1. Resolve the satellite vault share token and custody address using the same
   routing data used for the satellite trade.
2. Read the custody account's raw share balance immediately before
   `PositionManager.close_position()` creates the redemption trade.
3. Compare that balance with the position's available trading quantity. Build
   the close for the smaller amount and retain both values, token address and
   custody address under `vault_test_attempt.outcome_data`.
4. When the observed balance is zero, stop before broadcast with a typed
   `redemption_unavailable` result and a useful detail. Never construct a zero
   quantity sell.
5. Retain the existing post-redemption USDC reconciliation for CCTP bridge-back.
   Share reconciliation happens before `redeem()`; USDC reconciliation happens
   after the successful receipt and protects the subsequent burn amount.

Tests:

- A focused unit test where recorded position shares exceed the observed
  custody balance and the generated sell is capped.
- A focused unit test where the observed balance is zero and no sell is signed.
- A Base fork regression for Aerodrome USDC confirming the full deposit,
  redemption and bridge-back lifecycle.

## Work item 2: make report aggregation command-owned

The chunked run currently requires external `jq` consolidation because each
chunk report contains only its own rows. Make the command persist a final
aggregate report after the final chunk, with exactly one row per requested
vault id and a deterministic sort by input position.

The aggregate must preserve the raw result string. Successful legacy attempts
whose result is absent must be exported as an explicit
`success_simulated`/`success_real` value, rather than requiring consumers to
interpret `null` as success.

Tests:

- Two synthetic chunk reports produce a correctly ordered aggregate report.
- A legacy successful attempt exports an explicit result without mutating the
  saved state record.
- Duplicate vault ids and missing requested ids fail the finalisation step
  clearly.

## Work item 3: persist cache and provider diagnostics

Attach run and per-attempt diagnostics to the JSON report:

- selected fork block and chain id;
- cache directory, whether it was warm before the run and byte counts before
  and after synchronisation;
- the simulated primary chain id and Anvil generation; and
- all upstream RPC provider domains contacted through the proxy, together with
  request count, failed request count and last redacted error per domain.

The finalisation path must synchronise the warm cache even when a command or
RPC call fails. Redact URLs, API keys and headers before serialising errors.

Tests:

- A failed simulated command still invokes cache synchronisation.
- Provider-domain aggregation preserves multiple upstream failures without
  storing provider URLs or credentials.
- A Base-only chunk includes the Ethereum primary chain and its pinned block.

## Work item 4: consume eth-defi follow-ups

After the two adapter changes land, update the eth-defi submodule pointer and
add no compatibility fallback. Trade-executor must only map the shared typed
`VaultFlowUnavailable` fields to its persisted result and report detail.

Rerun the three explicit vault ids at the same pinned blocks first, then rerun
the full 129-vault matrix with a warm cache. Publish the aggregate report and
the selected-block table in the PR comment.

## Acceptance criteria

- Aerodrome USDC no longer broadcasts a redemption larger than the satellite
  custody share balance.
- TAU and Chimi stop before broadcast with typed eth-defi outcomes once their
  adapter fixes are available.
- A chunked invocation emits one complete, deterministic report without an
  external consolidation command.
- Failed runs retain useful cache and provider-domain evidence.
- The 129-vault rerun records zero `infrastructure_failed` rows and one stable
  Ethereum primary chain across all chunks.
