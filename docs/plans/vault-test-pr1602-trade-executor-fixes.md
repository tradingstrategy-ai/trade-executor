# Vault-test PR #1602 trade-executor fixes

## Objective

Make one `vault-test-trade --auto-simulated` invocation reproducible and
diagnosable across the complete production vault list. This plan contains only
trade-executor changes. It does not assume that an on-chain vault revert is an
executor accounting bug until the recorded call evidence proves that ownership.

The target run boundary is one command invocation containing every requested
vault id. Manual shell-level chunking remains useful for research, but it is not
part of the command's report contract and will not be supported by an implicit
cross-process aggregation protocol.

## Confirmed baseline

The following behaviour is already implemented and must be preserved:

- operation-scoped trade cursors prevent an earlier lifecycle operation from
  determining a later failure result;
- mined status-0 receipts are classified as `transaction_reverted`;
- typed preflight, receipt-analysis, state-inference and infrastructure results
  remain distinct;
- unknown historical result strings survive state round-trip; and
- attempts already record source revisions, fork blocks, Anvil generation and
  Lagoon deployment addresses.

The 2026-08-03 research rerun demonstrated one confirmed harness defect: using
the first vault in each shell chunk as the simulated Lagoon primary changed the
deployment topology according to chunk boundaries. Holding Ethereum as the
primary produced 13 successful chunks and zero `infrastructure_failed` rows for
that run.

The same run observed an Aerodrome USDC redemption revert. Its executor
ownership is not yet proven. Vault routing already reconciles planned shares
against the transaction builder's custody balance and rejects material
shortfalls, so the existing reconciliation path must be traced before changing
position accounting.

## Scope

In scope:

- stable simulated Lagoon topology;
- optional externally documented fixed fork blocks, automatic midnight fallback
  and persistent Foundry cache handling;
- upstream RPC provider-domain diagnostics across Anvil replacements;
- report rows tied to the current attempt and explicit success results; and
- a pinned-fork Aerodrome investigation that assigns ownership from evidence.

Out of scope:

- Yearn, IPOR Fusion, 40acres or other protocol-specific transaction logic;
- decoding protocol custom errors in trade-executor;
- treating a status-0 receipt as success;
- aggregating unrelated command invocations; and
- persisting RPC URLs, credentials, headers or signed transactions.

## Work item 1: make the simulated primary explicit

The simulated deployment and runtime currently derive the primary from
`vault_specs[0]`. Replace this implicit contract with one selected primary that
is passed through setup, RPC filtering, fork selection and deployment.

1. Select Ethereum when `json_rpc_ethereum` is configured. If it is absent,
   fall back to the first requested vault chain.
2. Keep the selected primary RPC when filtering configuration, even when the
   requested vault list contains no vault on that chain.
3. Build the selected chain list as primary first, followed by requested vault
   chains with stable deduplication.
4. Permit the primary chain to have no vault specifications. It still owns the
   simulated Lagoon vault, source Safe and CCTP hub.
5. Pass the selected primary explicitly to `start_simulated_vault_runtime()` and
   `deploy_simulated_lagoon_multichain()`; do not recalculate it downstream.

Tests:

- A Base-only list with Ethereum RPC selects Ethereum and keeps only Ethereum
  and Base RPCs.
- A mixed list produces the same primary regardless of vault ordering.
- Without Ethereum RPC, the first requested chain is the documented fallback.
- A runtime integration test starts Ethereum and Base forks for a Base-only
  list and records `primary_chain_id = 1`.

## Work item 2: accept fixed blocks and define cache ownership

The command needs two fork-block modes. Integration tests and published
experiments use an explicit fixed map. Ad hoc command runs may resolve a new map
from the preceding naive UTC midnight. Both modes produce one immutable map
before generation one and reuse it unchanged for every Anvil replacement.

### Fixed mode

1. Add a repeatable CLI option:

   ```text
   --simulation-fork-block CHAIN_ID:BLOCK_NUMBER
   ```

   Example:

   ```text
   --simulation-fork-block 1:25670641 \
   --simulation-fork-block 8453:49462926
   ```

2. Parse the values with one setup helper into `dict[ChainId, int]`. Reject
   unknown chain ids, malformed or non-positive block numbers and duplicate
   chain entries before creating any Web3 or Anvil process.
3. Fixed mode is enabled when at least one option is supplied. Require exactly
   one block for every selected chain, including the simulated primary. Reject
   missing and unrelated chain entries instead of combining fixed and automatic
   blocks in one run.
4. The block values are experiment inputs. Keep their durable documentation in
   the relevant GitHub issue, PR comment or plan/report file and pass them to
   the command explicitly. Do not add one experiment's blocks to production
   constants.
5. Add an optional
   `--simulation-fork-block-reference TEXT` value for the external issue URL,
   PR comment URL or repository document path. Persist it as provenance; do not
   fetch or interpret it during execution.

### Automatic mode

1. When no fixed block option is supplied, choose the most recent naive UTC
   midnight once at command setup.
2. Resolve the latest block at or before that timestamp for every selected chain
   through `create_multi_provider_web3()` and a timestamp binary search.
3. Fail setup with the chain id and redacted provider-domain diagnostics when a
   selected chain cannot resolve or serve the historical block. Do not silently
   fall back to a live tip.

### Shared provenance and cache lifecycle

1. Persist the complete block map in run and attempt provenance with
   `fork_block_source` equal to `explicit` or `automatic_midnight`. In automatic
   mode also persist the selected midnight timestamp. In fixed mode persist the
   optional external reference.
2. Pass the complete map to every Anvil generation. Replacement code must accept
   the map from generation one and must not parse options or query upstream tips
   again.
3. Use `~/.tradingstrategy/vaults/rpc-cache` as the durable experiment cache.
   Because Foundry cache entries include the fork block, different documented
   experiments can safely share this directory.
4. Seed Foundry's actual cache directory from the durable cache before generation one.
5. Account for the current Foundry behaviour that writes fork `storage.json`
   under `~/.foundry/cache/rpc`: seed that directory from the durable cache
   before startup and copy it back with overwrite after shutdown.
6. Perform copy-back in cleanup on successful completion, setup failure,
   attempt failure and Anvil replacement. A cache-copy error is reported without
   masking the original command error.
7. Do not override `FOUNDRY_RPC_CACHE_DIR`: current Foundry writes fork cache
   data to `~/.foundry/cache/rpc`, so this workflow explicitly mirrors that
   directory instead of claiming an environment override changes Anvil output.

Tests:

- CLI parsing accepts a complete Ethereum/Base fixed map and rejects malformed,
  duplicate, missing and unrelated entries.
- A Base-only vault list still requires fixed blocks for both Base and its
  Ethereum primary.
- A fixed-mode integration test uses checked-in midnight block values and
  asserts that no automatic block-resolution call occurs.
- Automatic resolution returns the latest block whose timestamp is not after
  the target for sparse and dense block sequences.
- Provenance distinguishes explicit and automatic maps and retains the external
  reference only as inert text.
- Every replacement receives the exact generation-one block map.
- Success and injected setup/close failures all execute cache copy-back.

## Work item 3: retain provider diagnostics before teardown

An infrastructure result must identify all upstream provider domains involved
in the failed generation without exposing complete URLs.

1. Before replacing or closing a failed runtime, harvest each Anvil RPC proxy's
   per-provider request count, failure count, method counts and recent redacted
   errors.
2. Accumulate observations by vault id, simulation generation, chain id and
   provider domain. Do not discard generation-one observations when a retry
   fails in generation two.
3. For a failed chain with no proxy counters, record the configured upstream
   domains with `observed_failure_count = 0` and a diagnostic explaining that
   the proxy did not classify the failure. This covers single-provider setups
   and HTTP-200 invalid responses without inventing a provider failure.
4. Attach accumulated diagnostics only to a terminal infrastructure failure.
   Clear them after a successful clean rerun.
5. Apply the existing vault-test redaction to every stored error field and test
   that API keys, URL paths, query strings and headers are absent.

Tests:

- Failures from two domains and two generations are both retained.
- A single configured provider is recorded as involved without a fabricated
  non-zero failure count.
- A successful retry clears transient provider-failure metadata.
- Runtime replacement harvests diagnostics before closing the old proxies.

## Work item 4: make the current invocation's report authoritative

One invocation already knows its complete ordered vault list and receives one
row from the runner for each handled result. Tighten this existing report path;
do not add cross-invocation chunk aggregation.

1. Add a run id at command setup and copy it into every attempt and report row.
2. Resolve report attempt metadata by the row's attempt id and run id, rather
   than selecting the latest historical position for the vault address.
3. Validate that the final rows contain each requested vault id exactly once and
   in request order. Report a command-level incomplete status when an unhandled
   command failure prevents this invariant.
4. Export an explicit `success_simulated` or `success_real` presentation result
   when a successful legacy attempt has no raw result. Do not write this
   normalisation back to historical state.
5. Write a partial report from the command cleanup path when setup progressed
   far enough to establish the run id and requested list. Include attempted and
   unattempted ids explicitly.
6. Define report stability as identical membership, ordering, selected blocks
   and result classification. UUIDs, timestamps, transaction hashes and
   deployment addresses are intentionally volatile.

Tests:

- A state containing an older result for the same vault exports the attempt
  belonging to the current row.
- Duplicate and missing current-run rows produce an incomplete report instead
  of silently using stale state.
- Result-less simulated and real successes export explicit presentation values
  while their persisted raw metadata remains unchanged.
- A controlled command failure writes attempted and unattempted rows with a
  command-level incomplete status.

## Work item 5: diagnose Aerodrome before assigning a fix

Reproduce Base Aerodrome USDC through fixed CLI mode using the externally
documented Ethereum block `25670641` and Base block `49462926`. The integration
test may pass the same fixed map directly to the runtime setup helper. Add
diagnostics at the generic vault routing boundary; do not add a 40acres branch.

Capture before broadcast:

- accounting position quantity and planned raw redemption shares;
- transaction builder delivery address;
- resolved vault share-token address and raw custody balance;
- output or exception from `reconcile_vault_redemption_amount()`;
- the manager-generated target, selector and raw share amount; and
- the result of an `eth_call` of the exact wrapped redemption from the actual
  sender at the pre-broadcast block.

Then assign ownership using this decision table:

| Evidence | Ownership and next action |
|---|---|
| Custody shares are materially below planned shares and routing does not stop | Fix generic trade-executor reconciliation or custody-address selection. Preserve the real-execution material-shortfall guard and prove state contains no phantom closed quantity. |
| Custody shares cover the request and the exact preflight call reverts | The failure is a vault/adapter availability problem. Move it to the eth-defi follow-up and make no executor accounting change. |
| Exact preflight succeeds but the unchanged transaction reverts | Trace sender, wrapper, calldata and state mutations between preflight and broadcast; fix the generic executor boundary demonstrated by the mismatch. |
| Reconciled shares differ only within the configured epsilon | Verify raw/decimal conversion and receipt-accounted minted shares. Change epsilon behaviour only with a protocol-neutral rounding invariant and regression. |

Any executor fix selected by this gate must include a focused unit test and the
pinned Base fork regression. A partial close is not acceptable unless the plan
also defines the remaining position quantity and proves simulated cleanup does
not hide it.

## Implementation order

1. Land explicit primary selection and its runtime regression.
2. Land fixed-block CLI input, automatic midnight fallback and cache lifecycle
   handling.
3. Harvest and persist provider diagnostics across replacements.
4. Tie reporting to the current run and make success presentation explicit.
5. Reproduce Aerodrome with the diagnostic gate and implement only the fix
   supported by its evidence.
6. Document one complete fixed block map in the PR, pass that map to the full
   129-vault invocation with a warm cache, and publish the report summary.

## Acceptance criteria

- Vault ordering cannot change the simulated primary when Ethereum RPC is
  configured.
- Fixed experiments and integration tests use the externally documented block
  map without querying block discovery; automatic runs record their resolved
  preceding-midnight map.
- Every generation within a run uses the same complete block map.
- Cache copy-back runs on every exit path without masking the primary error.
- A terminal infrastructure result lists every involved provider domain and
  distinguishes observed proxy failures from configured-only providers.
- The report contains one ordered row per requested vault, tied to the current
  run, with an explicit presentation result.
- Aerodrome ownership and any resulting code change are supported by captured
  preflight and transaction evidence.
- The 2026-08-04 fixed-block warm-cache rerun recorded 129 unique vaults and
  zero `infrastructure_failed` results. It retained two transaction reverts:
  TAU InfiniFi Pointsmax (custom sell error) and Aerodrome USDC (ERC-20 balance
  shortfall during satellite close). This is evidence for that experiment, not
  a claim about future provider availability.
