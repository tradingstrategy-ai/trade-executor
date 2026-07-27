# Historical repair fork tests

Each repair test reproduces one recorded production accounting or execution
failure from its original block. The pinned block and matching state dump are
part of the regression case: do not replace either with the chain tip or a
shared latest-block fixture.

These tests are marked `warm_rpc_test_group`. They use an isolated Anvil process
because repair commands change on-chain state, while the project Foundry cache
preloads the fixed-block RPC responses before Anvil starts. The `xdist_group`
contains the chain and original block, so tests never share a mutated fork.

The current repair blocks are:

- Base: `27664435`, `30814817`, `32040184`, `32092657`
- Ethereum: `20377193`, `20409979`, `20438662`
- Polygon: `60719175`, `60855854`, `62255643`

The legacy Polygon broadcast repair at `49132512` also carries an isolated
warm-RPC marker, but requires a dedicated signing key and is normally skipped.

To add a repair case:

1. Capture the production state and record the chain and failure block in the test.
2. Fork exactly that block and keep the test's state file immutable by copying it
   to `tmp_path` when the repair command writes it.
3. Mark the module with `warm_rpc_test_group` and a unique `xdist_group`.
4. Generate and commit `tests/rpc_cache_seed/<network>/<block>/storage.json` by
   running the focused test with an empty external `FOUNDRY_RPC_CACHE_DIR`.
