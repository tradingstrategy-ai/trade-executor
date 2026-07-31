# CI test audit – 2026-07-31

## Scope and method

This audit covers every `test_*.py` file on `master` and the last successful
pull-request runs after the CI speed improvements:

- Automated suite: [run 30606701094](https://github.com/tradingstrategy-ai/trade-executor/actions/runs/30606701094)
- Slow suite: [run 30606701090](https://github.com/tradingstrategy-ai/trade-executor/actions/runs/30606701090)

The repository contains 324 test files. The successful CI runs exercised 1,494
collected tests: 1,323 passed and 171 skipped. Static inspection found 202
`skip` or `skipif` annotations, 47 `flaky` annotations, and 42
`slow_test_group` annotations. The static skip figure is intentionally higher:
it includes conditionally skipped tests that are enabled when the matching
secret, RPC endpoint, platform, or feature is available.

## Findings

### Permanently skipped or retired coverage

The highest-confidence removal candidates are tests that are already disabled
with an explicit retirement reason. They add little runtime because pytest
skips them, but they obscure the maintained test surface and still impose
collection and maintenance cost.

- Enzyme test modules are marked `No longer maintained`.
- Velvet test modules are marked `Velvet no longer supported`.
- Old 1delta test modules are marked `The old 1delta API is no longer supported`.
- Individual Lagoon CLI tests are marked `No longer supported` or
  `Unmaintained test`.
- `test_data_age` explicitly says that its live integration tests need a
  rewrite.
- `test_dummy_execution`, token-tax/token-mapper, and the RAM-heavy EMA
  backtest are explicitly unsupported or legacy.

Removal must be coupled with confirmation that the matching live integration
and user-facing command have been retired. A conditional skip caused by missing
credentials, chain availability, or a feature flag is not a removal candidate.

### High-cost active tests

The following tests are active in CI and are good candidates for a nightly or
path-triggered acceptance workflow. This means omitting them from the default
PR selector, not adding a permanent `pytest.skip` marker. Their full coverage
remains available on `master`, nightly, and explicit dispatch.

| Subarea | Test file | Test name | Current duration | Recommended routing |
| --- | --- | --- | ---: | --- |
| Backtest visual output | `tests/backtest/test_backtest_chart.py` | `test_backtest_charts` | 84.76s | Nightly/path-triggered visual regression suite |
| Hyperliquid release candidate | `tests/test_hyperliquid_waterfall_notebook.py` | `test_hyperliquid_waterfall_release_candidate_notebook` | 70.64s | Nightly and Hyperliquid/notebook path changes |
| Capped waterfall notebook | `tests/test_capped_waterfall_notebook.py` | `test_capped_waterfall_notebook` | 56.45s | Nightly and CCTP/notebook path changes |
| Vault-yield backtest | `tests/strategy_tests/test_vault_yield_manager.py` | `test_backtest_vault_yield_manager` | 55.64s | Nightly and vault/strategy path changes |
| Phase-aware waterfall notebook | `tests/test_capped_waterfall_phase_aware_notebook.py` | `test_capped_waterfall_phase_aware_notebook` | 33.63s | Nightly and phase-aware/notebook path changes |

The two longest mainnet-fork repair tests are deliberately excluded from that
recommendation: `test_rebalance_vault_yield` (103.39s) and
`test_repair_vault_position_open_failed` (78.02s) exercise financial recovery
behaviour. Their value is high enough that they should remain in a protected
coverage path.

### Flaky-test follow-up

The 47 `flaky` annotations need a separate reliability audit. A retry marker is
not evidence that a test is low value; it may be protecting an important
integration. Each marker should instead be classified as one of:

1. deterministic regression that should have its fixture or assertion fixed;
2. external-integration acceptance test that belongs in nightly/path-triggered CI;
3. retired feature coverage that can be removed with its retired integration.

## CI routing constraint

The present critical path is 12:43. The slow job is only 11 seconds behind the
automated job, so removing a slow test cannot improve the PR gate by more than
11 seconds until the automated job is also shortened. Runner queueing has also
been observed on the shared Beefy runner group; splitting the suite into more
parallel jobs should be evaluated against queue time, not only execution time.

## Proposed sequence

1. Remove the explicitly permanently skipped test modules only after validating
   retirement of the corresponding product integration.
2. Add a `nightly_test_group` marker and a nightly/manual workflow for the five
   expensive acceptance tests above, with conservative path selectors.
3. Audit every `flaky` annotation and either fix, route, or remove it.
4. Profile fixture/setup reuse in the automated suite, which remains the
   practical gate bottleneck after the prior optimisations.
