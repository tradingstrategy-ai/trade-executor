# Externalised vault fee backtest evidence

Validated on 2 November 2026. The shared backtester now automatically calculates
externalised management and performance fees on partial and full withdrawals.
Quotes, position NAV, synchronous proceeds and asynchronous settlement use the
same calculation. Fee inputs and realised components persist on existing trades.

Vault operators may deduct externalised fees before withdrawal. This simulator
currently collects them only on withdrawal. Future work may model management
fee deductions weekly or monthly; periodic collection is not simulated here.
Management uses remaining acquisition principal and a 365.25-day year;
performance fees apply to positive profit after management. Weighted-average
basis and constant current rates are explicit scenario assumptions. Protocol
high-water marks, historic rate changes and periodic crystallisation are not
reproduced. Internalised fees remain in observed prices and are not charged again.

For a reduction with allocated principal `C` and gross proceeds `G`, management
`M` accrues on principal over time, performance is `rate × max(G − C − M, 0)`,
and the fixed withdrawal charge applies once to the remaining non-negative
proceeds. Entry costs are not added to the fee basis.

## Real-data samples

Metadata was downloaded once through the authenticated vault data client.
Prices are a small unchanged slice of the authenticated NB05 capture from the
same day; their source and SHA256 are in the fixture receipt. Daily candles
resample observed prices and forward-fill gaps. The checked-in price sample is
Zstandard-compressed Parquet with typed timestamps and unchanged numeric values.
Metadata fee modes are the
published provider classifications, consistent with the pinned fee registry;
this is simulator evidence, not an independent on-chain audit of each vault's
operator accounting. Lighter's profitable observed window was selected to
exercise a non-zero performance fee, not to estimate investment performance.
Each vault is simulated separately with its actual chain/address identity and
10,000 starting USDC, using simulated funding. No live pool funding or Guard
compatibility is claimed.

| Family | Chain / identity | Management | Performance | Internalised |
|---|---|---:|---:|---|
| lagoon | 1 / `0x3be67ba2d3fec744d1d2b5d564c83f57372578e4` | 2.75% | 21.00% | False |
| ipor | 1 / `0x08dd57a51025a266cecc85154860aa4e71e48ee8` | 0.30% | 2.00% | True |
| lighter | 9998 / `lighter-pool-281474976489165` | 0.00% | 10.00% | False |
| hyperliquid | 9999 / `0x0034cd90f5a6195a2e282282a9e551502b7b516c` | 0.00% | 10.00% | False |

## Withdrawal results

All amounts are USDC. Each sample deposits 1,000, tops up 500, withdraws half,
then exits. The constructed strategy contains no fee calculation or mutation.

| Family | Trade | Principal | Gross | Management | Performance | Net cash |
|---|---:|---:|---:|---:|---:|---:|
| lagoon | 3 | 750.000000 | 756.340664 | 0.790554 | 1.165523 | 754.384586 |
| lagoon | 4 | 750.000000 | 658.513992 | 1.355236 | 0.000000 | 657.158756 |
| ipor | 3 | — | 749.992936 | 0.000000 | 0.000000 | 749.992936 |
| ipor | 4 | — | 749.992936 | 0.000000 | 0.000000 | 749.992936 |
| lighter | 3 | 750.000000 | 853.580505 | 0.000000 | 10.358051 | 843.222455 |
| lighter | 4 | 750.000000 | 789.903011 | 0.000000 | 3.990301 | 785.912710 |
| hyperliquid | 3 | 750.000000 | 785.671184 | 0.000000 | 3.567118 | 782.104066 |
| hyperliquid | 4 | 750.000000 | 794.647545 | 0.000000 | 4.464754 | 790.182790 |

IPOR's fees are already internalised, so no externalised-fee basis is recorded.
The tests recalculate each withdrawal independently from the captured inputs.

## Validation

The broader focused checks passed 186 cases covering fees, asynchronous vaults,
CCTP funding, metadata, timing, CLI action selection and phase-aware allocation.
These local results are not CI, funded or deployment certification evidence.

- Seven fee tests pass: four real-data strategy scenarios through the normal
  module loader and default generic router, plus Lagoon through a freshly
  created legacy pricing model each cycle; an independent numeric oracle and
  invalid-data/legacy-wrapper refusal; and pending withdrawal reload/recovery.
- 51 affected async vault, metadata translation and cash-aware CCTP tests pass.
- Three canonical NB05 strategy tests pass after removing the external-fee exclusion.
- The oracle verifies 1,000 principal, 1,200 gross, 2% annual management and 20%
  performance after 365.25 days: management 20, performance 36, net 1,144.
  Half withdrawal gives 10, 18 and 572. A separately configured 0.5% exit cost
  reduces the full withdrawal to 1,138.28, once. Delayed settlement recomputes fees
  against changed prices and additional time. Losses cannot create negative cash.
- Every real withdrawal is independently recalculated from successful purchases
  and proportional reductions. Final restored equity equals final cash within
  0.000001 USDC. The restart test preserves request evidence, refuses early
  settlement and verifies repeated recovery leaves state and wallet unchanged.
- Saved pre-exit NAV is independently checked for the real samples, including
  the legacy pricing factory. Factories and stop-loss pricing receive the same
  portfolio as execution; no separate fee ledger is introduced.
- Existing regressions cover pending-proceeds funding gates and Atoma's fixed
  withdrawal charge. Those funding tests reuse the existing fee-independent
  gating mechanism; they are distinct from the new fee numeric scenarios.

Reproduce with the installed Poetry environment, sourced `.local-test.env`,
worktree `PYTHONPATH` (including the pinned dependency trees) and these commands:

```sh
poetry run pytest tests/backtest/test_externalised_vault_fees.py -n auto -q
poetry run pytest tests/backtest/test_backtest_async_vault.py tests/ethereum/test_cctp_bridge_cash_aware.py tests/test_vault_metadata_translation.py -n auto -q
poetry run pytest /home/mikko/code/strategies/tests/test_master_vault_nb05.py -q
```

The fixtures and their SHA256 receipt are in
`tests/backtest/fixtures/vault_externalised_fees/`. Offline tests assert the input
hashes and calculate expectations independently from the trade ledger.

## Compatibility and caller cleanup

The canonical `hyper-ai-v13.py` no longer wraps pricing, mutates dynamic
performance-fee taxes, or maintains its own cost-basis replay. Its explicit
capital-friction allowance remains separate. NB05 now admits otherwise eligible
externalised-fee vaults. The subsequent
[eligibility-corrected NB05 run](nb05-eligibility-corrected-backtest-2026-11-02.md)
records the financial result and fees paid per vault after restoring supported
routes and correcting admission.
The executor refuses the known old Hyper-AI wrapper before execution to prevent
stacking charges. Legacy pending redemption-rate snapshots retain their prior
behaviour. Old hand-built universes with no fee metadata preserve their previous
behaviour; partially specified or invalid externalised inputs fail explicitly.
Older bundles without a fee-mode field use the pinned adapter registry where
their protocol and address establish it; translation persists the resolved mode.
Historical Hyper-AI versions remain unchanged. The canonical `hyper-ai.py`
symlink still selects v12 as of this check; changing the deployed version is
outside this work. Backtesting a version with the old wrapper requires migrating
that caller to the shared calculation.
No additional persistence system or production transactions were introduced.

Fixed entry/withdrawal costs retain the existing token-tax configuration. The
automatic calculation does not infer a fixed withdrawal tax from metadata;
NB05 explicitly models Atoma and excludes other unmodelled entry/exit costs.
The supported fee scenarios use vault pairs with zero AMM trading fees and no
`trading_fee_override`. Combining exchange-fee overrides with manager fees is
not validated by these results. A direct resolver call can settle a legacy
saved redemption-rate request even if fee-mode metadata is missing. Resuming
the normal backtest runner requires that metadata first: it values open holdings
before resolving pending requests and refuses to guess their fee mode.

## Input hashes and revisions

Metadata SHA256: `54d60b1032dc49071544dbfcf2c4cf951822e1f9127e1a5fc4bbc31428262a9d`.

Prices Parquet SHA256: `64a07883a5da9fd453cfed053c6618b48a3acdcaa0e4878ddb5fb6eb0268606a`.

Executor base: `ee7b1fd325f28359a149ac5a1a54d371c6849ad2`; implementation is in
[PR #1700](https://github.com/tradingstrategy-ai/trade-executor/pull/1700).

web3-ethereum-defi: `27c16e26dd9325427bb35af8e1c5e2efcb63db60`.

trading-strategy: `f80f684758413d27b5350a57718d3ca594a91574`.
