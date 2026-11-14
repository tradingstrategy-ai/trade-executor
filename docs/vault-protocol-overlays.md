# Vault protocol overlays

`DEXPair` already carries `VaultMetadata` in `other_data["token_metadata"]`,
including published fees, protocol identity and feature flags. Pair translation
reuses these and the existing asynchronous-deposit, delayed-redemption and
settlement-estimate helpers; no extra upstream metadata schema is needed.
`DEXPair.buy_tax`/`sell_tax` describe token-transfer taxes and are translated onto
the base asset. They do not describe operation queues or fee-support profiles;
vault economics must not mutate that shared asset metadata.

`VaultMetadata.vault_protocol_rules` holds the typed derived execution profile
and simulation defaults after the executor resolves them from loaded metadata.
The optional field defaults to `None`: raw data loaders do not select simulation
policy. Raw `DEXPair` metadata remains unchanged during executor translation;
only the translated pair receives the enriched metadata instance.
`VaultProtocolRules` is defined in `tradingstrategy.vault`, so the data
package has no dependency on trade-executor.

`pair.get_vault_protocol_rules()` reads this metadata field first. Full
`VaultMetadata` is transient in State JSON, so translation also stores the small
resolved snapshot as plain values in `other_data["vault_protocol_rules"]`.
Restored position pairs use that snapshot without losing fees or timing. There
is no dedicated `TradingPairIdentifier` schema field. Older/manual pairs with
neither source return `None`.

Pure overlays in
`tradeexecutor/strategy/vault_protocol/` resolve published fee data and existing
simulation assumptions and explicit manual vault-data corrections. They perform
no acquisition or account permission checks.

Shared framework code must consume generalised capabilities and populated
metadata fields, rather than branch on protocol names, feature flags or vault
addresses. Put manual exceptions and defaults in the protocol overlay and
select them in the enrichment dispatcher. Keep ABI details and dynamic
owner-specific reads in the eth-defi adapter. Extend the existing rule object
when a new capability is needed; do not add a parallel metadata registry or
protocol-specific execution path.

- `atoma.py` describes the two reviewed standard-fee public deployments. Their
  zero-entry/0.005-exit profile is distinct from mechanical support for a fee.
  Deposits are immediate; the existing fallback redemption delay is seven days.
  It also marks `redemption_delay_is_deposit_lock`, so fork tests can query the
  manager's actual custody-specific unlock time before requesting withdrawal.
  Actual live fee-recipient exemptions and ticket claimability remain adapter-owned.
- `autopool.py` populates `simulation_max_holding_seconds` with 30 minutes because
  Chainlink prices can expire after one hour while feed updates remain frozen.
  Both public vault test commands use this rule for home-chain and satellite
  exits. Explicit shorter intervals, including zero, are preserved. This is a
  fork-test interval, not a live strategy cycle, redemption lock or historical
  backtest settlement delay. Oracle freshness checks remain enforced.
- `d2.py` provides D2 window capability and a bounded early HYPE++ assumption:
  fourteen-day cadence, two open days, anchor 30 March 2025. It applies only
  from first price coverage on 20 November 2024 until 12 February 2026. It is
  labelled synthetic evidence, never a live rule, and never replaces a known
  flag, capacity or restriction. Later missing HYPE++ deposit history fails closed;
  other D2 vaults retain the existing unknown-state policy. The existing
  unknown-redemption policy is preserved.
- `hyperliquid.py` records manual native-vault overrides. Growi HF-2
  (`9999`, `0x9b5b1931fa5838a891973f9877c7648c93ee80cc`) is permanently closed
  for deposits at the operator's explicit instruction on 10 November 2026.
  This all-time override precedes the pre-11-April-2026 assumed-open policy,
  synthetic windows and provider/captured-open responses. It also applies to
  live/replay/simulated HyperCore pricing and the final buy execution preflight.
  Evidence is labelled
  `manual_override` / `permanently_closed`; it is not a newly recovered historical
  observation. Other native vaults retain their existing admission rules.
  Price histories remain loaded and redemption checks remain operation-specific.
  Earlier NB04 results using assumed Growi admission predate this correction;
  they must not be described as results of the corrected engine.
- The dispatcher retains existing Lagoon/D2 event gates, metadata-estimate
  eligibility, protocol/default delays and Ostium's next-day 18:00 UTC rule.
  The shared settlement helper applies explicit backtest overrides first;
  overrides retain their historical both-directions-queued behaviour.

D2 parquet availability is placed at the next decision bucket (`ceil`, exact
boundaries unchanged). It uses the existing single timestamp column and the
last sample in each bucket. There are no additional observation clocks.

For a positive fork-test holding interval, the CLI honours the manager's
`get_redemption_delay_over(custody_owner)` result when the loaded rules set
`redemption_delay_is_deposit_lock`. It waits until the later of
the test interval or one second past that onchain unlock time. This lets Atoma
Index's deposit shares unlock before a request is submitted; its separate
operator epoch settlement still follows the saved ticket. The query uses the
pricing route's actual custody resolver on the vault's own chain. Live trading
never advances the clock, and historical backtest settlement rules are unchanged.
An explicit zero interval retains its no-wait behaviour. The capability defaults
to false: the manager method can mean a settlement schedule rather than a deposit
lock. The optional `simulation_max_holding_seconds` defaults to no cap. Both
fields travel through enriched metadata and the existing saved pair snapshot;
the shared consumers contain no protocol predicates.
Snapshots written before these fields existed use their defaults. Public
test-trade commands select pairs from a freshly enriched universe, so they
receive the current overlay values rather than old snapshot defaults.

## Strategy-facing APIs

`pricing_model.get_vault_settlement_expectation(timestamp, pair,
operation="deposit")` returns `(queued, earliest_due_time)`. The runner binds
the same overrides/default delay used by execution, including on GenericPricing.
A pair-level estimate set before request creation remains supported. The due time
is a minimum estimate; historical events, grace rules and live managers still
control actual settlement. Use deposit timing for deposit horizon decisions.

Fee amounts are fractions, not basis points or pool fees. Per-pair fee getters
avoid modifying shared share-token metadata. Explicit existing token-tax settings
retain precedence for compatibility. Quotes and forecasts are pure; execution
and settlement apply the shared calculation's net amount exactly once. A pending
simulation request retains its fee rates in the existing trade record.

`validate_vault_fee_data(universe, allow_missing=False)` asserts with selected
vault identities and missing fields before a converted backtest starts. With
`allow_missing=True`, incomplete vaults stay data-only; no unknown fee becomes
zero. Known unsupported nonzero mechanisms still refuse. This policy is called
explicitly by the converted strategy; unrelated strategies do not acquire a new
global loader assertion. Avantis is excluded from this conversion for now.

Optional `entry_eligibility` callbacks on `PhaseAwareAlphaModel` and
`YieldRuleset` apply strategy entry policy to promotions and positive parking
adjustments. Refusal releases stale park earmarks and leaves reserve cash liquid;
negative adjustments remain available to exit existing holdings.

## Verification

`tests/backtest/test_vault_protocol_overlays.py` loads a hardcoded multi-protocol
universe through the actual metadata parser and lazy pair translation, verifies
operation timing/override parity, executes an Atoma deposit and delayed exit,
and checks $1,000 gross redemption produces $995 net and one $5 charge. It also
covers strict/overridden missing fees, real D2 bucket conversion, observed versus
assumed windows, explicit zero capacity, nullable reasons and public unknown-state
checks. Existing phase-aware/yield regressions check entry refusal and release.

These fixtures establish engine wiring and accounting, not economic performance
or live deployment readiness. No old-State migration framework is introduced.
See the existing vault lifecycle and state-persistence guides for claim/recovery
and accounting invariants.

`tests/units_tests/test_hyperliquid_protocol_overrides.py` checks the actual NB04
purchase dates through backtest pricing, precedence over optimistic provider and
window assumptions, direct live/simulation pricing, exact venue/address scope,
retained history and independently permitted redemptions. No new historical
backtest result is implied by this correction.
