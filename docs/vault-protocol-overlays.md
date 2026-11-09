# Vault protocol overlays

`translate_trading_pair()` attaches immutable `VaultProtocolRules` after loading
vault metadata and before caching the pair. Strategies read
`pair.get_vault_protocol_rules()` for capabilities and use pricing for dated
availability and settlement expectations. Existing pair constructors may leave
the optional field unset; existing strategy APIs remain supported.

Rules are declared in `tradeexecutor/state/vault_protocol.py`. Pure overlays in
`tradeexecutor/strategy/vault_protocol/` resolve published fee data and existing
simulation assumptions. They perform no acquisition or account permission checks.

- `atoma.py` describes the two reviewed standard-fee public deployments. Their
  zero-entry/0.005-exit profile is distinct from mechanical support for a fee.
  Deposits are immediate; the existing fallback redemption delay is seven days.
  Actual live fee-recipient exemptions and ticket claimability remain adapter-owned.
- `d2.py` provides D2 window capability and a bounded early HYPE++ assumption:
  fourteen-day cadence, two open days, anchor 30 March 2025. It applies only
  from first price coverage on 20 November 2024 until 12 February 2026. It is
  labelled synthetic evidence, never a live rule, and never replaces a known
  flag, capacity or restriction. Later missing deposit history fails closed;
  existing unknown-redemption policy is preserved.
- The dispatcher retains existing Lagoon/D2 event gates, metadata-estimate
  eligibility, protocol/default delays and Ostium's next-day 18:00 UTC rule.
  The shared settlement helper applies explicit backtest overrides first;
  overrides retain their historical both-directions-queued behaviour.

D2 parquet availability is placed at the next decision bucket (`ceil`, exact
boundaries unchanged). It uses the existing single timestamp column and the
last sample in each bucket. There are no additional observation clocks.

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
