# HyperCore funding through CCTP

HyperCore dataset chain 9999 uses HyperEVM (999) USDC for physical funding.
`get_funding_chain_id()` resolves that account without changing source identity.
When the universe declares a HyperEVM bridge, the CCTP planner includes native
sales and purchases in its existing satellite netting. Full and partial exits
already sort before bridge-backs. State uses the same bridge ledger; simulation
uses its physical destination USDC rather than the synthetic quote.

Standalone Hyper AI (HyperEVM primary, no bridge pairs) keeps its original
planner bypass. Historical universes without a HyperEVM route retain their
labelled assumed-primary convention. Live cross-chain trading requires the
physical route. A native sell with a declared route but no open or closed
funding ledger is rejected before planner mutations. Existing wallets opened
under the former instant-primary convention need evidence-led reconciliation;
this change does not migrate balances automatically.

Sequential live execution checks settled source capital before signing bridge
transfers and their dependent buys, even with optional balance checks disabled.
A reduced payout cannot finance an oversized burn. Unfunded remaining trades
are expired and checkpointed through existing state. Pending CCTP retains the
existing batch halt, saved transaction and startup retry behaviour. Automatic
daemon continuation and calibrated bridge-delay backtests are separate work.

The offline regression `tests/backtest/test_hypercore_cctp_funding.py` uses fixed
vaults, $1 NAV candles, a real AlphaModel/YieldManager and the normal backtest
runner. It starts with $1,000, buys native and cash vaults, then rotates into
hub/Base vaults. The original planner fails with a $500 shortfall; the corrected
run finishes with $400/$500 holdings and $100 cash. A transparent simulator spy
checks native cash never appears in the hub before bridging. Refusal coverage
checks reduced actual payout, shared native/HyperEVM capital, save/reload and
missing-ledger atomicity. Neither RPC nor downloaded market data is required.

The simple backtest valuation model now also supports bridge positions at $1,
matching the existing CCTP valuation convention; this is needed to exercise the
normal runner with an offline hardcoded universe. Historical transfer topology
is an explicit zero-latency assumption, not proof of route availability at the
historical date. No strategy ranking, universe policy or buffer is changed.

The unchanged-input NB07 rerun additionally exposed gross native reductions
being used as bridge funding even though settlement deducts externalised vault
fees. The planner now uses the existing ledger-derived `calculate_vault_fees()`
net estimate for native sells. This matches simulation without duplicating the
fee formula or changing live Hyper AI's standalone pricing. The refusal test
also covers a profitable native exit with a 10% performance fee and asserts the
projected bridge-back equals net proceeds. Actual live proceeds remain subject
to the source-spending guard.
