# Native vault funding through CCTP

HyperCore dataset chain 9999 uses HyperEVM (999) USDC for physical funding;
Lighter dataset chain 9998 uses Ethereum (1). `get_funding_chain_id()` resolves
the custody account without changing the source identity. With the corresponding
bridge route, native trades participate in existing CCTP netting and State
capital booking. Simulation uses physical USDC, not the synthetic dataset quote.

Standalone Hyper AI with no bridge pairs retains its planner bypass. Historical
universes without a HyperEVM route retain their assumed-primary convention;
this is not evidence of a physical cross-chain route. The live strategy runner
refuses native satellite trades without that route. A configured HyperCore exit
without an open or closed funding ledger is diagnosed before planner mutations;
old instant-primary wallets require reconciliation rather than automatic migration.

The planner estimates net HyperCore sale proceeds using the existing
ledger-derived externalised-fee calculation when fee-mode metadata is present.
Factory-created live pairs carrying only a performance rate keep a gross
forecast with a warning; no historical fee-mode assertion is applied to them.
The sequential executor requires State's existing allocation checks for bridges
and dependent purchases, even when optional on-chain balance checks are disabled.
State handles funding-source selection, idle collateral and sub-raw-unit buy
rounding in one place. Bridge-backs retain an explicit source-capital check,
because State does not allocate their source cash. This is an accounting
guard, not an independent RPC balance proof. The native router still verifies
actual payouts. Fees, NAV changes or withdrawal caps can reduce proceeds; an
oversized spend then expires all remaining planned work and checkpoints state.
Sub-raw-unit buy shortages use Portfolio's existing dust snap; larger deficits
halt the batch rather than automatically resizing it.
Internal exchange-account transfers use account capital, not reserve cash.

Pending CCTP keeps its existing batch halt, signed-transaction recovery and
startup retry. A live route can complete in one batch if each receive confirms
within the router wait. Otherwise the loop requires recovery/restart and later
replanning. Automatic daemon continuation and calibrated transfer-delay
simulation are outside this change.

Lighter purchase budgets include accounted idle collateral. State consumes that
contribution; simulation debits only `get_lighter_buy_funding()` from physical
Ethereum cash. The purchase's executed cost remains the full investment budget.
Manual pool buys that bypass the planner use the same State allocation path.
A $5 buy with $1 idle collateral therefore checks $4 of satellite Safe funding.
Planner diagnosis, simulation and bridge reopening share Portfolio's latest
closed-bridge lookup; looking up the ledger does not reopen or mutate it.
The simple backtest valuator prices CCTP positions at $1; allocated capital is
still accounted for by the existing bridge position, not counted twice.

## Verification and scope

`tests/backtest/test_hypercore_cctp_funding.py` builds five fixed-price vaults
and two bridge routes without RPCs or downloaded data. Its main case uses the
normal runner, AlphaModel and YieldManager: $1,000 becomes $600 native holdings,
$300 cash parking and $100 reserve, then rotates to $400 hub/$500 Base holdings
and $100 reserve. The original planner omitted the native exit and failed with
a $500 shortfall. The corrected simulation observes native cash on HyperEVM
before bridging; it does not establish live service availability.

Focused cases cover standalone primary-chain compatibility, a missing-ledger
refusal, net-fee forecasts, reduced payout refusal, state reload, shared physical
and native capital, Lighter idle funding, and internal account transfers in a
CCTP batch. The live sequential method uses mocked executor/transport wiring
with real State and account completion. It is not a funded live rehearsal.

Historical routing remains an idealised zero-latency assumption, including
periods where route availability has not been established. No strategy ranking,
universe policy, parameter, dependency pin or persistent state schema changed.
The NB07 execution record lives in the sibling vault-notebooks research lineage;
its completed financial run is separate from later saved-state reporting repairs.
