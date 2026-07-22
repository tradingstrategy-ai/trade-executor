# Vault test trade plan

## Objective

Add a standalone `vault-test-trade` command for testing explicitly selected
external vaults through a deployed Lagoon vault. It must not load a strategy
module or invoke `create_trading_universe()` / `decide_trades()`.

## Interfaces

- `vault-test-trade --id ID` uses `state/{id}.json` by default.
- Manual and `--auto-real` modes require the matching
  `state/{id}.deployment.json`, which provides the source Lagoon vault, guard
  module and satellite modules. `--auto-simulated` instead deploys an ephemeral
  Lagoon topology on fresh Anvil forks and removes its artefact at shutdown.
- `--vault-id` / `VAULT_ID` accepts an ordered comma-separated list of
  eth-defi `chain_id-address` vault specifications.
- `--amount` is the test reserve amount. For CCTP, the satellite deposit uses
  the amount actually available after bridge losses.
- `--auto-real` performs sequential real deposits or available redemptions.
- `--auto-simulated` performs sequential fork attempts. Instant vaults are
  deposited and redeemed; async vaults are deposited only and persisted as
  `simulation_unsupported_async` diagnostics.
- `--rerun` permits a fresh attempt after a terminal result. Automatic modes
  never retry a pending request.
- A failed local Anvil is infrastructure, not a vault result. Simulation tears
  down the complete multichain runtime and reruns the interrupted vault once on
  fresh forks. A second infrastructure failure is recorded and the batch
  continues; protocol and transaction failures are never retried.

## State model

- One dedicated vault-test executor state file contains all attempts using the
  normal `State` / `TradingPosition` serialisation; there is no second file for
  simulated results.
- `TradingPosition.simulated` identifies fork-only diagnostics. Simulated
  positions are always closed before being merged into this dedicated state.
  The flag is interpreted only by vault-test helpers and the Textual UI; normal
  portfolio analysis and accounting code does not branch on it.
- Position `other_data["vault_test_attempt"]` records vault id, execution
  mode, phase, diagnostic result and detail.
- The latest position for a vault is authoritative. Statuses are `success`,
  `deposited`, `deposit pending`, `redemption pending`, `deposit closed`, and
  explicit diagnostic failures such as `simulation unsupported async` and
  `infrastructure failed`.
- State-held `BlockchainTransaction` objects continue to own nonce tracking;
  no directory-wide nonce allocation is introduced.

## Execution flow

1. Download the full vault universe through `trading-strategy`.
2. For each selected action, construct a minimal executable universe from its
   selected vault and generate CCTP bridge pairs when it is on a satellite
   chain.
3. Construct generic routing, pricing and valuation models for that action.
4. Call the shared programmatic `perform_test_trade()` operation to open or
   close the position.
5. Before a real automatic action, resolve already-settled async requests.
   Pending requests are displayed and skipped without an automatic retry.
6. Stop the batch if no reserve cash remains. Print a `tabulate()` outcome row
   after every requested vault.

## Implementation layout

- `commands/vault_test_trade.py` contains only Typer option validation,
  high-level orchestration, table output and resource cleanup.
- `vault_test_trade_setup.py` owns client/universe loading, real or simulated
  runtime construction, dedicated state loading and startup settlement.
- `vault_test_trade_runner.py` owns the sequential per-vault lifecycle state
  machine, action selection, execution and result recording.
- `vault_test_trade_simulation.py` owns disposable Anvil generations,
  infrastructure classification and snapshot helpers.
- `vault_test_trade_tui.py` owns manual selection and typeahead widgets.
- `vault_test_trade.py` contains shared deployment, universe and special-state
  helper functions used by setup, runner and TUI modules.

## Manual interface

The Textual main screen lists vaults already tested with vault name, chain,
protocol, status, mode and position id. Enter selects an open deposit for a
redemption attempt. `n` opens a searchable typeahead over the complete
downloaded vault universe for a new deposit.

## Verification

- Unit tests cover vault-id parsing, real-mode deployment artefact topology,
  state isolation, disposable Anvil replacement and the Textual typeahead.
- Async settlement and CCTP continuation tests cover the shared test-trade
  behaviour.
- A dedicated Lagoon/Anvil black-box test exercises the Typer command with a
  real deployed Lagoon artefact and an instant ERC-4626 deposit/redemption.
- A second Lagoon/Anvil black-box test exercises the ephemeral simulated
  deployment without a pre-existing deployment file or production key.
