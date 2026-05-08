# CCTP auto-rebalance roadmap

Automated cross-chain CCTP rebalance for xchain master vault strategies.
Full plan at `.claude/plans/spicy-fluttering-lobster.md`.

## PR dependency graph

```
PR 1 (state foundation) ──┬──> PR 3 (settlement) ──> PR 4 (lifecycle)
                           │                                  │
PR 2 (custody bug fix) ────┤                                  │
                           │                                  v
                           └──> PR 5 (backtest) ──> PR 6 (planner) ──> PR 7 (integration test)
```

PR 1 and PR 2 have no cross-dependencies and can be developed in parallel.

## PR 1: Foundation — state layer changes

**Scope**: `TradeStatus.cctp_in_transit` + `mark_expired()` + sort phases

- Add `TradeStatus.cctp_in_transit` to enum
- Add `cctp_in_transit_at` field on `TradeExecution`
- Update `get_status()` priority chain (`executed_at` before `cctp_in_transit_at`)
- Add `TradeExecution.mark_expired(ts)` method (only from `planned` status)
- Add `State.mark_bridge_in_transit(trade)` with bridge-back `bridge_capital_allocated` adjustment
- Add CCTP bridge phases (-30M/+30M) in `get_execution_sort_position()` before `self.closing`
- Exclude `cctp_in_transit` from `is_unfinished()` and generic repair
- Unit tests for status transitions and sort ordering

**Files**:
- `tradeexecutor/state/trade.py` — enum, field, `get_status()`, `mark_expired()`, sort phases
- `tradeexecutor/state/state.py` — `mark_bridge_in_transit()`
- `tradeexecutor/cli/rebroadcast.py` — exclude `cctp_in_transit`

**No external dependencies**, pure state layer.

## PR 2: Custody resolver — fix `mint_recipient` bug

**Scope**: Direction-aware custody address resolution (P1 bug fix, Lagoon-first)

- Add `custody_address_resolver` parameter to `CctpBridgeRouting.__init__()`
- Build resolver from `satellite_vaults` + primary vault in `EthereumPairConfigurator`
- Replace `tx_builder.get_erc_20_balance_address()` with direction-aware resolver call
- Hot wallet fallback (returns same address for all chains)
- Test custody resolution for Lagoon (per-chain Safe) and hot wallet paths

**Files**:
- `tradeexecutor/ethereum/cctp/routing.py` — `__init__()` and `setup_trades()`
- `tradeexecutor/ethereum/ethereum_protocol_adapters.py` — pass resolver at construction

**Standalone bug fix**, can ship early and independently.

## PR 3: Multi-phase settlement — attest + receiveMessage automation

**Scope**: Extend settlement to poll attestation and broadcast receiveMessage

- Extend `CctpBridgeRouting.settle_trade()` with attestation polling and receive tx
- Store `_hot_wallet` on routing model during `setup_trades()`
- Clone hot wallet per destination chain (`HotWallet(account)` + `sync_nonce`) for nonce safety
- Load `message_transmitter = get_message_transmitter_v2(dest_web3)` for `sign_transaction()` contract arg
- Sign receive tx via `HotWalletTransactionBuilder`, append before broadcast
- Use `HexBytes(signed_bytes)` for broadcast, `native_datetime_utc_now()` for `broadcasted_at`
- `CctpBridgeRouting.needs_sequential_trade_execution()` returns `True`
- In-transit fallback: call `mark_bridge_in_transit()` when attestation times out or receive reverts

**Files**:
- `tradeexecutor/ethereum/cctp/routing.py` — settlement extension
- `tradeexecutor/strategy/generic/generic_router.py` — sequential flag delegation

**Depends on**: PR 1 (uses `mark_bridge_in_transit`)

## PR 4: In-transit lifecycle — valuation, halt, cleanup, startup check

**Scope**: Complete the `cctp_in_transit` lifecycle for production use

- `Portfolio.get_in_transit_value()` — sum `planned_reserve` for all in-transit trades
- Include in-transit value in `get_total_equity()` and `calculate_total_equity_chain()` (attribute to destination chain)
- Sequential executor halt on `cctp_in_transit`: expire remaining planned trades, remove orphan zero-quantity positions, raise `ExecutionHaltableIssue`
- CCTP startup check: scan for in-transit trades, attempt retry, halt if unresolved
- Post-restart retry utility: inspect N receive txs (handle confirmed/reverted/unbroadcast/pending), clone wallet, retry
- Bridge-back `bridge_capital_allocated` reversal on retry success before `mark_success(force=True)`
- Clear `cctp_in_transit_at` only on successful resolution (invariant)

**Files**:
- `tradeexecutor/state/portfolio.py` — `get_in_transit_value()`, equity methods
- `tradeexecutor/ethereum/execution.py` — halt + cleanup in sequential executor
- `tradeexecutor/ethereum/cctp/retry.py` — new retry utility
- `tradeexecutor/cli/loop.py` or `tradeexecutor/strategy/runner.py` — startup check

**Depends on**: PR 1 + PR 3

## PR 5: Backtest sequential path

**Scope**: Fix `BacktestExecution` for CCTP-dependent trade batches

- Detect `is_cctp_bridge()` trades in `BacktestExecution.execute_trades()`
- Switch to per-trade sequential start/simulate/settle when bridge trades present
- Existing batch path unchanged when no bridge trades (no regression)

**Files**:
- `tradeexecutor/backtest/backtest_execution.py` — sequential path

**Depends on**: PR 1 (sort phases)

## PR 6: Bridge trade injection planner

**Scope**: Automatically inject CCTP bridge trades from alpha model rebalance decisions

- New module: analyse alpha model trade list, inject bridge trades
- Group sells/buys by chain, compute net flows per satellite chain
- Size bridge amounts correctly (net of same-chain sells/buys)
- Consume `carry_forward_non_redeemable_positions()` results (skip locked capital)
- Integration point after `alpha_model.generate_rebalance_trades_and_triggers()`, before `execute_trades()`

**Files**:
- `tradeexecutor/ethereum/cctp/planner.py` — new module
- `tradeexecutor/strategy/runner.py` — call planner from tick

**Depends on**: PR 1 + PR 5

## PR 7: Integration test + dummy strategy + README

**Scope**: End-to-end verification across all components

**New files**:
- `strategies/test_only/xchain-cctp-cycling-test.py` — 3-cycle deterministic strategy (open all chains, rebalance, close all)
- `tests/mainnet_fork/test_xchain_cctp_auto_rebalance.py` — Lagoon 3-chain Anvil integration test (primary)
- `tests/mainnet_fork/test_xchain_cctp_auto_rebalance.py::test_..._hot_wallet` — hot wallet variant (secondary)
- `tests/backtest/test_xchain_backtest_bridge_ordering.py` — backtest ordering and accounting

**Test coverage**:
- Correct bridge trade injection per cycle
- Correct sort order (bridge-backs after vault redeems, bridge-outs before vault deposits)
- Bridge position lifecycle and `bridge_capital_allocated` tracking
- Total equity conservation across cycles
- Lagoon: burn txs `lagoon_vault` type, receive txs `hot_wallet` type, correct per-chain Safe mint_recipient
- In-transit failure: 4a mock `fetch_attestation()` timeout, 4b disable fork attester for receiveMessage revert
- README-cctp.md: new "Trade executor integration" section

**Depends on**: all previous PRs
