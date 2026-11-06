# Lighter recovery and gas fixes

Four defects found while funding the LIX vault on 2026-11-05. All live data below
comes from that vault's state and Ethereum receipts.

| # | Item | Finding |
| --- | --- | --- |
| 1 | Recover a Lighter deposit that was never mined | No command can do it; it took a hand edit of the state |
| 2 | Swapped gas fields | Two positional callers pass price and gas in the wrong order |
| 3 | Estimate the remaining fixed gas limits | Three limits are 5 to 10 times real use |
| 4 | Report Etherscan verification of the module | Verification is never attempted, so nothing reports it |

Each item is a separate commit, so any one can be dropped in review. Everything
fits in `trade-executor`. No `web3-ethereum-defi` change or submodule bump is needed.

## 1. Recover an unmined Lighter deposit

### Current behaviour

`LighterRouting` signs an approval and a deposit together. If the deposit never
reaches a block, `wait_and_broadcast_multiple_nodes()` raises `ConfirmationTimedOut`
and trade stays `broadcasted`, with reserves already allocated. `check_if_clean()`
then refuses `start`.

- `rebroadcast_all()` skips it, because `is_exchange_account_transfer_pending()` is true.
- `repair_trade()` raises `RepairAborted` for exchange-account transfers.
- `reconcile_verified_lighter_transfers()` needs every receipt, and returns nothing.
- It expires only `planned` trades, which hold no reserve.

On 2026-11-05 trade #10 was stuck like this, and the hot wallet's nonce was then
consumed by another transaction. The edit that resolved it is recorded in
`docs/deployment-and-manual-operations.md` in the `strategies` repository. This work
replaces that edit with a supported path.

### Change

Add `recover_unmined_lighter_deposits()` to `ethereum/lighter/transfer_verification.py`,
called by `repair` before `reconcile_verified_lighter_transfers()`. It does not run
from `check-accounts` or `correct-accounts`, as it can send transactions.

**Scope.** It handles external exchange-account deposits routed by `LighterRouting`
on Ethereum: pending Lighter buy trades with status `broadcasted`, with
`is_external_account_transfer()` true and a recorded `deposit` transaction.

Pool trades are excluded. They are `vault_settlement_pending` before broadcast, can be
funded from idle collateral or a bridge position, and `planned_reserve` is not what
they took from reserves. Other chains are excluded too, because `finalized` semantics
are only established for Ethereum. Anything out of scope stays pending, with a log line.

**Classification** uses every recorded transaction, not only those without a receipt.
Let `approve` and `deposit` be the trade's two transactions.

| Evidence | Case | Action |
| --- | --- | --- |
| `deposit` has a receipt | Not this path | Leave it to the existing reconciliation |
| `approve` has a receipt with status 0 | Undetermined | Leave pending, as a deposit could revert or need a state the reconciler refuses |
| No `deposit` receipt, and the unmined transactions' nonces start exactly at the wallet's mined nonce and have no gaps | Sendable | Rebroadcast unchanged |
| No `deposit` receipt, and a different transaction provably consumed its nonce | Void | Repair the trade |
| Anything else, including any read failure or contradiction | Undetermined | Leave pending and log why |

Undetermined is always safe: the trade blocks `start` until someone looks at it, which
is today's behaviour.

**Proof for void.** A consumed nonce alone proves nothing, because the consumer could
be the deposit itself and a stale node may not return its receipt. So find the
consumer, using block-pinned reads:

1. Read the `finalized` block number `f`. Require `getTransactionCount(sender, f) > nonce`.
2. Search backwards from `f` with growing steps, then bisect, for the lowest block where
   the count exceeds `nonce`.
3. Read that block with full transactions, and take the sender's transaction with that nonce.
4. Compare hashes. The same hash means it was mined, so leave it to reconciliation.
   A different hash means the signed deposit can never be mined, but not that the funds
   stayed: a fee-bumped copy would perform the same deposit.
5. Read the consumer's receipt. If it holds a reserve-token `Transfer` from the Safe, the
   trade is undetermined, whether or not the transaction reverted. Otherwise it moved
   nothing from the Safe.

Historic reads that fail, a missing transaction, or any inconsistency give undetermined.
There is no fallback to `latest`.

As corroboration only, the Safe's USDC at `latest` must be at least state reserves plus
`trade.planned_reserve`. Lighter collateral is not used: it can stay low for unrelated
reasons, so it proves nothing here.

**Sendable.** Check `keccak(signed_bytes) == tx_hash` for each unmined transaction, and
that the wallet holds `gas × maxFeePerGas` for them all. If it does not, stop and say how
much ETH to send, and that it must be an incoming transfer because anything sent from
the wallet consumes the nonce. Broadcast with `wait_and_broadcast_multiple_nodes()`,
keeping `check_nonce_validity=True` so its retrying nonce check runs. Transactions are
never re-signed or re-estimated. The following `reconcile_verified_lighter_transfers()`
call completes the trade from its receipts. If Lighter has not indexed the deposit, the
trade stays pending and rerunning `repair` finishes it.

**Void accounting.** Reuse repaired-trade semantics, which handle quantities, reserves
and consumers correctly. Hand-setting `expired_at` would leave planned quantities that
`get_position_quantity()` and `analysis/single_pair.py` still count.

- Split the body of `repair_trade()` into a helper, and add
  `repair_unmined_exchange_deposit(portfolio, trade, note)` that calls it. `repair_trade()`
  keeps its refusal of exchange-account transfers, so only this verified path gets through.
- The helper's existing buy branch returns `planned_reserve` to reserves exactly once.
  `start_execution()` allocated that amount for an external transfer, so the two match.
- Timestamps are kept, and the note records the consuming transaction's hash and block.
- `make_counter_trade()` gives the counter-trade no transfer flag, so `get_action_verb()`
  needs a repair branch, or `repr()` of the trade raises.

A repaired trade is no longer pending, so a rerun selects nothing and cannot credit
reserves twice. Call `store.sync(state)` after each repaired or rebroadcast trade, before
another trade's reconciliation can raise. An approval left behind is harmless, as the
next deposit's approval overwrites it.

Both actions print what they will do and ask for confirmation unless `--auto-approve`
is set, matching the other steps of `repair`.

`repair_zero_quantity()` also runs in `repair`. It does not close LIX's empty accounts,
because `can_be_closed()` is false for exchange accounts. Add a test for that.

### Left out

A deposit that was mined and reverted, `failed` trades, pool trades and non-Ethereum
chains. All stay blocked as they are today, and need their own evidence.

## 2. Swapped gas fields

`BlockchainTransaction.set_confirmation_information()` takes
`realised_gas_units_consumed` then `realised_gas_price`. Two callers pass them
positionally in the opposite order:

- `tradeexecutor/ethereum/execution.py:1068`
- `tradeexecutor/ethereum/tx.py:162`

The four other callers use keywords and are correct. LIX trade #11 stores
`realised_gas_units_consumed=4652987018` and `realised_gas_price=200146`.

Change both to keyword arguments, including `status` and `revert_reason`, and make
everything after `block_hash` keyword-only so this cannot recur. Correct the field comment, which says gwei; it holds wei.

**Consumers.** `analysis/position.py` shows wrong `Gas` and `Price (GWei)` columns.
`analysis/hypercore_rebalance_cost.py` and `ethereum/vault/hypercore_routing.py`
multiply the two, so their totals were right.

**Existing state files keep their swapped values.** There is no migration, because
totals are unaffected, and stored records cannot always be told apart from correct
ones on chains with very low gas prices. State the cut-off release in the field comment.

## 3. Gas limits

Measured on LIX, with the limit a node sees:

| Transaction | Limit | Used | At 11.92 gwei the wallet must hold |
| --- | --- | --- | --- |
| Approve | 1,000,000 | 72,717 to 92,629 | 0.0119 ETH |
| L1 withdraw | 1,000,000 | 156,655 | 0.0119 ETH |
| Pending withdrawal claim | 1,500,000 | 146,151 | 0.0179 ETH |
| Deposit | 900,000 | 195,358 to 223,281 | 0.0107 ETH |

`LagoonTransactionBuilder.sign_transaction()` always adds `extra_gnosis_gas`, 500,000
on these paths, so lowering a constant cannot go below that.

- Add `sign_transaction(..., total_gas_limit: int | None = None)`, which is used
  instead of `gas_limit + extra_gnosis_gas` when given.
- Add `LagoonTransactionBuilder.estimate_gas(args_bound_func) -> int`, which estimates
  the outer module call from the hot wallet, including the Safe and guard. It signs
  nothing and allocates no nonce. `details["gas"]` keeps the signed total.
- Add a helper in `lighter_routing.py` for approve, L1 withdraw and claim:
  `ceil(estimate × 1.5)`. It is not capped, because signing below a successful estimate
  would only produce a reverting transaction. If estimation fails, log a warning and use
  the old total, which is today's behaviour, so a failed estimate cannot stop a transfer.
- Use the same helper at the matching calls in `lighter_pool_routing.py`.

**The deposit cannot be estimated.** It is signed before its approval is mined, so
`transferFrom` would revert in the estimate. State overrides would need each token's
storage layout. Instead, pass `total_gas_limit=LIGHTER_DEPOSIT_GAS_LIMIT`, so 400,000
is the whole limit, as was intended in #1709. The highest use measured is the first
deposit to a new Lighter account, 223,281 gas through the guard (LIX nonce 55), so this
leaves about 1.8 times that. If it ever ran out, the transaction would revert and the
funds would stay in the Safe.

A claim estimate is advisory, because the relayer can pay in between. The existing
relayer reconciliation handles that outcome unchanged.

`LIGHTER_L1_WITHDRAW_GAS_LIMIT` and the `eth_defi.lighter.lagoon` helpers used by
`lighter-move-funds` and `lagoon-lighter-test-trade` stay as they are, as they need an
eth-defi change.

## 4. Report Etherscan verification of the module

`deploy_safe_trading_strategy_module()` refuses `use_forge` because of library linking,
and deploys through `deploy_contract()`, which never verifies. The LIX logs show no
attempt. So a failure is not the problem: operators assume `--etherscan-api-key`
covered the module, and nothing says otherwise.

Add `fetch_etherscan_verification_status(chain_id, address, api_key)` to
`ethereum/lagoon/deploy_report.py`. It calls the Etherscan v2 `getsourcecode` action,
and returns `verified`, `not_verified` or `unknown`. Only `status == "1"` with a result
row counts: empty `SourceCode` in that row is `not_verified`. HTTP errors, API errors,
rate limits and malformed bodies are `unknown`. It uses a 10-second timeout, so it cannot
stall a deployment. Etherscan can match known bytecode itself, which is why this reads the
status instead of hardcoding it.

Call it at the end of `lagoon-deploy-vault`, after the deployment artefacts are written,
for every deployed chain and module, when the verifier is `etherscan` and the run is not a
simulation. Log `not_verified` as a warning that names the module and says verification
was not attempted. Log `unknown` as a warning too. Add the status to the Markdown report.

It must never fail a finished deployment. The API key is in the URL, so log only the
exception type, never the URL, body or traceback.

Actually verifying the module is left out. It needs an eth-defi change, and the request
was to report it.

## Tests

Follow `CLAUDE.md`: one happy path and one refusal for each behaviour.

| Item | Test |
| --- | --- |
| 1 | `tests/exchange_account/test_lighter_unmined_deposit.py`, using real `State` and trade objects with mocked chain reads, as the existing recovery tests do. Void repairs and restores reserves once, and survives a save, reload and rerun. A deposit with a receipt but no approval receipt is not void. A failed approval with a sendable deposit is undetermined. A consumer with the same hash is not void. Failed historic reads are undetermined. Sendable refuses when the wallet is short. `repair_trade()` still refuses exchange transfers, and `repair_zero_quantity()` leaves empty exchange accounts open |
| 1 | The same file, using a local Anvil: the real consumer search with a signed transaction whose nonce is taken by another. This proves the mechanics, not provider behaviour on Ethereum |
| 2 | A test for each fixed path, asserting the realised fields against a receipt |
| 3 | Unit tests for the limit helper, including estimate failure and the cap. Extend the Lighter deposit fork test to assert the real limits |
| 4 | `verified`, `not_verified`, an HTTP-200 API error and a timeout, with the HTTP call mocked because the live response needs a key. The error test asserts the key is not logged. The real response shape was read on 2026-11-05 for the LIX contracts: the vault verified, the module and libraries not |

New `sign_transaction()` parameters must not break test doubles, as #1707 did. Before
the PR, run the Lighter tests and these shared ones:

- `tests/test_execution_sequencing.py`
- `tests/exchange_account/`
- `tests/hyperliquid/test_hypercore_deposit_settlement.py`
- `tests/ethereum/test_vault_settlement_lifecycle.py`

## Documentation

- `tradeexecutor/exchange_account/README-Lighter.md`: add a recovery bullet for an unmined deposit.
- `CHANGELOG.md`: one dated line per item.

The `strategies` guide's manual state edit can be replaced once a release contains
this work. That is a follow-up in that repository.
