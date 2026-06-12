# Vault deposit and redeem flows

A strategy can hold another vault as a trading position. Buying the position
means depositing the reserve currency (e.g. USDC) into the vault and receiving
its share token; selling means redeeming the shares back to the reserve. From
the strategy's point of view a vault position looks like any other spot
position — the complications come from *how* vaults hand the money over.

This document explains the two families of vault flows we support, walks
through the lifecycle of an asynchronous deposit and redemption, describes what
a pending request means for the portfolio's books, and then shows how the flow
is simulated in backtesting and executed in live trading. The final sections
map the relevant code and tests.

**Scope note.** This is about the strategy acting as a *depositor into an
external vault*. It is not about the strategy's own treasury being a Lagoon
vault (external investors depositing into *us* — `has_async_deposits()`,
`pending_redemptions`). The two systems share vocabulary but are separate code
paths, and confusing them is the most common mistake in this area.

## Instant and queued vaults

Most ERC-4626 vaults are **synchronous**: `deposit()` mints shares in the same
transaction, `redeem()` pays out immediately. Trading them is no different from
swapping a token — the trade executes and settles within one strategy cycle.

Some vaults cannot do that. A vault whose capital sits in slow markets
(perpetual liquidity pools, RWA, cross-chain positions) cannot price and honour
a deposit on the spot. These vaults are **asynchronous**: you *request* a
deposit or redemption, the request waits in a queue, and only after the vault
settles the queue can you *claim* the result. ERC-7540 standardises this
request/claim pattern on top of ERC-4626.

We support:

| System | Example protocols | How settlement happens |
|---|---|---|
| Synchronous ERC-4626 | IPOR, Morpho, Euler | Immediately, in the deposit/redeem transaction |
| ERC-7540 | Lagoon | The vault's operator settles the queue manually — there is **no schedule** and the wait is unknowable in advance |
| Ostium V1.5 | Ostium OLP | A time window (epoch) passes — the claimable timestamp is known when the request is made |
| Hypercore-native | Hyperliquid HLP | Not on-chain ERC-20 flows at all; handled by a separate Hypercore code path and out of scope here |

The protocol differences are hidden behind one interface: every vault exposes a
`VaultDepositManager` (eth_defi), and routing simply asks
`has_synchronous_deposit()` / `has_synchronous_redemption()` to choose between
the instant path and the queued path.

## The asynchronous lifecycle

An asynchronous deposit takes three steps, usually spread over several strategy
cycles:

1. **Request.** The strategy approves and calls `requestDeposit()`. The USDC
   leaves our wallet into the vault's pending silo. We have paid, but received
   nothing yet — no shares exist for us at this point.
2. **Settle.** The vault's asset manager values the vault and processes the
   queue (for Lagoon: `updateNewTotalAssets()` followed by `settleDeposit()`).
   For Ostium this is replaced by an epoch boundary passing. We take no action;
   we can only watch the request move from *pending* to *claimable*.
3. **Claim.** We call the claim form of `deposit()`, the vault mints our
   shares, and the position is finally backed by real tokens.

Redemption is the mirror image — `requestRedeem()`, settle, claim with
`redeem()` — with one important twist: **the share tokens leave our wallet at
request time**, moved into the vault's escrow. While a redemption is pending we
own a position whose tokens are visibly *not* in our wallet.

Inside the trade executor each step maps onto the trade lifecycle. A trade that
has broadcast its request but not yet claimed sits in the dedicated status
`vault_settlement_pending` (between `broadcasted` and `success`). Each strategy
cycle a settlement resolver checks every pending trade against the chain and
claims whatever has become claimable; only then is the trade marked `success`
with the actually-executed amounts.

ERC-7540 has one querying subtlety worth knowing: `pendingDepositRequest(0,
owner)` aggregates *all* of an owner's outstanding requests, while
`claimableDepositRequest(request_id, owner)` is per-request. A status check
must look at the request-specific claimable amount first — otherwise, when two
requests are outstanding, an already-claimable one is misreported as still
pending.

## What a pending request means for the books

Between request and claim the capital is in limbo, and the portfolio
accounting is built around three invariants:

**Equity stays continuous.** A pending deposit's position has quantity zero
(no shares yet), so it values to nothing — but the committed capital is
counted back into total equity as *vault settlement pending value*. Total
equity therefore does not dip when a deposit enters the queue, and does not
jump when it settles. A pending redemption needs no such correction: the
position still holds its shares (in escrow) and keeps its normal valuation
until the claim pays out.

**Committed capital cannot be spent twice.** Reserves are debited the moment
the trade starts, so `get_cash()` already excludes the money sitting in a
deposit queue. Symmetrically, shares locked in a pending redemption are
subtracted from the position's *available trading quantity*, so a strategy
cannot request the same shares twice.

**Reconciliation expects the wallet to disagree.** On-chain balance checks
(`check-accounts`, `correct-accounts`) compare our books against the wallet.
In live trading a pending deposit leaves both sides in agreement (cash gone
on both, shares zero on both). A pending redemption would *not* agree
naively: our books hold the full share quantity while the wallet holds none,
because the vault escrowed them. The expected-balance calculation therefore
subtracts escrowed-in-redemption shares, so a clean state reads as clean —
and an account correction can never mistake an escrow for a lost position
and close it mid-settlement. (Backtests balance the same books differently:
the simulated wallet is not touched until settlement, as described in the
backtesting section below.)

### Writing a strategy against an async vault

**AlphaModel-based strategies need no special handling** — the framework is
settlement-aware:

- `carry_forward_non_redeemable_positions()` pins any position with an
  in-flight settlement at its committed value (settled shares plus pending
  deposit capital) and counts it into the returned `locked_position_value`,
  so the standard `deployable = portfolio_target − locked` pattern
  automatically excludes in-flight capital from fresh allocation.
- `update_old_weights()` values pending deposit capital into the position's
  old value, so adjustment sizing never sees a phantom zero-value position.
- Trade generation skips any signal whose position has a pending settlement
  (the `settlement_pending` flag), even for strategies that never call
  carry-forward. The skipped signal carries `pending_deposit_usd` /
  `pending_redemption_usd` diagnostics, shown as columns in the
  `format_signals()` table and the `alpha_model_diagnostics` chart, and the
  `pending_vault_settlements` chart plots the queued buffers over time.
- Same-cycle financing is settlement-aware: rebalance buys are normally
  funded by the cycle's sells executing first, but a redemption requested
  from an async vault pays out only after settlement. The alpha model scales
  buys down to the cash that actually arrives this cycle (the
  `capped_by_pending_settlement_cash` flag) and redeploys the withheld
  capital on a later cycle once the redemption settles. Async vaults are
  recognised by their feature flags (`pair.is_async_vault()`) or, for vaults
  simulated as async only via a backtest delay override, by the
  `vault_async_flow` stamp on the position's earlier settlement requests
  (`position.has_async_vault_flow()`).

For hand-rolled (non-AlphaModel) `decide_trades` the manual rules still
apply:

- After requesting a deposit the position already exists and `is_any_open()`
  is true; the position simply has quantity zero until settlement. Do not
  open it again. Check `position.has_pending_vault_settlement()` before
  touching a position.
- `get_cash()` is already safe to spend — committed capital is excluded.
- If you size positions from `calculate_total_equity()`, subtract
  `get_vault_settlement_pending_value()` first, otherwise you allocate the
  in-flight capital a second time.
- The simplest robust pattern is to skip rebalancing entirely while any
  trade is in `vault_settlement_pending`, and rebalance on the next cycle
  once capital has actually moved.

## Backtesting

A backtest has no vault operator, so the settlement wait must be declared.
Two parameters on the backtest entry points (`run_backtest_inline()` and
friends) control it:

- `vault_settlement_delay` — the default delay for every async vault.
  Defaults to **two days** (`DEFAULT_VAULT_SETTLEMENT_DELAY`): with the
  common one-day decision cycle a one-day delay would settle every request
  exactly on the next cycle, hiding the multi-cycle pending window the
  simulation exists to model.
- `vault_settlement_delay_overrides` — per-vault delays, keyed by vault
  address. These take precedence over everything else, and giving a vault an
  override also marks it as asynchronous even if its metadata carries no
  async feature flags, which is convenient for synthetic test universes.

Ostium-style vaults get a schedule instead of a fixed delay: with no
override, a request settles **the next day at the epoch settlement hour**
(`OSTIUM_BACKTEST_SETTLEMENT_HOUR`, 18:00 UTC), approximating Ostium's daily
on-chain epochs.

**Note a behaviour change for older backtests.** The two-stage simulation
switches on *automatically* for any vault whose metadata carries async
features — there is no separate opt-in. A vault pair that used to settle
instantly in an old backtest now requests on one cycle and settles days
later (at the settlement-time price), so results for such pairs change once
their dataset metadata gains async feature flags. The defaults are
deliberately non-zero to make this visible; pass an explicit
`vault_settlement_delay`/override to control it precisely (an explicit zero
still settles no earlier than the next cycle).

Otherwise a vault counts as asynchronous when its feature flags include
`erc_7540_like`, `lagoon_like` or `ostium_like`. The flags are read via
`pair.get_vault_features()`, which accepts either a plain
`other_data["vault_features"]` list or the features carried inside the pair's
`VaultMetadata` object (`other_data["token_metadata"]`) —
`translate_vault_to_trading_pair()` writes both. One caveat for hand-built
test universes: the pairs DataFrame round-trip preserves the `VaultMetadata`
object but drops a bare `vault_features` key, so synthetic pairs should carry
their features in `VaultMetadata`.

The simulation mirrors the live lifecycle:

- On the request cycle the trade is marked `vault_settlement_pending` with a
  settlement due time of *cycle timestamp + delay*. The simulated wallet is
  deliberately not touched yet — only the cash ledger is debited — so wallet
  and position reconciliation stay consistent through the pending window.
- On each later cycle the backtest's settlement resolver (the same hook the
  live runner calls) settles every trade whose due time has arrived. The
  earliest a trade can settle is the *next* cycle, even with a zero delay,
  because requests are created after the resolver step within a cycle.
- Settlement executes at the price valid *at settlement time*, not at request
  time, so a long queue delay realises whatever the share price did in
  between — exactly the economics the delay exists to model. After settling,
  the position is revalued at the settlement price so equity is immediately
  correct.

## Live trading

In live trading nobody knows when a Lagoon operator will settle, so the
design is: persist everything needed to finish the trade, then poll.

When the request transaction confirms, the routing layer serialises a
*settlement ticket* into the trade's `other_data`, via
`State.mark_vault_settlement_pending()`: `vault_chain_id` and
`vault_direction` (`"deposit"` or `"redeem"`) for dispatch,
`vault_request_tx_count` so the resolver can tell request transactions from a
later claim transaction, and the deposit manager's own ticket fields (vault
address, owner, raw amounts, the ERC-7540 request id). Raw token amounts are
stored as strings because 18-decimal values exceed the JavaScript
safe-integer limit that the state file enforces. Because the ticket lives in
the persisted state, a restarted executor reconstructs it and carries on; no
settlement is lost to a crash or redeploy.

Every cycle the runner invokes the settlement resolver
(`ExecutionModel.resolve_pending_vault_settlements()` — the same polymorphic
hook the backtest overrides). The live implementation rebuilds the ticket,
asks the vault's deposit manager for the request status, and acts:

- *pending* — do nothing, check again next cycle.
- *claimable* — sign and broadcast the claim, read the actual share/asset
  amounts from the claim events, and mark the trade `success`.
- *reclaimable* (Ostium only; Lagoon has no reclaim) — recover the funds and
  mark the trade failed, restoring reserves.

The resolver is idempotent: it recognises an already-confirmed claim,
rebroadcasts a stuck one, and discards a reverted one before retrying.

The resolver also runs once at executor startup, so settlements left pending
by a previous run are picked up before the first cycle. Unlike a stuck CCTP
bridge transfer — which halts startup, because in-transit bridge funds are an
error state — an unresolved vault settlement does not block: queues can
legitimately take hours or days, and the trade simply resolves on a future
tick.

### Commands cannot wait for settlement

An ERC-7540 queue has no deadline — a Lagoon operator might settle in a
minute or in a week. A one-off CLI command therefore must never block waiting
for a request to become claimable. Every command instead leans on the same
foundation: the settlement ticket is persisted in the state file, so *any
later process* can finish the job. From that foundation the commands use
three completion routes:

- **The `start` daemon completes it.** The canonical route. Any pending trade
  left behind by a one-off command is picked up by the daemon's startup retry
  and per-cycle resolver. A command's job is only to get the request on-chain
  and report honestly that settlement is in flight.
- **Re-running the command completes it.** Commands are idempotent: a later
  run first sweeps anything that has become claimable, then continues. This
  is how an operator without a running daemon walks a position through the
  queue across days.
- **Tests force the queue.** On an Anvil fork the command itself can play the
  vault operator and settle immediately — never on a real chain.

How each command or context takes a request from pending to settled:

- **`start`** — fully automatic. The request is created during a cycle's
  trade execution; every subsequent cycle (and every restart) polls the
  request status and claims when the operator has settled. No operator action
  is needed on our side.
- **Backtests** — deterministic: the simulated clock reaches the configured
  delay and the resolver settles the trade on that cycle, as described in the
  backtesting section.
- **`perform-test-trade`** — on an Anvil fork it force-settles the vault
  queue as the operator (`_force_vault_settlement_and_resolve()`) and resolves
  in the same run, so a test trade completes its whole cycle in one command.
  On a real chain it leaves the trade pending and prints
  "re-run perform-test-trade after settlement" — the second run (or the
  daemon) claims it.
- **`trade-ui`** — observability plus the test-trade route: the position
  table shows the settlement ETA for pending deposits (a real timestamp for
  Ostium; `pending` for Lagoon, where no ETA exists). Test trades executed
  from the UI follow the `perform-test-trade` path above.
- **`check-wallet`** — observability only: reports pending and
  settled-but-unclaimed vault amounts (`maxDeposit`/`maxRedeem`/
  `pendingRedeemRequest`) so the operator can see where in the lifecycle the
  capital sits. It never resolves trades.
- **`check-accounts` / `correct-accounts`** — neither resolves; they
  *tolerate*. Expected balances account for queue escrow, so a position
  mid-settlement reads as clean and is never "corrected" away. Settlement
  itself still happens via the daemon or a re-run of the originating command.
- **`repair` / `retry`** — not involved: they act on *failed* trades, and a
  pending settlement is not a failure. Both explicitly leave
  `vault_settlement_pending` trades alone.
- **`close-position` / `close-all`** — the full re-run pattern. Each run
  first sweeps claimable settlements (so a re-run after the operator settles
  claims the earlier redeem and may close the position with no new on-chain
  action), then skips positions whose settlement is still in flight with a
  clear message, and only then requests new redeems. A new redeem that lands
  in `vault_settlement_pending` is force-settled on Anvil; on a real chain
  the command reports the request is on-chain and exits cleanly — the
  position closes on a later re-run or via the daemon. Walking a position
  through the queue is therefore: run once (request), wait for the operator,
  run again (claim and close).

The own-treasury sibling command `lagoon-redeem` (out of scope here, but the
origin of the idempotent-sweep pattern) shows the re-run route in full: each
invocation first claims any settled-but-unclaimed deposits and redemptions
from previous runs, then proceeds with the new redemption.

## Where the code lives

The protocol layer, in eth_defi:

| Module | Role |
|---|---|
| `eth_defi/vault/deposit_redeem.py` | `VaultDepositManager` interface, request status enum, ticket (de)serialisation |
| `eth_defi/erc_4626/vault_protocol/lagoon/deposit_redeem.py` | ERC-7540 implementation (Lagoon) |
| `eth_defi/erc_4626/vault_protocol/gains/deposit_redeem.py` | Ostium V1.5 implementation |
| `eth_defi/erc_4626/vault_protocol/lagoon/testing.py` | `force_lagoon_settle()` for acting as the operator in tests |

The execution layer, in trade-executor:

| Module | Role |
|---|---|
| [tradeexecutor/ethereum/vault/vault_routing.py](../../tradeexecutor/ethereum/vault/vault_routing.py) | Chooses instant vs queued flow, builds request transactions, persists the ticket |
| [tradeexecutor/ethereum/vault/settlement_retry.py](../../tradeexecutor/ethereum/vault/settlement_retry.py) | Live settlement resolver: poll, claim, reclaim |
| [tradeexecutor/strategy/execution_model.py](../../tradeexecutor/strategy/execution_model.py) | The polymorphic settlement hook called by the runner each cycle |
| [tradeexecutor/backtest/backtest_execution.py](../../tradeexecutor/backtest/backtest_execution.py) | Simulated requests and delayed settlement; delay configuration |
| [tradeexecutor/state/trade.py](../../tradeexecutor/state/trade.py), [state.py](../../tradeexecutor/state/state.py) | The `vault_settlement_pending` status and its bookkeeping |
| [tradeexecutor/state/portfolio.py](../../tradeexecutor/state/portfolio.py), [position.py](../../tradeexecutor/state/position.py) | `get_vault_settlement_pending_value()` in equity; `has_pending_vault_settlement()` and the per-position pending value; available-quantity guard against double redemption |
| [tradeexecutor/strategy/alpha_model.py](../../tradeexecutor/strategy/alpha_model.py) | Settlement-aware carry-forward pinning, pending-inclusive old weights, the `settlement_pending` rebalance skip, and the pending columns in `format_signals()` |
| [tradeexecutor/strategy/chart/standard/vault.py](../../tradeexecutor/strategy/chart/standard/vault.py) | `pending_vault_settlements` chart: queued deposit/redemption buffers per cycle |
| [tradeexecutor/strategy/asset.py](../../tradeexecutor/strategy/asset.py) | Escrow-aware expected balances for account reconciliation |
| [tradeexecutor/ethereum/vault/vault_utils.py](../../tradeexecutor/ethereum/vault/vault_utils.py) | Turning a vault into a tradeable pair identifier |
| [tradeexecutor/cli/loop.py](../../tradeexecutor/cli/loop.py) | Startup-time settlement retry (non-halting, unlike CCTP) |
| [tradeexecutor/cli/trade_ui_tui.py](../../tradeexecutor/cli/trade_ui_tui.py) | Settlement ETA display for pending deposits |
| [tradeexecutor/cli/testtrade.py](../../tradeexecutor/cli/testtrade.py) | Forced settlement of test trades on Anvil forks |

## Tests to read

Each test doubles as a worked example:

- [tests/backtest/test_backtest_async_vault.py](../../tests/backtest/test_backtest_async_vault.py)
  — the backtest behaviour end to end: the two-stage lifecycle cycle by
  cycle, settlement pricing, several vaults with different delays, partial
  redemptions, the double-deposit guard, the alpha-model wait-for-settlement
  pattern, and a backtest that ends while a request is still queued.
  `test_backtest_async_vault_alpha_model_native_settlement_handling` is the
  worked example of the framework-native handling: an alpha model rebalances
  every cycle with no manual pending guard, gradually entering and exiting
  async vaults on different settlement schedules;
  `test_backtest_async_vault_naive_alpha_model_guarded` shows the
  trade-generation safety net for strategies without carry-forward.
- [tests/erc_4626/test_vault_async_lagoon_erc_7540.py](../../tests/erc_4626/test_vault_async_lagoon_erc_7540.py)
  — the live flow against a real Lagoon ERC-7540 vault on an Anvil fork. The
  test plays the vault operator, first holding the queue and then settling
  it, and verifies the on-chain escrow, the account checks during both
  pending windows, and that a persisted state survives a simulated restart.
- [tests/erc_4626/test_vault_async_ostium_v15.py](../../tests/erc_4626/test_vault_async_ostium_v15.py)
  — the same lifecycle for the time-window (Ostium) flavour.
- [tests/erc_4626/test_vault_async_cli_commands.py](../../tests/erc_4626/test_vault_async_cli_commands.py)
  — the flow driven through the CLI (`init`/`start`) as a black box.
- [strategies/test_only/async_vault_backtest_example.py](../../strategies/test_only/async_vault_backtest_example.py)
  — a minimal strategy module written against an async vault.
