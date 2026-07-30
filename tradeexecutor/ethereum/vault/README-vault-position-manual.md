# Manual vault position management

This document describes how to manually open, close, increase, and decrease positions in vaults using the trade-executor console.

## Launching the console

Start an interactive IPython session with all trading objects pre-loaded:

```bash
poetry run trade-executor console --strategy-file your_strategy.py
```

### Available objects in console

When you launch the console, these objects are automatically available:

| Object | Description |
|--------|-------------|
| `state` | Current trading state with all positions |
| `strategy_universe` | Trading universe with pairs and candles |
| `pricing_model` | Price information for trades |
| `execution_model` | For executing trades on-chain |
| `routing_model` | Trade routing configuration |
| `routing_state` | Current routing state |
| `sync_model` | Vault synchronisation model |
| `store` | State store for saving changes |
| `web3` | Web3 connection |
| `hot_wallet` | Hot wallet for signing |
| `vault` | Vault smart contract proxy (if using vault) |
| `datetime` | Python datetime module |
| `Decimal` | Python Decimal class |
| `ChainId` | Chain ID enum |

## Vault status

Use `print_vault_rebalance_status()` to display vault allocations and `print_vault_deposit_status()` to display Lagoon deposit/redemption queue status.

### Console example

```python
from tradeexecutor.analysis.vault_rebalance import print_vault_rebalance_status, print_vault_deposit_status

# Print vault allocations
df = print_vault_rebalance_status(state, strategy_universe)

# Print Lagoon deposit/redemption queue status (if connected to Lagoon vault)
if vault is not None:
    print_vault_deposit_status(vault, state)
```

Example output for `print_vault_rebalance_status()`:

```
================================================================================
VAULT REBALANCE STATUS
================================================================================

Current cash:           $45,230.50
Total vault value:      $154,769.50
Total portfolio value:  $200,000.00

Vault Allocations (sorted by value, largest first):
--------------------------------------------------------------------------------
Vault                        Protocol   Address              Available  Position ID  Value USD     Weight %  Shares       1M CAGR
---------------------------  ---------  -------------------  ---------  -----------  ------------  --------  -----------  -------
IPOR USDC Lending Optimizer  ipor       0x45aa96f0...58216   Yes        1            $85,432.10    42.72%    82,543.2100  8.5%
Morpho Blue USDC             morpho     0x8eB67A509...7b2c3  Yes        2            $69,337.40    34.67%    67,892.5000  6.2%
Aave v3 USDC                 aave       0x4e65fE4D...a9f21   Yes        -            $0.00         0.00%     -            4.1%

================================================================================
```

Example output for `print_vault_deposit_status()`:

```
================================================================================
LAGOON VAULT DEPOSIT STATUS
================================================================================

Vault:                  My Lagoon Vault
Address:                0x6a5ea384e394083149ce39db29d5787a658aa98a
Safe:                   0xAD1241Ba37ab07fFc5d38e006747F8b92BB217D5

Block:                  12,345,678
Block time:             2024-01-15 14:35:00 UTC
Last settled:           2024-01-15 14:30:00 UTC

Total assets:           $200,000.00
Total supply:           195,000.0000 shares
Share price:            $1.025641

Queue status:
----------------------------------------
Pending deposits:       $10,000.00
Pending redemptions:    $5,000.00
  (shares pending:      4,850.5000)

================================================================================
```

### Getting raw data

```python
from tradeexecutor.analysis.vault_rebalance import get_vault_rebalance_status

# Get raw DataFrame without printing
df, cash = get_vault_rebalance_status(state, strategy_universe)

# Access specific columns
for _, row in df.iterrows():
    print(f"{row['Vault']}: ${row['Value USD']:,.2f} ({row['Weight %']:.1f}%)")
```

### Output columns

- **Vault**: Vault name
- **Protocol**: Vault protocol slug (ipor, morpho, aave, etc.)
- **Address**: Vault contract address (truncated for display)
- **Available**: Whether vault is in current trading universe
- **Position ID**: Position ID if we have an open position, "-" otherwise
- **Value USD**: Current position value in USD
- **Weight %**: Percentage of total portfolio value
- **Shares**: Number of vault shares held
- **1M CAGR**: One month annualised return (requires vault metadata from JSON blob)

Results are sorted by value (largest position first).

## Loading vaults with metadata

Use `load_vault_universe_with_metadata()` to load vaults from the JSON blob API with full performance metrics:

```python
from tradeexecutor.strategy.trading_strategy_universe import load_vault_universe_with_metadata
from tradingstrategy.chain import ChainId

# Load all vaults with metadata
vault_universe = load_vault_universe_with_metadata(client)

# Or limit to specific vaults
vault_universe = load_vault_universe_with_metadata(
    client,
    vaults=[
        (ChainId.base, "0x45aa96f0b3188d47a1dafdbefce1db6b37f58216"),
    ]
)

# Access metadata including 1-month CAGR
for vault in vault_universe.iterate_vaults():
    if vault.metadata:
        print(f"{vault.name}: 1M CAGR = {vault.metadata.one_month_cagr}")
```

The VaultUniverse can then be passed to `load_partial_data()`:

```python
dataset = load_partial_data(
    client=client,
    vaults=vault_universe,  # Pass VaultUniverse with metadata
    # ... other parameters
)
```

## Creating a position manager

Before opening or closing positions, create a `PositionManager` instance:

```python
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager

ts = datetime.datetime.utcnow()
pm = PositionManager(ts, strategy_universe, state, pricing_model)
```

## Opening a position

### Basic vault position

```python
# Get vault pair by name
pair = strategy_universe.get_pair_by_vault_name("IPOR USDC Lending Optimizer")

# Open $100 position
trades = pm.open_spot(pair=pair, value=100)

# Sync hot wallet nonce and execute the trades
execution_model.tx_builder.hot_wallet.sync_nonce(web3)
execution_model.execute_trades(
    ts, state, trades, routing_model, routing_state
)

# Save state
store.sync(state)
```

## Closing a position

### HyperCore deposit and redemption thresholds

HyperCore native vault deposits and redemptions use different client-side
validation:

- The client **deposit** path requires at least 5 USDC. The
  `MINIMUM_VAULT_DEPOSIT` constant and deposit encoder enforce this.
- The `vaultTransfer` withdrawal encoder has no equivalent check. No backend
  redemption minimum has been confirmed, so the client permits redemptions
  below 5 USDC and relies on settlement verification to detect a no-op.
- A redemption must still be positive, within live `max_withdrawable`, and
  below fresh vault equity after the safety margin. It must also wait for the
  depositor lock-up to expire.

`correct-accounts` uses these rules to redeem undersized HyperCore positions
directly. It never tops up a position being closed, because a deposit would
extend the lock-up. Cleanup uses an adaptive safety margin of at most 0.10 USDC
instead of the normal 1.50 USDC full-close margin, then falls back to 0.25,
0.50, and 1.00 USDC verified retry margins if HyperCore silently no-ops.
Positions at or below 0.30 USDC are closed locally because they cannot produce
enough balance movement to verify every withdrawal phase safely. Larger
residuals remain tracked for another direct redemption pass. A planned trade
that has not entered execution is expired before cleanup replaces it; started
or broadcast trades are left alone.

Preview the live actions without changing persisted state or broadcasting
transactions:

```shell
poetry run trade-executor correct-accounts --dry-run
```

Dry-run uses isolated state copies to perform live lock-up and withdrawal
preflight checks, supersede planned trades that have not entered execution,
create the replacement close trade, route it, and construct and sign the
phase-1 transaction. It stops before broadcast and settlement.

Do not use `--skip-save` as a dry-run substitute: it only skips the final state
write and may still broadcast transactions.

### Recovering an interrupted HyperCore deposit

A HyperCore deposit moves USDC Safe → EVM escrow → spot → perp → vault. An EVM
receipt does not prove that every HyperCore action applied. If a trade reports
`hypercore_stranded_usdc` or `hypercore_deposit_capital_at_risk`, do not run
the generic repair logic for that marked trade or re-submit the last transfer.
The guarded `repair` command deliberately leaves that position frozen and does
not create a counter-trade for it, but it may safely repair an unrelated
planned/started trade with no transaction. Run it first when
`correct-accounts` preflight reports an unfinished trade; then inspect the
marked HyperCore trade with `correct-accounts --dry-run`. If every repair
candidate is protected, `repair` deliberately makes no state change and shows
no confirmation prompt.

1. Run `check-hypercore-user.py` for the Safe and inspect Safe EVM USDC, EVM
   escrow, spot USDC, perp withdrawable USDC and vault equity.
2. If USDC is confirmed in spot, either complete spot → perp → vault or bridge
   spot → EVM.
3. If USDC is confirmed in perp, either complete perp → vault or first move
   **perp → spot**, then use `spotSend` to bridge back to EVM. Perp USDC cannot
   be sent directly to EVM.
4. If the recorded location is `*_or_*`, recheck live balances before acting:
   the queued action may still settle after the original timeout.
5. Reconcile state only after exactly one destination is confirmed: increased
   vault equity or USDC returned to the Safe.

#### Correct-accounts recovery for stranded Safe-level USDC

`correct-accounts` has a HyperCore-specific Safe-level transit recovery before
ordinary reserve accounting. It exists because of HyperAI Citadel trade #1486
(2026-07-30; PR #1593): the HyperEVM CoreWriter receipt succeeded, but the
combined HyperCore request moved 48.884068 USDC only from spot to perp. A
generic ERC-20 correction cannot see that perp balance and must not restore it
as Safe cash.

First inspect the recovery plan without a private key or any transaction:

```shell
poetry run trade-executor correct-accounts --dry-run
```

The dry run fetches the Safe's EVM, spot and perp balances and prints any
`perp_to_spot` and `spot_to_evm` actions. It refuses recovery if the Safe has
an active HyperCore perp position. It does not sign, broadcast, save or back up
state. The live command verifies the spot increase after the first action and
the EVM USDC increase after the second before proceeding to normal accounting.

The normal recovery is enabled by default for eligible HyperCore vault
strategies. For the live #1486 balances, the ordinary dry run plans exactly
48.884068 USDC from perp to spot and then from spot to EVM, while preserving
the earlier 0.50 USDC perp dust and 0.217882 USDC spot balance. The spot
balance exceeds HyperCore's 0.01 USDC bridge-fee headroom, so this incident
does not need any bridge-fee adjustment.

If a smaller future transit balance cannot cover the bridge-fee headroom and
the normal balance-verification tolerance, `correct-accounts` leaves it in
perp rather than sending only the first `perp_to_spot` leg. A later recovery
can safely include that dust once the balance is sufficient.

When resolving an ambiguous deposit, inspect vault equity before recovery: a
late vault settlement changes which destination should retain the USDC. The
transit helper is deliberately Safe-level, because an old state file may
predate the failed trade's metadata; it preserves its configured spot/perp dust
margin rather than assuming every internal USDC cent belongs to one failed
trade. After recovery, explicitly expire or reconcile stale planned trades
before restarting the executor. `repair` cannot determine the destination of
an ambiguous HyperCore deposit and intentionally preserves that evidence for
the account-correction planner.

### Close single position (for single-pair strategies)

```python
# Get current open position
position = pm.get_current_position()

# Close it
trades = pm.close_position(position, notes="Manual close from console")

# Sync hot wallet nonce and execute
execution_model.tx_builder.hot_wallet.sync_nonce(web3)
execution_model.execute_trades(ts, state, trades, routing_model, routing_state)
store.sync(state)
```

### Close position for a specific vault

```python
# Get vault pair by name
pair = strategy_universe.get_pair_by_vault_name("IPOR USDC Lending Optimizer")

position = pm.get_current_position_for_pair(pair)
trades = pm.close_position(position, notes="Closing vault position")

execution_model.tx_builder.hot_wallet.sync_nonce(web3)
execution_model.execute_trades(ts, state, trades, routing_model, routing_state)
store.sync(state)
```

### Close with custom slippage (for illiquid tokens)

```python
trades = pm.close_position(
    position,
    slippage_tolerance=0.05,  # 5% slippage for illiquid token
    notes="Closing illiquid position"
)

execution_model.tx_builder.hot_wallet.sync_nonce(web3)
execution_model.execute_trades(ts, state, trades, routing_model, routing_state)
store.sync(state)
```

## Increasing a position

```python
import datetime
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager

# Create position manager with current timestamp
ts = datetime.datetime.utcnow()
pm = PositionManager(ts, strategy_universe, state, pricing_model)

# Get the vault pair by name
pair = strategy_universe.get_pair_by_vault_name("Hyperithm USDC")

# Increase position by $100 using adjust_position()
# - dollar_delta: positive value = buy more
# - quantity_delta: not needed for buys, set to None
trades = pm.adjust_position(
    pair=pair,
    dollar_delta=1200,
    quantity_delta=None,
    weight=1,
    notes="Rebalancing"
)

# Sync hot wallet nonce and execute trades on-chain
execution_model.tx_builder.hot_wallet.sync_nonce(web3)
execution_model.execute_trades(ts, state, trades, routing_model, routing_state)
store.sync(state)
```

## Decreasing a position

```python
import datetime
from decimal import Decimal
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager

# Create position manager with current timestamp
ts = datetime.datetime.utcnow()
pm = PositionManager(ts, strategy_universe, state, pricing_model)

# Get the vault pair by name
pair = strategy_universe.get_pair_by_vault_name("IPOR USDC Lending Optimizer")

position = pm.get_current_position_for_pair(pair)
current_price = position.get_current_price()

# Calculate quantity to sell for $50 worth
quantity_to_sell = Decimal(str(50 / current_price))

# Decrease position by $50 using adjust_position()
# - dollar_delta: negative value = sell
# - quantity_delta: must be negative for sells
trades = pm.adjust_position(
    pair=pair,
    dollar_delta=-50,
    quantity_delta=-quantity_to_sell,
    weight=1,
    notes="Rebalancing"
)

# Sync hot wallet nonce and execute trades on-chain
execution_model.tx_builder.hot_wallet.sync_nonce(web3)
execution_model.execute_trades(ts, state, trades, routing_model, routing_state)
store.sync(state)
```

## Querying positions

### Check available cash

```python
cash = pm.get_current_cash()
print(f"Available cash: ${cash:.2f}")
```

### List all open positions

```python
for pos_id, position in state.portfolio.open_positions.items():
    pair = position.pair
    qty = position.get_quantity()
    value = position.get_value()
    print(f"Position {pos_id}: {pair.base.token_symbol} - {qty} tokens, ${value:.2f}")
```

### Display positions as table

```python
from tradeexecutor.analysis.position import display_positions

df = display_positions(state.portfolio.open_positions.values())
print(df)
```

### Check position details

```python
position = pm.get_current_position()

print(f"Quantity: {position.get_quantity()}")
print(f"Current price: ${position.get_current_price():.4f}")
print(f"Opening price: ${position.get_opening_price():.4f}")
print(f"Is open: {position.is_open()}")
print(f"Stop loss: {position.stop_loss}")
print(f"Take profit: {position.take_profit}")
```

## Updating stop loss and take profit

```python
position = pm.get_current_position()

# Update stop loss
pm.update_stop_loss(position, 1500.0)

# Update take profit
pm.update_take_profit(position, 2500.0)

# Save state
store.sync(state)
```

## CLI commands for position management

In addition to the console, you can use CLI commands:

### Close a specific position

```bash
poetry run trade-executor close-position \
    --strategy-file strategy.py \
    --position-id 1 \
    --close-by-sell true \
    --slippage 0.05
```

### Close all positions

```bash
poetry run trade-executor close-all \
    --strategy-file strategy.py
```

## Complete workflow example

```python
import datetime
from decimal import Decimal
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager

# 1. Create position manager
ts = datetime.datetime.utcnow()
pm = PositionManager(ts, strategy_universe, state, pricing_model)

# 2. Get a vault pair by name
pair = strategy_universe.get_pair_by_vault_name("IPOR USDC Lending Optimizer")

# 3. Check current cash
cash = pm.get_current_cash()
print(f"Available cash: ${cash:.2f}")

# 4. Sync hot wallet nonce and open a position
execution_model.tx_builder.hot_wallet.sync_nonce(web3)
trades = pm.open_spot(pair=pair, value=100)
execution_model.execute_trades(ts, state, trades, routing_model, routing_state)
store.sync(state)

# 5. Check position
position = pm.get_current_position()
print(f"Position quantity: {position.get_quantity()}")
print(f"Entry price: ${position.get_opening_price():.4f}")

# 6. Increase position
execution_model.tx_builder.hot_wallet.sync_nonce(web3)
trades = pm.adjust_position(pair=pair, dollar_delta=50, quantity_delta=None, weight=1, notes="Rebalancing")
execution_model.execute_trades(ts, state, trades, routing_model, routing_state)
store.sync(state)

# 7. Close position
execution_model.tx_builder.hot_wallet.sync_nonce(web3)
trades = pm.close_position(position, notes="Rebalancing")
execution_model.execute_trades(ts, state, trades, routing_model, routing_state)
store.sync(state)

print("Done!")
```

## Managing vault positions via CoW Swap

Instead of using `PositionManager` which deposits/redeems through the ERC-4626
`deposit()`/`redeem()` functions, you can trade vault share tokens directly on the
open market via CoW Swap. This bypasses the vault's deposit/redeem mechanism and
instead swaps through DEX liquidity that CoW Swap routes through.

Intended for vault share tokens traded on secondary markets, such as Staked USDAi (sUSDAi)
and LLama Lend pools.

This approach is useful when:
- The vault has a deposit queue or settlement delay (e.g. Lagoon vaults)
- You want to acquire or exit shares immediately without waiting for settlement
- The share token has sufficient DEX liquidity

**Requirements**: This only works for Lagoon vaults with CoW Swap integration enabled
via `TradingStrategyModuleV0`, and requires the vault share token to have liquidity on
DEXes that CoW Swap can route through.

### Opening, increasing, closing and reducing positions

```python
from tradeexecutor.ethereum.cowswap.swap_to_vault import (
    open_vault_position_cowswap,
    close_vault_position_cowswap,
)

# Open a new $100 position (or increase existing)
open_vault_position_cowswap(
    locals(),
    vault_name="IPOR USDC Lending Optimizer",
    amount_usd=100.0,
)

# Close the entire position
close_vault_position_cowswap(
    locals(),
    vault_name="IPOR USDC Lending Optimizer",
)

# Or reduce by selling a specific number of shares
close_vault_position_cowswap(
    locals(),
    vault_name="IPOR USDC Lending Optimizer",
    shares_to_sell=50.0,
)
```

### Important notes

- **Share token liquidity**: The vault share token must have liquidity on DEXes
  (e.g. Uniswap, Balancer) that CoW Swap can route through. Not all vault share tokens
  are traded on secondary markets.
- **Price difference**: The market price of the share token on DEXes may differ from
  the vault's NAV-based share price (available via `deposit()`). Check both prices
  before choosing this approach.
- **The `vault` object** in the console is the Lagoon vault that owns the strategy (the
  "outer" vault), not the target vault whose shares you are buying.

See [swap_to_vault.py](../../ethereum/cowswap/swap_to_vault.py) for the full implementation.

## Related integration tests

For more examples, see these test files:

### Enzyme vault tests

- [test_enzyme_end_to_end.py](../../../tests/enzyme/test_enzyme_end_to_end.py) - End-to-end Enzyme vault trading
- [test_enzyme_trade.py](../../../tests/enzyme/test_enzyme_trade.py) - Basic Enzyme trade execution
- [test_enzyme_end_to_end_multipair.py](../../../tests/enzyme/test_enzyme_end_to_end_multipair.py) - Multi-pair trading
- [test_enzyme_realised_profit.py](../../../tests/enzyme/test_enzyme_realised_profit.py) - Profit tracking

### Lagoon vault tests

- [test_lagoon_e2e.py](../../../tests/lagoon/test_lagoon_e2e.py) - End-to-end Lagoon vault trading
- [test_lagoon_swap.py](../../../tests/lagoon/test_lagoon_swap.py) - Lagoon swap execution
- [test_lagoon_deposit.py](../../../tests/lagoon/test_lagoon_deposit.py) - Deposit handling

### ERC-4626 vault tests

- [test_vault_trading_e2e.py](../../../tests/erc_4626/test_vault_trading_e2e.py) - ERC-4626 vault trading
- [test_vault_deposit_redeem.py](../../../tests/erc_4626/test_vault_deposit_redeem.py) - Deposit and redeem operations
- [test_vault_pricing.py](../../../tests/erc_4626/test_vault_pricing.py) - Vault pricing

### Position manager tests

- [test_position_manager.py](../../../tests/test_position_manager.py) - PositionManager API examples
- [test_backtest_position_trigger.py](../../../tests/backtest/test_backtest_position_trigger.py) - Stop loss/take profit triggers

### Mainnet fork tests

- [test_rebalance_vault_yield.py](../../../tests/mainnet_fork/test_rebalance_vault_yield.py) - Vault rebalancing
- [test_enzyme_guard_perform_test_trade.py](../../../tests/mainnet_fork/test_enzyme_guard_perform_test_trade.py) - Guard-protected trades

## Related source files

- [position_manager.py](../../strategy/pandas_trader/position_manager.py) - PositionManager class
- [console.py](../../cli/commands/console.py) - Console command implementation
- [close_position.py](../../cli/commands/close_position.py) - Close position CLI command
- [close_all.py](../../cli/commands/close_all.py) - Close all positions CLI command
- [vault_rebalance.py](../../analysis/vault_rebalance.py) - Vault rebalance status display
