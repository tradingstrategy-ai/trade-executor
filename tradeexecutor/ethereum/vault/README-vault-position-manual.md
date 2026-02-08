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

## Vault rebalance status

Use `print_vault_rebalance_status()` to display all vaults in the universe with current allocations.

### Console example

```python
from tradeexecutor.analysis.vault_rebalance import print_vault_rebalance_status

# Print vault status to console
df = print_vault_rebalance_status(state, strategy_universe)
```

Example output:

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

### Basic spot position

```python
# Get trading pair from universe
pair = strategy_universe.get_pair_by_human_description(
    (ChainId.base, "uniswap-v3", "WETH", "USDC")
)

# Open $100 position
trades = pm.open_spot(pair=pair, value=100)

# Execute the trades
execution_model.execute_trades(
    ts, state, trades, routing_model, routing_state
)

# Save state
store.sync(state)
```

### Position with stop loss and take profit

```python
trades = pm.open_spot(
    pair=pair,
    value=1000,
    stop_loss_pct=0.95,      # Stop loss at 5% below entry
    take_profit_pct=1.10,    # Take profit at 10% above entry
    notes="Manual console entry"
)

execution_model.execute_trades(ts, state, trades, routing_model, routing_state)
store.sync(state)
```

### Position with custom slippage

```python
trades = pm.open_spot(
    pair=pair,
    value=500,
    slippage_tolerance=0.02,  # 2% slippage tolerance
    notes="High slippage trade"
)

execution_model.execute_trades(ts, state, trades, routing_model, routing_state)
store.sync(state)
```

## Closing a position

### Close single position (for single-pair strategies)

```python
# Get current open position
position = pm.get_current_position()

# Close it
trades = pm.close_position(position, notes="Manual close from console")

# Execute and save
execution_model.execute_trades(ts, state, trades, routing_model, routing_state)
store.sync(state)
```

### Close position for a specific pair

```python
pair = strategy_universe.get_pair_by_human_description(
    (ChainId.base, "uniswap-v3", "WETH", "USDC")
)

position = pm.get_current_position_for_pair(pair)
trades = pm.close_position(position, notes="Closing WETH position")

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

execution_model.execute_trades(ts, state, trades, routing_model, routing_state)
store.sync(state)
```

## Increasing a position

Use `adjust_position()` with a positive `dollar_delta`:

```python
pair = strategy_universe.get_pair_by_human_description(
    (ChainId.base, "uniswap-v3", "WETH", "USDC")
)

# Increase position by $100
trades = pm.adjust_position(
    pair=pair,
    dollar_delta=100,       # Positive = buy more
    quantity_delta=None,    # Not needed for buys
    weight=1,
    notes="Manual increase from console"
)

execution_model.execute_trades(ts, state, trades, routing_model, routing_state)
store.sync(state)
```

## Decreasing a position

Use `adjust_position()` with a negative `dollar_delta`:

```python
pair = strategy_universe.get_pair_by_human_description(
    (ChainId.base, "uniswap-v3", "WETH", "USDC")
)

position = pm.get_current_position_for_pair(pair)
current_price = position.get_current_price()

# Calculate quantity to sell for $50 worth
quantity_to_sell = Decimal(str(50 / current_price))

trades = pm.adjust_position(
    pair=pair,
    dollar_delta=-50,                 # Negative = sell
    quantity_delta=-quantity_to_sell, # Must be negative for sells
    weight=1,
    notes="Manual decrease from console"
)

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

# 2. Get a trading pair
pair = strategy_universe.get_pair_by_human_description(
    (ChainId.base, "uniswap-v3", "WETH", "USDC")
)

# 3. Check current cash
cash = pm.get_current_cash()
print(f"Available cash: ${cash:.2f}")

# 4. Open a position
trades = pm.open_spot(pair=pair, value=100, stop_loss_pct=0.95)
execution_model.execute_trades(ts, state, trades, routing_model, routing_state)
store.sync(state)

# 5. Check position
position = pm.get_current_position()
print(f"Position quantity: {position.get_quantity()}")
print(f"Entry price: ${position.get_opening_price():.4f}")

# 6. Increase position
trades = pm.adjust_position(pair=pair, dollar_delta=50, quantity_delta=None, weight=1)
execution_model.execute_trades(ts, state, trades, routing_model, routing_state)
store.sync(state)

# 7. Close position
trades = pm.close_position(position)
execution_model.execute_trades(ts, state, trades, routing_model, routing_state)
store.sync(state)

print("Done!")
```

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
- [test_enzyme_credit_position.py](../../../tests/mainnet_fork/test_enzyme_credit_position.py) - Credit positions
- [test_enzyme_guard_perform_test_trade.py](../../../tests/mainnet_fork/test_enzyme_guard_perform_test_trade.py) - Guard-protected trades

## Related source files

- [position_manager.py](../../strategy/pandas_trader/position_manager.py) - PositionManager class
- [console.py](../../cli/commands/console.py) - Console command implementation
- [close_position.py](../../cli/commands/close_position.py) - Close position CLI command
- [close_all.py](../../cli/commands/close_all.py) - Close all positions CLI command
- [vault_rebalance.py](../../analysis/vault_rebalance.py) - Vault rebalance status display
