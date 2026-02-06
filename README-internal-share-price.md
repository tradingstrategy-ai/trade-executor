# Internal share price profit calculation

This document describes the internal share price method for calculating position profit, inspired by ERC-4626 vault mechanics.

## Overview

The internal share price method tracks profit by maintaining a virtual share price for each position:

- **Buys** mint new shares at the current share price
- **Sells** burn shares proportionally
- **Price movements** affect total assets but not total supply, causing share price to change
- **Profit** = `(current_share_price / initial_share_price) - 1`

This approach isolates returns from capital flows, providing a cleaner profit metric for positions with multiple buys/sells at different prices.

## Key formula

```
share_price = total_assets / total_supply
profit_pct = (current_share_price / initial_share_price) - 1
```

- Initial share price starts at 1.0
- When buying: `shares_minted = buy_value / current_share_price`
- When selling: shares are burned proportionally based on quantity sold
- Total assets = current position value at mark price

## Files

### State dataclass

**`tradeexecutor/state/position_internal_share_price.py`**

Contains `SharePriceState` - the running state stored on each position for incremental tracking:

```python
@dataclass
class SharePriceState:
    current_share_price: float      # Current share price
    total_supply: float             # Total shares outstanding
    cumulative_quantity: float      # Total quantity held
    total_invested: float           # Total USD invested
    peak_total_supply: float        # Peak shares (for closed positions)
    initial_share_price: float      # Starting price (1.0)
    last_updated_at: datetime       # Last update timestamp
```

### Strategy functions

**`tradeexecutor/strategy/position_internal_share_price.py`**

Functions for managing share price state:

- `create_share_price_state(trade)` - Create initial state from first trade
- `update_share_price_state(state, trade)` - Update state with new trade
- `migrate_share_price_state(position)` - Rebuild state for legacy positions
- `backfill_share_price_state(state)` - Migrate all positions in portfolio

### Profit calculation

**`tradeexecutor/strategy/pnl.py`**

- `SharePriceData` - Result dataclass with profit metrics
- `calculate_share_price_pnl(position)` - Full recalculation from trade history

### Position method

**`tradeexecutor/state/position.py`**

- `TradingPosition.share_price_state` - Cached state field
- `TradingPosition.get_share_price_profit()` - Get profit using cached state (O(1)) or fallback to full recalculation

### Statistics integration

**`tradeexecutor/state/statistics.py`**

`PositionStatistics` includes share price fields:

- `internal_share_price` - Current share price
- `internal_total_supply` - Shares outstanding
- `internal_profit_pct` - Profit percentage from share price method
- `internal_profit_usd` - Profit in USD from share price method

## Usage

### Get profit for a position

```python
# Uses cached state for O(1) lookup, falls back to recalculation
share_data = position.get_share_price_profit()

print(f"Profit: {share_data.profit_pct:.2%}")
print(f"Profit USD: ${share_data.profit_usd:.2f}")
print(f"Share price: {share_data.current_share_price:.4f}")
```

### Migrate legacy positions

For positions created before this feature was added:

```python
from tradeexecutor.strategy.position_internal_share_price import backfill_share_price_state

# Migrate all positions in state
migrated_count = backfill_share_price_state(state)
print(f"Migrated {migrated_count} positions")
```

## How it works

### On trade execution

When a trade is executed via `State.mark_trade_success()`, the share price state is automatically updated:

1. **First trade**: Creates initial state with `shares = trade_value / 1.0`
2. **Subsequent buys**: Mints shares at current share price
3. **Sells**: Updates share price based on sell price, then burns proportional shares

### Incremental vs full calculation

- **Incremental (O(1))**: Uses cached `position.share_price_state` updated on each trade
- **Full recalculation (O(n))**: Replays all trades if no cached state exists

The incremental approach follows the same pattern as `TradingPosition.loan` for margin positions.

## Example

```
Position timeline:
1. Buy 1 ETH at $1700 -> mint 1700 shares at price 1.0
2. Buy 0.5 ETH at $1800 -> mint 900 shares at price 1.0 (total: 2600 shares)
3. ETH price rises to $1900
4. Current value: 1.5 ETH * $1900 = $2850
5. Share price: $2850 / 2600 = 1.096
6. Profit: (1.096 / 1.0) - 1 = 9.6%
```

## Testing

Run share price tests:

```bash
source .local-test.env && poetry run pytest tests/test_state.py -k "share_price" -v
```

Run legacy state file migration tests (requires env var):

```bash
source .local-test.env && RUN_INTERNAL_SHARE_PRICE_TESTS=true poetry run pytest tests/test_state.py -k "legacy_file or large_portfolio" -v
```

## Comparison with legacy method

| Aspect | Legacy (`get_total_profit_percent`) | Share price method |
|--------|-------------------------------------|-------------------|
| Multiple buys | Weighted average cost basis | Share minting |
| Partial sells | FIFO/average cost | Proportional burn |
| Capital flows | Affects cost basis | Isolated from returns |
| Complexity | O(n) per call | O(1) with cached state |
