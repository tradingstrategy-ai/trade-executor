# CCXT exchange account integration

This document describes how to set up and use CCXT integration with the trade executor for tracking external CEX account positions (e.g. Aster futures).

## Overview

Exchange account positions allow tracking capital deployed to external exchanges. The trade executor monitors the account value via CCXT and syncs profit/loss changes as balance update events.

## Environment variables

Configure these environment variables for CCXT integration:

| Variable | Required | Description |
|----------|----------|-------------|
| `CCXT_EXCHANGE_ID` | Yes | CCXT exchange identifier (e.g. `aster`, `binance`, `bybit`) |
| `CCXT_OPTIONS` | Yes | JSON string with CCXT exchange constructor options (includes `apiKey`, `secret`, etc.) |
| `CCXT_SANDBOX` | No | Set to `true` for sandbox/testnet mode (default: `false`) |

Example `CCXT_OPTIONS` value:

```json
{"apiKey": "your-api-key", "secret": "your-api-secret"}
```

## Strategy setup

### 1. Create exchange account trading pair

In your strategy's `create_trading_universe()` function, define an exchange account pair:

```python
from tradeexecutor.state.identifier import (
    AssetIdentifier,
    TradingPairIdentifier,
    TradingPairKind,
)

# Quote asset (USDC for USD-denominated value)
usdc = AssetIdentifier(
    chain_id=1,
    address="0x0000000000000000000000000000000000000001",
    token_symbol="USDC",
    decimals=6,
)

# Base asset represents the exchange account value
aster_account = AssetIdentifier(
    chain_id=1,
    address="0x0000000000000000000000000000000000000002",
    token_symbol="ASTER-ACCOUNT",
    decimals=6,
)

# Exchange account trading pair with CCXT metadata
exchange_account_pair = TradingPairIdentifier(
    base=aster_account,
    quote=usdc,
    pool_address="0x0000000000000000000000000000000000000003",
    exchange_address="0x0000000000000000000000000000000000000004",
    internal_id=1,
    internal_exchange_id=1,
    fee=0.0,
    kind=TradingPairKind.exchange_account,
    exchange_name="ccxt_aster",
    other_data={
        "exchange_protocol": "ccxt",
        "ccxt_account_id": "aster_main",
        "ccxt_exchange_id": "aster",
        "exchange_is_testnet": False,
    },
)
```

### 2. Strategy configuration

Exchange account strategies use `TradeRouting.ignore` since trading happens on the external exchange:

```python
from tradeexecutor.strategy.default_routing_options import TradeRouting

trade_routing = TradeRouting.ignore
```

### 3. Example strategy modules

- `strategies/test_only/ccxt_exchange_account_strategy.py` - Full exchange account strategy with Aster pair (uses `ChainId.ethereum`)
- `strategies/test_only/minimal_ccxt_strategy.py` - Minimal strategy for CLI integration testing on Anvil chain

## CLI commands

### correct-accounts

Syncs exchange account positions and corrects any drift between tracked and actual values:

```bash
export CCXT_EXCHANGE_ID=aster
export CCXT_OPTIONS='{"apiKey": "your-key", "secret": "your-secret"}'

trade-executor correct-accounts \
    --strategy-file my_strategy.py \
    --state-file state/my_strategy.json
```

The command will:
1. Connect to the exchange via CCXT using configured credentials
2. Fetch current account value (e.g. `totalMarginBalance` for Aster futures)
3. Generate balance update events for any profit/loss changes
4. Save updated state

## Python API

### Creating account value function

```python
import os
from tradeexecutor.exchange_account.ccxt_exchange import (
    create_ccxt_exchange,
    create_ccxt_account_value_func,
    aster_total_equity,
)

# Create authenticated exchange
exchange = create_ccxt_exchange("aster", {
    "apiKey": os.environ["ASTER_API_KEY"],
    "secret": os.environ["ASTER_API_SECRET"],
})

# Create account value function
exchanges = {"aster_main": exchange}
account_value_func = create_ccxt_account_value_func(exchanges)

# Get account value for a position
value = account_value_func(exchange_account_pair)
```

### Using sync model

```python
import datetime
from tradeexecutor.exchange_account.sync_model import ExchangeAccountSyncModel

sync_model = ExchangeAccountSyncModel(account_value_func)

# Sync positions and get balance updates
events = sync_model.sync_positions(
    timestamp=datetime.datetime.utcnow(),
    state=state,
    strategy_universe=universe,
    pricing_model=pricing_model,
)

for evt in events:
    print(f"Position {evt.position_id}: PnL change {evt.quantity}")
```

### Custom value extractor

For exchanges other than Aster, provide a custom `value_extractor`:

```python
from decimal import Decimal

def bybit_total_equity(exchange):
    """Extract total equity from Bybit unified account."""
    response = exchange.privateGetV5AccountWalletBalance({"accountType": "UNIFIED"})
    return Decimal(str(response["result"]["list"][0]["totalEquity"]))

account_value_func = create_ccxt_account_value_func(
    exchanges,
    value_extractor=bybit_total_equity,
)
```

## Module structure

- `ccxt_exchange.py` - CCXT exchange factory and account value functions
- `derive.py` - Derive-specific account value function
- `sync_model.py` - `ExchangeAccountSyncModel` for syncing position values
- `pricing.py` - `ExchangeAccountPricingModel` for valuation
- `valuation.py` - `ExchangeAccountValuationModel` for portfolio valuation
- `tradeexecutor/cli/commands/correct_accounts.py` - CLI command with unified Derive/CCXT dispatcher

## Aster specifics

### Account value

The `aster_total_equity()` function reads `totalMarginBalance` from Aster's `/fapi/v4/account` endpoint. This value equals `totalWalletBalance + totalUnrealizedProfit` and is denominated in USDT.

### API authentication

Aster uses HMAC SHA256 signatures. The CCXT options dict needs `apiKey` and `secret`:

```python
exchange = create_ccxt_exchange("aster", {
    "apiKey": "your-api-key",
    "secret": "your-api-secret",
})
```

## Installation

CCXT is an optional dependency. Install with:

```bash
poetry install -E ccxt
```

## Testing

### Environment variables

Set these to run the Aster integration tests:

| Variable | Description |
|----------|-------------|
| `ASTER_API_KEY` | Aster API key |
| `ASTER_API_SECRET` | Aster API secret |

Tests are skipped automatically when these variables are not set or when `ccxt` is not installed.

### Test files

- `tests/exchange_account/test_ccxt_aster_integration.py` - API-level integration tests with real Aster (2 tests: `test_aster_total_equity_returns_value`, `test_ccxt_account_value_func_with_real_aster`)
- `tests/exchange_account/test_correct_accounts_ccxt.py` - CLI integration tests (`test_correct_accounts_aster`, `test_aster_cli_start`)

### Running tests

```bash
source .local-test.env && poetry run pytest tests/exchange_account/test_ccxt_aster_integration.py -v
source .local-test.env && poetry run pytest tests/exchange_account/test_correct_accounts_ccxt.py -v
```

## See also

- [CCXT documentation](https://docs.ccxt.com/) - CCXT library docs
- [Aster API documentation](https://docs.asterdex.com/product/aster-perpetuals/api/api-documentation) - Official Aster API docs
- [README-Derive.md](README-Derive.md) - Derive exchange account integration
