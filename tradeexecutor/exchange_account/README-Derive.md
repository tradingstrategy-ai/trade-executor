# Derive exchange account integration

This document describes how to set up and use Derive.xyz integration with the trade executor for tracking external perp DEX positions.

## Overview

Exchange account positions allow tracking capital deployed to external perpetual DEXes like Derive. The trade executor monitors the account value via the Derive API and syncs profit/loss changes as balance update events.

## Environment variables

Configure these environment variables for Derive integration:

| Variable | Required | Description |
|----------|----------|-------------|
| `DERIVE_OWNER_PRIVATE_KEY` | Yes | Private key of the Ethereum wallet that owns the Derive account |
| `DERIVE_SESSION_PRIVATE_KEY` | Yes | Private key for the Derive session key (used for API authentication) |
| `DERIVE_WALLET_ADDRESS` | No | Derive wallet address (auto-derived from owner if not provided) |
| `DERIVE_NETWORK` | No | Network selection: `mainnet` or `testnet` (default: `mainnet`) |

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
    chain_id=1,  # Ethereum mainnet
    address="0x0000000000000000000000000000000000000001",  # Synthetic address
    token_symbol="USDC",
    decimals=6,
)

# Base asset represents the exchange account value
derive_account = AssetIdentifier(
    chain_id=1,
    address="0x0000000000000000000000000000000000000002",  # Synthetic address
    token_symbol="DERIVE-ACCOUNT",
    decimals=6,
)

# Exchange account trading pair with Derive metadata
exchange_account_pair = TradingPairIdentifier(
    base=derive_account,
    quote=usdc,
    pool_address="0x0000000000000000000000000000000000000003",
    exchange_address="0x0000000000000000000000000000000000000004",
    internal_id=1,
    internal_exchange_id=1,
    fee=0.0,
    kind=TradingPairKind.exchange_account,
    exchange_name="derive",
    other_data={
        "exchange_protocol": "derive",
        "exchange_subaccount_id": 143246,  # Your Derive subaccount ID
        "exchange_is_testnet": False,
    },
)
```

### 2. Strategy configuration

Exchange account strategies typically use `TradeRouting.ignore` since trading happens on the external exchange:

```python
from tradeexecutor.strategy.default_routing_options import TradeRouting

trade_routing = TradeRouting.ignore
```

### 3. Example strategy module

See `strategies/test_only/exchange_account_strategy.py` for a complete example.

## CLI commands

### correct-accounts

Syncs exchange account positions and corrects any drift between tracked and actual values:

```bash
trade-executor correct-accounts \
    --strategy-file my_strategy.py \
    --state-file state/my_strategy.json
```

The command will:
1. Connect to Derive API using configured credentials
2. Fetch current account value for each exchange account position
3. Generate balance update events for any profit/loss changes
4. Save updated state

## Python API

### Creating account value function

```python
from eth_account import Account
from eth_defi.derive.authentication import DeriveApiClient
from eth_defi.derive.account import fetch_subaccount_ids
from eth_defi.derive.onboarding import fetch_derive_wallet_address

from tradeexecutor.exchange_account.derive import create_derive_account_value_func

# Set up authentication
owner_account = Account.from_key(os.environ["DERIVE_OWNER_PRIVATE_KEY"])
derive_wallet = fetch_derive_wallet_address(owner_account.address, is_testnet=False)

client = DeriveApiClient(
    owner_account=owner_account,
    derive_wallet_address=derive_wallet,
    is_testnet=False,
    session_key_private=os.environ["DERIVE_SESSION_PRIVATE_KEY"],
)

# Get subaccount IDs
subaccount_ids = fetch_subaccount_ids(client)
client.subaccount_id = subaccount_ids[0]

# Create account value function
clients = {client.subaccount_id: client}
account_value_func = create_derive_account_value_func(clients)

# Get account value for a position
value = account_value_func(exchange_account_pair)
```

### Using sync model

```python
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

## Module structure

- `derive.py` - Derive-specific account value function and `DeriveNetwork` enum
- `sync_model.py` - `ExchangeAccountSyncModel` for syncing position values
- `pricing.py` - `ExchangeAccountPricingModel` for valuation
- `valuation.py` - `ExchangeAccountValuationModel` for portfolio valuation

## Derive concepts

### Owner wallet

The Ethereum wallet that controls the Derive account. Used to derive the Derive wallet address and sign authentication requests.

### Session key

A separate key pair used for API authentication. Created during Derive onboarding. Allows API access without exposing the owner wallet private key.

### Subaccount

Derive accounts can have multiple subaccounts for isolated margin. Each subaccount has a unique integer ID. The `exchange_subaccount_id` in the trading pair metadata specifies which subaccount to track.

### Derive wallet address

A smart contract wallet address on the Derive L2 chain. Automatically derived from the owner wallet address using a deterministic formula.

## Troubleshooting

### "Derive credentials required" error

Ensure both `DERIVE_OWNER_PRIVATE_KEY` and `DERIVE_SESSION_PRIVATE_KEY` are set.

### "Subaccount not found" warning

The subaccount ID in the trading pair metadata doesn't match any subaccounts on the Derive account. Verify the correct `exchange_subaccount_id` value.

### Network mismatch

Ensure `DERIVE_NETWORK` matches your setup:
- Use `testnet` for Derive testnet (chain ID 901)
- Use `mainnet` for Derive mainnet

## See also

- [eth-defi Derive module](https://github.com/tradingstrategy-ai/web3-ethereum-defi/tree/master/eth_defi/derive) - Low-level Derive API client
- [Derive documentation](https://docs.derive.xyz/) - Official Derive docs
