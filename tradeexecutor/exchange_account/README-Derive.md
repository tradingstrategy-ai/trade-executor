# Derive exchange account integration

This document describes how to set up and use Derive.xyz integration with the trade executor for tracking external perp DEX positions.

## Overview

Exchange account positions allow tracking capital deployed to external perpetual DEXes like Derive. The trade executor monitors the account value via the Derive API and syncs profit/loss changes as balance update events.

For deploying a Lagoon vault on Derive chain and linking it to a Derive trading account, see the [Derive vault deployment guide](./README-Derive-Vault.md).

## Derive v3

Derive v3 replaced the v2 API at `api.lyra.finance` with `https://api.derive.xyz/v3` and settles on Ethereum. The integration uses the v3 API through `eth_defi.derive`. The main changes for operators are:

- **No separate Derive wallet.** The wallet that owns the account is the owner EOA or multisig itself. When `DERIVE_WALLET_ADDRESS` is not set, the owner EOA address is used.
- **Accounts start with a deposit.** A wallet has no v3 account until its first deposit. v2 accounts and session keys do not work against v3 until they are set up again.
- **Multisig owners use onchain actions.** A Safe registers its session key by calling `OnchainActionManager.submit()` on Ethereum, because it cannot sign the API's EIP-712 actions. See [smart contracts and multisigs](https://docs.derive.xyz/authentication/contract-owned-accounts).

See the [v3 breaking changes](https://docs.derive.xyz/migrating/breaking-changes) for the full list.

## Environment variables

Configure these environment variables for Derive integration:

| Variable | Required | Description |
|----------|----------|-------------|
| `DERIVE_SESSION_PRIVATE_KEY` | Yes | Private key for the Derive session key (used for API authentication) |
| `DERIVE_OWNER_PRIVATE_KEY` | No* | Private key of the Ethereum wallet that owns the Derive account |
| `DERIVE_WALLET_ADDRESS` | No* | Wallet that owns the Derive account (defaults to the owner EOA address). For Lagoon vault deployments, set this to the Safe multisig address. |
| `DERIVE_NETWORK` | No | Network selection: `mainnet` or `testnet` (default: `mainnet`) |

\* Either `DERIVE_OWNER_PRIVATE_KEY` or `DERIVE_WALLET_ADDRESS` must be provided. If the wallet address is provided directly, the owner private key is not needed. For Lagoon vault deployments, the Derive exchange account is registered against the Safe multisig address, so `DERIVE_WALLET_ADDRESS` must be set to the Safe address (not the vault contract address).

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
    exchange_name="Derive",
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

from tradeexecutor.exchange_account.derive import create_derive_account_value_func

# Set up authentication
owner_account = Account.from_key(os.environ["DERIVE_OWNER_PRIVATE_KEY"])
# In Derive v3 the owner EOA is the wallet that owns the account
derive_wallet = owner_account.address

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

The Ethereum wallet that owns the Derive account. In Derive v3 its address is the account's wallet address.

### Session key

A separate key pair used for API authentication. Created during Derive onboarding. Allows API access without exposing the owner wallet private key.

For Safe-owned Lagoon vault accounts, see [Rotating an existing Derive session key](./README-Derive-Vault.md#rotating-an-existing-derive-session-key) for the Safe multisig transaction flow.

### Subaccount

Derive accounts can have multiple subaccounts for isolated margin. Each subaccount has a unique integer ID. The `exchange_subaccount_id` in the trading pair metadata specifies which subaccount to track.

### Derive wallet address

In Derive v2 this was a LightAccount smart contract wallet on Derive Chain, derived from the owner address. Derive v3 removed it: the owner EOA or multisig is the wallet, and v2 Derive wallets were moved to their owners.

## Troubleshooting

### "Derive credentials required" error

Ensure `DERIVE_SESSION_PRIVATE_KEY` is set, and either `DERIVE_OWNER_PRIVATE_KEY` or `DERIVE_WALLET_ADDRESS` is provided.

### "Subaccount not found" warning

The subaccount ID in the trading pair metadata doesn't match any subaccounts on the Derive account. Verify the correct `exchange_subaccount_id` value.

### Network mismatch

Ensure `DERIVE_NETWORK` matches your setup:
- Use `testnet` for Derive testnet (chain ID 901)
- Use `mainnet` for Derive mainnet

## See also

- [eth-defi Derive module](https://github.com/tradingstrategy-ai/web3-ethereum-defi/tree/master/eth_defi/derive) - Low-level Derive API client
- [Derive documentation](https://docs.derive.xyz/) - Official Derive docs
