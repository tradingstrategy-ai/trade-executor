# Orderly Network Integration

This folder contains the trade-executor integration for [Orderly Network](https://orderly.network/), a decentralized orderbook exchange with an on-chain vault for deposits and withdrawals.

## Overview

Orderly Network is a CEX-like orderbook exchange that combines:
- **On-chain vault** for deposits and withdrawals
- **Off-chain orderbook** for trading (high performance, low latency)
- **Broker system** for integration partners

The integration bridges on-chain assets (via vault) with off-chain Orderly trading accounts.

## Architecture

### Two-Layer Design

The integration is split across two layers following clean architecture principles:

#### 1. Low-Level Layer: `eth_defi` (in deps/web3-ethereum-defi)

**Location:** `deps/web3-ethereum-defi/eth_defi/orderly/`

**Purpose:** Basic blockchain interactions and vault operations

**Files:**
- `vault.py` - Simple `OrderlyVault` wrapper class
- `api.py` - Orderly API client for off-chain operations

**Key Components:**

```python
# Simple vault wrapper
class OrderlyVault:
    def __init__(self, web3: Web3, address: str):
        self.web3 = web3
        self.address = address
        self.contract = get_deployed_contract(web3, "orderly/Vault.json", address)

# Helper functions that return contract function objects
def deposit(...) -> tuple[ContractFunction, ContractFunction, ContractFunction]:
    # Returns: (approve_fn, get_deposit_fee_fn, deposit_fn)

def withdraw(...) -> tuple[ContractFunction, ContractFunction, ContractFunction]:
    # Returns: (approve_fn, get_withdraw_fee_fn, withdraw_fn)
```

**Characteristics:**
- No VaultBase inheritance (simple wrapper)
- Returns contract functions, not transactions
- Stateless operations
- Can be used standalone

#### 2. High-Level Layer: `tradeexecutor` (this folder)

**Location:** `tradeexecutor/ethereum/orderly/`

**Purpose:** Full vault pattern implementation with VaultBase integration

**Files:**
- `orderly_vault.py` - Full `VaultBase` implementation
- `orderly_routing.py` - Routing model for vault flows
- `orderly_execution.py` - Execution model
- `orderly_analysis.py` - Transaction log analysis
- `orderly_valuation.py` - Position valuation
- `orderly_live_pricing.py` - Live price feeds
- `tx.py` - Transaction builder

**Key Components:**

```python
# Full VaultBase implementation
class OrderlyVault(VaultBase):
    def __init__(
        self,
        web3: Web3,
        spec: VaultSpec,
        broker_id: str,
        supported_tokens: dict[str, str] | None = None,
    ):
        super().__init__()
        # Implements all VaultBase abstract methods

class OrderlyFlowManager(VaultFlowManager):
    # Manages deposit/redemption flows

class OrderlyVaultInfo(VaultInfo):
    # Vault metadata (broker_id, supported_tokens, etc.)
```

## How Orderly Integration Works

### Prerequisites: Delegate Signing Setup (Manual Process)

**Before deposits can work, a one-time delegate signing setup must be completed:**

This is a manual process that grants your vault contract permission to operate with Orderly on behalf of the user.

**Steps:**

1. **Deploy or identify your vault contract** (e.g., SimpleVaultV0)
   - This is the contract that will hold user funds
   - Must have the user's EOA as owner/asset manager

2. **Call `delegateSigner()` on Orderly vault contract**
   ```python
   # From vault contract, call Orderly vault's delegateSigner
   broker_hash = web3.keccak(text="woofi_pro")
   tx = orderly_vault.functions.delegateSigner(
       (broker_hash, hot_wallet_address)
   ).transact()
   ```

3. **Confirm delegation via Orderly API**
   ```python
   from eth_defi.orderly.api import OrderlyApiClient

   client = OrderlyApiClient(
       account=hot_wallet.account,
       broker_id="woofi_pro",
       chain_id=42161,
       is_testnet=True,
   )

   response = client.delegate_signer(
       delegate_contract=vault_address,
       delegate_tx_hash=tx.hex(),
   )
   # Returns: {"success": true, "data": {"account_id": "0x..."}}
   ```

4. **Register Ed25519 key for API access (optional)**
   ```python
   response = client.register_key(
       delegate_contract=vault_address,
   )
   # Save the returned secret for API authentication
   orderly_secret = response["data"]["secret"]
   ```

**Important Notes:**
- This setup is required ONCE per vault contract
- The delegation links your vault contract address with Orderly's system
- Without this, deposits will fail
- See `deps/web3-ethereum-defi/tests/orderly/manual_test_orderly_delegate.py` for reference implementation

### 1. Deposit Flow

**User wants to deposit USDC to trade on Orderly:**

```
1. User initiates deposit via trade-executor
   |
   v
2. OrderlyRouting.deposit_or_withdraw() prepares transactions:
   - tx_1: approve(vault, amount)
   - tx_2: vault.deposit(accountId, brokerId, tokenId, amount)
   |
   v
3. Transactions executed on-chain
   |
   v
4. Orderly vault contract emits Deposit event
   |
   v
5. Funds credited to user's Orderly account (off-chain)
   |
   v
6. User can now trade on Orderly orderbook
```

**Key Details:**
- Requires hashed `broker_id` and `token_id` (Keccak256)
- Deposit fee may be charged (queried via `getDepositFee()`)
- Two-transaction pattern: approve + deposit

### 2. Withdrawal Flow

**User wants to withdraw from Orderly back to wallet:**

```
1. User initiates withdrawal via trade-executor
   |
   v
2. OrderlyRouting.deposit_or_withdraw() prepares transactions:
   - tx_1: approve(vault, amount)  [if needed]
   - tx_2: vault.withdraw(accountId, brokerId, tokenId, amount)
   |
   v
3. Transactions executed on-chain
   |
   v
4. Orderly vault contract emits Withdraw event
   |
   v
5. Funds transferred from vault to user wallet
```

### 3. Trading (Off-Chain)

Trading happens externally on Orderly's orderbook exchange:
- Uses Orderly API (`eth_defi.orderly.api.OrderlyApiClient`)
- Requires Ed25519 key registration for signing
- Trading does NOT go through trade-executor
- PnL reflected in Orderly account balance

## Configuration

### Token ID Mapping

Orderly uses its own token identifiers. Configure mapping in routing:

```python
token_id_mapping = {
    "USDC": "USDC",
    "WETH": "WETH",
    "USDT": "USDT",
}

routing = OrderlyRouting(
    reserve_token_address=usdc.address,
    vault=vault,
    broker_id="woofi_pro",
    orderly_account_id=account_id,
    token_id_mapping=token_id_mapping,
)
```

## File Structure

```
tradeexecutor/ethereum/orderly/
|-- README.md                    # This file
|-- orderly_vault.py             # VaultBase implementation
|   |-- OrderlyVault             # Main vault class
|   |-- OrderlyFlowManager       # Deposit/withdrawal flow tracking
|   +-- OrderlyVaultInfo         # Vault metadata
|-- orderly_routing.py           # Routing model for vault operations
|   |-- OrderlyRouting           # Routes vault deposit/withdraw trades
|   |-- OrderlyRoutingState      # Per-cycle state management
|   +-- get_orderly_vault_for_pair()  # Vault instance retrieval
|-- orderly_execution.py         # Execution model
|   +-- OrderlyExecution         # Manages trade execution lifecycle
|-- orderly_analysis.py          # Transaction log analysis
|   +-- analyse_orderly_flow_transaction()  # Parses tx receipts
|-- orderly_valuation.py         # Position valuation
|   +-- OrderlyValuator          # Re-values positions
|-- orderly_live_pricing.py      # Live price feeds
|   +-- OrderlyPricing           # Fetches current prices
+-- tx.py                        # Transaction builder
    +-- OrderlyTransactionBuilder  # Builds signed transactions
```

## Work In Progress (WIP)

- Event parsing in transaction analysis (`orderly_analysis.py`)
- Flow manager event scanning (`OrderlyFlowManager` methods)
- Portfolio fetching from on-chain + off-chain sources
- NAV calculation (hybrid on-chain/off-chain)

## Testing

### Test Files

```
tests/orderly/
|-- conftest.py                  # Pytest fixtures
|-- test_orderly_routing.py      # Routing model tests
|-- test_orderly_execution.py    # Execution model tests (TODO)
|-- test_orderly_pricing.py      # Pricing tests (TODO)
+-- test_orderly_tx.py           # Transaction builder tests (TODO)
```

### Test Networks

**Arbitrum Sepolia (Testnet):**
- Vault: `0x0EaC556c0C2321BA25b9DC01e4e3c95aD5CDCd2f`
- USDC: `0x75faf114eafb1BDbe2F0316DF893fd58CE46AA4d`
- Chain ID: 421614
- Broker: `woofi_pro`

### Documentation
- [Deposit/Withdrawal Flow](https://orderly.network/docs/build-on-omnichain/user-flows/withdrawal-deposit)

## Future Integration Requirements

### PnL Settlement via API

Currently not implemented. Strategies will need to periodically settle PnL on Orderly to reconcile off-chain trading profits/losses with on-chain balances.

**What it does:**
- Realizes trading profits to on-chain vault balance
- Updates available withdrawal amounts
- Required for accurate NAV calculations

**Implementation:**
- Settlement logic should be implemented in `eth_defi.orderly.api` module
- Uses EIP-712 `SettlePnl` or `DelegateSettlePnl` message signing
- Integration point: Create custom `OrderlyVaultSyncModel` in tradeexecutor (similar to `LagoonVaultSyncModel`)
- Should run before withdrawals or during regular sync cycles

**References:**
- Message types: `deps/web3-ethereum-defi/eth_defi/orderly/constants.py`
- API client: `deps/web3-ethereum-defi/eth_defi/orderly/api.py`

