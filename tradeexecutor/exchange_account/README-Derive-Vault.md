# Derive vault deployment

## Overview

Derive Chain is an OP Stack L2 rollup built specifically for decentralised derivatives trading (options and perpetuals), offering institutional-grade infrastructure with unified portfolio margin, high-performance orderbook matching, and low gas costs optimized for frequent trading operations.

This document details the steps required to deploy a Trading Strategy vault to Derive Chain and link the vault's Safe multisig to a Derive trading account via session keys. This setup enables automated strategy execution on Derive's derivatives markets while maintaining non-custodial control of vault assets through the Safe.

## Deployment steps

### Step 1: Deploy Lagoon vault on Derive chain

The process of deploying a Lagoon vault on Derive is essentially the same as it is for other chains, with two Derive-specific requirements:
- **Derive deployer:** The account used for deployment must be the Derive deployer. This is a hot wallet that has been approved for deployment on Derive (via governance vote). Mikko has the private key.
- **Asset manager(s):** Since the same account is always used for deployment, you usually create one or more separate asset manager hot wallets per vault deployment.

The `ASSET_MANAGER` setting accepts an ordered comma-separated list. The first address is the primary asset manager and becomes the Lagoon valuation manager. Any later addresses are secondary asset managers: they receive guard sender permissions, but they do not become the Lagoon vault manager.

For the underlying Python role definitions, see `deps/web3-ethereum-defi/eth_defi/erc_4626/vault_protocol/lagoon/deployment.py`, especially `LagoonConfig.asset_managers` and the `deploy_automated_lagoon_vault()` role documentation.

#### 1.a. Create asset manager hot wallet

Follow the instructions [here](https://ethereum.stackexchange.com/questions/82926/how-to-generate-a-new-ethereum-address-and-private-key-from-a-command-line), or use `cast` if you have Foundry installed.

```bash
cast wallet new
```

> [!IMPORTANT]
> Store the key securely. Keep the address handy for step 1.b.

#### 1.b. Deploy Lagoon vault to Derive chain

The vault is deployed using the `trade-executor lagoon-deploy-vault` command.

<details>
<summary><strong>Deployment example</strong></summary>

```shell
# Derive chain values needed for deployment
export JSON_RPC_DERIVE="https://rpc.derive.xyz/"
export VERIFIER="blockscout"
export VERIFIER_URL="https://explorer.derive.xyz/api"

# USDC contract on Derive
export DENOMINATION_ASSET=0x6879287835A86F50f784313dBEd5E5cCC5bb8481

# Trading Strategy Derive Deployer key (request from Mikko)
export PRIVATE_KEY=0x123...

# Address list (NOT keys) for the asset manager hot wallet(s) created above.
# The first address is primary and becomes the Lagoon valuation manager.
export ASSET_MANAGER="0x234..., 0x345..."

# Update with correct vault-specific values
export FUND_SYMBOL=XYZ123
export FUND_NAME="Derive derivatives trading vault"
export MANAGEMENT_FEE=0
export PERFORMANCE_FEE=0

# See https://github.com/tradingstrategy-ai/scratchpad/wiki/Safe-multisig-signers
export MULTISIG_OWNERS="0xa7208b5c92d4862b3f11c0047b57a00Dc304c0f8, 0x5C46ab9e42824c51b55DcD3Cf5876f1132F9FbA9, 0x10df0a900a7c595c35e782e1dad71d333ce33038, 0xFFCb2e7196B786bff5F5eC6ab31b5C428C7Ae7eB"

trade-executor lagoon-deploy-vault \
  --vault-record-file="tmp/derive-vault-info.json" \
  --any-asset \
  --performance-fee="$PERFORMANCE_FEE" \
  --management-fee="$MANAGEMENT_FEE"  
```

</details>

### Step 2: Register Derive session key and approve deposits

In order to trade with funds from a safe wallet on Derive, you must register a Derive "session key" and approve the Derive deposit manager contract as spender of USDC. This is completed using the Safe transaction builder. See [Manual Gnosis safe onboarding](https://derivexyz.notion.site/Manual-Gnosis-safe-onboarding-24503b51517e80e1b75ef0db0096f6ff) **Enabling using Derive UI** for additional background.

> [!NOTE]
> [Step 3](#step-3-make-a-test-deposit-usdc) can be completed in parallel with step 2. If you are waiting on the Derive team to complete 2.a. or waiting on Trading Strategy team members to sign multisig transactions, proceed to step 3.

#### 2.a. Request safe address whitelist

Before step 2.c. below can be completed, the Derive team must whitelist the safe address (created in step 1) in their API. This is documented in the [safe onboarding guide](https://derivexyz.notion.site/Manual-Gnosis-safe-onboarding-24503b51517e80e1b75ef0db0096f6ff) (see "Add account to API"). Request this before completing other steps so it can be addressed in parallel. Mikko has a Telegram group with members of the Derive team.

#### 2.b. Create session key hot wallet

You'll need a separate hot wallet to act as the Derive session key. See [step 1.a.](#1a-create-asset-manager-hot-wallet) for instructions (but create a _new_ hot wallet for the session key).

#### 2.c. Build Safe transaction batch

Visit the [Derive Safe(Wallet) UI](https://safe.derive.xyz) and connect a wallet that's a signer on the vault safe. Click "use transaction builder" to build a new transaction batch with the following two transactions:

<details>
<summary><strong>Approve transaction</strong></summary>

- Address: `0x6879287835A86F50f784313dBEd5E5cCC5bb8481` (USDC on Derive)
- ABI:
   ```
   [{"inputs":[{"internalType":"address","name":"spender","type":"address"},{"internalType":"uint256","name":"amount","type":"uint256"}],"name":"approve","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"nonpayable","type":"function"}]
   ```
- Spender: `0x9B3FE5E5a3bcEa5df4E08c41Ce89C4e3Ff01Ace3` (Derive Deposit Module)
- Amount: `1000000000000000000000000000000000000` (infinite spend approval)
</details>

<details>
<summary><strong>Register session key transaction</strong></summary>

- Address: `0xeB8d770ec18DB98Db922E9D83260A585b9F0DeAD` (Derive Matching contract)
- ABI:
   ```
   [{"inputs":[{"internalType":"address","name":"sessionKey","type":"address"},{"internalType":"uint256","name":"expiry","type":"uint256"}],"name":"registerSessionKey","outputs":[],"stateMutability":"nonpayable","type":"function"}]
   ```
- Session key: `0x123...` (session key address from step 2.b.)
- Expiry: Unix timestamp in seconds, for example 30-90 days from now. See [Choose the expiry timestamp](#2-choose-the-expiry-timestamp).
</details>

After you've created and signed the transaction batch, request signatures from the other Trading Strategy Safe signers. Once there are sufficient signatures, use the Deployer hot wallet to execute (since this account has some ETH for gas).

### Rotating an existing Derive session key

Use this process when the current `DERIVE_SESSION_PRIVATE_KEY` is expiring, has been exposed, or needs to be replaced for a trader. The Derive exchange account is owned by the vault Safe, so the Safe must authorise the new session key on the Derive Matching contract. The session key wallet itself does not need ETH, because the Safe transaction executor pays gas.

#### Rotation checklist

1. Create a new session key wallet.
2. Register the new session key address from the Safe.
3. Update the runtime secret that contains the session key private key.
4. Verify Derive API access using the Safe address as `DERIVE_WALLET_ADDRESS`.
5. Optionally deregister the old session key.

#### 1. Create a new session key wallet

Create a fresh Ethereum account. Do not reuse the asset manager key, Derive deployer key, or any Safe signer key.

```bash
cast wallet new
```

Store both values:

- Address: used in the Safe transaction as `sessionKey`
- Private key: used later as `DERIVE_SESSION_PRIVATE_KEY`

If a human trader needs UI access, import the private key to Rabby or another wallet and use the Derive UI session key login flow in [Step 4.a](#4a-sign-into-derive-ui-with-session-key).

#### 2. Choose the expiry timestamp

`registerSessionKey()` takes an explicit Unix timestamp in seconds. There is no implicit Derive contract default: every Safe transaction must set `expiry`.

Our operational default is 90 days. Set this by changing `EXPIRY_DAYS` below before generating the timestamp:

```bash
poetry run python - <<'PY'
import time

EXPIRY_DAYS = 90

expiry = int(time.time()) + EXPIRY_DAYS * 24 * 60 * 60
print(expiry)
print(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(expiry)))
PY
```

Use a shorter value, such as 30 days, for high-risk rotations or temporary trader access. The Matching contract stores the expiry as a Solidity `uint256`, so the on-chain maximum is `2**256 - 1`, i.e. `115792089237316195423570985008687907853269984665640564039457584007913129639935`. Treat this as a technical limit, not an operational recommendation. Do not use a non-expiring or far-future key for production vaults; rotate keys before expiry.

#### 3. Register the new key from the Safe

You can prepare the Safe Transaction Builder inputs from a `trade-executor console` session. This snippet creates a new session key, sets a five-year expiry, and prints both raw calldata and ABI-based input values.

> [!IMPORTANT]
> The snippet prints the new session key private key. Store it in the deployment secret manager as `DERIVE_SESSION_PRIVATE_KEY` and do not paste it to Safe, Derive UI chat, pull requests, or tickets. Only the session key **address** and expiry go to the Safe transaction.

```python
import json
import os
import time

from eth_account import Account
from web3 import Web3


DERIVE_MATCHING_CONTRACT = Web3.to_checksum_address("0xeB8d770ec18DB98Db922E9D83260A585b9F0DeAD")
SAFE_ADDRESS = os.environ.get("DERIVE_WALLET_ADDRESS", "<vault Safe address>")
EXPIRY_YEARS = 5
EXPIRY_SECONDS = EXPIRY_YEARS * 365 * 24 * 60 * 60

REGISTER_SESSION_KEY_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "sessionKey", "type": "address"},
            {"internalType": "uint256", "name": "expiry", "type": "uint256"},
        ],
        "name": "registerSessionKey",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    }
]

session_key = Account.create()
session_key_private_key = session_key.key.hex()
session_key_address = Web3.to_checksum_address(session_key.address)
expiry = int(time.time()) + EXPIRY_SECONDS

matching = Web3().eth.contract(
    address=DERIVE_MATCHING_CONTRACT,
    abi=REGISTER_SESSION_KEY_ABI,
)
raw_hex = matching.encode_abi(
    "registerSessionKey",
    args=[session_key_address, expiry],
)

print("Secret material, store in deployment secrets")
print("DERIVE_SESSION_PRIVATE_KEY=" + session_key_private_key)
print()

print("Safe Transaction Builder - raw hex mode")
print("Safe / from: " + SAFE_ADDRESS)
print("To: " + DERIVE_MATCHING_CONTRACT)
print("Value: 0")
print("Data: " + raw_hex)
print()

print("Safe Transaction Builder - ABI mode")
print("Contract address: " + DERIVE_MATCHING_CONTRACT)
print("ABI:")
print(json.dumps(REGISTER_SESSION_KEY_ABI, separators=(",", ":")))
print("Function: registerSessionKey")
print("sessionKey: " + session_key_address)
print("expiry: " + str(expiry))
print("expiry UTC: " + time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(expiry)))
```

In Gnosis Safe Transaction Builder, either paste `To`, `Value`, and `Data` as a raw transaction, or paste `Contract address`, `ABI`, `Function`, `sessionKey`, and `expiry` as an ABI-based contract interaction. Make sure the selected Safe is the vault Safe printed as `Safe / from`.

Open the [Derive Safe(Wallet) UI](https://safe.derive.xyz), select the vault Safe, and open Transaction Builder.

Add a contract interaction:

- Address: `0xeB8d770ec18DB98Db922E9D83260A585b9F0DeAD` (Derive Matching contract)
- Value: `0`
- ABI:
   ```
   [{"inputs":[{"internalType":"address","name":"sessionKey","type":"address"},{"internalType":"uint256","name":"expiry","type":"uint256"}],"name":"registerSessionKey","outputs":[],"stateMutability":"nonpayable","type":"function"}]
   ```
- `sessionKey`: new session key address from step 1
- `expiry`: Unix timestamp from step 2

Submit the batch, collect enough Safe signatures, then execute it with a wallet that has ETH on Derive chain for gas. The Trading Strategy Derive Deployer hot wallet is commonly used for execution because it is already funded.

> [!NOTE]
> The Matching contract stores session key permissions as `sessionKeys(sessionKey, owner) -> expiry`, where `owner` is the Safe address. Registering a different new key does not require deregistering the old key first.

#### 4. Update trade executor secrets

Update the deployed environment after the Safe transaction has executed:

```bash
export DERIVE_SESSION_PRIVATE_KEY="0x..."  # New session key private key
export DERIVE_WALLET_ADDRESS="0x..."       # Vault Safe address, not the vault contract
export DERIVE_NETWORK="mainnet"
```

For Lagoon vault deployments, `DERIVE_OWNER_PRIVATE_KEY` is not needed when `DERIVE_WALLET_ADDRESS` is set to the Safe address. If you also use the integration-test variables, update `DERIVE_INTEGRATION_TEST_SESSION_KEY` to the new private key.

Restart the trade executor process after updating the secret.

#### 5. Verify the new key

Run a read-only API check from the repository root:

```bash
source .local-test.env
export DERIVE_SESSION_PRIVATE_KEY="0x..."
export DERIVE_WALLET_ADDRESS="0x..."  # Vault Safe address
export DERIVE_NETWORK="mainnet"

poetry run python - <<'PY'
from tradeexecutor.exchange_account.derive import discover_derive_subaccount_id

print(discover_derive_subaccount_id())
PY
```

The command should print the Safe-owned Derive subaccount ID. A `403 Forbidden` response usually means the new key has not been registered for this Safe, the wrong Safe address was used, or the old private key is still configured.

You can also verify on the Matching contract by reading:

- Function: `sessionKeys(address signer, address owner)`
- `signer`: new session key address
- `owner`: vault Safe address

The return value is the registered expiry timestamp.

#### 6. Deregister the old key

If the old key has not yet expired and should no longer trade, create another Safe transaction:

- Address: `0xeB8d770ec18DB98Db922E9D83260A585b9F0DeAD` (Derive Matching contract)
- Value: `0`
- ABI:
   ```
   [{"inputs":[{"internalType":"address","name":"sessionKey","type":"address"}],"name":"deregisterSessionKey","outputs":[],"stateMutability":"nonpayable","type":"function"}]
   ```
- `sessionKey`: old session key address

`deregisterSessionKey()` does not revoke instantly. It sets the old key to expire after Derive's deregistration cooldown, currently 10 minutes on mainnet. If the old key has already expired, omit this transaction because it will revert.

If you need to shorten the expiry of the same session key address, deregister it first, wait for the cooldown to pass, and then register it again with the lower expiry. Calling `registerSessionKey()` with an expiry lower than or equal to the existing expiry reverts with `OV_NeedDeregister`.

### Step 3: Make a test deposit (USDC)

To verify that the vault is set up correctly and make some funds available for trading, you'll need to deposit a small amount of USDC into the vault. This process involves: (a) funding wallets with ETH for gas; (b) bridging USDC to Derive; (c) performing the test deposit.

#### 3.a. Fund wallets with ETH for gas

Both the depositor EOA and the asset manager wallet need a small ETH balance on Derive for gas fees. Gas fees are very low on Derive — `0.00005` ETH should be sufficient for each. One easy solution is to transfer from the Deployer hot wallet.

**Wallets that need ETH:**
- **Depositor EOA:** signs approval, requests deposit
- **Primary asset manager:** (from step 1) signs the settlement transaction (post NAV + settle)

#### 3.b. Bridge USDC to Derive chain

See Derive documentation [Depositing collateral assets](https://docs.derive.xyz/reference/deposit-to-lyra-chain#depositing-collateral-assets-usdc-weth-wbtc-etc) and [Manual Gnosis safe onboarding](https://derivexyz.notion.site/Manual-Gnosis-safe-onboarding-24503b51517e80e1b75ef0db0096f6ff) section titled  **Bridging to Derive**.

<details>
<summary><strong>Bridge USDC from Arbitrum</strong></summary>

**Addresses:**
- [Arbitrum USDC bridge:](https://arbiscan.io/address/0x5e027ad442e031424b5a2c0ad6f656662be32882) `0x5e027ad442e031424b5a2C0ad6f656662Be32882`
- [USDC on Arbitrum:](https://arbiscan.io/token/0xaf88d065e77c8cc2239327c5edb3a432268e5831) `0xaf88d065e77c8cC2239327C5EDb3A432268e5831`
- [Connector address:](https://arbiscan.io/token/0x17Fc4c7ea8267044b6D0ACC17a6C049Bed6F8B21) `0x17Fc4c7ea8267044b6D0ACC17a6C049Bed6F8B21`

**Bridge steps:**
1. From the wallet account that you're bridging USDC from, approve the amount you want to bridge on [USDC contract](https://arbiscan.io/token/0xaf88d065e77c8cc2239327c5edb3a432268e5831#writeProxyContract) with the bridge address as spender. Bridge address: `0x5e027ad442e031424b5a2c0ad6f656662be32882`

2. Use the [bridge contract](https://arbiscan.io/address/0x5e027ad442e031424b5a2c0ad6f656662be32882#readContract#F5) `getMinFees` to get the estimated bridge price. Enter the above connector address and `100000` for gas limit.
3. Use the [bridge contract](https://arbiscan.io/address/0x5e027ad442e031424b5a2c0ad6f656662be32882#writeContract#F2) `depositToAppChain` bridge the asset:
   - `payableAmount`: [convert](https://arbiscan.io/unitconverter) the amount from step 2 to ether: `0.0000025269`
   - `receiver`: address you're bridging to (same as address you're bridging from): ``
   - `amount`: amount USDC to bridge * 10^6 (e.g., `25` USDC = `25000000`)
   - `msgGasLimit`: `100000`
   - `connector`: `0x17Fc4c7ea8267044b6D0ACC17a6C049Bed6F8B21`
</details>

#### 3.c. Perform the test deposit

The test deposit uses the `trade-executor console` command with a minimal strategy config and a deposit script. The full ERC-7540 async deposit flow is: approve USDC, request deposit, settle (asset manager), finalise deposit.

<details>
<summary><strong>Test deposit steps</strong></summary>

Run the following from a local `trade-executor` instance:

**Set up environment variables:**

```bash
source .local-test.env
export PRIVATE_KEY="0x..."              # Asset manager private key (needs ETH for gas)
export DEPOSITOR_PRIVATE_KEY="0x..."    # EOA wallet private key with USDC on Derive (needs ETH for gas)
export VAULT_ADDRESS="0x..."            # "Vault" address from step 1
export VAULT_ADAPTER_ADDRESS="0x..."    # "Trading strategy module" address from step 1
export JSON_RPC_DERIVE="https://rpc.derive.xyz"
```

**Launch the console:**

```bash
trade-executor console \
    --strategy-file=strategies/test_only/derive-lagoon-vault.py \
    --asset-management-mode=lagoon
```

> [!NOTE]
> You will see warnings about routing and statistics not being available. This is expected — the minimal strategy config does not include real trading pairs, and these are not needed for deposit operations.

**Run the deposit script:**

Once in the IPython console, paste the contents of `scripts/derive-lagoon-test-deposit.py`. This will:

1. Set up the depositor wallet and register it as a signer
2. Check USDC and ETH balances
3. Approve the vault to spend 25 USDC
4. Request deposit (ERC-7540 phase 1)
5. Initialise the sync model and settle via the asset manager (posts NAV onchain + settles deposits)
6. Finalise the deposit (ERC-7540 phase 2) — moves share tokens to the depositor wallet
7. Print the resulting VEGA share token balance

After a successful deposit you should see output similar to:

```
Depositor share token balance: 25.0
Done!
```

</details>

#### 3.d) Deposit on Derive explorer directlty

You can deposit using *Write contract* on Derive explorer

1) Approve USDC

2) Call requestDeposit 

- https://explorer.derive.xyz/address/0x0C81AD1825826eECB11E46Cb0C730b1747f07e0B?tab=write_proxy

- Asset: `1500000000` (USDC amount on Derive)
- Controll

### Step 4: Sign into Derive UI and make test trades

#### 4.a. Sign into Derive UI with session key

- Open the [Derive Trading UI](https://app.derive.xyz/)
- Click "connect", connect using the session key wallet from [step 2.b.](#2b-create-session-key-hot-wallet)
- After connecting, a "Verify wallet" dialog should open. If not, click "Verify" button.
- In the Verify dialog, click the "Account" dropdown and select "Session Key" (with Safe address)
- Click "Verify Wallet" and sign the signature request.

#### 4.b. Create Derive subaccount

The funds in the safe wallet need to be moved to a Derive subaccount in order to deposit/trade.

- In left-nav, click "Subaccounts"
- Click "+ Create Subaccount"
- Enter a name, leave the default Margin Type
- Click "Enable Trading" and sign the signature request in your wallet
- Click "Create Subaccount" and sign the additional request

#### 4.c. "Recover" USDC from safe wallet to subaccount

Now you can move the USDC from the vault's safe wallet to the subaccount for trading.

- In left-nav, click "Recover ⚠️"
- You should see the USDC that was deposited in [step 3](#step-3-make-a-test-deposit-usdc)
- Click "Recover"
- The "Recover Balance" dialog should default to the subaccount from step 4.b. and the full amount of USDC.
- Click "Recover USDC" and sign the signature request in your wallet

#### 4.d. Make a test trade

- In left-nav, click "Trade"
- The "Balances" tab in the bottom pane should now show the recovered USDC balance
- Select "Options" or "Perps" in the top-nav and make a test trade using the "Trade Form" pane on the right
- The "Positions" tab in the bottom pane should now display the open position

**Congratulations 🎉**
