"""Test deposit script for Derive lagoon vault.

Set up environment variables:

    source .local-test.env
    export PRIVATE_KEY="0x..."              # Asset manager private key (used as hot_wallet, needs ETH for gas)
    export DEPOSITOR_PRIVATE_KEY="0x..."    # EOA wallet private key with USDC on Derive (needs ETH for gas)
    export JSON_RPC_DERIVE="https://rpc.derive.xyz"
    export VAULT_ADDRESS="0x0C81AD1825826eECB11E46Cb0C730b1747f07e0B"
    export VAULT_ADAPTER_ADDRESS="0xd7D0fb5F147f6dbe792247104598c677421bbCf5"

Run inside the trade-executor console:

    trade-executor console \
        --strategy-file=strategies/test_only/derive-lagoon-vault.py \
        --asset-management-mode=lagoon

Once in the console, paste the code below.
"""

import os
from decimal import Decimal

from eth_defi.compat import native_datetime_utc_now
from eth_defi.hotwallet import HotWallet
from eth_defi.trace import assert_transaction_success_with_explanation
from tradeexecutor.monkeypatch.web3 import construct_sign_and_send_raw_middleware
from tradeexecutor.state.identifier import AssetIdentifier

# --- Set up depositor wallet ---

depositor_key = os.environ["DEPOSITOR_PRIVATE_KEY"]
depositor_wallet = HotWallet.from_private_key(depositor_key)
depositor_wallet.sync_nonce(web3)

# Register depositor as a signer so transact({"from": depositor}) works
web3.middleware_onion.add(construct_sign_and_send_raw_middleware(depositor_wallet.account))

depositor = depositor_wallet.address
print(f"Depositor: {depositor}")
print(f"Asset manager (hot_wallet): {hot_wallet.address}")

# Ensure asset manager nonce is synced (needed for sign_bound_call_with_new_nonce).
# sync_model has its own HotWallet instance, so sync both.
hot_wallet.sync_nonce(web3)
sync_model.hot_wallet.sync_nonce(web3)

# --- Check balances ---

usdc = vault.underlying_token
usdc_balance = usdc.fetch_balance_of(depositor)
gas_balance = web3.eth.get_balance(depositor) / 10**18
print(f"Depositor USDC balance: {usdc_balance}")
print(f"Depositor ETH balance: {gas_balance}")

# --- Step 1: Approve vault to spend USDC ---

usdc_amount = Decimal(25)
raw_amount = usdc.convert_to_raw(usdc_amount)

print(f"\nApproving {usdc_amount} USDC for vault {vault.vault_address}...")
tx_hash = usdc.approve(vault.vault_address, usdc_amount).transact({"from": depositor})
assert_transaction_success_with_explanation(web3, tx_hash)
print(f"Approve tx: {tx_hash.hex()}")

# --- Step 2: Request deposit (ERC-7540 phase 1) ---

print(f"\nRequesting deposit of {usdc_amount} USDC...")
tx_hash = vault.request_deposit(depositor, raw_amount).transact({"from": depositor})
assert_transaction_success_with_explanation(web3, tx_hash)
print(f"Request deposit tx: {tx_hash.hex()}")

# --- Step 3: Settle (post NAV + settle via trading strategy module) ---
# This must be done by the asset manager

# Initialise the sync model if this is a fresh state
usdc_asset = AssetIdentifier(
    chain_id=web3.eth.chain_id,
    address=vault.underlying_token.address,
    token_symbol="USDC",
    decimals=6,
)
sync_model.sync_initial(state, reserve_asset=usdc_asset, reserve_token_price=1.0)

print(f"\nSettling deposits via asset manager...")
events = sync_model.sync_treasury(native_datetime_utc_now(), state, post_valuation=True)
print(f"Settlement events: {events}")

# --- Step 4: Finalise deposit (ERC-7540 phase 2) ---

print(f"\nFinalising deposit for {depositor}...")
tx_hash = vault.finalise_deposit(depositor).transact({"from": depositor})
assert_transaction_success_with_explanation(web3, tx_hash)
print(f"Finalise deposit tx: {tx_hash.hex()}")

# --- Verify ---

share_balance = vault.vault_contract.functions.balanceOf(depositor).call()
share_decimals = vault.vault_contract.functions.decimals().call()
print(f"\nDepositor share token balance: {share_balance / 10**share_decimals}")
print(f"Done!")
